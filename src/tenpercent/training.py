from __future__ import annotations

import json
import math
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from .accounting import Budget, cubic_group_count
from .constants import MODEL_SHA256, PARAMETER_CAP, SEED, SOURCE_TARGET
from .data import data_commitment, load_split
from .guard import verify_public_commit
from .io import append_record, object_hash, read_json, safe_relative, sha256_file, write_json
from .masking import frozen_hash, replace_linears, set_global_mask
from .model import Runner, load_base, load_tokenizer
from .teacher import TeacherCache, distillation_loss
from .vocabulary import CompactTokenizer, Vocabulary, trim_model


def require_preregistration(root: Path):
    path = root / "runs" / "preregistration_attestation.json"
    if not path.is_file():
        raise PermissionError("Mask search requires a pushed preregistration and attestation")
    attestation = read_json(path)
    verify_public_commit(root, attestation)
    manifest = read_json(safe_relative(root, attestation["manifest"]))
    if manifest.get("data_commitment") != data_commitment(root):
        raise PermissionError("Public preregistration refers to different splits")
    if "teacher_commitment.json" not in manifest["files"]:
        raise PermissionError("Public preregistration must commit the complete teacher cache")
    for relative, digest in manifest["files"].items():
        if sha256_file(safe_relative(root, relative)) != digest:
            raise PermissionError("Preregistered input changed")
    size = len(load_split(root, "banking/working", purpose="baseline"))
    identity = read_json(root / "runs" / f"identity-{size}.json")
    if not all(
        identity.get(key) for key in ("full_working_set", "logits_bit_equal", "predictions_equal")
    ):
        raise PermissionError("Full working-set all-one identity must pass before search")
    return manifest


def validate_search_config(
    registration,
    *,
    steps,
    learning_rate,
    accumulation,
    source_target,
    group_size,
    diagnostic_oversize,
):
    grid = registration.get("search")
    if not grid or any(
        (
            steps not in grid["steps"],
            learning_rate not in grid["learning_rates"],
            accumulation not in grid["accumulations"],
            source_target not in grid["source_targets"],
            group_size not in grid["group_sizes"],
            diagnostic_oversize != (source_target > PARAMETER_CAP),
            diagnostic_oversize and steps not in grid["diagnostic_steps"],
        )
    ):
        raise PermissionError("Search configuration is outside the public preregistration")


def save_checkpoint(
    path: Path,
    modules,
    optimizer,
    *,
    config_hash: str,
    step: int,
    order: np.ndarray,
    cursor: int,
    rng,
    best_rank: tuple,
):
    temporary = path.with_suffix(".tmp")
    payload = {
        "config_hash": config_hash,
        "step": step,
        "cursor": cursor,
        "order": torch.from_numpy(order),
        "rng_json": json.dumps(rng.bit_generator.state),
        "best_rank": best_rank,
        "scores": {n: m.scores.detach().cpu() for n, m in modules.items()},
        "optimizer": optimizer.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }
    torch.save(payload, temporary)
    os.replace(temporary, path)


def restore_checkpoint(path: Path, modules, optimizer, *, config_hash: str, device: str):
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload["config_hash"] != config_hash or set(payload["scores"]) != set(modules):
        raise ValueError("Resume checkpoint does not match this candidate")
    with torch.no_grad():
        for name, module in modules.items():
            if module.scores.shape != payload["scores"][name].shape:
                raise ValueError("Resume score shape mismatch")
            module.scores.copy_(payload["scores"][name])
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng"].cpu())
    if payload["cuda_rng"]:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload["cuda_rng"]])
    rng = np.random.default_rng()
    rng.bit_generator.state = json.loads(payload["rng_json"])
    return (
        payload["step"],
        payload["order"].cpu().numpy(),
        payload["cursor"],
        rng,
        tuple(payload["best_rank"]),
    )


@torch.no_grad()
def wanda_initialize(model, modules, adapter, rows, calibration_count=256):
    accumulator = {}
    hooks = []
    for name, module in modules.items():
        accumulator[name] = torch.zeros(module.in_features, device=module.weight.device)

        def hook(_module, args, key=name):
            x = args[0].detach().reshape(-1, args[0].shape[-1]).float()
            accumulator[key].add_(x.square().sum(0))

        hooks.append(module.register_forward_pre_hook(hook))
    ordered = sorted(rows, key=lambda r: object_hash({"seed": SEED, "id": r["id"]}))
    device = next(model.parameters()).device
    try:
        for row in ordered[:calibration_count]:
            ids = torch.tensor([adapter.prompt(row["text"])], device=device)
            model.model(ids, use_cache=False)
    finally:
        for hook in hooks:
            hook.remove()
    for name, module in modules.items():
        module.initialize(accumulator[name].sqrt())


def require_resource_pilot(root: Path, config: dict, device: str):
    for path in sorted((root / "runs/pilots").glob("*/resource_report.json")):
        report = read_json(path)
        pilot = read_json(path.parent / "config.json")
        if (
            report["headroom_pass"]
            and report["source_unchanged"]
            and report["optimizer_updates"] == 3
            and report["device"] == device
            and report["teacher_examples_scanned"] == config["teacher_examples"]
            and report["keep_group_phases"][0] == config["budget"]["total_groups"]
            and report["keep_group_phases"][-1] <= config["budget"]["keep_groups"]
            and all(
                pilot[key] == config[key]
                for key in (
                    "registration",
                    "teacher",
                    "source_frozen_hash",
                    "group_size",
                    "accumulation",
                )
            )
        ):
            return
    raise PermissionError("A matching fresh worst-input resource pilot must pass before search")


def run_search(
    root: Path,
    *,
    steps=2000,
    learning_rate=0.003,
    accumulation=32,
    source_target=SOURCE_TARGET,
    group_size=4,
    device="cuda",
    diagnostic_oversize=False,
    resume=False,
    resource_pilot=False,
):
    cold_started = time.perf_counter()
    if resource_pilot and resume:
        raise ValueError("A resource pilot must start in a fresh process, not resume")
    registration = require_preregistration(root)
    validate_search_config(
        registration,
        steps=steps,
        learning_rate=learning_rate,
        accumulation=accumulation,
        source_target=source_target,
        group_size=group_size,
        diagnostic_oversize=diagnostic_oversize,
    )
    if steps < 1 or accumulation < 1 or learning_rate <= 0:
        raise ValueError("Invalid optimizer configuration")
    vocabulary_path = root / "runs" / "vocabulary.json"
    vocabulary = Vocabulary.load(vocabulary_path)
    cache = TeacherCache(root / "runs" / "teacher.sqlite", readonly=True)
    if (
        cache.metadata["vocabulary_sha256"] != sha256_file(vocabulary_path)
        or cache.metadata["data_commitment"] != data_commitment(root)
        or cache.metadata.get("smoke_only")
        or cache.metadata.get("format") != 2
        or cache.metadata.get("source_sha256") != MODEL_SHA256
        or cache.metadata.get("temperature") != 2
        or cache.metadata.get("topk") != 64
    ):
        raise ValueError("Teacher cache is not eligible for registered search")
    if len(cache.ids()) != cache.metadata["examples"]:
        raise ValueError("Teacher cache is incomplete")
    teacher_commitment = read_json(root / "teacher_commitment.json")
    if teacher_commitment["metadata"] != cache.metadata or teacher_commitment[
        "sha256"
    ] != sha256_file(root / "runs/teacher.sqlite"):
        raise ValueError("Teacher cache differs from the public commitment")
    labels = read_json(root / "data" / "manifest.json")["labels"]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    model = load_base(root / "assets" / "base", device)
    trim_model(model, vocabulary)
    adapter = CompactTokenizer(load_tokenizer(root / "assets" / "base"), vocabulary)
    modules = replace_linears(model, group_size)
    budget = Budget(
        len(vocabulary.retained), group_size, source_target, diagnostic_oversize=diagnostic_oversize
    )
    if sum(module.scores.numel() for module in modules.values()) != budget.total_groups:
        raise ValueError("Maskable graph differs from budget")
    train_rows = load_split(root, "banking/train", purpose="search")
    wanda_initialize(model, modules, adapter, train_rows)
    original_hash = frozen_hash(model)
    config = {
        "registration": object_hash(registration),
        "steps": steps,
        "learning_rate": learning_rate,
        "accumulation": accumulation,
        "group_size": group_size,
        "budget": budget.report(),
        "teacher": object_hash(cache.metadata),
        "teacher_examples": cache.metadata["examples"],
        "source_frozen_hash": original_hash,
    }
    if resource_pilot:
        config["purpose"] = "non-selecting-resource-pilot"
    else:
        require_resource_pilot(root, config, device)
    candidate_id = object_hash(config)[:16]
    output = root / "runs" / ("pilots" if resource_pilot else "search") / candidate_id
    output.mkdir(parents=True, exist_ok=resume)
    if resume:
        if read_json(output / "config.json") != config:
            raise ValueError("Resume configuration changed")
    else:
        write_json(output / "config.json", config)
    parameters = [module.scores for module in modules.values()]
    optimizer = torch.optim.Adam(
        parameters, lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, foreach=False
    )
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    model.train()
    ids = cache.ids()
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(ids))
    max_tokens = None
    if resource_pilot:
        longest_index, max_tokens = max(
            ((i, len(cache.get(key)["input_ids"])) for i, key in enumerate(ids)),
            key=lambda item: (item[1], -item[0]),
        )
        order = np.full(accumulation, longest_index, dtype=np.int64)
    cursor = 0
    runner = Runner(model, adapter, labels)
    working = load_split(root, "banking/working", purpose="search")
    base_records = {}
    for mode in ("greedy", "constrained"):
        path = root / "runs" / "baseline" / f"working-{len(working)}-{mode}.jsonl"
        base_records[mode] = {
            r["example_id"]: r
            for r in (json.loads(line) for line in path.read_bytes().splitlines())
        }
        if set(base_records[mode]) != {r["id"] for r in working}:
            raise ValueError("Working baseline coverage differs from this candidate")
    winning_mode = max(
        ("greedy", "constrained"),
        key=lambda mode: sum(
            r["true_label"] == r["predicted_label"] for r in base_records[mode].values()
        ),
    )
    best_rank = (-1, -1, -budget.source_entries)
    start_step = 0
    if resume:
        start_step, order, cursor, rng, best_rank = restore_checkpoint(
            output / "resume.pt",
            modules,
            optimizer,
            config_hash=object_hash(config),
            device=device,
        )
    start = time.perf_counter()
    phases = (
        budget.total_groups,
        (budget.total_groups + budget.keep_groups) // 2,
        budget.keep_groups,
    )
    for step in range(start_step, 3 if resource_pilot else steps):
        count = (
            phases[step]
            if resource_pilot
            else cubic_group_count(step, steps, budget.total_groups, budget.keep_groups)
        )
        set_global_mask(modules, count)
        optimizer.zero_grad(set_to_none=True)
        progress = (step + 1) / steps
        lr_scale = min(1.0, progress / 0.05)
        if progress > 0.05:
            lr_scale *= 0.5 * (1 + math.cos(math.pi * (progress - 0.05) / 0.95))
        optimizer.param_groups[0]["lr"] = learning_rate * lr_scale
        total_loss = 0
        for _ in range(accumulation):
            if cursor == len(order):
                if not resource_pilot:
                    order = rng.permutation(len(ids))
                cursor = 0
            record = {k: v.to(device) for k, v in cache.get(ids[int(order[cursor])]).items()}
            cursor += 1
            hidden = model.model(
                record["input_ids"][None], use_cache=False, return_dict=True
            ).last_hidden_state[0]
            logits = model.lm_head(hidden[record["positions"]])
            loss = distillation_loss(
                logits,
                record["top_ids"],
                record["top_probabilities"],
                record["tail_probability"],
                record["targets"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite mask-search loss")
            (loss / accumulation).backward()
            total_loss += float(loss.detach()) / accumulation
        grad = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        append_record(
            output / "progress.jsonl",
            {
                "step": step + 1,
                "loss": total_loss,
                "score_grad_norm": float(grad),
                "keep_groups": count,
                "elapsed_seconds": time.perf_counter() - start,
                "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
                "cuda_peak_bytes": torch.cuda.max_memory_allocated()
                if device.startswith("cuda")
                else 0,
            },
        )
        if resource_pilot or (step + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "event": "mask_search",
                        "candidate": candidate_id,
                        "step": step + 1,
                        "steps": steps,
                        "loss": total_loss,
                    }
                ),
                flush=True,
            )
        if not resource_pilot and ((step + 1) % 100 == 0 or step + 1 == steps):
            model.eval()
            set_global_mask(
                modules,
                cubic_group_count(
                    step + 1,
                    steps,
                    budget.total_groups,
                    budget.keep_groups,
                ),
            )
            eligible = sum(int(m.hard.sum()) * group_size for m in modules.values())
            eligible += budget.dense_entries
            for mode in ("greedy", "constrained"):
                predictions = [runner.predict(row["text"], mode) for row in working]
                correct = sum(
                    pred == row["label"] for pred, row in zip(predictions, working, strict=True)
                )
                agreement = sum(
                    pred == base_records[winning_mode][row["id"]]["predicted_label"]
                    for pred, row in zip(predictions, working, strict=True)
                )
                evaluation = {
                    "step": step + 1,
                    "correct": correct,
                    "total": len(working),
                    "agreement": agreement,
                    "mode": mode,
                    "entries": eligible,
                    "config": candidate_id,
                    "diagnostic_only": diagnostic_oversize,
                }
                append_record(output / "working_evaluations.jsonl", evaluation)
                rank = (correct, agreement, -eligible)
                under_limit = eligible <= source_target and (
                    eligible <= PARAMETER_CAP or diagnostic_oversize
                )
                if under_limit and rank > best_rank:
                    best_rank = rank
                    masks = {n: m.hard.cpu().contiguous() for n, m in modules.items()}
                    save_file(masks, output / "selected_masks.safetensors")
                    write_json(output / "selection.json", evaluation)
            save_checkpoint(
                output / "resume.pt",
                modules,
                optimizer,
                config_hash=object_hash(config),
                step=step + 1,
                order=order,
                cursor=cursor,
                rng=rng,
                best_rank=best_rank,
            )
            model.train()
    if frozen_hash(model) != original_hash:
        raise RuntimeError("Source model values changed during score optimization")
    cache.close()
    if resource_pilot:
        reserved = torch.cuda.max_memory_reserved() if device.startswith("cuda") else 0
        total = (
            torch.cuda.get_device_properties(device).total_memory
            if device.startswith("cuda")
            else 0
        )
        report = {
            "format": 1,
            "purpose": "resource-pilot-only",
            "device": device,
            "optimizer_updates": 3,
            "accumulation": accumulation,
            "teacher_examples_scanned": len(ids),
            "max_input_tokens": max_tokens,
            "keep_group_phases": list(phases),
            "source_unchanged": True,
            "cold_elapsed_seconds": time.perf_counter() - cold_started,
            "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated() if total else 0,
            "cuda_peak_reserved_bytes": reserved,
            "cuda_total_bytes": total,
            "headroom_pass": not total or reserved * 10 <= total * 9,
        }
        write_json(output / "resource_report.json", report)
        return {"resource_pilot": candidate_id, "report": report}
    append_record(
        root / "runs" / "candidates.jsonl",
        {
            "id": candidate_id,
            "config": config,
            "working_correct": best_rank[0],
            "status": "diagnostic-only"
            if diagnostic_oversize
            else ("eligible" if best_rank[0] >= 0 else "no-eligible-checkpoint"),
        },
    )
    return {
        "candidate": candidate_id,
        "best_working_correct": best_rank[0],
        "working_total": len(working),
        "diagnostic_only": diagnostic_oversize,
    }
