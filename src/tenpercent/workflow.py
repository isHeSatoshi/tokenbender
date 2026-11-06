from __future__ import annotations

import shutil
from pathlib import Path

from .bundle import archive_bundle, bundle_manifest, load_bundle, pack_model, verify_lineage
from .constants import BASE_PARAMETERS, MODEL_SHA256, NEW_INTENTS, SOURCE_TARGET
from .data import data_commitment, load_feedback, load_split
from .evaluation import evaluate
from .guard import ExamSession
from .io import append_record, read_json, safe_relative, sha256_file, write_json
from .masking import frozen_hash, replace_linears
from .memory import FeedbackMemory, MemoryConfig, build_stream
from .metrics import claim_results
from .model import load_base
from .vocabulary import Vocabulary, trim_model


def pack_candidate(root: Path, candidate: str, *, device="cuda") -> dict:
    from safetensors.torch import load_file

    directory = safe_relative(root / "runs" / "search", candidate)
    config = read_json(directory / "config.json")
    selection = read_json(directory / "selection.json")
    if selection.get("diagnostic_only"):
        raise ValueError("A 30% diagnostic candidate cannot be submitted as a 10% model")
    model = load_base(root / "assets" / "base", device)
    vocabulary_path = root / "runs" / "vocabulary.json"
    vocabulary = Vocabulary.load(vocabulary_path)
    trim_model(model, vocabulary)
    modules = replace_linears(model, config["group_size"])
    if frozen_hash(model) != config["source_frozen_hash"]:
        raise ValueError("Selected candidate source weights changed")
    masks = load_file(directory / "selected_masks.safetensors", device=device)
    if set(masks) != set(modules):
        raise ValueError("Candidate mask graph mismatch")
    for name, module in modules.items():
        if masks[name].dtype.__str__() != "torch.bool" or masks[name].shape != module.hard.shape:
            raise ValueError("Invalid candidate mask")
        module.hard = masks[name]
    entries = sum(int(m.hard.sum()) * m.group_size for m in modules.values())
    entries += config["budget"]["hidden"] * len(vocabulary.retained)
    entries += config["budget"]["fixed_body_entries"]
    if entries != selection["entries"]:
        raise ValueError("Selected candidate size changed")
    destination = root / "runs" / "bundles" / "A"
    labels = read_json(root / "data" / "manifest.json")["labels"]
    manifest = pack_model(
        model,
        root / "assets" / "base",
        vocabulary_path,
        labels,
        destination,
        mode=selection["mode"],
    )
    proof = verify_lineage(destination, root / "assets" / "base" / "model.safetensors")
    write_json(root / "runs" / "A_bit_identity_report.json", proof)
    archive = archive_bundle(destination, destination.with_suffix(".tar.zst"))
    write_json(root / "runs" / "A_size_report.json", {**archive, "retained_entries": entries})
    return {"bundle": str(destination.relative_to(root)), "manifest": manifest, "archive": archive}


def replay_feedback(root: Path, state_a: Path, config_path: Path, *, device="cuda"):
    manifest_a = bundle_manifest(state_a)
    base_runner = load_bundle(state_a, device)
    if hasattr(base_runner, "memory"):
        raise ValueError("This replay requires an A without a training-memory booster")
    config = MemoryConfig(**read_json(config_path))
    labels = base_runner.labels + list(NEW_INTENTS)
    memory = FeedbackMemory(labels, config)
    feedback = load_feedback(root, state_a)
    corrections = [
        row
        for row in feedback
        if base_runner.predict(row["text"], manifest_a["mode"]) != row["label"]
    ]
    teaching = load_split(root, "clinc/train", purpose="memory")
    events = build_stream(teaching, corrections)
    stream_path = root / "runs" / "feedback_stream.jsonl"
    if stream_path.exists():
        raise FileExistsError("Feedback stream already exists; do not silently replace it")
    for event in events:
        append_record(stream_path, event)
    checkpoint_root = root / "runs" / "feedback"
    checkpoint_root.mkdir(exist_ok=False)
    new_count = 0
    for index, event in enumerate(events):
        hidden = base_runner.feature(event["text"], config.layer)
        action = memory.add(event["id"], event["text"], event["label"], hidden)
        append_record(
            checkpoint_root / "events.jsonl",
            {
                "event_index": index,
                "source_id": event["id"],
                "action": action,
                "numeric_entries": memory.numeric_entries,
            },
        )
        if event["kind"] == "new":
            new_count += 1
            if new_count in (12, 60, 120):
                name = {12: "1-shot", 60: "5-shot", 120: "B"}[new_count]
                destination = root / "runs" / "bundles" / name
                shutil.copytree(state_a, destination)
                memory.save(destination / "memory.json")
                manifest = dict(manifest_a)
                manifest["state"] = name
                manifest["auxiliary_entries"] = memory.numeric_entries
                manifest["retained_entries"] = manifest["source_entries"] + memory.numeric_entries
                manifest["feedback_events"] = index + 1
                manifest["files"] = {
                    p.relative_to(destination).as_posix(): sha256_file(p)
                    for p in destination.rglob("*")
                    if p.is_file() and p.name != "manifest.json"
                }
                write_json(destination / "manifest.json", manifest)
                bundle_manifest(destination)
                write_json(
                    checkpoint_root / f"{name}_size.json",
                    archive_bundle(destination, destination.with_suffix(".tar.zst")),
                )
    return {
        "events": len(events),
        "old_corrections": len(corrections),
        "new_examples": new_count,
        "memory_entries": memory.numeric_entries,
    }


def require_released_final_runtime():
    # Do not let an unfinished final path consume the single locked session.
    raise PermissionError(
        "Final evaluation is not released: automatic feedback calibration and "
        "source-isolated candidate workers must be implemented and validated first"
    )


def make_freeze_manifest(root: Path) -> dict:
    require_released_final_runtime()
    states = {}
    files = {}
    for name in ("A", "1-shot", "5-shot", "B"):
        relative = f"runs/bundles/{name}"
        directory = root / relative
        manifest = bundle_manifest(directory)
        states[name] = {
            "path": relative,
            "manifest_sha256": sha256_file(directory / "manifest.json"),
            "retained_entries": manifest["retained_entries"],
            "model_hash": manifest["model_hash"],
        }
        for path in directory.rglob("*"):
            if path.is_file():
                files[path.relative_to(root).as_posix()] = sha256_file(path)
    if len({s["model_hash"] for s in states.values()}) != 1:
        raise ValueError("Feedback changed the frozen source model")
    for relative in (
        "data/manifest.json",
        "runs/feedback_stream.jsonl",
        "runs/environment.json",
        "runs/A_bit_identity_report.json",
        "runs/A_size_report.json",
        "uv.lock",
        "pyproject.toml",
        "PROTOCOL_AMENDMENTS.md",
    ):
        files[relative] = sha256_file(root / relative)
    for path in (root / "src").rglob("*.py"):
        files[path.relative_to(root).as_posix()] = sha256_file(path)
    manifest = {
        "format": 1,
        "states": states,
        "files": files,
        "data_manifest_sha256": sha256_file(root / "data" / "manifest.json"),
        "data_commitment": data_commitment(root),
        "source_sha256": MODEL_SHA256,
        "bootstrap_draws": 10000,
        "crash_policy": "Resume unchanged pending rows only; no code or candidate changes",
    }
    write_json(root / "final_manifest.json", manifest, exclusive=True)
    return manifest


def final_evaluate(root: Path, attestation: Path, *, resume=False, device="cuda"):
    require_released_final_runtime()
    from .model import Runner, TokenizerAdapter, load_tokenizer

    session = ExamSession(root, attestation, resume=resume)
    try:
        old = session.rows("banking/exam")
        new = session.rows("clinc/test")
        labels = read_json(root / "data" / "manifest.json")["labels"]
        output = root / "runs" / "final"
        model = load_base(root / "assets" / "base", device)
        runner = Runner(model, TokenizerAdapter(load_tokenizer(root / "assets" / "base")), labels)
        predictions = {}
        for mode in ("greedy", "constrained"):
            predictions[f"base_{mode}"], _ = evaluate(
                runner,
                old,
                output / f"base_{mode}.jsonl",
                mode=mode,
                state_hash=MODEL_SHA256,
            )
        del runner, model
        import gc

        import torch

        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        counts = {}
        for name in ("A", "1-shot", "5-shot", "B"):
            state = session.manifest["states"][name]
            directory = safe_relative(root, state["path"])
            runner = load_bundle(directory, device)
            manifest = bundle_manifest(directory)
            if name in ("A", "B"):
                predictions[f"{name}_old"], _ = evaluate(
                    runner,
                    old,
                    output / f"{name}_old.jsonl",
                    mode=manifest["mode"],
                    state_hash=state["manifest_sha256"],
                )
                counts[name] = manifest["retained_entries"]
            predictions[f"{name}_new"], _ = evaluate(
                runner,
                new,
                output / f"{name}_new.jsonl",
                mode=manifest["mode"],
                state_hash=state["manifest_sha256"],
            )
            del runner
            gc.collect()
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        result = claim_results(
            predictions["A_old"],
            predictions["B_old"],
            predictions["base_greedy"],
            predictions["base_constrained"],
            predictions["A_new"],
            predictions["B_new"],
            counts,
        )
        write_json(output / "results.json", result)
        session.finish(result)
        return result
    except BaseException:
        session.release_after_error()
        raise


def create_preregistration(root: Path) -> dict:
    """Generate measured local commitment inputs, without pretending they are public."""
    if not (root / "teacher_commitment.json").is_file():
        raise PermissionError("Complete the full teacher cache before preregistration")
    result = {
        "format": 1,
        "source_sha256": MODEL_SHA256,
        "data_commitment": data_commitment(root),
        "files": {},
        "search": {
            "steps": [200, 2000, 4000],
            "diagnostic_steps": [200],
            "learning_rates": [0.001, 0.003, 0.01],
            "accumulations": [32],
            "source_targets": [SOURCE_TARGET, BASE_PARAMETERS * 3 // 10],
            "group_sizes": [4, 2],
            "selection": [
                "working_correct",
                "base_agreement",
                "fewer_entries",
                "lexical_config_hash",
            ],
        },
        "publication_status": "Verified separately by public commit attestation before mask search",
    }
    write_json(
        root / "split_commitment.json", read_json(root / "data" / "manifest.json"), exclusive=True
    )
    for relative in (
        "data/manifest.json",
        "split_commitment.json",
        "runs/source_verification.json",
        "runs/environment.json",
        "runs/vocabulary.json",
        "runs/vocabulary_report.json",
        "runs/adaptation_plan.json",
        "teacher_commitment.json",
        "rules_snapshot.md",
        "runs/identity-770.json",
        "runs/baseline/working-770-greedy.summary.json",
        "runs/baseline/working-770-constrained.summary.json",
        "uv.lock",
        "PROTOCOL_AMENDMENTS.md",
        "playbook/PREREGISTRATION_TEMPLATE.md",
    ):
        result["files"][relative] = sha256_file(root / relative)
    for path in sorted((root / "src").rglob("*.py")):
        result["files"][path.relative_to(root).as_posix()] = sha256_file(path)
    write_json(root / "preregistration.json", result, exclusive=True)
    return result
