from __future__ import annotations

import json
from pathlib import Path

from .bundle import archive_bundle, pack_model, verify_lineage
from .constants import BASE_PARAMETERS, MODEL_SHA256, SOURCE_TARGET
from .data import data_commitment
from .guard import ExamSession
from .io import read_json, safe_relative, sha256_file, write_json
from .masking import frozen_hash, replace_linears
from .model import load_base
from .vocabulary import Vocabulary, trim_model


def pack_candidate(root: Path, candidate: str, *, device="cuda") -> dict:
    from safetensors.torch import load_file

    directory = safe_relative(root / "runs" / "search", candidate)
    config = read_json(directory / "config.json")
    selection = read_json(directory / "selection.json")
    if selection.get("diagnostic_only"):
        raise ValueError("A 30% diagnostic candidate cannot be submitted as a 10% model")
    completed = [
        json.loads(line) for line in (root / "runs/candidates.jsonl").read_bytes().splitlines()
    ]
    if not any(
        row["id"] == candidate and row["status"] == "eligible" and row["config"] == config
        for row in completed
    ):
        raise PermissionError("Only a completed registered eligible search can be packed as A")
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
    write_json(
        root / "runs/A_selection.json",
        {
            "candidate": candidate,
            "config": config,
            "selection": selection,
            "config_sha256": sha256_file(directory / "config.json"),
            "selection_sha256": sha256_file(directory / "selection.json"),
            "masks_sha256": sha256_file(directory / "selected_masks.safetensors"),
            "a_manifest_sha256": sha256_file(destination / "manifest.json"),
        },
        exclusive=True,
    )
    return {"bundle": str(destination.relative_to(root)), "manifest": manifest, "archive": archive}


def replay_feedback(
    root: Path,
    state_a: Path,
    config_path: Path,
    calibration_path: Path,
    *,
    device="cuda",
    resume=False,
):
    from .feedback import replay_feedback as replay

    return replay(root, state_a, config_path, calibration_path, device=device, resume=resume)


def require_released_final_runtime(root: Path, *, device=None):
    from .validation import require_final_ready

    return require_final_ready(root, device=device)


def make_freeze_manifest(root: Path) -> dict:
    proof = require_released_final_runtime(root)
    from .calibration import read_commitment

    read_commitment(root, root / "runs/bundles/A")
    files = {
        **proof["files"],
        "runs/standalone_validation.json": sha256_file(root / "runs/standalone_validation.json"),
    }
    manifest = {
        "format": 1,
        "states": proof["states"],
        "device": proof["device"],
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
    proof = require_released_final_runtime(root, device=device)
    from .execution import run_frozen_evaluations

    registration = read_json(attestation)
    frozen = read_json(safe_relative(root, registration["manifest"]))
    expected_files = {
        **proof["files"],
        "runs/standalone_validation.json": sha256_file(root / "runs/standalone_validation.json"),
    }
    if (
        frozen["states"] != proof["states"]
        or frozen["files"] != expected_files
        or frozen["device"] != device
        or frozen["source_sha256"] != MODEL_SHA256
    ):
        raise PermissionError("Final manifest does not match the validated frozen pipeline")
    session = ExamSession(root, attestation, resume=resume)
    try:
        old = session.rows("banking/exam")
        new = session.rows("clinc/test")
        result = run_frozen_evaluations(
            root,
            session.manifest["states"],
            old,
            new,
            root / "runs/final",
            device=device,
            source_sha256=MODEL_SHA256,
        )
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
            "resource_pilot": {
                "selects_candidate": False,
                "optimizer_updates": 3,
                "density_phases": ["all", "midpoint", "target"],
                "inputs": "repeat the longest complete teacher record, scanned over every cached ID",
                "minimum_free_device_fraction": 0.1,
                "includes": ["cold load", "Wanda", "exact top-k", "backward", "Adam allocation"],
            },
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
        "PREREGISTRATION.md",
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
