from __future__ import annotations

import gc
import shutil
import sys
import tempfile
from pathlib import Path

from .bundle import archive_bundle, bundle_manifest, load_bundle, verify_lineage
from .constants import MODEL_SHA256, NEW_INTENTS
from .data import load_split
from .execution import STATES, inventory, run_frozen_evaluations
from .io import object_hash, read_json, sha256_file, write_json
from .sandbox import IsolatedRunner


def host_runtime() -> dict:
    import torch
    import transformers

    executable = shutil.which("bwrap")
    if not executable:
        raise PermissionError("Source-isolated validation requires bubblewrap")
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bubblewrap_sha256": sha256_file(executable),
    }


def frozen_inputs(root: Path) -> dict:
    paths = set()
    for relative in (
        "src",
        "playbook",
        "runs/bundles",
        "runs/calibration",
        "runs/feedback",
        "runs/standalone/dry-run",
    ):
        directory = root / relative
        if directory.exists():
            for path in directory.rglob("*"):
                if path.is_file() and (
                    relative == "runs/bundles" or path.suffix in {".py", ".md", ".json", ".jsonl"}
                ):
                    paths.add(path)
    for relative in (
        "data/manifest.json",
        "uv.lock",
        "pyproject.toml",
        "PROTOCOL_AMENDMENTS.md",
        "rules_snapshot.md",
        "preregistration.json",
        "teacher_commitment.json",
        "feedback_commitment.json",
        "feedback_stream.jsonl",
        "runs/feedback_stream.jsonl",
        "runs/preregistration_attestation.json",
        "runs/feedback_attestation.json",
        "runs/environment.json",
        "runs/source_verification.json",
        "runs/identity-770.json",
        "runs/A_bit_identity_report.json",
        "runs/A_size_report.json",
        "runs/A_selection.json",
        "runs/adaptation_plan.json",
    ):
        if (root / relative).is_file():
            paths.add(root / relative)
    for directory in (root / "assets/base", root / "runs/baseline"):
        if directory.exists():
            paths.update(path for path in directory.iterdir() if path.is_file())
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in sorted(paths)}


def check_loader(root: Path, directory: Path):
    source = {
        path.relative_to(root / "src/tenpercent").as_posix(): sha256_file(path)
        for path in (root / "src/tenpercent").rglob("*.py")
    }
    bundled = {
        path.relative_to(directory / "loader/tenpercent").as_posix(): sha256_file(path)
        for path in (directory / "loader/tenpercent").rglob("*.py")
    }
    if not source or source != bundled:
        raise PermissionError("Frozen loader differs from the registered evaluation code")


def standalone_validate(root: Path, *, device="cuda", expected_source_sha256=MODEL_SHA256):
    """Development-only proof. This function has no test split or decryption entrypoint."""
    states = inventory(root)
    proof_path = root / "runs/standalone_validation.json"
    runtime = host_runtime()
    if proof_path.exists():
        previous = read_json(proof_path)
        if (
            previous["files"] != frozen_inputs(root)
            or previous["states"] != states
            or previous["runtime"] != runtime
            or previous["device"] != device
            or previous["source_sha256"] != expected_source_sha256
        ):
            raise PermissionError("Existing standalone proof no longer matches the frozen inputs")
        return previous
    old = load_split(root, "banking/working", purpose="memory")
    new = load_split(root, "clinc/validation", purpose="memory")
    train = load_split(root, "banking/train", purpose="memory")
    source = root / "assets/base/model.safetensors"
    reports = {}
    comparison_rows = None
    for name in STATES:
        state = states[name]
        directory = root / state["path"]
        check_loader(root, directory)
        lineage = verify_lineage(directory, source, expected_sha256=expected_source_sha256)
        archive_path = directory.with_suffix(".tar.zst")
        with tempfile.TemporaryDirectory(dir=root / "runs", prefix=".archive-proof-") as temporary:
            fresh = Path(temporary) / "state.tar.zst"
            archive = archive_bundle(directory, fresh)
            if archive_path.exists():
                if sha256_file(archive_path) != archive["sha256"]:
                    raise ValueError(
                        "Stored inference archive differs from the actual standalone bundle"
                    )
            else:
                fresh.replace(archive_path)
        reference = load_bundle(directory, device)
        labels = read_json(root / "data/manifest.json")["labels"] + list(NEW_INTENTS)
        if getattr(reference, "output_labels", reference.labels) != labels:
            raise ValueError("Standalone states must expose the same joint label namespace")
        if comparison_rows is None:
            tokenizer = reference.tokenizer
            longest = max(train, key=lambda row: len(tokenizer.prompt(row["text"])))
            synthetic = [
                {"id": f"synthetic:{index}", "text": text, "label": old[0]["label"]}
                for index, text in enumerate(
                    ("", "हिन्दी 中文 café", "\u0378💳unknown", longest["text"])
                )
            ]
            comparison_rows = old + new + synthetic
        expected = [reference.predict(row["text"], state["mode"]) for row in comparison_rows]
        del reference
        gc.collect()
        if device.startswith("cuda"):
            import torch

            torch.cuda.empty_cache()
        with IsolatedRunner(
            directory,
            state["manifest_sha256"],
            device=device,
            denied_paths=(source, root / ".secrets/exam.key", root / "data/manifest.json"),
        ) as worker:
            actual = [worker.predict(row["text"], state["mode"]) for row in comparison_rows]
            cold = worker.hello
        if actual != expected:
            raise ValueError(
                f"Standalone {name} inference differs from the selected-state reference"
            )
        reports[name] = {
            "lineage": lineage,
            "archive": archive,
            "rows": len(expected),
            "mismatches": 0,
            "prediction_sha256": object_hash(actual),
            "cold": cold,
            "complete": worker.profile,
            "working_correct": sum(actual[i] == row["label"] for i, row in enumerate(old)),
            "new_correct": sum(actual[len(old) + i] == row["label"] for i, row in enumerate(new)),
        }
    synthetic_old = [
        {"id": f"dry-old:{i}", "text": row["text"], "label": row["label"]}
        for i, row in enumerate(comparison_rows[-4:])
    ]
    synthetic_new = [
        {"id": f"dry-new:{label}", "text": f"synthetic example for {label}", "label": label}
        for label in NEW_INTENTS
    ]
    dry = run_frozen_evaluations(
        root,
        states,
        synthetic_old,
        synthetic_new,
        root / "runs/standalone/dry-run",
        device=device,
        source_sha256=expected_source_sha256,
    )
    proof = {
        "format": 1,
        "passed": True,
        "source_sha256": expected_source_sha256,
        "device": device,
        "runtime": runtime,
        "states": states,
        "reports": reports,
        "old_rows": len(old),
        "new_rows": len(new),
        "input_sha256": object_hash(comparison_rows),
        "synthetic_final": {
            "old_rows": dry["A"]["total"],
            "new_rows": dry["new"]["total"],
            "completed": True,
        },
        "files": frozen_inputs(root),
    }
    write_json(proof_path, proof, exclusive=True)
    return proof


def require_final_ready(root: Path, *, device=None) -> dict:
    error = "Final evaluation is not released: validated feedback and source-isolated proof are required"
    try:
        proof = read_json(root / "runs/standalone_validation.json")
        replay = read_json(root / "runs/feedback/replay_report.json")
        if (
            proof["format"] != 1
            or not proof["passed"]
            or proof["source_sha256"] != MODEL_SHA256
            or proof["files"].get("assets/base/model.safetensors") != MODEL_SHA256
            or proof["states"] != inventory(root)
            or proof["files"] != frozen_inputs(root)
            or proof["runtime"] != host_runtime()
            or device is not None
            and proof["device"] != device
            or not proof["synthetic_final"]["completed"]
            or proof["synthetic_final"]["new_rows"] != len(NEW_INTENTS)
        ):
            raise PermissionError(error)
        labels = read_json(root / "data/manifest.json")["labels"]
        if len(labels) != 77 or set(labels) & set(NEW_INTENTS):
            raise PermissionError(error)
        required = {
            "preregistration.json",
            "teacher_commitment.json",
            "feedback_commitment.json",
            "feedback_stream.jsonl",
            "runs/feedback_stream.jsonl",
            "runs/identity-770.json",
            "runs/preregistration_attestation.json",
            "runs/feedback_attestation.json",
        }
        if not required.issubset(proof["files"]):
            raise PermissionError(error)
        for name in STATES:
            report = proof["reports"][name]
            manifest = bundle_manifest(root / proof["states"][name]["path"])
            if (
                report["mismatches"]
                or report["lineage"]["mismatches"]
                or report["lineage"]["checked_source_entries"] != manifest["source_entries"]
                or manifest["source_sha256"] != MODEL_SHA256
                or report["rows"] != proof["old_rows"] + proof["new_rows"] + 4
                or report["cold"]["state_hash"] != proof["states"][name]["manifest_sha256"]
                or report["cold"]["denied_path_checks"] < 3
                or set(report["cold"]["network_interfaces"]) - {"lo"}
            ):
                raise PermissionError(error)
        b = replay["checkpoints"]["B"]
        if (
            not replay["g3_pass"]
            or replay["a_old"]["correct"] < replay["base_bar"]
            or b["old"]["correct"] < max(replay["a_old"]["correct"], replay["base_bar"])
            or b["new"]["correct"] * 5 < b["new"]["total"] * 4
            or b["new"]["correct"] <= replay["a_new"]["correct"]
            or replay["identity"]["state_a"] != proof["states"]["A"]["manifest_sha256"]
            or replay["identity"]["commitment"] != sha256_file(root / "feedback_commitment.json")
            or replay["final_model_hash"] != proof["states"]["B"]["model_hash"]
            or replay["a_old"]["correct"] != proof["reports"]["A"]["working_correct"]
            or replay["a_new"]["correct"] != proof["reports"]["A"]["new_correct"]
            or b["old"]["correct"] != proof["reports"]["B"]["working_correct"]
            or b["new"]["correct"] != proof["reports"]["B"]["new_correct"]
        ):
            raise PermissionError(error)
        return proof
    except (FileNotFoundError, KeyError, ValueError) as exc:
        raise PermissionError(error) from exc
