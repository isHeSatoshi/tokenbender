from __future__ import annotations

import time
from pathlib import Path

from .bundle import bundle_manifest
from .constants import NEW_INTENTS
from .evaluation import evaluate
from .io import read_json, safe_relative, sha256_file, write_json
from .metrics import claim_results, summarize
from .sandbox import IsolatedRunner

STATES = ("A", "1-shot", "5-shot", "B")


def inventory(root: Path) -> dict:
    result = {}
    for name in STATES:
        relative = f"runs/bundles/{name}"
        directory = root / relative
        manifest = bundle_manifest(directory)
        if manifest["state"] != name:
            raise ValueError("Packaged state name differs from its registered checkpoint")
        result[name] = {
            "path": relative,
            "manifest_sha256": sha256_file(directory / "manifest.json"),
            "retained_entries": manifest["retained_entries"],
            "model_hash": manifest["model_hash"],
            "mode": manifest["mode"],
        }
    if len({state["model_hash"] for state in result.values()}) != 1:
        raise ValueError("Feedback changed the frozen source model")
    if len({state["mode"] for state in result.values()}) != 1:
        raise ValueError("Feedback changed the frozen decoding mode")
    return result


def run_frozen_evaluations(
    root: Path,
    states: dict,
    old: list[dict],
    new: list[dict],
    output: Path,
    *,
    device: str,
    source_sha256: str,
) -> dict:
    """Shared by the synthetic dry run and the guarded exam, with no model in this process."""
    if set(states) != set(STATES):
        raise ValueError("Final session needs exactly the four registered states")
    labels = read_json(root / "data/manifest.json")["labels"]
    predictions, counts = {}, {}
    original = root / "assets/base"
    loader = safe_relative(root, states["A"]["path"]) / "loader"
    with IsolatedRunner(
        original,
        source_sha256,
        device=device,
        source=True,
        loader=loader,
        labels=labels,
        denied_paths=(root / "data/manifest.json",),
    ) as worker:
        for mode in ("greedy", "constrained"):
            key = f"base_{mode}"
            predictions[key], _ = evaluate(
                worker, old, output / f"{key}.jsonl", mode=mode, state_hash=source_sha256
            )
    write_json(
        output / f"base-resource-{time.time_ns()}.json",
        {"cold": worker.hello, "complete": worker.profile},
        exclusive=True,
    )
    for name in STATES:
        state = states[name]
        directory = safe_relative(root, state["path"])
        manifest = bundle_manifest(directory)
        if (
            sha256_file(directory / "manifest.json") != state["manifest_sha256"]
            or manifest["mode"] != state["mode"]
        ):
            raise PermissionError("Frozen state changed before worker launch")
        with IsolatedRunner(
            directory,
            state["manifest_sha256"],
            device=device,
            denied_paths=(
                original / "model.safetensors",
                root / ".secrets/exam.key",
                root / "data/manifest.json",
            ),
        ) as worker:
            if worker.output_labels != labels + list(NEW_INTENTS):
                raise ValueError("Candidate does not expose the shared old/new label interface")
            if name in ("A", "B"):
                predictions[f"{name}_old"], _ = evaluate(
                    worker,
                    old,
                    output / f"{name}_old.jsonl",
                    mode=state["mode"],
                    state_hash=state["manifest_sha256"],
                )
                counts[name] = state["retained_entries"]
            predictions[f"{name}_new"], _ = evaluate(
                worker,
                new,
                output / f"{name}_new.jsonl",
                mode=state["mode"],
                state_hash=state["manifest_sha256"],
            )
        write_json(
            output / f"{name}-resource-{time.time_ns()}.json",
            {"cold": worker.hello, "complete": worker.profile},
            exclusive=True,
        )
    result = claim_results(
        predictions["A_old"],
        predictions["B_old"],
        predictions["base_greedy"],
        predictions["base_constrained"],
        predictions["A_new"],
        predictions["B_new"],
        counts,
    )
    joint = labels + list(NEW_INTENTS)
    result["shot_checkpoints"] = {
        name: summarize(predictions[f"{name}_new"], joint) for name in STATES
    }
    result["mixed"] = {
        name: summarize(predictions[f"{name}_old"] + predictions[f"{name}_new"], joint)
        for name in ("A", "B")
    }
    write_json(output / "results.json", result)
    return result
