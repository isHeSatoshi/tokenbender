from __future__ import annotations

import gc
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import numpy as np

from .bundle import archive_bundle, bundle_manifest, load_bundle
from .calibration import load_events, macro_fraction, read_commitment, write_identical
from .constants import NEW_INTENTS, PARAMETER_CAP
from .data import load_split
from .evaluation import evaluate
from .io import (
    append_record,
    atomic_bytes,
    canonical,
    object_hash,
    read_json,
    sha256_file,
    write_json,
)
from .memory import FeedbackMemory, MemoryConfig, MemoryRunner


def pid_alive(pid):
    if type(pid) is not int or pid <= 0:
        raise PermissionError("Invalid replay lock owner")
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


class ReplayLock:
    def __init__(self, path: Path, identity: dict, resume: bool):
        self.path, self.identity, self.resume = path, identity, resume
        self.payload = {"pid": os.getpid(), "identity": object_hash(identity)}
        self.recovered = False

    def __enter__(self):
        if self.path.exists():
            old = read_json(self.path)
            if (
                not self.resume
                or old["identity"] != self.payload["identity"]
                or pid_alive(old["pid"])
            ):
                raise PermissionError("Feedback replay is active or has a different identity")
            self.path.unlink()
            self.recovered = True
        write_json(self.path, self.payload, exclusive=True)
        return self

    def __exit__(self, *args):
        if read_json(self.path) != self.payload:
            raise PermissionError("Feedback replay lock ownership changed")
        self.path.unlink()


def replay_events(
    events,
    runner,
    initial: FeedbackMemory,
    directory: Path,
    identity: dict,
    *,
    mode,
    source_entries,
    resume=False,
    on_checkpoint=None,
    after_save=None,
):
    directory.mkdir(parents=True, exist_ok=True)
    state_path, journal_path = directory / "replay_state.json", directory / "events.jsonl"
    replay_id = object_hash(identity)
    if len({e["id"] for e in events}) != len(events):
        raise ValueError("Duplicate source IDs in feedback stream")
    with ReplayLock(directory / "replay.lock", identity, resume) as lock:
        if state_path.exists():
            if not resume:
                raise FileExistsError("Replay already exists; use identical --resume")
            state = read_json(state_path)
            if state["identity"] != identity or state["format"] != 1:
                raise PermissionError("Replay checkpoint belongs to a different frozen run")
            memory = FeedbackMemory.from_payload(state["memory"])
            cursor = state["next_event"]
            if (
                type(cursor) is not int
                or not 0 <= cursor <= len(events)
                or memory.labels != initial.labels
                or memory.config != initial.config
            ):
                raise ValueError("Replay checkpoint cursor or memory configuration differs")
        else:
            if resume and not (lock.recovered and not journal_path.exists()):
                raise PermissionError("No matching replay checkpoint to resume")
            memory, cursor = initial.clone(), 0
            write_json(
                state_path,
                {"format": 1, "identity": identity, "next_event": 0, "memory": memory.payload()},
                exclusive=True,
            )
        raw = journal_path.read_bytes() if journal_path.exists() else b""
        complete, partial = (
            (raw, b"")
            if not raw or raw.endswith(b"\n")
            else (raw[: raw.rfind(b"\n") + 1], raw[raw.rfind(b"\n") + 1 :])
        )
        records = [json.loads(line) for line in complete.splitlines()]
        if len(records) not in (cursor, cursor + 1) or (partial and len(records) != cursor):
            raise ValueError("Replay journal does not match the atomic checkpoint")
        for index, record in enumerate(records):
            event = events[index]
            if (
                record["event_index"] != index
                or record["replay"] != replay_id
                or record["source_id"] != event["id"]
                or record["kind"] != event["kind"]
                or record["true_label"] != event["label"]
            ):
                raise PermissionError("Replay journal differs from the frozen stream")
            expected_before = (
                object_hash(initial.payload()) if index == 0 else records[index - 1]["after_memory"]
            )
            if record["before_memory"] != expected_before:
                raise ValueError("Replay journal memory lineage is broken")
        expected_memory = (
            object_hash(initial.payload()) if cursor == 0 else records[cursor - 1]["after_memory"]
        )
        if object_hash(memory.payload()) != expected_memory:
            raise ValueError("Replay memory checkpoint differs from its journal")
        if source_entries + memory.numeric_entries > PARAMETER_CAP:
            raise ValueError("Replay starts over the retained-entry cap")
        new_count = sum(e["kind"] == "new" for e in events[:cursor])
        for index in range(cursor, len(events)):
            event = events[index]
            if event["kind"] not in ("new", "correction"):
                raise ValueError("Unknown feedback event kind")
            before = object_hash(memory.payload())
            prediction = MemoryRunner(runner, memory, mode).predict(event["text"])
            hidden = runner.feature(event["text"], memory.config.layer)
            action = memory.add(event["id"], event["text"], event["label"], hidden)
            if source_entries + memory.numeric_entries > PARAMETER_CAP:
                raise ValueError("Feedback would exceed the retained-entry cap")
            record = {
                "replay": replay_id,
                "event_index": index,
                "source_id": event["id"],
                "kind": event["kind"],
                "prediction_before_reveal": prediction,
                "true_label": event["label"],
                "action": action,
                "numeric_entries": memory.numeric_entries,
                "before_memory": before,
                "after_memory": object_hash(memory.payload()),
            }
            if index < len(records):
                if records[index] != record:
                    raise ValueError("Recomputed pending feedback event differs")
            elif partial:
                if not canonical(record).startswith(partial):
                    raise ValueError("Partial feedback journal differs from the pending event")
                atomic_bytes(journal_path, complete + canonical(record))
                partial = b""
                records.append(record)
            else:
                append_record(journal_path, record)
                records.append(record)
            if event["kind"] == "new":
                new_count += 1
                if on_checkpoint and new_count in (12, 60, 120):
                    on_checkpoint(
                        {12: "1-shot", 60: "5-shot", 120: "B"}[new_count], memory.clone(), index + 1
                    )
            next_kind = events[index + 1]["kind"] if index + 1 < len(events) else None
            corrections_done = (
                new_count == 12
                and next_kind != "correction"
                and (
                    event["kind"] == "correction" or (event["kind"] == "new" and next_kind == "new")
                )
            )
            if corrections_done and on_checkpoint:
                on_checkpoint("after-corrections", memory.clone(), index + 1)
            write_json(
                state_path,
                {
                    "format": 1,
                    "identity": identity,
                    "next_event": index + 1,
                    "memory": memory.payload(),
                },
            )
            if after_save:
                after_save(index + 1)
        return memory


def save_snapshot(root: Path, state_a: Path, name: str, memory: FeedbackMemory, count: int):
    manifest_a = bundle_manifest(state_a)
    destination = root / "runs/bundles" / name
    if destination.exists():
        manifest = bundle_manifest(destination)
        if (
            manifest["state"] != name
            or manifest["model_hash"] != manifest_a["model_hash"]
            or manifest["feedback_events"] != count
            or (destination / "memory.json").read_bytes() != canonical(memory.payload())
        ):
            raise PermissionError("Existing feedback snapshot differs")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "bundle"
            shutil.copytree(state_a, staging)
            memory.save(staging / "memory.json")
            manifest = {
                **manifest_a,
                "state": name,
                "feedback_events": count,
                "auxiliary_entries": memory.numeric_entries,
                "retained_entries": manifest_a["source_entries"] + memory.numeric_entries,
            }
            manifest["files"] = {
                p.relative_to(staging).as_posix(): sha256_file(p)
                for p in staging.rglob("*")
                if p.is_file() and p.relative_to(staging).as_posix() != "manifest.json"
            }
            write_json(staging / "manifest.json", manifest)
            bundle_manifest(staging)
            os.rename(staging, destination)
    archive_path = destination.with_suffix(".tar.zst")
    with tempfile.TemporaryDirectory(prefix=".archive-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle.tar.zst"
        archive = archive_bundle(destination, staging)
        if archive_path.exists():
            if sha256_file(archive_path) != archive["sha256"]:
                raise PermissionError("Existing feedback archive differs")
        else:
            os.rename(staging, archive_path)
    write_identical(root / f"runs/feedback/{name}_size.json", canonical(archive))
    return destination


def checkpoint_metrics(records, labels):
    truth = np.asarray([labels.index(r["true_label"]) for r in records])
    prediction = np.asarray(
        [
            labels.index(r["predicted_label"])
            if r["predicted_label"] in labels and r["valid"]
            else -1
            for r in records
        ]
    )
    result = {
        "correct": int((truth == prediction).sum()),
        "total": len(records),
        "false_new_intents": sum(
            r["predicted_label"] in NEW_INTENTS and r["true_label"] not in NEW_INTENTS
            for r in records
        ),
    }
    if all(r["true_label"] in NEW_INTENTS for r in records):
        macro = macro_fraction(prediction, truth, [labels.index(label) for label in NEW_INTENTS])
        result["macro_accuracy"] = float(macro)
        result["macro_fraction"] = str(macro)
    return result


def replay_feedback(
    root: Path,
    state_a: Path,
    config_path: Path,
    calibration_path: Path,
    *,
    device="cuda",
    resume=False,
):
    commitment = read_commitment(root, state_a)
    calibration = read_json(calibration_path)
    config = MemoryConfig(**read_json(config_path))
    identity = {
        "state_a": sha256_file(state_a / "manifest.json"),
        "commitment": sha256_file(root / "feedback_commitment.json"),
    }
    if (
        calibration["identity"] != identity
        or not calibration["selected"]
        or calibration["selected"]["config"] != asdict(config)
    ):
        raise PermissionError("Replay configuration was not selected by this frozen calibration")
    identity["config"] = object_hash(asdict(config))
    manifest_a = bundle_manifest(state_a)
    runner = load_bundle(state_a, device)
    if hasattr(runner, "memory"):
        raise ValueError("This replay requires A without a training-memory booster")
    labels = runner.labels + list(NEW_INTENTS)
    memory = FeedbackMemory(labels, config)
    events = load_events(root, commitment)
    working = load_split(root, "banking/working", purpose="memory")
    validation = load_split(root, "clinc/validation", purpose="memory")
    directory = root / "runs/feedback"
    old_a, _ = evaluate(
        runner,
        working,
        directory / "A_old.jsonl",
        mode=manifest_a["mode"],
        state_hash=identity["state_a"],
    )
    new_a, _ = evaluate(
        runner,
        validation,
        directory / "A_new.jsonl",
        mode=manifest_a["mode"],
        state_hash=identity["state_a"],
    )
    metrics_path = directory / "checkpoint_metrics.json"
    metrics = read_json(metrics_path) if metrics_path.exists() else {}

    def measure(name, current, count):
        if name == "after-corrections":
            path = directory / "after-corrections.json"
            write_identical(path, canonical(current.payload()))
            reloaded = MemoryRunner(runner, FeedbackMemory.load(path), manifest_a["mode"])
            state_hash = object_hash({"A": identity["state_a"], "memory": current.payload()})
        else:
            path = save_snapshot(root, state_a, name, current, count)
            reloaded = load_bundle(path, device)
            state_hash = sha256_file(path / "manifest.json")
        old_records, _ = evaluate(
            reloaded,
            working,
            directory / f"{name}_old.jsonl",
            mode=manifest_a["mode"],
            state_hash=state_hash,
        )
        value = {
            "old": checkpoint_metrics(old_records, labels),
            "memory_entries": current.numeric_entries,
            "corrections_stored": sum(
                label not in NEW_INTENTS for label in (labels[i] for i in current.label_ids)
            ),
            "quarantined": len(current.quarantined),
            "events": count,
        }
        if name != "after-corrections":
            new_records, _ = evaluate(
                reloaded,
                validation,
                directory / f"{name}_new.jsonl",
                mode=manifest_a["mode"],
                state_hash=state_hash,
            )
            value["new"] = checkpoint_metrics(new_records, labels)
        if name in metrics and metrics[name] != value:
            raise ValueError("Recomputed checkpoint metrics changed")
        metrics[name] = value
        write_json(metrics_path, metrics)
        del reloaded
        gc.collect()

    final = replay_events(
        events,
        runner,
        memory,
        directory,
        identity,
        mode=manifest_a["mode"],
        source_entries=manifest_a["source_entries"],
        resume=resume,
        on_checkpoint=measure,
    )
    a_old, a_new = checkpoint_metrics(old_a, labels), checkpoint_metrics(new_a, labels)
    base_bar = max(
        read_json(root / f"runs/baseline/working-{len(working)}-{mode}.summary.json")["correct"]
        for mode in ("greedy", "constrained")
    )
    b = metrics["B"]
    passed = (
        b["old"]["correct"] >= max(a_old["correct"], base_bar)
        and b["new"]["correct"] * 5 >= 4 * b["new"]["total"]
        and Fraction(b["new"]["macro_fraction"]) >= Fraction(4, 5)
        and b["new"]["correct"] > a_new["correct"]
        and manifest_a["source_entries"] + final.numeric_entries <= PARAMETER_CAP
    )
    report = {
        "format": 1,
        "identity": identity,
        "a_old": a_old,
        "a_new": a_new,
        "base_bar": base_bar,
        "checkpoints": metrics,
        "g3_pass": bool(passed),
        "final_memory_entries": final.numeric_entries,
        "final_model_hash": bundle_manifest(root / "runs/bundles/B")["model_hash"],
        "a_model_hash": manifest_a["model_hash"],
    }
    write_identical(directory / "replay_report.json", canonical(report))
    return report
