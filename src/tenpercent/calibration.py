from __future__ import annotations

import hashlib
import itertools
import json
import sqlite3
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from .bundle import bundle_manifest, load_bundle
from .constants import NEW_INTENTS, PARAMETER_CAP
from .data import data_commitment, load_feedback, load_split
from .evaluation import evaluate
from .guard import verify_public_commit
from .io import (
    atomic_bytes,
    canonical,
    jsonl_bytes,
    object_hash,
    read_json,
    safe_relative,
    sha256_file,
)
from .memory import FeedbackMemory, MemoryConfig, build_stream


def config_grid() -> list[MemoryConfig]:
    return [
        MemoryConfig(layer=layer, neighbors=k, radius=radius, blend=blend, temperature=tau)
        for layer, k, radius, blend, tau in itertools.product(
            (7, 14, 21, 28), (1, 3, 5), (0.05, 0.10, 0.20, 0.30), (0.5, 0.75, 1.0), (0.05, 0.10)
        )
    ]


def write_identical(path: Path, payload: bytes):
    if path.exists():
        if path.read_bytes() != payload:
            raise PermissionError("An existing feedback commitment or stream differs")
    else:
        atomic_bytes(path, payload, exclusive=True)


def prepare_feedback(root: Path, state_a: Path, *, device="cuda"):
    manifest = bundle_manifest(state_a)
    if manifest["state"] != "A" or manifest["feedback_events"]:
        raise PermissionError("Feedback requires an immutable pre-feedback A")
    runner = load_bundle(state_a, device)
    if hasattr(runner, "memory"):
        raise ValueError(
            "Feedback calibration currently requires A without a training-memory booster"
        )
    feedback = load_feedback(root, state_a)
    state_hash = sha256_file(state_a / "manifest.json")
    review_path = root / "runs/feedback_review.jsonl"
    records, _ = evaluate(
        runner, feedback, review_path, mode=manifest["mode"], state_hash=state_hash
    )
    correction_ids = {r["example_id"] for r in records if r["predicted_label"] != r["true_label"]}
    corrections = [row for row in feedback if row["id"] in correction_ids]
    events = build_stream(load_split(root, "clinc/train", purpose="memory"), corrections)
    public = []
    for event in events:
        item = {key: value for key, value in event.items() if key != "text"}
        item["text_sha256"] = hashlib.sha256(event["text"].encode()).hexdigest()
        public.append(item)
    write_identical(root / "feedback_stream.jsonl", jsonl_bytes(public))
    write_identical(root / "runs/feedback_stream.jsonl", jsonl_bytes(events))
    commitment = {
        "format": 1,
        "state_a_manifest_sha256": state_hash,
        "data_commitment": data_commitment(root),
        "events": len(events),
        "old_reviewed": len(feedback),
        "old_corrections": len(corrections),
        "new_examples": 120,
        "grid": [asdict(c) for c in config_grid()],
        "feature_location": "hidden_states[layer], last prompt token; layer 28 is post-final-RMSNorm",
        "full_label_batch_size": 8,
        "files": {},
    }
    for relative in (
        "feedback_stream.jsonl",
        "runs/feedback_stream.jsonl",
        "runs/feedback_review.jsonl",
        "data/manifest.json",
    ):
        commitment["files"][relative] = sha256_file(root / relative)
    for path in sorted((root / "src").rglob("*.py")):
        commitment["files"][path.relative_to(root).as_posix()] = sha256_file(path)
    write_identical(root / "feedback_commitment.json", canonical(commitment))
    return {
        "commitment": "feedback_commitment.json",
        "public_stream": "feedback_stream.jsonl",
        "events": len(events),
        "old_corrections": len(corrections),
        "status": "must be publicly committed before calibration",
    }


def read_commitment(root: Path, state_a: Path):
    attestation = read_json(root / "runs/feedback_attestation.json")
    if attestation["manifest"] != "feedback_commitment.json":
        raise PermissionError("Feedback attestation names a different manifest")
    verify_public_commit(root, attestation)
    if sha256_file(root / attestation["manifest"]) != attestation["manifest_sha256"]:
        raise PermissionError("Feedback commitment hash changed")
    commitment = read_json(root / attestation["manifest"])
    if (
        commitment["state_a_manifest_sha256"] != sha256_file(state_a / "manifest.json")
        or commitment["data_commitment"] != data_commitment(root)
        or commitment["grid"] != [asdict(c) for c in config_grid()]
    ):
        raise PermissionError("Frozen A, data, or registered memory grid changed")
    for relative, digest in commitment["files"].items():
        if sha256_file(safe_relative(root, relative)) != digest:
            raise PermissionError("Precommitted feedback input or code changed")
    return commitment


def load_events(root: Path, commitment: dict) -> list[dict]:
    path = root / "runs/feedback_stream.jsonl"
    if sha256_file(path) != commitment["files"]["runs/feedback_stream.jsonl"]:
        raise PermissionError("Private feedback stream changed")
    events = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(events) != commitment["events"] or len({e["id"] for e in events}) != len(events):
        raise ValueError("Feedback stream count or IDs differ")
    public = [
        json.loads(line) for line in (root / "feedback_stream.jsonl").read_bytes().splitlines()
    ]
    expected = [
        {
            **{k: v for k, v in e.items() if k != "text"},
            "text_sha256": hashlib.sha256(e["text"].encode()).hexdigest(),
        }
        for e in events
    ]
    if public != expected:
        raise PermissionError("Public/private feedback streams do not match")
    return events


@dataclass
class NeighborTable:
    labels: np.ndarray
    distances: np.ndarray


def neighbor_table(memory: FeedbackMemory, features: np.ndarray, count=5) -> NeighborTable:
    labels = np.zeros((len(features), count), dtype=np.int64)
    distances = np.full(labels.shape, np.inf, dtype=np.float32)
    for row, hidden in enumerate(features):
        order, d = memory.nearest(hidden, count)
        labels[row, : len(order)] = np.asarray(memory.label_ids, dtype=np.int64)[order]
        distances[row, : len(d)] = d
    return NeighborTable(labels, distances)


def batch_predict(
    table: NeighborTable,
    config: MemoryConfig,
    model_probabilities: np.ndarray,
    fallback: np.ndarray,
):
    probabilities = np.asarray(model_probabilities, dtype=np.float64)
    if (
        probabilities.ndim != 2
        or len(probabilities) != len(fallback)
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1)
    ):
        raise ValueError("Invalid cached model probabilities")
    labels = table.labels[:, : config.neighbors]
    distances = table.distances[:, : config.neighbors]
    admitted = distances <= config.radius
    nearest = np.where(admitted[:, 0], distances[:, 0], 0)
    safe = np.where(admitted, distances, nearest[:, None])
    weights = np.exp(-(safe - nearest[:, None]) / config.temperature)
    weights[~admitted] = 0
    vote = np.zeros_like(probabilities)
    np.add.at(vote, (np.broadcast_to(np.arange(len(vote))[:, None], labels.shape), labels), weights)
    total = vote.sum(1, keepdims=True)
    vote = np.divide(vote, total, out=np.zeros_like(vote), where=total != 0)
    prediction = ((1 - config.blend) * probabilities + config.blend * vote).argmax(1)
    return np.where(total[:, 0] > 0, prediction, fallback)


def macro_fraction(predictions, truth, label_ids):
    values = []
    for label in label_ids:
        selected = truth == label
        if not selected.any():
            raise ValueError("A registered new intent has no validation examples")
        values.append(
            Fraction(int((predictions[selected] == truth[selected]).sum()), int(selected.sum()))
        )
    return sum(values, Fraction()) / len(values)


def choose_config(
    configs,
    tables,
    probabilities,
    fallback,
    truth,
    old_count,
    new_label_ids,
    memory_entries,
    *,
    source_entries,
    base_bar,
):
    if old_count < 1 or old_count >= len(truth):
        raise ValueError("Both old working and new validation rows are required")
    a_correct = int((fallback[:old_count] == truth[:old_count]).sum())
    assessments, eligible = [], []
    for config in configs:
        predictions = batch_predict(tables[config.layer], config, probabilities, fallback)
        old_correct = int((predictions[:old_count] == truth[:old_count]).sum())
        new_correct = int((predictions[old_count:] == truth[old_count:]).sum())
        macro = macro_fraction(predictions[old_count:], truth[old_count:], new_label_ids)
        entries = memory_entries[config.layer]
        valid = (
            old_correct >= a_correct
            and old_correct >= base_bar
            and source_entries + entries <= PARAMETER_CAP
        )
        value = {
            "config": asdict(config),
            "config_hash": object_hash(asdict(config)),
            "old_correct": old_correct,
            "old_total": old_count,
            "new_correct": new_correct,
            "new_total": len(truth) - old_count,
            "new_macro_accuracy": float(macro),
            "macro_fraction": str(macro),
            "memory_entries": entries,
            "eligible": valid,
        }
        assessments.append(value)
        if valid:
            eligible.append(((-macro, config.radius, entries, value["config_hash"]), value))
    selected = min(eligible, key=lambda pair: pair[0])[1] if eligible else None
    passed = bool(
        selected
        and selected["new_correct"] * 5 >= 4 * selected["new_total"]
        and Fraction(selected["macro_fraction"]) >= Fraction(4, 5)
    )
    return {
        "a_old_correct": a_correct,
        "base_bar": base_bar,
        "selected": selected,
        "g3_calibration_pass": passed,
        "grid": assessments,
    }


class FeatureCache:
    def __init__(self, path: Path, identity: dict):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("CREATE TABLE IF NOT EXISTS identity (payload TEXT)")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS examples (id TEXT PRIMARY KEY, payload BLOB, prediction TEXT)"
        )
        encoded = canonical(identity).decode()
        saved = self.connection.execute("SELECT payload FROM identity").fetchone()
        if saved and saved[0] != encoded:
            self.connection.close()
            raise ValueError("Calibration cache identity changed")
        if not saved:
            self.connection.execute("INSERT INTO identity VALUES (?)", (encoded,))
            self.connection.commit()

    def get(self, row, runner, layers, mode, *, scoring=False):
        key = object_hash({"id": row["id"], "text": row["text"]})
        saved = self.connection.execute(
            "SELECT payload,prediction FROM examples WHERE id=?", (key,)
        ).fetchone()
        tensors, prediction = (load_tensors(saved[0]), saved[1]) if saved else ({}, None)
        changed = not saved
        if not tensors:
            tensors = {
                f"layer_{layer}": torch.from_numpy(vector).contiguous()
                for layer, vector in runner.features(row["text"], layers).items()
            }
        if scoring and "scores" not in tensors:
            tensors["scores"] = runner.label_logprobs(row["text"]).contiguous()
            prediction = runner.predict(row["text"], mode)
            changed = True
        if changed:
            self.connection.execute(
                "INSERT OR REPLACE INTO examples VALUES (?,?,?)",
                (key, save_tensors(tensors), prediction),
            )
            self.connection.commit()
        return tensors, prediction

    def close(self):
        self.connection.close()


def calibrate_feedback(root: Path, state_a: Path, *, device="cuda"):
    commitment = read_commitment(root, state_a)
    manifest = bundle_manifest(state_a)
    runner = load_bundle(state_a, device)
    if hasattr(runner, "memory"):
        raise ValueError("Calibration requires A without a training-memory booster")
    events = load_events(root, commitment)
    configs = config_grid()
    layers = sorted({c.layer for c in configs})
    labels = runner.labels + list(NEW_INTENTS)
    working = load_split(root, "banking/working", purpose="memory")
    validation = load_split(root, "clinc/validation", purpose="memory")
    rows = working + validation
    identity = {
        "state_a": commitment["state_a_manifest_sha256"],
        "commitment": sha256_file(root / "feedback_commitment.json"),
    }
    directory = root / "runs/calibration" / object_hash(identity)[:16]
    directory.mkdir(parents=True, exist_ok=True)
    cache = FeatureCache(directory / "features.sqlite", identity)
    try:
        event_values = [cache.get(row, runner, layers, manifest["mode"])[0] for row in events]
        query_values = [
            cache.get(row, runner, layers, manifest["mode"], scoring=True) for row in rows
        ]
        tables, entries = {}, {}
        for layer in layers:
            config = next(c for c in configs if c.layer == layer)
            memory = FeedbackMemory(labels, config)
            for row, values in zip(events, event_values, strict=True):
                memory.add(row["id"], row["text"], row["label"], values[f"layer_{layer}"].numpy())
            features = np.stack([v[0][f"layer_{layer}"].numpy() for v in query_values])
            tables[layer] = neighbor_table(memory, features)
            entries[layer] = memory.numeric_entries
        probabilities, fallback = [], []
        for tensors, prediction in query_values:
            scores = tensors["scores"].numpy()
            p = np.exp(scores - scores.max())
            p /= p.sum()
            full = np.zeros(len(labels))
            full[: len(p)] = p
            probabilities.append(full)
            fallback.append(labels.index(prediction) if prediction in labels else -1)
        base_bar = max(
            read_json(root / f"runs/baseline/working-{len(working)}-{mode}.summary.json")["correct"]
            for mode in ("greedy", "constrained")
        )
        report = choose_config(
            configs,
            tables,
            np.stack(probabilities),
            np.asarray(fallback),
            np.asarray([labels.index(row["label"]) for row in rows]),
            len(working),
            [labels.index(label) for label in NEW_INTENTS],
            entries,
            source_entries=manifest["source_entries"],
            base_bar=base_bar,
        )
        report.update(
            format=1,
            identity=identity,
            label_count=len(labels),
            old_rows=len(working),
            new_rows=len(validation),
        )
        write_identical(directory / "report.json", canonical(report))
        if report["selected"]:
            write_identical(
                directory / "memory_config.json", canonical(report["selected"]["config"])
            )
        return {
            "report": str((directory / "report.json").relative_to(root)),
            "config": str((directory / "memory_config.json").relative_to(root))
            if report["selected"]
            else None,
            "g3_calibration_pass": report["g3_calibration_pass"],
        }
    finally:
        cache.close()
