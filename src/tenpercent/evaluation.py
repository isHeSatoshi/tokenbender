from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import torch

from .io import append_record, object_hash, read_json, write_json
from .metrics import summarize


def evaluate(runner, rows: list[dict], output: Path, *, mode: str, state_hash: str):
    expected = {r["id"]: r for r in rows}
    records = []
    if output.exists():
        records = [json.loads(line) for line in output.read_bytes().splitlines()]
        for record in records:
            if (
                record["example_id"] not in expected
                or record["mode"] != mode
                or record["state_hash"] != state_hash
                or record["true_label"] != expected[record["example_id"]]["label"]
            ):
                raise ValueError("Existing predictions do not match this frozen evaluation")
        if len({r["example_id"] for r in records}) != len(records):
            raise ValueError("Duplicate prediction journal records")
    seen = {r["example_id"] for r in records}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        if row["id"] in seen:
            continue
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = runner.predict(row["text"], mode=mode)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        record = {
            "example_id": row["id"],
            "true_label": row["label"],
            "predicted_label": prediction,
            "valid": prediction in runner.labels,
            "mode": mode,
            "state_hash": state_hash,
            "seconds": time.perf_counter() - start,
        }
        append_record(output, record)
        records.append(record)
        if len(records) % 25 == 0:
            # No text or labels are exposed in progress output.
            print(
                json.dumps(
                    {"event": "evaluation_progress", "completed": len(records), "total": len(rows)}
                ),
                flush=True,
            )
    report = summarize(records, runner.labels)
    report["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if torch.cuda.is_available():
        report["cuda_peak_allocated"] = torch.cuda.max_memory_allocated()
        report["cuda_peak_reserved"] = torch.cuda.max_memory_reserved()
    report["state_hash"] = state_hash
    report["mode"] = mode
    write_json(output.with_suffix(".summary.json"), report)
    return records, report


def baseline(
    root: Path, *, split="working", limit: int | None = None, device="cuda", vocabulary_only=False
):
    from .constants import MODEL_SHA256
    from .data import load_split
    from .model import Runner, TokenizerAdapter, load_base, load_tokenizer

    rows = load_split(root, f"banking/{split}", purpose="baseline")
    if limit is not None:
        if limit < 1:
            raise ValueError("Limit must be positive")
        rows = rows[:limit]
    model_path = root / "assets" / "base"
    model = load_base(model_path, device=device)
    labels = read_json(root / "data" / "manifest.json")["labels"]
    runner = Runner(model, TokenizerAdapter(load_tokenizer(model_path)), labels)
    vocabulary_hash = None
    if vocabulary_only:
        from .io import sha256_file
        from .vocabulary import CompactTokenizer, Vocabulary, trim_model

        vocabulary_path = root / "runs" / "vocabulary.json"
        vocabulary = Vocabulary.load(vocabulary_path)
        trim_model(model, vocabulary)
        runner = Runner(model, CompactTokenizer(load_tokenizer(model_path), vocabulary), labels)
        vocabulary_hash = sha256_file(vocabulary_path)
    identity_fields = {
        "weights": MODEL_SHA256,
        "ids": [r["id"] for r in rows],
        "device": device,
        "backend": "eager",
    }
    if vocabulary_only:
        identity_fields["vocabulary_sha256"] = vocabulary_hash
    identity = object_hash(identity_fields)
    results = {}
    for mode in ("greedy", "constrained"):
        stem = f"{split}-vocabulary" if vocabulary_only else split
        output = root / "runs" / "baseline" / f"{stem}-{len(rows)}-{mode}.jsonl"
        _, results[mode] = evaluate(runner, rows, output, mode=mode, state_hash=identity)
    return results
