from __future__ import annotations

import json
import resource
import time
from pathlib import Path

import torch

from .io import append_record, atomic_bytes, canonical, object_hash, read_json, write_json
from .metrics import summarize


def evaluate(runner, rows: list[dict], output: Path, *, mode: str, state_hash: str):
    output_labels = getattr(runner, "output_labels", runner.labels)
    cuda = torch.cuda.is_available() and str(getattr(runner, "device", "cpu")).startswith("cuda")
    expected = {r["id"]: r for r in rows}
    if len(expected) != len(rows):
        raise ValueError("Evaluation input IDs must be unique")
    records = []
    pending_path = output.with_suffix(".pending.json")
    partial = b""
    if output.exists():
        lines = output.read_bytes().splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            partial = lines.pop()
        records = [json.loads(line) for line in lines]
        if len({r["example_id"] for r in records}) != len(records):
            raise ValueError("Duplicate prediction journal records")

    def validate(record):
        if (
            record["example_id"] not in expected
            or record["mode"] != mode
            or record["state_hash"] != state_hash
            or record["true_label"] != expected[record["example_id"]]["label"]
            or record["valid"] != (record["predicted_label"] in output_labels)
        ):
            raise ValueError("Existing predictions do not match this frozen evaluation")

    for record in records:
        validate(record)
    if pending_path.exists():
        pending = read_json(pending_path)
        validate(pending)
        previous = next((r for r in records if r["example_id"] == pending["example_id"]), None)
        if previous is not None:
            if previous != pending or partial:
                raise ValueError("Pending prediction disagrees with the committed journal")
        else:
            if partial:
                if not canonical(pending).startswith(partial):
                    raise ValueError(
                        "Truncated prediction journal does not match its pending record"
                    )
                atomic_bytes(output, b"".join(canonical(r) for r in records) + canonical(pending))
            else:
                append_record(output, pending)
            records.append(pending)
        pending_path.unlink()
    elif partial:
        raise ValueError("Truncated prediction journal has no durable pending record")
    seen = {r["example_id"] for r in records}
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    for row in rows:
        if row["id"] in seen:
            continue
        if cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        prediction = runner.predict(row["text"], mode=mode)
        if cuda:
            torch.cuda.synchronize()
        record = {
            "example_id": row["id"],
            "true_label": row["label"],
            "predicted_label": prediction,
            "valid": prediction in output_labels,
            "mode": mode,
            "state_hash": state_hash,
            "seconds": time.perf_counter() - start,
        }
        # Persist the result before appending, so a torn append never reruns this query.
        write_json(pending_path, record)
        append_record(output, record)
        pending_path.unlink()
        records.append(record)
        if len(records) % 25 == 0:
            # No text or labels are exposed in progress output.
            print(
                json.dumps(
                    {"event": "evaluation_progress", "completed": len(records), "total": len(rows)}
                ),
                flush=True,
            )
    report = summarize(records, output_labels)
    report["host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    if cuda:
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
