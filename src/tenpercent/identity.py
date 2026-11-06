from __future__ import annotations

from pathlib import Path

import torch

from .constants import MODEL_SHA256
from .data import load_split
from .io import read_json, write_json
from .masking import frozen_hash, replace_linears
from .model import Runner, TokenizerAdapter, load_base, load_tokenizer


def check_identity(root: Path, *, limit=8, device="cuda"):
    rows = load_split(root, "banking/working", purpose="baseline")
    if limit < 1:
        raise ValueError("Identity check requires at least one example")
    rows = rows[:limit]
    model = load_base(root / "assets" / "base", device)
    adapter = TokenizerAdapter(load_tokenizer(root / "assets" / "base"))
    runner = Runner(model, adapter, read_json(root / "data" / "manifest.json")["labels"])
    reference = []
    with torch.inference_mode():
        for row in rows:
            inputs = torch.tensor([adapter.prompt(row["text"])], device=device)
            logits = model(inputs, use_cache=False).logits[0, -1].clone()
            reference.append((logits, runner.predict(row["text"], "greedy")))
    modules = replace_linears(model)
    logits_equal = predictions_equal = True
    max_difference = 0
    with torch.inference_mode():
        for row, (expected, prediction) in zip(rows, reference, strict=True):
            inputs = torch.tensor([adapter.prompt(row["text"])], device=device)
            actual = model(inputs, use_cache=False).logits[0, -1]
            logits_equal &= torch.equal(actual, expected)
            max_difference = max(
                max_difference, float((actual.float() - expected.float()).abs().max())
            )
            predictions_equal &= prediction == runner.predict(row["text"], "greedy")
    result = {
        "source_sha256": MODEL_SHA256,
        "examples": len(rows),
        "full_working_set": len(rows)
        == len(load_split(root, "banking/working", purpose="baseline")),
        "logits_bit_equal": logits_equal,
        "predictions_equal": predictions_equal,
        "max_logit_difference": max_difference,
        "mask_matrices": len(modules),
        "candidate_groups": sum(m.scores.numel() for m in modules.values()),
        "frozen_hash": frozen_hash(model),
    }
    write_json(root / "runs" / f"identity-{len(rows)}.json", result)
    if not logits_equal or not predictions_equal:
        raise RuntimeError(f"All-one-mask identity failed: {result}")
    return result
