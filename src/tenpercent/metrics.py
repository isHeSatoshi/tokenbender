from __future__ import annotations

from collections import defaultdict

import numpy as np

from .constants import PARAMETER_CAP, SEED


def summarize(records: list[dict], labels: list[str]) -> dict:
    if not records or len({r["example_id"] for r in records}) != len(records):
        raise ValueError("Predictions must be nonempty with unique IDs")
    by_label = defaultdict(lambda: [0, 0])
    for row in records:
        correct = int(row["predicted_label"] == row["true_label"])
        by_label[row["true_label"]][0] += correct
        by_label[row["true_label"]][1] += 1
    correct = sum(pair[0] for pair in by_label.values())
    per_label = {
        label: {"correct": a, "total": b, "accuracy": a / b}
        for label, (a, b) in sorted(by_label.items())
    }
    return {
        "correct": correct,
        "total": len(records),
        "accuracy": correct / len(records),
        "invalid": sum(row["predicted_label"] not in labels for row in records),
        "macro_accuracy": float(np.mean([v["accuracy"] for v in per_label.values()])),
        "per_label": per_label,
    }


def aligned_correctness(*groups: list[dict]) -> list[np.ndarray]:
    maps = [{r["example_id"]: r for r in rows} for rows in groups]
    if any(len(m) != len(rows) for m, rows in zip(maps, groups, strict=True)):
        raise ValueError("Duplicate prediction IDs")
    ids = sorted(maps[0])
    if any(set(m) != set(ids) for m in maps):
        raise ValueError("Prediction sets differ; never drop missing predictions")
    for key in ids:
        if len({m[key]["true_label"] for m in maps}) != 1:
            raise ValueError("Ground truth differs between prediction records")
    return [
        np.asarray([m[key]["predicted_label"] == m[key]["true_label"] for key in ids]) for m in maps
    ]


def paired_comparison(candidate, base_greedy, base_constrained, *, draws=10_000, seed=SEED):
    a, b, c = aligned_correctness(candidate, base_greedy, base_constrained)
    if not len(a) or draws < 1:
        raise ValueError("Empty comparison or invalid bootstrap count")
    rng = np.random.default_rng(seed)
    deltas = np.empty(draws)
    for start in range(0, draws, 128):
        stop = min(draws, start + 128)
        ix = rng.integers(0, len(a), size=(stop - start, len(a)))
        deltas[start:stop] = 100 * (a[ix].mean(1) - np.maximum(b[ix].mean(1), c[ix].mean(1)))
    low, high = np.percentile(deltas, [2.5, 97.5], method="linear")
    bar_count = max(int(b.sum()), int(c.sum()))
    return {
        "correct": int(a.sum()),
        "total": len(a),
        "bar_correct": bar_count,
        "strict_pass": int(a.sum()) >= bar_count,
        "difference_pp": 100 * (int(a.sum()) - bar_count) / len(a),
        "paired_ci_95_pp": [float(low), float(high)],
        "secondary_noninferiority": bool(low > -1.0),
        "bootstrap_draws": draws,
    }


def claim_results(old_a, old_b, base_g, base_c, new_a, new_b, counts: dict) -> dict:
    a = paired_comparison(old_a, base_g, base_c)
    b = paired_comparison(old_b, base_g, base_c)
    na, nb = aligned_correctness(new_a, new_b)
    size_a = counts["A"] <= PARAMETER_CAP
    size_b = counts["B"] <= PARAMETER_CAP
    learned = int(nb.sum()) * 5 >= len(nb) * 4 and int(nb.sum()) > int(na.sum())
    return {
        "A": a,
        "B": b,
        "entry_counts": counts,
        "claim_A": a["strict_pass"] and size_a,
        "claim_B": b["strict_pass"] and size_b and learned,
        "new": {
            "before_correct": int(na.sum()),
            "after_correct": int(nb.sum()),
            "total": len(nb),
            "learned": learned,
        },
    }
