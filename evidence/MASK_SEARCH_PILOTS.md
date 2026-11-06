# Development pilot results, not a challenge result

Snapshot: 2026-11-06, after the six registered 200-step pilots and the
non-selecting Wanda-only ablation. **Every pilot failed its quality gate.
No qualifying A or B exists, and neither locked exam has been opened.**

The original public registration is commit
`fe4ea30df3ea6bc436f65ab5c691317bcf24b9f1`. The preregistration SHA-256 remains
`d17e2c59ff1c94e741c25be76c7389fc8f2a73cba435773c501e542b8fb96cf8`.
This evidence publication changes no registered input, search algorithm,
baseline, split, or budget.

## Completed comparisons

All scores below use the same 770-row Banking77 working split. Both original
base protocols score **719/770**. Their predictions agree on all 770 rows.
Vocabulary-only evaluation also scored 719/770 in both modes.

Each learned pilot used four-entry groups, 200 optimizer steps, accumulation
32, the registered 50,003-example teacher cache, and seed `20261106`. Scores
shown here are the selected mode at the final target-size checkpoint.
The machine-readable report retains both modes rather than hiding the worse
one.

| Learning rate | 30% correct | 30% base agreement | 10% correct | 10% base agreement |
|---|---:|---:|---:|---:|
| 0.001 | 492/770 | 500/770 | 11/770 | 13/770 |
| 0.003 | 633/770 | 644/770 | 59/770 | 62/770 |
| 0.01 | 667/770 | 674/770 | 481/770 | 482/770 |

The 30% diagnostics retained **463,114,288** source entries and cannot qualify
as A. The 10% pilots retained **152,800,000** source entries. Meeting the entry
budget did not make their accuracy acceptable. No deployment archive or
complete-container byte result is claimed here.

G1 requires at least **763/770** base agreement at 30% total size.
G2 targets at least **717/770** correct and **751/770** agreement at the
ten-percent entry budget. None passed. These working targets do not replace
the strict final requirement to meet both original baselines on the sealed
exam.

### Wanda-only ablation

This applies the registered training-only Wanda initialization and exact
global selection, with **zero optimizer updates**.

| Retained entries | Literal greedy correct | Constrained correct |
|---|---:|---:|
| 463,114,288 | 0/770 | 14/770 |
| 152,800,000 | 0/770 | 10/770 |

Every literal Wanda-only prediction was invalid. Invalid outputs count as
wrong. The constrained decoder returns valid labels but does not recover
accuracy. This comparison shows a substantial improvement from learned masks,
not a successful compression result.

All 3,080 ablation records were independently checked against working IDs,
input hashes, truth labels, both baseline agreements, and committed journal
hashes. Frozen source hashes matched the verified resource-pilot source.
The exact diagnostic wrapper is archived as
`wanda_only_ablation.py.txt`; its intended local path is
`runs/ablations/wanda_only.py`. It uses the unchanged registered modules and
requires the locally provisioned source, data, and attestation. No model or
dataset payload is published.

## Resources and recovery

The fresh worst-input pilot scanned all 50,003 teacher records and exercised
32-example accumulated updates at full, midpoint, and target size. Its
longest actual input was **125 tokens**. Peak CUDA reservation was
**9,344,909,312 bytes**, approximately **8.70 GiB**. The headroom and
unchanged-source checks passed.

For five completed searches, the last 100 regular same-shape updates averaged
9.35–9.66 seconds per update. Evaluation/save transitions are excluded.
Density varied: these are **not** measurements of 100 fixed-density updates.
`mask_search_resources.json` records exact values, scope, and provenance.
Stored entry counts do not imply sparse-kernel acceleration or a tenfold VRAM
reduction.

The LR-0.01 ten-percent pilot was interrupted at step 158 and recovered from
its step-100 checkpoint. Across 58 replayed updates, logged loss,
score-gradient norm, and kept-group count matched exactly. Full score-tensor
bit equality was not measured. Final checkpoint metadata was reloaded for
each of the five resource-report searches; actual CUDA optimizer restoration
was exercised by this recovery.

## Next registered attempt

At this snapshot, a fresh **2,000-step**, LR-**0.01** ten-percent search was
started as the ordered score-search fallback:
`09157f8862a30a4d`. It is not included above as a completed result.

Its cubic schedule first reaches the source target at evaluation step
**1,600**. Earlier baseline-matching scores are above budget and are not
qualifying ten-percent results. A longer run may still fail. No feedback
calibration, real-model standalone proof, final freeze, or locked exam is
justified by these pilot results.
