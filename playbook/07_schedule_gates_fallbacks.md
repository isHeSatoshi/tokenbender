# 07. Relative schedule, gates, and fallbacks

Calendar dates and wall-clock timestamps are deliberately omitted from the schedule. Day 1 is the day execution starts. Days 6–7 are contingency, not permission to postpone a complete proof.

## Sequence

| Relative day | Work and deliverables | Gate |
|---|---|---|
| Day 1 | Environment and source locks; split/seal; commit and push preregistration; working baselines; original-vocabulary all-one mask identity; teacher cache after vocabulary freeze | G0 |
| Day 2 | 30% and 10% pilots, three learning rates; fresh-process first-step and steady-state profiling | G1 |
| Days 3–4 | Full 10% search; working ablations and candidate registry; implement memory while the GPU runs | G2 |
| Day 5 | Feedback calibration/replay; package both states; identity and size proof; synthetic dry run; push frozen final manifest; one final session; publish results | G3, G4 |
| Days 6–7 | Pre-exam contingency only, or documentation/report cleanup after final evaluation | No exam-based retuning |

The executor should finish and publish the proof as soon as all gates pass. Estimates must be revised from measured seconds per step and memory peaks; there is no guaranteed compute duration in this document.

## Gates, with explicit consequences

### G0: trustworthy setup

Require all of:

* Exact source hash/count verified; environment locked and GPU/backend recorded.
* Expected clean test count checked; actual split counts/hashes committed; exam sealed.
* Complete preregistration, split script, source lock, and commitment pushed before adaptation.
* Both working baselines measured without exam access.
* Original-vocabulary 100%-kept mask matches base logits and generated outputs exactly under the reference deterministic backend.
* Vocabulary fallback round-trip and vocabulary-only ablation complete.
* Teacher cache built from train and train-derived variants only; schema, support, and probability normalization tested.
* Lock guard negative tests and source-weight freeze tests pass.

**Fail:** fix infrastructure using training/synthetic data. No mask pilots while identity or exam isolation is unresolved.

### G1: pilot is viable

At **30% total retained size**, require **≥99% working agreement** with the working-set winning base mode. Report agreement with the other mode too. With $N_w$ working examples, this is `agreements >= ceil(0.99 * N_w)`, not a rounded percentage.

Run 10%-total pilots too, and compare score initialization and the three registered learning rates. Both pilot targets include dense embeddings, norms, biases, and any auxiliary numbers.

Measure initialization, first optimizer update, density transition, evaluation, and serialization in fresh processes. No OOM, no mutated source weights, exact global group budget, and enough resource headroom are required.

**Fail:** apply the fallback ladder in order. A 30% run is a diagnostic, not a qualifying submission.

### G2: freeze an eligible A

Target **working accuracy no more than 0.3 percentage point below the working base bar**, and **≥97.5% working agreement** with its winning base mode, at **≤154,371,430 entries** including any A booster.

Use exact integer checks:

* `100 * (bar_correct - A_correct) <= 0.3 * N_w`.
* `agreements >= ceil(0.975 * N_w)`.

The 0.3-point gate is a search target, **not** the final ≥ claim. Prefer a clear working margin above the bar; a merely close working score does not predict a strict final pass. If only a below-bar candidate reaches G2, state the risk before consuming the one exam session.

Select A by working accuracy, then base agreement, then smaller retained count, then lexical config hash. Freeze A before revealing correction feedback or learning new-intent memory.

**Fail:** use preregistered fallbacks, or publish the development accuracy–size curve without opening the exam. Do not describe 97.5% agreement as 97.5% accuracy.

### G3: feedback and old-intent retention

Using frozen memory hyperparameters, replay from a fresh A. Final B must:

* Score **≥80%** on the new-intent **validation** set; **85%** is the stretch goal.
* Have **no working accuracy drop versus A** and score **≥ the working base bar**.
* Remain inside the retained-entry cap, with all intermediate state sizes reported.
* Pass duplicate/conflict, radius, resume, serialization, and capacity tests.

**Fail:** adjust only within the registered memory calibration procedure using working/CLINC validation; replay from A. Do not change the mask using CLINC or feedback labels. A source-model change requires restarting A selection and feedback, still without exam access.

### G4: final proof

Before opening: standalone bundles load without base/network access; exhaustive BF16 lineage has zero mismatches; state counts pass; actual bytes are reported; full manifest pushed and guard verified.

During the one final session:

* A old correct count **≥ max(two base old correct counts)**.
* B old correct count **≥ the same maximum**.
* B new-intent test accuracy **≥80%** and **strictly greater than A's**.
* A and B each have **≤154,371,430 retained numeric entries**.
* Claim the secondary byte target only for states with **≤308,746,714 bytes**.

The 95% bootstrap CI and −1.0-point non-inferiority result are secondary. If either strict old-score test fails, publish the failed claim. There is no “within noise” substitution.

## Ordered fallback ladder

Every fallback is a named candidate in the registry. Use working data only for selection, preserve the original baseline and size denominators, and disclose changes. No fallback permits new weight values.

1. **Retune score search.** Use the registered LR sweep, 4,000-step extension, cubic schedule, and optional hidden-state matching loss. For that loss, cache teacher training features only, coefficient grid `{0.01, 0.1}`, and add no trainable projection. Try groups of **2** as a distinct run; twice as many scores can break GPU fit, so profile first.
2. **Structured pre-pruning.** Select whole layers or attention heads to remove, using training calibration and working selection. Preserve residual shapes and GQA grouping, copy kept source slices exactly, and recheck the all-kept identity of the remaining reference graph. Removing candidates can reduce score memory, but accuracy may worsen. Record which structural pieces were discarded.
3. **Training-exemplar nearest neighbors.** Add a counted A memory built from train only. Adopt only for **≥0.3 percentage-point working gain**. Reserve its full numeric count by shrinking the model target before freezing A; 10,003 256-D int8 keys alone exceed the default reserve.
4. **Deterministic rewording ensemble.** Average label probabilities over at most three fixed, publicly specified rewrites (original plus two). Adopt only for **≥0.3 percentage-point working gain**. Freeze rewrites before the exam; no remote paraphraser or teacher at deployment. Report latency and exact text transforms.
5. **Score every complete label.** Use the EOS-inclusive, unnormalized sequence log probabilities in [03](03_baseline_harness.md). Select on working; do not change either base protocol or call this greedy decoding.
6. **Middle-layer truncation plus nearest neighbors.** Retain a source embedding and prefix layers, classify through counted exemplar memory, and remove the LM head if not used. Report this as a materially different feature-classifier fallback, not a successful 90%-sparse full-network result. No trained head or projection.
7. **Honest development curve.** If none works, publish accuracy versus retained entries and bytes for the measured candidates, with missing/failed configurations shown. Final exam data, if ever opened, cannot select a new point on the curve.

Steps 1–2 changing the source architecture or score space require refreshed cache/budget/lineage tests. Steps 3–6 change A's inference path and must be frozen before feedback. Any search expansion beyond the registered grids gets a public pre-exam amendment, not a retroactive preregistration.

## Main risks

| Risk | Detection / response |
|---|---|
| Frozen-weight selection fails at high sparsity | Compare vocabulary-only, Wanda, 30%, 10%; stop or publish failure |
| Base near the accuracy ceiling | Seek working margin; retain strict final tests, no noise exemption |
| Global ranking, activations, or Adam exceeds VRAM | Measure first update and transitions; chunk exact top-k, checkpoint, pre-prune |
| Byte overhead defeats the 10% file target | Pack early, measure masks/tokenizer and BF16 compression; report separate outcome |
| Vocabulary fallback changes unseen tokenization | Byte-level round trips, vocabulary-only ablation, multilingual tests |
| Memory confuses old and new intents | 89-way validation, radius gating, old working no-drop constraint |
| Working-set overfitting | Finite grids, complete registry, no locked-set candidate selection |
| Guard gives a false sense of secrecy | Custodian/encryption where possible; disclose public reconstruction risk |
| Model/data/license or upstream-contamination uncertainty | Pin sources, preserve notices, state what is and is not established |
| Another entrant completes first | Finish early when valid; do not assert organizer acceptance |

Planning notes describe tokenbender's own high-compression attempt as stalling. Treat that as anecdotal context, not an independently verified numerical benchmark. There is no direct source here proving this exact recipe reaches parity.
