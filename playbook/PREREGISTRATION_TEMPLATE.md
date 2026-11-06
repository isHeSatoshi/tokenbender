# Preregistration: frozen-weight ten-percent Banking77

**Status: protocol prepared, not yet registered.** The choices below are filled from the plan. Replace every `REQUIRED_AT_EXECUTION` field with measured identifiers and commit/push this document before adaptation. This docs-only delivery does not claim a public commitment already exists.

## 1. Identity and commitment

| Field | Registered value |
|---|---|
| Execution owner and authorized public repository | `REQUIRED_AT_EXECUTION` |
| Protocol version and pre-search commit | `REQUIRED_AT_EXECUTION` |
| Remote confirmation / separate attestation location | `REQUIRED_AT_EXECUTION` |
| Rules snapshot SHA-256 | `REQUIRED_AT_EXECUTION` |
| Base | `johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent` |
| Base revision | `6c13ee7e6a4d229615dccdd089869818ad389388` |
| Weights SHA-256 | `2084675316db424191877b0ee41689ba28b83e07c8444fc1f44f037c769e82dc` |
| Source parameters / weights bytes | 1,543,714,304 / 3,087,467,144 |
| Download verification report and source-lock hash | `REQUIRED_AT_EXECUTION` |
| Environment lock, Python patch version, CUDA/driver/backend, hardware | `REQUIRED_AT_EXECUTION` |

Time is organized as relative Days 1–7. No calendar deadline is part of this protocol. Fixed seed `20261106` is not a schedule timestamp.

## 2. Data and blindness

* Banking77 revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`, config `default`.
* CLINC revision `155b9c710419136e17307b80d0a13e68cd46b4ec`, config `plus`.
* Remove test rows whose exact UTF-8 text appears in official train. Expected clean test count: **3,076**, subject to verification.
* Per-label 50/25/25 exam/working/feedback; largest-remainder allocation, ties exam/working/feedback; within-label SHA-256 ordering of seed plus stable source ID, as in [02](02_setup_data_locked_exam.md).
* Ordinary code refuses locked rows. Working/feedback **text only** may determine vocabulary. Only train and train-derived EDA are used for score gradients and teacher caches. Working labels select candidates. Feedback labels are reserved for the post-A stream.
* CLINC is excluded from vocabulary and mask learning. CLINC train teaches memory; validation calibrates it; test remains sealed until the final session.
* Preparation audits and seals duplicates/overlaps before tuning, with no exam text or model analysis exposed. The default test splitter aborts on within-test exact duplicates.
* Public deterministic splitting provides procedural blinding, not cryptographic secrecy or a proof of absence from foundation-model pretraining.

| Required commitment | Value |
|---|---|
| Actual source/split counts and per-label quota report | `REQUIRED_AT_EXECUTION` |
| Splitter code and source-lock SHA-256 | `REQUIRED_AT_EXECUTION` |
| Locked canonical JSONL SHA-256 | `REQUIRED_AT_EXECUTION` |
| Working and feedback file SHA-256 | `REQUIRED_AT_EXECUTION` |
| CLINC eligible teaching IDs, validation and sealed test hashes/counts | `REQUIRED_AT_EXECUTION` |
| Sealing/custodian method and guard-test report | `REQUIRED_AT_EXECUTION` |

## 3. Primary outcomes

Let the final old-intent bar be the larger of the unmodified base's card-protocol greedy exact-match and valid-label-constrained greedy accuracies, measured on the same locked rows.

* **Claim A:** frozen pre-feedback A has at least as many correct old-intent predictions as **each** baseline.
* **Claim B:** frozen post-feedback B has at least as many correct old-intent predictions as **each** baseline; new-intent test accuracy is **≥80%** and strictly exceeds A's.
* Both states have **≤154,371,430 retained numeric entries**. All stored memory payload coordinates/scales/label IDs/settings count.
* Secondary storage outcome: each complete inference container is **≤308,746,714 bytes**. A parameter pass does not imply a byte pass.
* Secondary inference: paired percentile-bootstrap 95% CIs, 10,000 draws, seed `20261106`, linear quantiles, resampling examples jointly and recomputing the maximum baseline each draw.
* Secondary non-inferiority: lower CI endpoint **>−1.0 percentage point**. This cannot rescue a failed strict headline.
* Report B−A on old intents, all exact counts and invalids, new-label false positives, and results for 1-, 5-, and 10-shot states.

Invalid/missing predictions are wrong. No rounded-percentage decisions, abstention filtering, exam reweighting, post-hoc label repair, or removal of hard examples.

## 4. Frozen baseline procedures

System prompt: `Classify the banking customer message into its intent label.`

Pinned tokenizer/chat template, user text unchanged, greedy `do_sample=False`, one beam, `max_new_tokens=20`, decode generated suffix only, `skip_special_tokens=True`, literal decoded-string equality. No trimming or normalization in the primary comparator.

Second baseline restricts each next token with a valid-label trie and EOS, with ties by lowest original ID. It is not full-label scoring. Preserve unusual label capitalization and punctuation.

| Development baseline evidence | Value |
|---|---|
| Prompt/tokenizer/template hashes and working prediction arrays | `REQUIRED_AT_EXECUTION` |
| Working correct counts for both base modes | `REQUIRED_AT_EXECUTION` |
| All-one original-vocabulary identity and vocabulary-only reports | `REQUIRED_AT_EXECUTION` |

## 5. Adaptation and candidate selection

* Keep source weights frozen and BF16-bit-identical. No LoRA, quantization, trained projection/head, reconstruction updates, or numeric weight programming.
* Trim vocabulary using special/template/label tokens, all original token IDs from permitted Banking77 text, IDs `0..8191`, and all byte leaves. Recursively decompose dropped BPE tokens using the pinned parent map.
* Four-entry input-dimension groups, exact global top-k; Wanda training-only initialization standardized per matrix; straight-through identity surrogate.
* Source-model target **152,800,000**, reduced pre-search if auxiliary reserve is insufficient. End density is computed from actual vocabulary, not fixed at 10% body density.
* Adaptation: official train plus approximately 40,000 deterministic train-only EDA variants, base-labeled for variants.
* Dense-body vocabulary-trimmed teacher at label/EOS positions, temperature **2**, top-64 retained-token buckets plus tail; $T^2$ KL plus **0.3** full-vocabulary target CE.
* Adam scores only, betas `(0.9, 0.999)`, eps `1e-8`, zero weight decay, clip norm `1.0`; LR grid `{0.001, 0.003, 0.01}`; batch 1 accumulated to 32.
* Pilots: 200 steps at 30% and 10% total-size budgets. Full candidates: 2,000 steps, then 4,000 if needed. Evaluate every 100 steps; only final-target-size checkpoints qualify.
* Density schedule: 10% hold full, 70% cubic decay, 20% hold target. LR: 5% warmup then cosine.
* Select highest working accuracy among count-eligible completed candidates; ties by working-base agreement, fewer entries, lexical config hash.
* Training-memory or up-to-three-view deterministic rewrite boosters need **≥0.3 percentage-point working gain**. All cost counts.
* Ordered fallbacks: retune/hidden-state match/groups of two; structured pre-pruning; counted train memory; rewrite averaging; full-label scoring; middle-layer feature kNN; honest development curve.

Record expansions to grids, structured candidates, hidden-state coefficients `{0.01, 0.1}`, or rewrite definitions as public pre-exam amendments, never as if chosen on Day 1. No mask learning on feedback or CLINC.

| Search inputs committed before pilot | Value |
|---|---|
| Actual vocabulary $V$, ID-list/fallback hash, source/body/auxiliary budgets | `REQUIRED_AT_EXECUTION` |
| Actual adaptation/EDA counts, resource versions, data/teacher-cache hashes | `REQUIRED_AT_EXECUTION` |
| Exact runtime flags, initial working baselines, resource pilot evidence | `REQUIRED_AT_EXECUTION` |

## 6. Feedback protocol

Twelve labels: `credit_score`, `improve_credit_score`, `apr`, `routing`, `direct_deposit`, `interest_rate`, `credit_limit`, `credit_limit_change`, `min_payment`, `pay_bill`, `rewards_balance`, `redeem_rewards`.

Nested first 1/5/10 eligible CLINC train examples per label by seeded stable-ID hash. One-shot round, frozen-A error corrections from Banking77 feedback ordered by ID, four more rounds to five-shot, five more rounds to ten-shot B. No working/test exemplars in memory.

Last non-padding prompt-token hidden state, layer grid `{7,14,21,28}`; untrained seeded 256-D Rademacher projection; per-key scaled int8 representation. If A already has a training-memory booster, reuse its frozen feature layer/projection rather than searching a new layer for B. Gate by cosine radius; nearest-neighbor vote blends with EOS-inclusive full-label model probabilities. New labels have zero model-branch probability; all evaluations use the joint 89-label interface.

Calibration grid: neighbors `{1,3,5}`, radius `{0.05,0.10,0.20,0.30}`, blend `{0.5,0.75,1.0}`, vote temperature `{0.05,0.10}`. Maximize final 10-shot new-intent validation macro-accuracy subject to final old working no-drop versus A, final old working ≥ bar, and size. Intermediate checkpoints are reported, not used to choose a shot count. Ties by smaller radius, smaller memory, lexical config hash. Freeze settings and replay from fresh A.

Minimal memory count: 258 numeric entries/example plus all actual additional state. Duplicate source IDs are no-ops, contradictory identical exemplars are quarantined, overflow is failure unless a deterministic capacity policy was committed beforehand. Record intermediate regressions.

G3 requires final new validation **≥80%**, old working **≥A** and **≥working base bar**. **85%** is a stretch, not a moving pass threshold. Claim B's test criteria remain those in section 3.

## 7. Final freeze addendum, before opening anything

This addendum follows A selection and memory calibration, not Day-1 invented results:

| Required freeze field | Value |
|---|---|
| Selected A config, working counts, candidate-registry hash | `REQUIRED_AT_EXECUTION` |
| Projection PRNG specification/hash and chosen memory settings | `REQUIRED_AT_EXECUTION` |
| Feedback stream, A, 1-shot, 5-shot, B hashes and event counts | `REQUIRED_AT_EXECUTION` |
| Size reports, exhaustive bit-identity report, standalone dry run | `REQUIRED_AT_EXECUTION` |
| Evaluation code/runtime hashes, final manifest and remote commit | `REQUIRED_AT_EXECUTION` |
| Receipt location, custodian release, guard verification | `REQUIRED_AT_EXECUTION` |

The loader verifies the pushed manifest commit through a separate attestation before opening the exam. One final session computes both bases, A, B, and predeclared new-intent checkpoints. Exact crash continuation may finish unprocessed rows under identical hashes; changes after access compromise confirmation.

Publish strict pass/fail outcomes, secondary CIs, byte failures, amendments, interruptions, and the accuracy–size evidence. No retries to choose a better model after seeing the exam.

## 8. Authority

The executor may publish the implementation and evidence to its explicitly authorized executor-owned public repository. No spending, organizer contact, X post, challenge issue/PR, or publication to another owner's repository without separate approval. The docs build itself performs no external writes.
