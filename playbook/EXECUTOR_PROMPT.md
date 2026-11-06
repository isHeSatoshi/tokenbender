# Executor handoff

Copy the prompt below into the implementation agent. The playbook is docs only; implementing this handoff is a **separate task** that will create code and, if authorized, public artifacts.

---

Implement the experiment specified in this playbook directory. Read `README.md`, then `01_rules_claims_size.md` through `08_references.md`, and `PREREGISTRATION_TEMPLATE.md`. Use relative Days 1–7 and gates G0–G4; ignore calendar dates and timestamps as deadlines. Keep seed `20261106`.

## Goal and boundaries

Select a compact subnet of `johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent`, revision `6c13ee7e6a4d229615dccdd089869818ad389388`. Its root weights SHA-256 must be:

```text
2084675316db424191877b0ee41689ba28b83e07c8444fc1f44f037c769e82dc
```

Denominators are 1,543,714,304 unique source parameters and 3,087,467,144 weights-file bytes.

* **Never update source-model values.** No fine-tuning, LoRA, kept-value quantization, weight reconstruction, new trained head/projection, or numeric adaptor optimization. Only disposable mask scores are optimized.
* Primary cap: **154,371,430 retained numeric entries**, including every stored memory payload number. Aim for **152,800,000 source-model entries** before auxiliary state.
* Secondary cap: **308,746,714 bytes** for the complete standalone inference container, including tokenizer, masks, loader, and memory. Report this separately; do not equate parameters with bytes.
* Claim A requires frozen pre-feedback A old-intent accuracy **≥ the higher of the two final base scores**. Claim B requires the same for post-feedback B plus new-intent test **≥80% and >A**. The 85% new-intent figure is a stretch.
* The two base modes are literal card-prompt greedy exact match and valid-label-constrained greedy. The reported 93.4% is not the measured final bar.
* A paired 95% bootstrap CI and −1.0-point non-inferiority result are secondary and never replace a strict failure.

## First actions

1. Inspect project instructions and existing user files. Work in the designated execution repository; do not overwrite user work or alter this protocol silently. Create a visible todo list and a candidate/event registry.
2. Record actual hardware and set up uv/Python 3.12/CUDA-compatible wheels. Resolve and freeze exact dependencies; suggested pins in the playbook are not tested guarantees. Do not spend money or rent compute without approval.
3. Download pinned sources and verify file hashes, BF16 counts, shapes, and tied storage. Use the merged root checkpoint, not the separate LoRA folder.
4. Prepare and seal Banking77 before any adaptation. Expect 3,076 exact train-decontaminated test rows; split each label 50/25/25 exam/working/feedback using the exact rounding and seeded ordering in 02. Abort on unexpected counts or within-test duplicates until resolved pre-search.
5. Complete the preregistration's measured fields and publish the split script, source lock, exam hash, and registration to the explicitly authorized executor-owned public repo before pilots. The example repo name is not proof that you own it or permission to use another account.

If public publishing credentials or repository ownership are unresolved, ask for that specific authorization. You may develop training/synthetic smoke tests, but G0 and registered search are blocked. Do not claim a push happened when it did not.

## Implement the real interfaces

Create tested commands with the following responsibilities, using one documented CLI layout of your choice:

| Interface | Required responsibility |
|---|---|
| `prepare-splits` | Pinned source load, decontamination, deterministic allocation, hashes, sealing; one-shot only |
| `baseline` | Working-only evaluation under both fixed base modes |
| `build-vocabulary` | Permitted Banking77 text only, required tokens, tested recursive BPE fallback |
| `cache-teacher` | Train/EDA only, retained-vocabulary dense teacher, top-64 plus tail at label/EOS |
| `search-mask` | Frozen BF16 masters, score-only Adam, exact global top-k, count-aware schedule |
| `replay-feedback` | Immutable A, registered correction/new-intent stream, joint 89-label memory |
| `pack` | Sparse selected values, lossless codec, complete standalone loader and ledgers |
| `verify` | Exhaustive source-bit lineage, standalone inference and guard tests |
| `freeze` | Final manifest, separate commit attestation, public remote confirmation |
| `final-evaluate` | Only authorized locked-split loader; one session for both bases and frozen states |
| `report` | Exact counts, strict claims, CIs, byte outcome, failures and caveats |

These command names are requirements to implement, not installed tools. Never call a placeholder command and interpret its absence as an experiment failure.

## Method and checkpoints

Implement vocabulary trim, then input-dimension four-entry groups. Exact body budget depends on actual vocabulary and auxiliary reserve. There are 327,548,928 candidate body groups, not a guaranteed fixed 10% body density. Learn global masks with Wanda initialization and the straight-through surrogate. Distill at temperature 2, top-64 retained-token buckets plus tail, with 0.3 target CE, on train plus approximately 40k base-labeled EDA variants.

Use the exact hyperparameter grids and schedules in 04 and the preregistration. Test custom backward against a tiny reference; recompute masked weights rather than saving them. Measure fresh-process first-update/transition/evaluation peaks and top-k cost before full runs. Estimated 5.24 GB score/gradient/Adam payload is not proof of 16 GB GPU fit.

G0 requires original-vocabulary all-one-mask identity. Vocabulary trimming is a separate measured ablation and need not preserve unconstrained outputs. G1 requires ≥99% working agreement at 30% total size. G2 targets within 0.3 point of working base accuracy and ≥97.5% agreement at the entry cap.

Select A only by working results and the registered tie-breakers. Boosters need ≥0.3-point working gain and full accounting. Follow the ordered fallback ladder in 07 rather than improvising forbidden weight edits.

Freeze A before correction labels or CLINC examples enter memory. Implement 256-D scaled-int8 exemplar keys and distance-gated blending, not GRACE's trained adaptor. Use exactly the twelve new intents and nested 1/5/10-shot stream in 05. CLINC never enters mask learning or vocabulary building. G3 requires ≥80% new validation and no old working loss, still ≥working bar.

Keep the same source-model and mask bytes in A and B. Verify memory replay, duplicate/conflict behavior, capacity, restart, and immutable state snapshots.

## Exam and proof

The exam is not a debug resource. No exam text for token statistics, vocabulary, caching, augmentation, memory, thresholds, or candidate selection. Do not reconstruct the deterministic public split to evade the guard.

Before opening, fully pack A and B, count all numbers/bytes, compare every kept BF16 word to its source, and test a fresh standalone load without base paths or network access. Publish the final manifest and prove its commit exists on the authorized remote. Use a separate attestation to avoid a commit hashing itself.

Then run one final session containing both fixed base modes, A, B, and the predeclared new-intent checkpoints. All mixed-domain predictions expose 89 labels with no oracle router. Use exact crash continuation only as preregistered; no code changes or model retries after access. Publish failures honestly.

## Communication and permission

Keep progress brief: current gate, exact metrics, resource measurements, next bounded action, blockers. Never fabricate model scores, source hashes, file sizes, public commitments, or organizer acceptance. A gap at 10% should lead to a clear failed claim and accuracy–size curve.

Only pushes to the expressly designated executor-owned public repo are authorized by this handoff. No organizer messages, X posts, challenge issue/PR, other-owner publication, paid services, or destructive cleanup without additional approval. Prepare a submission draft for the user rather than posting it yourself.

Return the implementation and evidence paths, achieved gates, exact claim/size outcomes, checks run, checks skipped, and any remaining blockers. Do not call this a winning proof unless the measured strict criteria pass.
