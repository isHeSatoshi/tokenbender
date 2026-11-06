# 01. Rules, claims, and size

## What the challenge actually says

The [challenge README](https://github.com/DeusExperiments/ten-percent-challenge) asks for one trained model, about 10% of its size, no weight training, matching the base, an unseen exam set aside before building or choosing, and feedback learning that keeps the old ability. Only the first complete working proof qualifies.

Our protocol is deliberately stricter in some places. **Selecting, never changing, source weights** and the two exact 10% ceilings below are our commitments, not a quoted organizer ruling. The README does not resolve parameters versus bytes, statistical noise, or the treatment of a numeric feedback memory. Disclose these interpretations. Optionally ask the organizer, with approval, but never silently weaken the registered claims.

The starting model is the already trained **merged Banking77 checkpoint**, not the general Qwen model and not an 18M-parameter LoRA file. Its previous fine-tuning predates the experiment. The executor must not repeat fine-tuning or merge a new adapter.

## Immutable identity

| Item | Value |
|---|---|
| Model | `johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent` |
| Full revision | `6c13ee7e6a4d229615dccdd089869818ad389388` |
| Weights | Root `model.safetensors` |
| SHA-256 | `2084675316db424191877b0ee41689ba28b83e07c8444fc1f44f037c769e82dc` |
| Unique stored source parameters | 1,543,714,304 |
| Weights-file size | 3,087,467,144 bytes |
| Raw BF16 payload | 3,087,428,608 bytes |
| Safetensors overhead | 38,536 bytes |

The full hash and counts were checked against pinned Hugging Face metadata. The executor must independently hash the downloaded file and enumerate its tensors before using it. Stop on any mismatch.

## Claims and statistics

Let $a_g$ and $a_c$ be the two unmodified-base accuracies on the same $N$ locked examples. Set $b=\max(a_g,a_c)$. Every example counts; invalid replies, timeouts, and missing predictions are wrong, not dropped rows.

* **Claim A, step 3:** accuracy of frozen state A is at least $b$.
* **Claim B, step 5:** accuracy of frozen state B after all feedback is at least $b$, and the new-intent criteria in [05](05_feedback_memory.md) pass.
* Both states must meet the retained-entry ceiling. State B must not be exempted because the added numbers are called “memory.”
* The strict score comparison is integer: `correct_A >= max(correct_base_g, correct_base_c)`, and likewise for B. Do not round percentages to decide.
* Report a paired percentile-bootstrap 95% CI for each difference, in percentage points, with 10,000 replicates and seed `20261106`. Resample the same example indices for candidate and both baselines; recompute the maximum of the two baseline accuracies in each replicate.
* Secondary non-inferiority passes only if the CI's lower endpoint is **greater than −1.0 percentage point**. A secondary pass with a negative point estimate is still a headline failure.
* Also report B minus A on old intents, exact counts, per-label accuracy, invalid rate, and working-set base agreement. Accuracy is micro-accuracy; macro-accuracy is diagnostic.

The bar is first known in the final session. Pre-final gates use the corresponding working-set maximum as a proxy. The published 93.4% is not a target to copy into final results.

## Retained-entry ledger

The primary ceiling is `floor(1,543,714,304 / 10) = 154,371,430`.

Count unique source-model values physically retained, including zero-valued kept weights, norms, biases, embeddings, and any retained head. Count tied embeddings once **only if the deployed package and loader really share them**. Count duplicate numerical copies if actually stored twice.

Also count every stored numeric payload used by feedback: int8 key coordinates, scales, label IDs, counts, radii, mixture settings, cached probabilities, random-projection coefficients if stored, and token IDs if stored as exemplars. One int8 coordinate is one entry, not one-sixteenth of a BF16 parameter. No dividing memory by its bit width.

Structural representation fields, such as sparse-mask bits, tensor shapes, source indices, and a vocabulary ID map, are not source-model parameters. Itemize them separately as metadata and count all their bytes. To prevent hidden memory in “metadata,” also give an auxiliary all-numeric-fields count. Learned mask scores must not ship.

State the convention for textual labels and optional raw exemplar text. Do not hide unbounded learned information in strings: retain only keys and label IDs in the main memory design, not a free text database. Fixed output-label names are metadata and count toward bytes.

The random projection is untrained algorithmic machinery. Prefer regeneration from a specified integer PRNG, with its seed and settings counted; if a matrix is shipped, count every coefficient. Report regenerated dimensionality separately so readers see the computational size too.

### Exact checkpoint arithmetic

For hidden size 1,536, vocabulary 151,936, 28 layers, 12 query heads, and 2 KV heads:

| Component | Entries |
|---|---:|
| Tied embedding/output matrix | 233,373,696 |
| Dense norms and attention biases | 144,896 |
| Maskable transformer matrix entries | 1,310,195,712 |
| Four-entry candidate groups | 327,548,928 |

Keep norms and biases unchanged. Let $V$ be the actual retained vocabulary rows, $M$ all auxiliary numeric entries, $g$ the group width, and $K$ the kept groups. Then

$$P=1536V+144896+gK+M.$$

For a **source-model** target of 152,800,000 entries, group width four, and an illustrative $V=16,384$, there are 25,165,824 embedding entries and room for 31,872,320 groups, or 127,489,280 body entries. Body density is about 9.7306%, **not automatically 10%**. The global density schedule must end at the budget computed from the actual vocabulary, not the headline fraction.

This leaves 1,571,430 entries for auxiliary state under the primary ceiling. If vocabulary grows or feedback exceeds the reserve, recompute the body budget **before** search and freezing. Do not prune A further after seeing the exam.

## Byte ledger, a separate claim

The secondary ceiling is `floor(3,087,467,144 / 10) = 308,746,714` bytes, using decimal bytes, not MiB. Include all task-specific inference files: values, masks/indices, tokenizer and BPE fallback data, config, labels, memory, projection state, custom loader, and codec tables. Include the deterministic archive/container overhead. Exclude training checkpoints, teacher caches, reports, and generic installed runtime dependencies, which must be listed separately.

At 152,800,000 BF16 model entries, raw values already use 305,600,000 bytes, leaving only 3,146,714 bytes before masks or tokenizer files. A raw four-group bitmask costs 40,943,616 bytes. In the illustrative vocabulary case its ideal binary-entropy size is about 18,850,262 bytes, **a theoretical estimate, not a measured codec result**.

Thus this recipe budgets for lossless BF16 compression plus mask compression; do not promise that it will reach the byte cap. At an illustrative 11 bits per retained value, the value payload alone would be 210,100,000 bytes. Actual values selected by importance can compress differently from full dense weights. Measure both states after round-trip verification.

## Allowed and forbidden

Allowed: discard source values; copy selected values unchanged; rearrange tensors with a reversible source-index map; learn disposable mask scores; use frozen-teacher outputs; deterministic token decomposition; lossless byte compression; countable exemplar memory; deterministic inference-time averaging or full-label scoring if selected before the exam.

Forbidden:

* Fine-tuning any source parameter, LoRA, adapter training, optimizer updates to source weights, or a newly trained classifier/head.
* Quantizing kept weights, changing their dtype in storage, rounding, clipping, rescaling, weight averaging, bias correction, or modifying retained norms.
* SparseGPT-style weight reconstruction or compensation that changes surviving values. A method's name is not permission to use its weight-changing stages.
* Soft inference masks or learned gates that multiply kept values by anything other than one. Final masks are binary.
* Weight-space programming that writes new values. It may propose selection indices only.
* Teacher/base inference, a remote model, undeclared retrieval, or the full source checkpoint as a deployment dependency.
* Exam text in vocabulary building, augmentation, teacher caches, memory, calibration, hyperparameter selection, or error-driven retries.
* Training projection weights, numeric adapters, or feedback values by gradient descent. Our GRACE-inspired memory is not the full GRACE training procedure.

“Bit-identical” applies to every retained **source-model value**. Memory keys are derived data, not selected source weights; int8 key storage is therefore not weight quantization. This distinction is an explicit rule interpretation. If the organizer disallows numeric memory, report that limitation rather than relabeling it as permitted weights.
