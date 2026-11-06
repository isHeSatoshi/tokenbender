# 04. Vocabulary and frozen-weight mask search

This is an experimental recipe, not evidence that 90% removal will retain accuracy. Implement the reference dense masking path first; optimize only after its correctness tests pass.

## 1. Trim vocabulary without learning weights

Build the retained original-ID set from:

* Every actual special token and chat-template token.
* All tokens in the exact system prompt, assistant prefix, the 77 label continuations, and EOS.
* Original-tokenizer IDs appearing in Banking77 train, working, and feedback **text**, not locked text.
* Original IDs `0..8191`.
* All byte-alphabet leaf tokens required to represent arbitrary UTF-8. Assert their presence; do not assume their IDs fall below 8,192.

No CLINC text or label tokens enter this construction. Its new output names will be returned by the memory's external label dictionary. Hash and freeze the retained-ID list before teacher caching or search. Record actual $V$; 16,384 rows is an accounting illustration, not a vocabulary-size guarantee.

Use the original tokenizer to tokenize incoming text. For a dropped non-special token, recursively replace it by its two BPE parent tokens until every leaf is retained. Build a deterministic parent map from the pinned merge table: use the earliest ranked merge producing the token's byte string, whose children have lower merge rank. Detect ambiguity, missing parents, and cycles. Byte leaves must terminate the recursion. Special tokens are atomic and always retained.

```python
def expand_token(token_id, keep, parents):
    if token_id in keep:
        return [token_id]
    left, right = parents[token_id]  # missing parent is a build error
    return (expand_token(left, keep, parents) +
            expand_token(right, keep, parents))
```

Test on all non-special original token IDs and synthetic multilingual/byte-heavy input. Compare concatenated **token bytes**, not isolated-token Unicode decoding, which can insert replacement characters. Require `decode(expanded_sequence) == decode(original_sequence)` on complete valid text, and test the prompt/template round trip.

Store selected embedding rows in ascending original-ID order with original↔compact maps. Retie the output head to the same compact storage. Preserve original-ID ordering for deterministic argmax ties. Do not renumber special IDs in just one config file and leave the generation code or trie inconsistent.

**Limits:** decomposition preserves text, not hidden states. New inputs using dropped tokens get different token sequences. Even on unchanged input IDs, deleting output rows changes softmax normalization and may change unconstrained greedy output. Therefore G0's exact base-identity test runs **before vocabulary trimming**; measure vocabulary-only behavior separately.

Keep the original tokenizer data if needed by the fallback; its bytes count in the deployed package. A smaller equivalent tokenizer/fallback representation is allowed only after equivalence tests. Do not discard rare tokens from the fallback path just to reduce the reported size.

## 2. Allocate the actual body budget

Dense tied embedding rows, norms, and biases stay unmodified. Mask the seven transformer matrices per layer: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and `down_proj`.

Group each matrix along its **input dimension** into four adjacent entries, without crossing output-row boundaries. Dimensions in this checkpoint divide by four. This is **block selection with groups of four**, not NVIDIA 2:4 sparsity and not a promise of hardware sparse acceleration.

After all auxiliary-state reservations, calculate the target using the formula in [01](01_rules_claims_size.md):

```python
SOURCE_TARGET = 152_800_000
fixed = 1536 * actual_vocab_rows + 144_896
K = (SOURCE_TARGET - fixed) // 4
assert 0 <= K <= 327_548_928
body_density = K / 327_548_928
```

If auxiliary state in A would exhaust the 1,571,430-entry reserve, reduce the **pre-search** source target further. If the actual vocabulary makes the budget impossible, fail the gate; do not turn off required tokens. A 30%-total-size pilot uses the analogous total budget, not a hard-coded 30% matrix density.

## 3. Scores and exact selection

Freeze all source parameters: `requires_grad_(False)`, immutable master BF16 buffers, and optimizer groups containing score tensors only.

Initialize each matrix from Wanda-style importance on 256 deterministically sampled **training** prompts. Accumulate input-channel squared activations over non-padding positions. Use $|W_{ij}|\sqrt{\sum_t x_{tj}^2}$, sum over each four-entry group, then standardize group scores **within each matrix** with population standard deviation and an epsilon of `1e-8`. Do not calibrate on working, feedback, or CLINC examples.

Assign one FP32 scalar per candidate group. Choose a **global** top-$K$ across all matrices at every optimizer step. Tie by stable group ID: `(tensor_name sorted lexically, output_row, input_group)`. Learned scores can be signed; rank scores directly, not their absolute values.

Use an exact threshold and exact tie allocation. A “threshold ≥ kth score” can accidentally keep too many groups. A memory-bounded implementation may histogram scores, refine the boundary bucket, then select the exact residual count within that bucket. Unit-test its results against a full stable sort on small tensors. Never substitute a stale or approximate top-k without a logged, pre-final search amendment.

A forward mask is binary. In the score backward pass, use the straight-through surrogate $\partial m/\partial s=1$. Gradients flow through frozen operations to score choices; weights do not accumulate gradients. Save each forward's hard mask or equivalent immutable selection so checkpoint recomputation uses the **same** mask, not the next optimizer step's threshold.

## 4. Teacher and adaptation data

Use the official Banking77 training rows plus approximately **40,000** deterministic EDA variants of those rows. Start with replacement/insertion/swap rate `0.05` and deletion probability `0.05`; preserve nonempty text and never edit working or feedback examples into training data. Pin any synonym resource and its hash; omit synonym operations if no reproducible resource is available, with a pre-search record.

Keep original training labels for originals. Assign variants pseudo-labels using the frozen base's valid-label-constrained greedy protocol. Log how often the pseudo-label changes. Do not discard variants by working-set performance or locked-set similarity. Deduplicate within adaptation data only; report the actual count if fewer than 40,000 remain.

Freeze the vocabulary before producing teacher outputs. Original variants with dropped input tokens use the registered fallback. To isolate mask learning, cache a **dense-body, vocabulary-trimmed teacher** on exactly those compact input sequences, constructed only by selecting base rows. Keep a small full-base comparison cache on training examples as a diagnostic. Neither teacher is a deployment dependency or a substitute for the two unmodified final baselines.

At every teacher-forced label-token position and at EOS, cache the top-64 original token IDs **within the retained vocabulary**, their temperature-2 probabilities, and the aggregate remaining probability. Cache the normalization needed to reconstruct probabilities, example and position IDs, target tokens, source hashes, and dtype. Use FP32 cache probabilities unless a measured cache-precision ablation justifies otherwise.

This retained-vocabulary teacher is an explicit adaptation of “distill the base”: full-base top-64 entries can contain dropped IDs, making KL to a zero-support student invalid. Do not silently map such teacher mass to arbitrary surviving tokens. The baseline remains the unmodified full base.

## 5. Loss and initial hyperparameters

For each supervised position, use the teacher's 64-token buckets plus one “other retained tokens” bucket. Aggregate the student's temperature-2 softmax over exactly the same buckets. Compute KL including the tail bucket, multiply by $T^2$, and average over non-padding label/EOS positions. Add full-vocabulary cross-entropy on the target token, weight **0.3**:

$$L=2^2\,\mathrm{KL}(p_{\mathrm{teacher},2}\Vert p_{\mathrm{student},2})
  +0.3\,\mathrm{CE}(y,p_{\mathrm{student},1}).$$

Both distributions use the same retained output support. Use numerically stable log-sum-exp for the tail, and verify bucket probabilities sum to one. No loss on prompt tokens. No teacher responses from the working or locked sets.

| Setting | Initial registered values |
|---|---|
| Group size | 4; fallback 2 only as recorded candidate |
| Optimizer | Adam on scores, betas `(0.9, 0.999)`, eps `1e-8`, weight decay `0` |
| Pilot learning rates | `0.001`, `0.003`, `0.01` |
| Pilot budgets | 30% and 10% total retained entries; 200 optimizer steps each |
| Full lengths | 2,000 steps initially; one 4,000-step candidate if needed |
| Microbatch / effective batch | 1 / 32 examples using gradient accumulation |
| Gradient clipping | Global score-gradient norm `1.0` |
| LR schedule | 5% linear warmup, then cosine decay |
| Evaluation | Every 100 optimizer steps on working only |
| Seed | `20261106` |
| Sequence handling | Dynamic padding; measure maximum fallback-expanded length, no silent truncation |

For density, hold all groups for the first 10% of optimizer steps; decay over the next 70%; hold the final target for the last 20%. During the decay, use

$$d(u)=d_f+(1-d_f)(1-u)^3,\qquad 0\leq u\leq1.$$

Round to an exact group count, clamp within valid bounds, and hit the precomputed $K$ exactly at the end. Use the union of fixed embedding/norm/bias entries plus kept groups for size checks. Checkpoints above the cap are useful diagnostics, not eligible deployment candidates.

## 6. GPU memory and recomputation

The rough “332M scores” from planning becomes **327,548,928 scores** when only these transformer matrices are maskable. FP32 scores, gradients, and two Adam moments take **5,240,782,848 bytes** together (about 4.88 GiB), before optimizer temporaries. The full base payload is about 2.88 GiB. This arithmetic suggests room on a 16 GB GPU but **does not prove fit** with activations, logits, KV/cache state, top-k, or optimizer workspace.

Use gradient checkpointing with `use_cache=False` during search, a microbatch of one, and a non-foreach Adam path if foreach temporaries cause a spike. Cache teacher outputs on disk, stream batches, avoid a second teacher model in VRAM, and calculate large-vocabulary logits only at supervised positions when the model implementation supports it.

Avoid autograd retaining a separate dense masked weight for every matrix. A custom linear function can keep references to frozen $W$, input $x$, and the hard group mask, then reconstruct one masked matrix in backward:

```python
# Reference equations; production backward must tile the grad_W calculation.
Wm = torch.where(expanded_hard_mask, W, torch.zeros_like(W))
y = x @ Wm.T
# Backward, with batch/token axes flattened:
grad_x = grad_y @ Wm
grad_W = grad_y.T @ x
grad_scores = (grad_W.float() * W.float()).reshape(rows, -1, 4).sum(-1)
# Return grad_x, None for frozen W, grad_scores, None for hard mask.
```

Implement this as `torch.autograd.Function`, not the above ordinary autograd graph. Save no masked matrix or `grad_W` across calls. Tile output rows for the score gradient to bound temporaries; support noncontiguous tensors, padding, dtype casts, accumulation, and a frozen optional bias. With `torch.where`, kept BF16 values, including signed zero, are copied exactly. Dropped entries reconstruct as canonical positive zero.

Test forward and surrogate gradients against a tiny ordinary-autograd reference **before** optimization. With activation checkpointing, verify mask identity on recomputation. Do not cache hidden states across masks, since they become stale.

Before choosing full-run length, measure a fresh process through initialization, the first Adam update (which allocates state), a density transition, 100 steady steps, working evaluation, and save/reload. Record GPU peak allocation/reservation, host peak RSS, seconds per step including top-k, and disk/cache use at the longest observed input. Leave practical headroom, roughly 1.5 GiB VRAM and 4 GiB host RAM where possible; a tensor-size estimate is not a benchmark.

If global top-k or recomputation dominates runtime, fix that implementation or use the disclosed fallback ladder. Do not claim an unmeasured finish time. Structured pre-pruning and two-entry groups change the search space and must appear as distinct candidates.

## 7. Selection, ablations, and freeze

Write a candidate registry with config hash, source data hashes, actual counts, peak resources, working correct count, agreement, and rejection reason. Compare dense base, vocabulary-only, Wanda-only, learned 30%, and learned 10% to expose where quality is lost.

Among complete candidates under the **retained-entry cap**, select highest working-set accuracy. Ties prefer agreement with the working-set winning base mode, then fewer retained entries, then lexical config hash. If claiming the secondary byte target, only a byte-eligible candidate may make that extra claim; do not hide a byte failure of the primary selected candidate.

Adopt a training-exemplar memory or deterministic rewording ensemble only if it improves working accuracy by **at least 0.3 percentage point**, with its entire deployed cost counted. Since the working set is small, decide from exact counts: require `100 * (new_correct - old_correct) >= 0.3 * N_work`. No rounded-percentage pass.

Before feedback, freeze A's model, tokenizer, decoding mode, any adopted boosters, and any pre-feedback memory. A may include a counted training-exemplar booster, but **no feedback-pool or CLINC examples**. B starts from A, with the same source model and updated memory only.
