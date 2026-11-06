# 03. Baseline and evaluation harness

Implement this before mask search. All commands and modules here are **planned executor interfaces**, not files supplied by the playbook.

## Exact card-protocol baseline

Use the pinned merged model in BF16 with its own tokenizer and chat template. Keep original dataset label spelling, particularly `Refund_not_showing_up` and `reverted_card_payment?`. Do not use sorted label positions as dataset IDs without an explicit mapping.

System message, exactly:

```text
Classify the banking customer message into its intent label.
```

User content is exactly the dataset message. No label list, demonstrations, retrieval, or explanation request.

```python
messages = [
    {"role": "system",
     "content": "Classify the banking customer message into its intent label."},
    {"role": "user", "content": text},
]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True,
    return_tensors="pt", return_dict=True,
)
# Move inputs to model.device. Slice output after each prompt, not after padding.
generated = model.generate(
    **inputs, do_sample=False, max_new_tokens=20, num_beams=1,
)
```

Decode only generated tokens with `skip_special_tokens=True`. The primary **exact match** compares the decoded string directly with the dataset label. No `.strip()`, lowercasing, punctuation removal, fuzzy mapping, or label extraction. Record raw outputs. Whitespace-tolerant matching may be a named diagnostic only.

The card specifies the prompt, greedy decoding, and a 20-token limit but does not completely document its original string-normalization code. Our literal comparator is the registered exact-string operationalization; report this limitation rather than claiming perfect replication of every card implementation detail.

The card reports 93.4% on 3,076 clean official test examples. Do **not** reproduce that number by opening official test data during development. Reproduce the stated procedure on the working set; its score will differ. Measure the card protocol on the locked exam only in the final session.

## Second baseline: valid-label-constrained greedy

Keep the same prompt, tokenizer, weights, and deterministic backend. Build a prefix trie of the 77 label token sequences, using the exact assistant-prefix boundary. Test tokenization against the complete serialized conversation; do not assume tokenizing a string in isolation always gives the correct continuation.

At each step allow only tokens that extend a valid label prefix. Permit the configured EOS token at a complete label; if that label also prefixes a longer label, permit both EOS and the valid continuation tokens. Choose the highest original logit among the allowed tokens. Break exact ties by lowest original token ID. Use no beam search and do not sum full-label scores here.

Use `max_new_tokens=20` after verifying all registered label-plus-EOS paths fit. If a path does not fit, resolve before registration, not after observing an exam failure. A capped incomplete answer counts wrong. The trie must prevent malformed labels, not repair them after generation.

The two frozen baseline procedures are both evaluated on the final locked rows. The higher score is the bar, even if it is not the better mode on the working set.

## Compact model modes

Evaluate both greedy and constrained greedy on the working set. Register one deployment mode through the selection rule before exam access. Allowed full-label scoring and deterministic rewording ensembles are **candidate-only** alternatives, not replacements for either baseline.

Where probabilities are needed, assign each full label $\ell$ its teacher-forced sequence log probability **including EOS**, with no length normalization. Apply softmax across those 77 sequence scores to obtain a label distribution. This can support memory blending without changing the registered greedy prediction outside the memory's radius. Report the extra forward-pass cost.

Full-label scoring as a fallback predicts the highest such sequence score. It is not constrained greedy: their answers can differ.

## Records and metrics

Each prediction record has `example_id`, `true_label`, `predicted_label`, validity, decoding-mode ID, model-state hash, and elapsed inference time. Store generated text in development logs only where permitted; final exam logs are released only after the sealed run finishes and licensing is checked.

For each state and split report:

* Correct count / total and unrounded micro-accuracy; invalid count and coverage.
* Per-label accuracy, macro-accuracy, and confusion counts as diagnostics.
* Agreement with each base mode. Working selection ties use agreement with the **working-set winning** base mode, ties favoring the card protocol.
* Warm and cold latency, tokens processed, peak VRAM, host peak RSS, and artifact bytes. No speedup claim from sparsity alone.

Bootstrap the final differences as defined in [01](01_rules_claims_size.md). Pseudocode:

```python
# a, b, c are correctness arrays aligned by stable example ID.
rng = np.random.default_rng(20261106)
deltas = []
for _ in range(10_000):
    ix = rng.integers(0, len(a), size=len(a))
    deltas.append(100 * (a[ix].mean() -
                        max(b[ix].mean(), c[ix].mean())))
ci = np.percentile(deltas, [2.5, 97.5], method="linear")
```

Repeat for A and B. Keep bootstrap computation outside model inference; it must not trigger more exam queries or choose a different candidate.

## Required development tests

1. **Source and prompt:** verify file hash, tensor counts, BF16 dtype, tied embedding storage, literal system prompt, chat-template hash, and 77 exact labels.
2. **Greedy:** single-example versus padded-batch results; correct prompt slicing; EOS and token cap; invalid reply counts; two deterministic repeats.
3. **Trie:** all labels reachable, label-prefix collisions, EOS handling, correct original/compact ID mapping, deterministic ties, and equality against a tiny hand-constructed logits/trie example.
4. **100%-kept identity:** use the **original vocabulary** and an all-one body mask. Require byte-identical retained values, equal logits under the same deterministic dense operations, and exactly equal generated outputs on the working set. If arithmetic order differs, first use the reference dense path; do not excuse drift by rounding. This is G0.
5. **Vocabulary-only ablation:** with the dense body, compare retained-row logits for inputs whose tokenization is unchanged. Record any unconstrained prediction changes. Trimming can remove the winning output token, so full-output identity is not guaranteed.
6. **Mask gradient:** finite-difference/check analytic surrogate gradients on a tiny linear example with a **fixed mask**. Check global top-k and tie behavior separately. A numerical derivative of the discontinuous hard top-k itself is not a valid test of the straight-through surrogate.
7. **Freeze:** source weights have no optimizer membership or gradients; rehash their bytes after each pilot and at checkpoints. Only scores can change.
8. **Isolation:** training/cache/tokenizer/memory loaders refuse locked rows and CLINC test; source-ID assertions prevent cross-partition use.
9. **State lifecycle:** serialize/reload A; add feedback; serialize/reload B; reload A again and verify it did not mutate. Predictions and memory counts survive restart.
10. **Final dry run:** synthetic exam only, through the real standalone loader, receipt/guard, all four main predictions, and report generator.

Do not debug an inference failure by opening the locked exam. Every edge case above can be tested on public training messages or synthetic inputs.
