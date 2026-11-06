# 06. Packaging, proof, and the final session

## Standalone inference artifact

Implement the following conceptual layout for each frozen state. These are future executor outputs, not files created by this docs build.

```text
bundle/
  manifest.json          # format, hashes, entry counts, source lineage
  values.bin             # losslessly encoded selected BF16 bit patterns
  mask.bin               # exact selected groups / structural indices
  vocabulary.bin         # retained original token IDs and maps
  tokenizer/             # all data needed by tokenizer and BPE fallback
  memory.bin             # keys, scales, labels, optional A memory
  labels.json            # exact 77 + 12 labels, ID ordering
  config.json            # architecture, special IDs, decode/memory settings
  loader/                # custom inference and decompression code
  LICENSES/              # notices required for bundled assets
```

Include all task-specific codec tables and scripts, and a precise runtime dependency lock. Exclude training scores, optimizer state, dense master weights, teacher caches, calibration text, and remote-model calls. Evaluation tooling may live outside the bundle, but inference must not reach through it to task-specific data or model parameters.

Store kept values in stable `(tensor name, row, input position)` order. Reconstruct discarded entries as zeros at load or per layer, never by downloading the base. Preserve tied embeddings, token maps, and source-index provenance. If whole layers/heads were removed, record the exact graph change and its original-index mapping.

Lossless BF16 entropy coding is allowed. DFloat11 and ZipNN motivate exploiting exponent redundancy; their reported compression ratio is not guaranteed on selected weights. A practical fallback is a pinned, documented byte-plane codec with lossless compression and separate checksums. The decoder must return exactly the original 16-bit words, not a numerically close float representation.

Try a compressed group bitset and a compressed list of kept group IDs; select the smaller lossless representation before the final manifest. Do not call an ideal entropy estimate a file size. Measure tokenizer and BPE-parent-table costs too.

Make the canonical deployment container deterministic: fixed file order, no host paths or credentials, normalized metadata, stable compression settings, no dependence on wall-clock timestamps. Count the **actual complete container bytes**. Also report sum of component bytes and unpacked size.

## Two ledgers, two states

Emit `size_report.json` and a human-readable table for A and B:

| Category | Retained numeric entries | Unpacked bytes | Packed bytes |
|---|---:|---:|---:|
| Selected embedding rows / tied head | Count once if shared | Measure | Measure |
| Selected transformer values | Include kept zeros | Measure | Measure |
| Norms and biases | 144,896 in main recipe | Measure | Measure |
| Feedback/training memory and scalar state | Every numeric payload entry | Measure | Measure |
| Mask, maps, tokenizer, code, config, notices | Structural fields separately itemized | Measure | Measure |
| Container framing | Not parameters | Measure | Measure |
| Total | Must be ≤154,371,430 | Informational | Secondary: ≤308,746,714 |

Give the auxiliary all-numeric-fields count from [01](01_rules_claims_size.md), so metadata cannot quietly conceal a memory store. If a shared A/B download is published, count each runnable state with all its required shared dependencies included, and additionally report combined release size.

Size claims describe **stored information**, not dense logical shape, runtime VRAM, or latency. Reconstructing a dense tensor full of zeros can use far more than one-tenth the RAM and may not accelerate inference. Publish cold-load peak RSS/VRAM and working-set latency from a fresh process; include decompression.

## Bit-identical lineage proof

The source checkpoint is available to this **offline verifier**, not to deployed candidate inference.

1. Hash the downloaded base and match the full SHA-256 in [01](01_rules_claims_size.md).
2. Enumerate source tensors and aliases; validate unique parameter totals and expected shapes/dtypes.
3. Decode the complete bundle in a fresh process. Reject unexplained tensors, learned gates, extra heads, or hidden numeric side files.
4. Map every retained value to a unique allowed source location. Compare **raw uint16 BF16 words** in chunks, including signed zeros. Float equality alone is insufficient; it conflates signed zeros and handles NaNs poorly.
5. Rebuild sparse/reference matrices with zeros at discarded positions and check exact group counts, embedding ordering, norms, biases, and head sharing.
6. Compare frozen-master and source bytes before/after mask search and before/after feedback. B's source model and mask hashes must equal A's; only allowed memory state and its settings differ as preregistered.
7. Round-trip values, masks, vocabulary, and memory. Compare predictions from the unpacked selected-state reference and packaged loader on working and synthetic inputs. A lossless file round trip does not, by itself, establish inference equivalence.
8. Run standalone inference with networking disabled and no base checkpoint path/cache mounted. Confirm no teacher or dense-source fallback. Test out-of-vocabulary, multilingual, empty, and maximum-development-length inputs.

Store the verifier version, input hashes, exact totals, mismatch count (must be zero), and exit status in `bit_identity_report.json`. Hash the report and verifier in the final manifest. This verification must cover **all retained values**, not a random sample.

## Freeze manifest and publication gate

Before exam access, produce and commit `final_manifest.json` containing:

* Source revision/hash and source parameter/byte denominators.
* Split and sealed-exam hashes, CLINC selection/config and sealed-test hash.
* A, B, and predeclared shot-checkpoint bundle hashes, byte sizes, numeric counts, and mode/config hashes.
* Feedback-stream and event-order hash; memory hyperparameters; any A booster.
* Tokenizer/fallback, codec, loader, runtime lock, evaluation code, and prediction-schema hashes.
* Working selection registry and exact selection rule, all preregistration amendments.
* Size and bit-identity reports; synthetic final-run dry-run result.
* Fixed metrics, bootstrap settings, pass/fail inequalities, and crash policy.

Push that commit to the executor-owned public remote. Verify the remote branch contains it and record its SHA in `final_attestation.json` separately. No self-referential manifest hash. Guard checks must not depend only on a marker file that any training script could fabricate.

After this point, do not change weights, masks, vocabulary, memory, inference modes, thresholds, exam membership, or evaluation logic. A bug found **before** opening may be fixed only through a new public manifest/attestation and a documented reason. Once the first exam receipt starts, no candidate replacement is eligible.

## Single final exam session

The planned `final-evaluate` command:

1. Checks the manifest/remote/receipt guard and obtains any custodian release.
2. Atomically creates the started receipt and opens the sealed Banking77 exam and CLINC test. This is one authorized session, not a development loader.
3. Computes both unmodified-base protocols, frozen A, and frozen B on the exact same old-intent rows. Each candidate is run with its registered mode. Keep baseline processes separate from candidate processes to prevent source-checkpoint access by candidates.
4. Computes A, the 1-shot and 5-shot checkpoints, and B (10-shot) on the frozen new-intent test. All use the same 89-label interface and no oracle routing.
5. Uses those saved per-example predictions for mixed-domain summaries, the bar, strict claims, paired bootstrap CIs, non-inferiority diagnostics, retention, and size results. No extra query is needed for a mixed metric.
6. Emits immutable prediction arrays, reports, and a finished receipt. Publish failures as well as passes. Do not tune on the output.

“Once” means one final session with the predeclared set of model states and modes, not a ban on comparing multiple frozen states. The exam may be streamed into those evaluators within that session; do not subsequently reopen it to choose a winner.

Precommit a crash policy: atomic per-example prediction journaling permits an **exact continuation of the same session**, same hashes and settings, on unfinished rows only. Record the crash, row IDs, and recovery. Do not rerun completed predictions, add candidates, patch the evaluator, or tune from partial output. If recovery needs any code/config change, the confirmatory run is compromised; publish it as interrupted and label later results exploratory.

## Results and submission

Publish a compact table including:

| State | Old correct / N | Old accuracy | Difference vs bar, CI | New-intent test | Entries | Bundle bytes |
|---|---|---|---|---|---:|---:|
| Base card greedy | Measured | Measured | Baseline | Not required | 1,543,714,304 | 3,087,467,144 weights only |
| Base constrained greedy | Measured | Measured | Baseline | Not required | Same | Same |
| A | Measured | Measured | Measured | Measured | Measured | Measured |
| B | Measured | Measured | Measured | Measured | Measured | Measured |

The base byte denominator is its **weights file**, whereas the candidate byte claim includes the complete inference artifact. Label that conservative asymmetry. Report other base assets separately.

Include reproducible setup, data attribution, known pretraining-contamination limits, candidate history, feedback replay, runtime/resource evidence, and whether the byte target failed. Check Apache-2.0 model notices, Banking77 CC BY 4.0 attribution, and CLINC CC BY 3.0 attribution before distributing assets or prediction text.

The executor can push only to its authorized public repository. It must **not** post on X, open a challenge issue/PR, or message tokenbender on its own. Prepare a submission draft with links, exact claims, and caveats for the user's approval. Do not claim to be the first qualified entrant without external confirmation.
