# 05. Learn new intents and keep old ones

Feedback changes stored examples, not source weights, masks, projection coefficients, or a trained classifier. Call this a **GRACE-inspired distance-gated exemplar memory**, not a reproduction of GRACE's trainable adaptor values.

## Representation and memory format

Extract the compact model hidden state at the **last non-padding prompt token**, before any answer token. Choose layer index from `{7, 14, 21, 28}`, where zero denotes embeddings and 28 the last transformer output, using working and CLINC validation only. Register the exact pre-/post-norm location.

If A already has a training-exemplar booster, its feature layer and projection were selected on working data before A froze. Reuse that feature pipeline for B, restricting the layer grid to A's chosen layer. Do not silently re-encode A's keys under a different representation.

L2-normalize, project from 1,536 to **256 dimensions**, then normalize again. Define a deterministic untrained Rademacher projection: generate `1536 * 256` signs in row-major order with a pinned, explicitly implemented integer PRNG (for example SplitMix64), seed `20261106`, and divide by `sqrt(256)`. Ship the algorithm and golden-output hash; do not rely on a library's default RNG staying unchanged.

The projection is algorithmic, not fitted to examples. Regenerate it at load time to avoid storing 393,216 coefficients; report that dimensionality anyway. If coefficients are stored, include all 393,216 entries and their bytes.

For each normalized projected key $z$, store symmetric int8 coordinates
$q_i=\operatorname{clip}(\operatorname{round}(z_i/s),-127,127)$
with $s=\max_i|z_i|/127$. Specify ties-to-even rounding. Store a FP32 scale, integer label ID, and fixed metadata. For a zero vector use an explicitly registered nonzero scale and a zero key; test this path.

At lookup, reconstruct `q.float() * scale`, normalize, and use cosine distance `1 - dot(query, key)`. Do **not** compare a float query to raw, unscaled int8 keys. Brute-force search is sufficient for this small memory and avoids an opaque learned index.

A minimal exemplar stores **256 coordinates + 1 scale + 1 label ID = 258 numeric entries**, regardless of their byte widths. Any additional counters, IDs, radii, timestamps, cached distributions, or deduplication fingerprints must be itemized. Do not ship timestamps unless actually needed. Memory exemplar text is not required at inference.

## Prediction rule

For nearest distance $d$, defer to A's registered prediction whenever the memory is empty or $d>r$. Inside radius $r$, compute a distance-weighted vote over the nearest $k$ admitted exemplars, weights `exp(-distance / tau)`, normalized to a label distribution.

When constructing an optional training-memory booster for A, its fallback is the compact decoder without that booster. B's fallback is the complete frozen A, including any booster, not a recursive call to B's updated memory. Shared immutable training keys may use one stored prefix; duplicated payloads must be counted twice.

Blend with the compact model's normalized **full-label sequence probabilities** from [03](03_baseline_harness.md):

$$p_{\mathrm{blend}}=(1-\lambda)p_{\mathrm{model}}+\lambda p_{\mathrm{memory}}.$$

Extend the label space from 77 to **89 labels**. Give the 12 new labels zero probability in the model branch; their mass can come only from memory. Outside the gate return the original prediction, not a newly substituted full-label-scoring argmax. Inside the gate use the blend's argmax, ties by fixed label ID.

This avoids requiring CLINC label tokens in the trimmed LM output vocabulary. New label names are output metadata; no CLINC text or label token is added to the model's vocabulary.

Calibrate on working old-intent rows and CLINC validation rows, never CLINC test:

| Setting | Finite calibration grid |
|---|---|
| Feature layer | `7, 14, 21, 28` |
| Neighbors $k$ | `1, 3, 5` |
| Cosine radius $r$ | `0.05, 0.10, 0.20, 0.30` |
| Blend $\lambda$ | `0.5, 0.75, 1.0` |
| Vote temperature $\tau$ | `0.05, 0.10` |

Use macro-accuracy across the 12 new intents at the **final 10-shot validation checkpoint** as the calibration objective, subject to **no old working-set accuracy loss versus A**, no loss versus the working base bar at that final checkpoint, and the size limit. Intermediate checkpoints are reported, not used to pick a different shot count. Ties prefer smaller radius, smaller memory, then lexical config hash. If no setting satisfies the constraints, G3 fails; do not lower the floor by reading test results.

Choose one configuration by replaying the precommitted stream, then replay it from a fresh copy of A. This tuning replay is development, not evidence of a one-pass online guarantee. The final reported stream uses frozen hyperparameters, with no feedback-driven hyperparameter search.

## Exactly twelve new intents

Use CLINC150 `plus`, pinned as in [02](02_setup_data_locked_exam.md):

```text
credit_score
improve_credit_score
apr
routing
direct_deposit
interest_rate
credit_limit
credit_limit_change
min_payment
pay_bill
rewards_balance
redeem_rewards
```

These are absent from Banking77's label set. Their semantics can still overlap banking messages; disclose that “new” means new output classes and feedback examples, not new foundation-model knowledge.

Freeze this list before any CLINC test evaluation. Do not choose easy labels after inspecting results. Confirm each name exists in the pinned feature schema; look up IDs by name rather than copying integers from a different dataset configuration.

For each intent, order eligible official train IDs by `SHA256("20261106|clinc|" + id)`, tie by ID. The first ten examples form the pool; the 1-, 5-, and 10-shot conditions are nested prefixes. No test-aware replacement or paraphrase mining.

Remove exact overlaps between CLINC calibration/teaching and new-intent test data in the initial preparation process, before sealing. Keep cross-dataset exclusions in [02](02_setup_data_locked_exam.md) consistent. Report actual remaining test counts, not a guaranteed 360. If a class loses all test examples or has fewer than ten eligible training examples, stop before tuning and record an amendment.

## Stream and checkpoints

Commit `feedback_stream.jsonl`, its hash, and the ordering policy before feedback calibration. Each item carries a source ID, label, and event type; keep raw text outside the inference bundle.

1. Start from immutable A with zero new-intent exemplars.
2. Reveal one example for each new intent, in the fixed list order. Save checkpoint **1-shot**.
3. Insert the remaining Banking77 feedback rows where A's frozen prediction is incorrect, ordered by stable ID. Predict-before-reveal and log each correction. Correctly predicted feedback rows do not become memory by default; log them as reviewed and skipped.
4. Reveal four more examples per new intent, round-robin in the fixed list order. Save checkpoint **5-shot**.
5. Reveal five more examples per new intent the same way. Save checkpoint **10-shot**, which is B.

For correction selection, A is evaluated on feedback only **after A is frozen**. Feedback labels never train the mask. Stream construction may use those labels at this stage; it must not inspect locked rows.

Append examples deterministically with duplicate suppression by source ID and documented conflict handling. Repeated identical feedback is a no-op; contradictory labels for an identical exemplar are quarantined and logged, not silently overwritten. Do not delete difficult old examples based on working or test errors. If capacity would overflow, fail or use a capacity rule committed before replay, never improvise eviction after examining a score.

Save and reload every checkpoint before measuring it. Keep A immutable, not a mutable object reference to B's growing dictionary.

## Size reserve and lifecycle checks

At most 120 new-intent exemplars are added, plus at most the actual feedback-pool count in old corrections. For illustration, 769 corrections plus 120 new examples means 889 entries in the exemplar list:

* Numeric payload: `889 * 258 = 229,362` entries.
* Minimal bytes with uint16 label IDs: `889 * (256 + 4 + 2) = 232,918` bytes.
* Extra metadata, projection seed/settings, mixture/radius parameters, and serialization overhead are additional.

These examples fit within the 1,571,430-entry reserve only if all other auxiliary state fits too. A training-example memory containing 10,003 keys would alone use 2,580,774 numeric entries at 258 each, exceeding that reserve. If fallback memory is needed, reduce the source-model target **before freezing A**.

Test empty memory, one entry, zero keys, ties, radius boundary (`distance <= r` is admitted), duplicate updates, conflicting updates, mixed new/old neighbors, serialization, max capacity, repeated stream application, and interruption/resume. Resume from an event index committed with state; never double-insert after a crash.

## What passes and what gets reported

For each 1-, 5-, and 10-shot checkpoint, report new-intent validation accuracy, old working accuracy, memory entries/bytes, correction uptake, and new-label false positives on old inputs. Measure retention after the correction block as well, not just at the end.

**G3, before opening tests:** B has at least **80%** new-intent validation accuracy, no old working accuracy drop versus A, and old working accuracy **≥ the working base bar**. **85%** is the stretch target. Intermediate regressions must be disclosed; the minimum claim concerns final B, not a promise that every intermediate state is monotone.

During the single frozen final session, evaluate all predeclared shot checkpoints on the CLINC test once per checkpoint, with **all 89 labels available** and no oracle domain routing. For A and B, also summarize the mixed set of locked Banking77 rows and CLINC test rows using their saved predictions, not a second routed query. Intermediate shot states use working data for old-intent retention, not additional locked-exam evaluations. Do not report a 12-way classifier as if it were an 89-way task.

For **Claim B's new-learning component**, require final B's new-intent test accuracy **≥80%** and strictly higher than A's, while final B's old locked accuracy is **≥ the final bar**. Report 85% attainment separately. A's new-label accuracy under this fixed output design is normally zero; state that rather than presenting it as a strong open-set baseline.

Also report final old B versus old A, confusion matrices and false new-intent rates. A ≥ bar and B ≥ bar can both hold while B is worse than A; do not describe that as “no forgetting.” The working-set no-drop constraint is not proof of locked-set no-drop.
