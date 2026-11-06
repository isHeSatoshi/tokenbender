# Initial implementation evidence

Execution owner and approved public repository: `isHeSatoshi/tokenbender`.

This is a working implementation under development, not a completed challenge
submission. The full 50,003-example train-only teacher cache is complete.
Its commitment and the pre-search registration accompany this implementation.
No real mask search or locked evaluation preceded this registration.

Measured so far:

* 80 automated tests pass, and Ruff checks pass.
* Both original base decoding modes score 719/770 on working data.
* Vocabulary-only decoding also scores 719/770 in each mode and agrees with
  the corresponding original base on all 770 predictions.
* All-one masking has bit-equal logits and identical greedy predictions on
  all 770 working examples.
* The vocabulary retains 9,999 rows, with all 151,643 ordinary source-token
  expansions exhaustively checked.
* The adaptation pool contains 10,003 official training rows and 40,000
  deterministic variants. The maximum teacher-input bound is 130 tokens.
* A real eight-example teacher-cache smoke and tiny score-only optimizer,
  selection, packaging, and resume tests pass.
* All four packaged states pass a real CPU source-isolated/network-disabled
  synthetic final dry run. Pending prediction recovery avoids repeated queries.
  A tiny CUDA sandbox smoke is validated separately from full-size optimization.

Full-size score optimization has not run. Its resource fit and accuracy are
unmeasured. The 288-setting memory grid, crash-safe feedback replay, and isolated
workers pass synthetic tests, but no real A/B exists. The final exam CLI refuses
missing, failed, or stale real-model G3/standalone proofs.

JSON files here contain aggregate results, versions, hashes, and structural
metadata only. The `pending` list in `local_validation.json` records the
initial publication state; this status file records subsequent implementation
validation. No model payload, source utterance, sealed
exam, or decryption key is included.
