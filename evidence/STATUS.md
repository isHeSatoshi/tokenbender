# Initial implementation evidence

Execution owner and approved public repository: `isHeSatoshi/tokenbender`.

This is a working implementation under development, not a completed challenge
submission. The full train-only teacher-cache job is running locally. The
complete teacher commitment and preregistration will be published before any
real mask search. No locked exam has been opened.

Measured so far:

* 55 automated tests pass, and Ruff checks pass.
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

Full-size score optimization has not run. Its resource fit and accuracy are
unmeasured. Automatic memory calibration, crash-safe feedback replay, and
source-isolated final workers remain unfinished. The final exam CLI refuses
to run until those safeguards are implemented.

JSON files here contain aggregate results, versions, hashes, and structural
metadata only. The `pending` list in `local_validation.json` records the
pre-publication validation state; this status file records the publication
authorization received afterward. No model payload, source utterance, sealed
exam, or decryption key is included.
