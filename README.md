# Ten-percent Banking77 implementation

An executable frozen-weight subnetwork experiment. **No parity result is claimed.**
The locked exam has not been opened. Calendar schedules are not execution constraints.

## Run

```bash
uv sync --frozen
uv run tenpercent --help
uv run tenpercent doctor
uv run pytest -q
uv run ruff check src tests
```

All commands accept `--root /absolute/project/path` before the subcommand.
Large downloads, plaintext development data, sealed tests, caches, and bundles
are ignored by Git. `.secrets/exam.key` is local-only; never publish it.

## Pipeline

```bash
uv run tenpercent download
uv run tenpercent prepare-splits
uv run tenpercent baseline
uv run tenpercent build-vocabulary
uv run tenpercent baseline --vocabulary-only
uv run tenpercent check-identity --limit 770
uv run tenpercent cache-teacher --variants 0 --limit 8
uv run tenpercent prepare-adaptation
uv run tenpercent cache-teacher
uv run tenpercent preregister
```

Split preparation is deliberately one-shot. It refuses to overwrite existing data.
Read `PROTOCOL_AMENDMENTS.md`: the pinned PolyAI source has 3,080 clean literal
test messages, not the model card's anticipated 3,076. Preparation produced
1,540 sealed exam, 770 working, and 770 feedback examples. CLINC test is sealed
separately. Development loaders cannot open either test split.

`cache-teacher --limit` creates a **smoke-only** cache that cannot train a registered
candidate. Full train/EDA caching uses `cache-teacher` without `--limit`.
The completed full cache writes `teacher_commitment.json`; preregistration
refuses an incomplete cache. `prepare-adaptation` records real counts and the
maximum fallback-expanded length without truncating or reading locked data.

Mask search requires a public preregistration commitment and a matching
`runs/preregistration_attestation.json`. It does not silently skip that gate.
Creating `preregistration.json` locally is not publication. Repository creation
and pushing need the user's explicit authorization for the named repository.

After that commitment:

```bash
uv run tenpercent search-mask --steps 200 --learning-rate 0.003
uv run tenpercent pack --candidate CANDIDATE_ID
uv run tenpercent verify --bundle /absolute/path/to/runs/bundles/A
```

`replay-feedback` is experimental. `freeze` and `final-evaluate` currently refuse
to run, before reading any sealed rows or creating an exam receipt. They will be
released only after automatic calibration and source-isolated evaluation workers
are implemented and tested. The single-session/hash guard itself has synthetic
tests. `report` only reads already-saved results.

## Measured local development evidence

The unmodified base and vocabulary-only model each score **719/770 (93.38%)**
on working data in both literal greedy and constrained-label modes. These are
development results, not locked-exam results or a ten-percent candidate.
The retained vocabulary has **9,999 rows**; all 151,643 ordinary source-token
expansions were checked. Original-vocabulary all-one masks pass bit-equal logits
and identical greedy predictions on **all 770 working examples**.

An eight-example real teacher-cache smoke and a tiny Qwen search pass.
The tiny search exercises score gradients, both decoding modes, candidate masks,
and optimizer/data-order resume. It is not a resource benchmark of the real search.

## Boundaries and current maturity

* The only optimized parameters are disposable mask scores. Retained source
  values remain BF16-bit-identical and are exhaustively verified at packing.
* Exact global top-k uses a memory-bounded radix threshold with deterministic ties.
* Standalone bundles contain selected values, compressed masks, tokenizer,
  recursive BPE fallback, config, labels, and the exact custom loader.
* A 256-D seeded, scaled-int8 exemplar memory supports 12 new intent labels.
  Numeric memory entries count toward the 154,371,430-entry cap.
* Full mask searches and the real feedback demo are **not yet validated**.
  Current feedback replay accepts an A without a training-memory booster.
  Automatic memory-grid calibration, interrupted feedback replay, and isolated
  final workers remain implementation work. Optimizer/data-order resume is
  tested on a tiny model, not yet a full-size run. Do not interpret CLI
  availability as gate completion.
* The optional 308,746,714-byte container target is measured independently.
  Sparse storage does not promise lower latency or lower dense runtime memory.

The `playbook/` directory preserves the detailed protocol. Implementation
amendments take precedence for source facts that were measured rather than assumed.
See `runs/` for actual local verification reports and prediction journals.
