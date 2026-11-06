# 02. Setup, data, and the locked exam

## Environment and downloads

Target machine from planning: one RTX 4060 Ti with 16 GB VRAM and approximately 31 GB host RAM. This docs build has **not** verified a GPU or installed a training environment. At execution, record CPU, RAM, GPU/VRAM, driver, disk space, Python, CUDA runtime, and package versions. Do not assume CUDA tooling exists.

Use `uv`, Python 3.12, and CUDA-enabled PyTorch wheels compatible with the installed driver. A starting setup is:

```bash
cd /absolute/path/to/execution-repo
uv python install 3.12
uv venv --python 3.12
uv pip install --python .venv/bin/python torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv/bin/python transformers==5.20.2 datasets accelerate safetensors huggingface_hub numpy
uv pip freeze --python .venv/bin/python > requirements.lock.txt
```

These are proposed starting pins, **not a tested compatibility claim**. The checkpoint config records Transformers 5.20.2. If resolution or model loading fails, stop and resolve a compatible combination on public data. Commit the resulting exact lock, wheel index, Python patch version, and deterministic settings before pilots. Do not silently use a different model revision, dtype, tokenizer, or remote code. Use `trust_remote_code=False`.

Set all seeds to `20261106`; disable TF32 and stochastic sampling; record attention backend and deterministic-algorithm flags. Use the same pinned backend for base and compact comparisons. Reproducibility does not mean kernels on different GPUs are bitwise identical.

| Source | Immutable revision/config |
|---|---|
| Base model, tokenizer, chat template | `6c13ee7e6a4d229615dccdd089869818ad389388` |
| `PolyAI/banking77` | `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`, config `default` |
| `clinc/clinc_oos` | `155b9c710419136e17307b80d0a13e68cd46b4ec`, config `plus` |

Resolve through `snapshot_download(..., revision=FULL_SHA)` / `load_dataset(..., revision=FULL_SHA)`. Download only required root model files, not `lora/`. Verify the weights SHA-256 in [01](01_rules_claims_size.md). Hash tokenizer, template, config, dataset source files, and label maps into `sources.lock.json`. Dataset commits are pinned; file hashes must still be filled from the actual downloads, never invented.

Keep assets under ignored `assets/`, `data/`, and `runs/`, outside the inference bundle. Budget disk from actual cache and checkpoint sizes; do not create many copies of the 3.09 GB base or the approximately 5.24 GB score/gradient/Adam state. No cloud spend without approval.

## Banking77 construction, before any adaptation

Only a one-shot preparation process may access official test text before the lock. Its job is deterministic decontamination, assignment, hashing, and sealing, not analysis or model inference.

1. Assign stable source IDs, such as `banking77:test:000000`, before filtering.
2. Form the exact UTF-8 text set of all official training rows. Remove official test rows with exactly the same text, regardless of label. Expect **3,076** surviving rows from 3,080. Preserve original text: no lowercasing or Unicode normalization in the primary decontamination rule.
3. Audit exact duplicate text within surviving test data. The default splitter requires uniqueness and aborts if violated. Resolve duplicates as groups, with an explicit pre-search preregistration amendment, rather than allowing a message into two partitions. Also report a normalized-overlap diagnostic; do not silently alter the primary dataset.
4. Partition separately within each of the 77 labels into **50% locked exam / 25% working / 25% feedback**. Use largest-remainder rounding: floor each quota, then award remaining rows by largest fractional part; ties go exam, working, feedback. This is approximate per-label allocation when counts are not divisible by four.
5. Within each label, order stable IDs by `SHA256("20261106|" + source_id)`, breaking ties by source ID. Assign consecutive slices using those quotas. Sort each output file by stable source ID.
6. Write canonical JSONL, UTF-8, one compact sorted-key object per row, `ensure_ascii=False`, LF line endings and a final LF. Required fields: `id`, `text`, `label`. Hash the **exact bytes**.
7. Produce split sizes per label, aggregate counts, overlap assertions, source revisions, preparation-script hash, and locked-file SHA-256. Do not print locked text, examples, token statistics, or label-level model outcomes.

Illustrative quota logic, to implement in the split script:

```python
def quotas(n):
    numerators = (2*n, n, n)  # denominator four
    q = [x // 4 for x in numerators]
    order = sorted(range(3), key=lambda j: (-(numerators[j] % 4), j))
    for j in order[:n - sum(q)]:
        q[j] += 1
    assert sum(q) == n
    return tuple(q)  # exam, working, feedback
```

Do not hard-code aggregate split sizes to 1,538/769/769. Those are ratio estimates; rounding and filtered label counts determine the actual counts. If the clean total differs from 3,076, stop and inspect source identity and the filtering rule before tuning, then document any justified amendment.

### Allowed uses

| Partition | Text can build vocabulary? | Mask gradient / teacher cache? | Memory / tuning? |
|---|---|---|---|
| Official Banking77 train | Yes | Yes | Optional pre-feedback memory |
| Working | Yes, text only for vocabulary | No | Candidate scoring, calibration; never stored exemplars |
| Feedback pool | Yes, text only for vocabulary | No | Labels revealed only in the committed feedback stream |
| Locked exam | No | No | Final evaluation only |
| CLINC train / validation | No | No | Feedback examples / memory calibration respectively |
| CLINC test | No | No | Frozen final new-intent evaluation only |

Seeing working and feedback text for vocabulary is an explicitly disclosed **transductive preprocessing** choice. It is not an unseen-text claim for those partitions. The locked exam remains excluded. The original base was already trained upstream; its card says the decontaminated test is absent from its fine-tuning data, but we cannot prove absence from foundation-model pretraining.

Generate approximately 40,000 EDA variants **from train only** after the split is sealed. Use no test-aware rejection of variants: querying the locked split to detect overlap is itself prohibited. A final auditor may report accidental overlaps after all states are frozen, without changing the primary results.

## Lock and public commitment

On Day 1, create a new executor-owned public repository, for example `isHeSatoshi/ten-percent-banking77`, only if that account/repository is actually designated and authorized. Publish the split script, source lock, split manifest including the locked-file hash, and the completed preregistration before mask search. Do not publish the exam plaintext with those artifacts.

The preparation process seals the exam outside normal development paths, preferably encrypted under a key held by a user/custodian. A code guard alone is not access control. Deterministic public splits can be reconstructed from public data, so this is **auditable procedural blinding**, not a claim that exam content is cryptographically unknowable. Do not regenerate the split during development.

Implement an exam loader that refuses access unless **all** conditions hold:

* It runs as `final-evaluate`, not a training, caching, tokenizing, notebook, or development command.
* `final_manifest.json` exists; a separate `final_attestation.json` identifies the commit containing it. Avoid embedding a commit's own hash inside itself.
* The local checkout and tracked evaluation files are clean and match the attested commit. The authorized remote branch is confirmed to contain that exact manifest commit, not just a local commit.
* Hashes of both frozen states, all inference dependencies, decoding settings, feedback stream, and evaluation code match the manifest.
* The sealed file's digest matches the Day-1 commitment, and any custodian has released access.
* No exam-use receipt already exists. Atomically create a started receipt **before** decrypting or reading plaintext.

Publish no secret keys, credentials, private paths, or user PII. Git history is not a place to store a passphrase.

Pre-test the entire guard with a synthetic “exam.” Test refusal for missing manifest, unpushed commit, modified files, wrong hashes, repeated access, and concurrent invocations. Ordinary loaders must reject both exam filenames and the locked split identifier.

If offline or publication is blocked, work may proceed on public train-only smoke tests, but G0 is incomplete and no registered mask search or exam access is allowed.

## CLINC preparation

Freeze the 12 intents in [05](05_feedback_memory.md), source revision, names, and selection seed on Day 1. Preparation may check schema and aggregate availability without feeding CLINC into the vocabulary or teacher pipeline.

Use official `train` only for the nested feedback samples and `validation` only for memory hyperparameters. Keep official `test` sealed by the same final-session policy. Test examples must not select intent names, layers, radii, examples, or the shot count. Record counts and any duplicate-removal policy before calibration.

For cross-dataset overlap, the preparation process may exclude CLINC train/validation rows that collide with Banking77 evaluation partitions, without revealing those Banking77 texts. Commit the resulting eligible IDs, their hash, and removed counts. Require each selected intent to retain at least ten training examples. No replacement intents chosen after inspecting test performance.
