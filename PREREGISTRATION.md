# Pre-search registration

This records choices before the first real mask pilot or search. It is not an
accuracy result. The source, splits, vocabulary, full teacher cache, and
development baselines already exist; no qualifying compressed model exists.
Neither locked test has been opened for evaluation.

## Authority and immutable inputs

The authorized executor repository is
<https://github.com/isHeSatoshi/tokenbender>. The public commit containing this
file and `preregistration.json` is the registration commit. Its exact SHA,
remote, branch, and manifest digest are recorded separately in local
`runs/preregistration_attestation.json` and checked anonymously against GitHub
before optimization. Merely generating these files is not publication.

The full protocol is in `playbook/`; `PROTOCOL_AMENDMENTS.md` corrects measured
source facts. This execution record replaces the historical template's
`REQUIRED_AT_EXECUTION` fields for the pre-search stage. A/B selection and final
freeze fields are deliberately deferred until those states actually exist.

* Source: `johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent`,
  revision `6c13ee7e6a4d229615dccdd089869818ad389388`.
* Weights SHA-256:
  `2084675316db424191877b0ee41689ba28b83e07c8444fc1f44f037c769e82dc`.
* Denominators: 1,543,714,304 parameters; 3,087,467,144 weights-file bytes.
* Entry cap: 154,371,430 including memory. Source target: 152,800,000.
  Secondary complete-container cap: 308,746,714 bytes.
* Seed: `20261106`. There is no calendar deadline.
* Runtime: Python 3.12.16, torch 2.6.0+cu124, eager BF16 Qwen2, deterministic
  seeds, TF32 disabled, up to eight CPU computation threads.
  NVIDIA RTX 4060 Ti, 16,380 MiB. Exact dependency/build/environment hashes are
  bound by `uv.lock`, the environment report, and `preregistration.json`.

`preregistration.json` hashes every implementation module, this narrative, the
rules snapshot, environment, source verification, adaptation plan, vocabulary,
full teacher commitment, split commitment, baseline summaries, and full
all-one identity report. Large local files are represented by hashes, not
published payloads. The rules snapshot is the fetched Markdown representation,
not claimed to be the original server's raw HTTP bytes.

## Data and existing evidence

Banking77 revision `1846984189f2bc0ac51c6fc3a600fa9e1714e0c6`, config `default`,
has 10,003 train and 3,080 test rows. Exact UTF-8 overlap removal removes zero.
Seeded per-label allocation gives 1,540 sealed exam, 770 working, and 770 sealed
feedback rows. Every label has 20/10/10 rows. The originally anticipated 3,076
count is superseded by amendment A1, not by deleting arbitrary cases.

CLINC revision `155b9c710419136e17307b80d0a13e68cd46b4ec`, config `plus`,
contributes 120 eligible teaching rows, 240 validation rows, and 360 sealed
test rows across the twelve registered new intents. `split_commitment.json`
publishes counts, labels, source revisions, and plaintext/ciphertext hashes,
not message text. Keys and sealed payloads remain local. Local AES-GCM plus
the one-session guard provides procedural blinding, not independent-custodian
secrecy.

Both full-base working protocols score **719/770**. Vocabulary-only predictions
agree with the corresponding original baseline on all 770 rows. The retained
vocabulary has **9,999** rows; all **151,643** ordinary-token fallback expansions
were checked. Original-vocabulary all-one masks have bit-equal logits and
identical greedy predictions on all 770 working rows.

Adaptation uses 10,003 official train examples plus exactly 40,000 seeded
train-only deletion/swap variants. The original full base supplies variant
pseudo-labels, changing 4,234 labels. No synonym download or CLINC text is used.
The completed 50,003-example, non-smoke teacher cache is committed by
`teacher_commitment.json`, with SHA-256
`1cc21b9e52c62fe3c02023cc5514e31f6145dab7e1121a1978c2c62ec3f97dd3`.
Maximum original/expanded prompt length is 121; label+EOS length is at most 10;
the full teacher-input upper bound is 130. No truncation is registered.

## Search, resource pilot, and selection

Only disposable FP32 mask scores are optimized. Retained BF16 source words
must remain bit-identical. No source training, quantization, trained projection,
trained head, LoRA, or reconstruction is permitted.

Use training-only Wanda initialization, exact global top-k over four-entry
input-dimension groups, and the registered straight-through score gradient.
Group size two is a separately profiled registered fallback. Teacher
distillation uses dense-body retained-vocabulary label/EOS probabilities,
temperature 2, top-64 plus tail, temperature-squared KL plus 0.3 target CE.

Adam settings are betas (0.9, 0.999), epsilon 1e-8, zero weight decay, gradient
clip 1, accumulation 32 at batch one, and learning rates 0.001/0.003/0.01.
Density holds full for 10%, decays cubically for 70%, and holds target for 20%.
Learning rate warms up for 5%, then follows cosine decay.

Registered runs are 200-step 30%-size diagnostics and 200/2,000/4,000-step
10%-size candidates. A 30% diagnostic cannot be packed as A. Evaluate both
literal greedy and constrained modes every 100 steps. Only target-size
checkpoints qualify. Select by working correct count, winning-base prediction
agreement, fewer entries, then lexical config hash. The exact finite grid is
machine-readable in `preregistration.json`.

Before full runs, a non-selecting resource pilot uses this same implementation
in a fresh process. It scans every teacher record, repeats the longest input for
32-example accumulated updates at full, midpoint, and target densities, and
includes cold loading, Wanda, exact ranking, backward, and initial Adam state
allocation. Report RSS, CUDA allocated/reserved peaks, and time. Seek at least
10% reserved-memory headroom. Pilot scores are discarded and cannot become A.
Resource results are not invented in this pre-pilot registration.

Training-memory, rewrite, structural, or hidden-state fallbacks follow the
playbook's ordered ladder. Any not-yet-implemented search expansion requires
a public pre-exam amendment before use. CLINC and feedback labels never train
mask scores.

## Feedback and the final session

Freeze A before revealing feedback labels. Publish the text-free stream and
its commitment before tuning memory. The nested stream is twelve one-shot
examples, A's old-feedback corrections, then the remaining rounds to five and
ten shots. Predict each event before its update.

The memory grid has 288 settings: layers 7/14/21/28, neighbors 1/3/5, radius
0.05/0.10/0.20/0.30, blend 0.5/0.75/1, and temperature 0.05/0.10. Use untrained
seeded 256-D projection and scaled-int8 exemplar keys; count all key coordinates,
scales, label IDs, and numeric settings. Full-label model log probabilities are
EOS-inclusive and evaluated in right-padded batches of eight. Layer 28 is
post-final-RMSNorm. Tune final validation macro-accuracy subject to old working
retention, the working base bar, and size; use the registered tie rules.
Replay the chosen config from fresh A, not a grid-search memory object.

All A/1-shot/5-shot/B evaluations expose the same 89-label namespace without
oracle routing. Before release, verify exhaustive source-word lineage,
unchanged model/mask hashes, actual archive bytes, selected-state/standalone
prediction equality, and a synthetic final dry run. Candidate processes mount
only generic runtime files and their read-only package, with no source,
dataset, key, or host-cache mounts and no external network. Original baselines
run in separate source-only workers.

The public final manifest and separate attestation must precede the single
locked session. Claim A requires old correct count at least both measured
original-base protocols. Claim B additionally requires new-test accuracy at
least 80% and strictly above A. Both states must satisfy the entry cap.
Invalid predictions are wrong. Secondary paired bootstrap uses 10,000 draws
and the fixed seed; confidence intervals cannot rescue failed strict claims.
Complete bytes are a separate secondary outcome.

Durable pending records and per-example journals permit unchanged continuation
of unfinished work after a crash. Kernel locks prevent concurrent sessions.
No code, model, threshold, or membership changes after exam access. Publish
failures as well as successes. No organizer contact, social posts, spending,
or challenge issue/PR is authorized.
