# 08. References and evidence limits

The pages below were retrieved successfully while building this playbook. A resolved page verifies its identity and accessible content, not experimental reproducibility. The base identity and arithmetic were independently checked; no model experiment was run for these docs.

## Rules and immutable source material

1. **[The 10% challenge](https://github.com/DeusExperiments/ten-percent-challenge)**, with the [raw README](https://raw.githubusercontent.com/DeusExperiments/ten-percent-challenge/main/README.md). Establishes the five steps, no weight training, held-out exam, and first-complete-proof condition. It does not specify our exact size ceilings, confidence margin, or memory accounting. Snapshot and hash the rules at execution; a live README can change.
2. **[Pinned Banking77 model card](https://huggingface.co/johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent/raw/6c13ee7e/README.md).** Supplies the literal short system prompt, greedy decoding, 20-token budget, unusual label spellings, merged-checkpoint distinction, and reported 93.4% on 3,076 decontaminated examples. That figure is self-reported and is not our locked-exam bar.
3. **[Pinned model metadata](https://huggingface.co/api/models/johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent/revision/6c13ee7e?blobs=true)** and **[config](https://huggingface.co/johnsonfzzz/Qwen2.5-1.5B-Banking77-Intent/raw/6c13ee7e/config.json)**. Verify full revision, weights SHA-256, file bytes, BF16 count, tied embeddings, hidden dimensions, attention/GQA shape, and layer count. Metadata is not a substitute for hashing the actual downloaded weights.
4. **[Pinned Banking77 dataset card](https://huggingface.co/datasets/PolyAI/banking77/raw/1846984189f2bc0ac51c6fc3a600fa9e1714e0c6/README.md)**. Supplies schema, canonical label IDs, split definition, attribution, and CC BY 4.0 notice.
5. **[Pinned CLINC150 dataset card](https://huggingface.co/datasets/clinc/clinc_oos/raw/155b9c710419136e17307b80d0a13e68cd46b4ec/README.md)**. Supplies intent names, configurations including `plus`, and CC BY 3.0 notice. Read names from the selected configuration rather than assuming IDs agree across loaders.

## Selection without source-weight updates

| Source | What it contributes here | What it does not establish |
|---|---|---|
| Jiang et al., **[Pruning Pre-trained Language Models Without Fine-Tuning](https://arxiv.org/abs/2210.06210)**, [ACL paper page](https://aclanthology.org/2023.acl-long.35/) | SMP's first-order mask learning with pretrained weights frozen; motivates disposable scores and gradual pruning | Does not prove 10%-total-size parity for this decoder, vocabulary, task, or feedback system |
| Ramanujan et al., **[What's Hidden in a Randomly Weighted Neural Network?](https://arxiv.org/abs/1911.13299)** | Edge-popup selection and straight-through score optimization | Results on random vision networks are not a Banking77 accuracy forecast |
| Sun et al., **[A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)** | Wanda's magnitude-times-input-activation criterion, usable without changing weights | Its original per-output pruning is not our global learned four-entry group scheme; no guarantee at this density |
| **[MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models](https://arxiv.org/abs/2409.17481)** | Evidence that frozen decoder weights plus learned masks can support downstream specialization | Studies 2:4 semi-structured sparsity; selecting whole four-entry blocks at roughly 10% is much more aggressive and different |
| Ushio et al., **[An Efficient Multilingual Language Model Compression through Vocabulary Trimming](https://arxiv.org/abs/2305.15020)** | Motivation for removing unused embedding/output rows without retraining | Does not validate our recursive Qwen BPE fallback or guarantee unchanged decoding on unseen text |

The method in [04](04_mask_search.md) combines these ideas. Its group size, global ranking, teacher support, schedule, and supervised task loss are design choices, not a verbatim reproduction of any one paper. A trained classification head in an upstream method is not permitted in this experiment.

## Memory, augmentation, and data

| Source | Contribution and limitation |
|---|---|
| Hartvigsen et al., **[Aging with GRACE: Lifelong Model Editing with Discrete Key-Value Adaptors](https://arxiv.org/abs/2211.11031)** | Local keys and deferral radii motivate distance-gated edits. Full GRACE optimizes adaptor values; this playbook instead stores exemplar labels and disallows that training. |
| **[kNN Prompting: Beyond-Context Learning with Calibration-Free Nearest Neighbor Inference](https://arxiv.org/abs/2303.13824)** | Supports exemplar-based prediction from frozen model representations. Our hidden-state projection and 89-way blend are not the paper's exact algorithm. |
| Wei and Zou, **[EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks](https://arxiv.org/abs/1901.11196)** | Supplies simple replacement, insertion, swap, and deletion ideas for train-only variants. Editing can change meaning; base pseudo-labels are noisy supervision, not human truth. |
| Casanueva et al., **[Efficient Intent Detection with Dual Sentence Encoders](https://arxiv.org/abs/2003.04807)** | Introduces Banking77's fine-grained single-domain benchmark. Encoder-based results do not establish this decoder's compression result. |
| Larson et al., **[An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction](https://arxiv.org/abs/1909.02027)** | Introduces CLINC150's 150 intents across ten domains and out-of-scope evaluation. Our 12 selected classes are only a limited incremental-learning demonstration. |

## Lossless storage, not weight quantization

* **[70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length Float (DFloat11)](https://arxiv.org/abs/2504.11651)** explains entropy coding of BF16 patterns with exact reconstruction. It motivates the secondary byte budget. Its roughly 30% reduction does not promise the same ratio for selected sparse values.
* **[ZipNN: Lossless Compression for AI Models](https://arxiv.org/abs/2411.05239)** motivates floating-point-aware lossless storage and transfer. All codec metadata and the tokenizer still count in our final container.

Neither paper compresses a full model to 10% through lossless coding alone. Here the model first loses parameters by selection, then its surviving bit patterns and masks are losslessly encoded.

## Claims not established by the references

There is no cited result establishing strict ≥ parity for this exact 1.54B source at 154,371,430 retained entries. There is no measured claim here that 327.5M score optimization fits comfortably in 16 GB, that 11 bits/value is achieved, or that 10-shot memory reaches 80–85% in the joint 89-label task. Those are executor gates or hypotheses.

The author's reported stalled high-compression experiment is planning context without an independently verified quantitative source in this playbook. Do not promote it into a baseline table.
