from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from .constants import MODEL_SHA256, SEED
from .data import data_commitment, load_split
from .io import append_record, object_hash, read_json, sha256_file, write_json
from .model import Runner, TokenizerAdapter, load_base, load_tokenizer
from .vocabulary import CompactTokenizer, Vocabulary, trim_model


def eda_variants(rows: list[dict], count: int = 40_000) -> list[dict]:
    if count < 0:
        raise ValueError("Negative augmentation count")
    rng = np.random.default_rng(SEED)
    existing = {row["text"] for row in rows}
    variants = []
    for _ in range(20):
        for row in rows:
            if len(variants) == count:
                return variants
            words = row["text"].split()
            if len(words) < 2:
                continue
            edited = words[:]
            if rng.random() < 0.5:
                keep = rng.random(len(words)) >= 0.05
                if keep.all():
                    keep[rng.integers(len(words))] = False
                edited = [word for word, retained in zip(words, keep, strict=True) if retained]
            else:
                for _ in range(max(1, round(0.05 * len(words)))):
                    a, b = rng.choice(len(words), size=2, replace=False)
                    edited[a], edited[b] = edited[b], edited[a]
            text = " ".join(edited)
            if not text or text in existing:
                continue
            existing.add(text)
            variants.append(
                {
                    "id": "banking77:aug:" + hashlib.sha256(text.encode()).hexdigest(),
                    "text": text,
                    "parent_id": row["id"],
                    "label": row["label"],
                }
            )
    return variants


def prepare_adaptation(root: Path, *, variants=40_000):
    rows = load_split(root, "banking/train", purpose="teacher")
    augmented = eda_variants(rows, variants)
    vocabulary_path = root / "runs/vocabulary.json"
    vocabulary = Vocabulary.load(vocabulary_path)
    identity = {
        "format": 1,
        "source_sha256": MODEL_SHA256,
        "data_commitment": data_commitment(root),
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "tokenizer_sha256": sha256_file(root / "assets/base/tokenizer.json"),
        "tokenizer_config_sha256": sha256_file(root / "assets/base/tokenizer_config.json"),
        "requested_variants": variants,
        "actual_variants": len(augmented),
        "official_train": len(rows),
        "total_examples": len(rows) + len(augmented),
        "training_inputs_sha256": object_hash(rows + augmented),
        "augmentation": "seeded training-only deletion/swap; no synonym resource",
    }
    path = root / "runs/adaptation_plan.json"
    if path.exists():
        report = read_json(path)
        if any(report.get(k) != v for k, v in identity.items()):
            raise PermissionError("Adaptation plan changed; an explicit amendment is required")
        return report
    tokenizer = load_tokenizer(root / "assets/base")
    original = TokenizerAdapter(tokenizer)
    compact = CompactTokenizer(tokenizer, vocabulary)
    labels = read_json(root / "data/manifest.json")["labels"]
    max_answer = max(len(compact.label("", label)) for label in labels)
    max_original = max_expanded = 0
    longest_id = None
    for row in rows + augmented:
        prompt = original.prompt(row["text"])
        expanded = len(vocabulary.encode(prompt))
        max_original = max(max_original, len(prompt))
        if expanded > max_expanded:
            max_expanded, longest_id = expanded, row["id"]
    report = {
        **identity,
        "max_original_prompt_tokens": max_original,
        "max_expanded_prompt_tokens": max_expanded,
        "max_label_eos_tokens": max_answer,
        "max_teacher_input_tokens_upper_bound": max_expanded + max_answer - 1,
        "longest_training_example_id": longest_id,
        "truncation": False,
    }
    write_json(path, report, exclusive=True)
    return report


class TeacherCache:
    def __init__(self, path: Path, metadata: dict | None = None, *, readonly=False):
        if readonly:
            self.connection = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(path)
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT)"
            )
            self.connection.execute(
                "CREATE TABLE IF NOT EXISTS examples (id TEXT PRIMARY KEY, payload BLOB)"
            )
        saved = self.connection.execute(
            "SELECT value FROM metadata WHERE key='identity'"
        ).fetchone()
        if metadata is not None:
            encoded = json.dumps(metadata, sort_keys=True)
            if saved and saved[0] != encoded:
                raise ValueError("Teacher cache identity mismatch")
            if not saved:
                self.connection.execute("INSERT INTO metadata VALUES ('identity',?)", (encoded,))
                self.connection.commit()
            self.metadata = metadata
        elif saved:
            self.metadata = json.loads(saved[0])
        else:
            raise ValueError("Cache has no source identity")

    def ids(self):
        return [row[0] for row in self.connection.execute("SELECT id FROM examples ORDER BY id")]

    def put(self, source_id, tensors):
        self.connection.execute(
            "INSERT INTO examples VALUES (?,?)", (source_id, save_tensors(tensors))
        )
        self.connection.commit()

    def get(self, source_id):
        row = self.connection.execute(
            "SELECT payload FROM examples WHERE id=?", (source_id,)
        ).fetchone()
        if row is None:
            raise KeyError(source_id)
        return load_tensors(row[0])

    def close(self):
        self.connection.close()


def _teacher_record(model, adapter, text: str, label: str, topk=64):
    prompt = adapter.prompt(text)
    answer = adapter.label(text, label)
    device = next(model.parameters()).device
    inputs = torch.tensor([prompt + answer[:-1]], device=device)
    with torch.inference_mode():
        hidden = model.model(inputs, use_cache=False, return_dict=True).last_hidden_state
        logits = model.lm_head(hidden[0, len(prompt) - 1 :]).float() / 2
        log_probabilities = logits.log_softmax(-1)
        top_logp, top_ids = torch.topk(log_probabilities, k=min(topk, logits.shape[-1]), dim=-1)
        selected = torch.zeros_like(log_probabilities, dtype=torch.bool).scatter_(-1, top_ids, True)
        tail = torch.logsumexp(log_probabilities.masked_fill(selected, -torch.inf), dim=-1).exp()
        original_ids = torch.tensor(
            [[adapter.original_id(int(token)) for token in position] for position in top_ids.cpu()],
            dtype=torch.int64,
        )
        # Keep tail in probability form: a zero-probability bucket has zero KL contribution.
        return {
            "input_ids": inputs[0].cpu().contiguous(),
            "targets": torch.tensor(answer, dtype=torch.int64),
            "positions": torch.arange(len(prompt) - 1, inputs.shape[1], dtype=torch.int64),
            "top_ids": top_ids.cpu().contiguous(),
            "top_original_ids": original_ids,
            "top_probabilities": top_logp.exp().cpu().contiguous(),
            "tail_probability": tail.cpu().contiguous(),
            "log_normalizer": logits.logsumexp(-1).cpu().contiguous(),
        }


def pseudo_label_variants(path: Path, variants: list[dict], runner) -> list[dict]:
    """Continue an identical deterministic journal, never relabel its completed prefix."""
    saved = []
    if path.exists():
        saved = [json.loads(line) for line in path.read_bytes().splitlines()]
        if len(saved) > len(variants):
            raise ValueError("Pseudo-label journal has extra examples")
        for original, record in zip(variants, saved):
            if (
                record.get("label") not in runner.labels
                or {**original, "label": record.get("label")} != record
            ):
                raise ValueError("Pseudo-label journal differs from deterministic augmentation")
    for index in range(len(saved), len(variants)):
        row = dict(variants[index])
        row["label"] = runner.predict(row["text"], mode="constrained")
        if row["label"] not in runner.labels:
            raise ValueError("Pseudo-label is not a valid base label")
        append_record(path, row)
        saved.append(row)
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {"event": "pseudo_labels", "completed": index + 1, "total": len(variants)}
                ),
                flush=True,
            )
    return saved


def commit_teacher(root: Path, path: Path, metadata: dict):
    commitment = {
        "format": 1,
        "metadata": metadata,
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
    }
    destination = root / "teacher_commitment.json"
    if destination.exists():
        if read_json(destination) != commitment:
            raise ValueError("A committed teacher cache changed")
    else:
        write_json(destination, commitment, exclusive=True)
    return commitment


def cache_teacher(root: Path, *, variants=40_000, limit=None, device="cuda"):
    rows = load_split(root, "banking/train", purpose="teacher")
    augmented = eda_variants(rows, variants)
    if limit is None:
        prepare_adaptation(root, variants=variants)
    if limit is not None:
        if variants or limit < 1:
            raise ValueError("A limited smoke cache requires a positive limit and --variants 0")
        rows = rows[:limit]
    model = load_base(root / "assets" / "base", device)
    labels = read_json(root / "data" / "manifest.json")["labels"]
    tokenizer = load_tokenizer(root / "assets" / "base")
    base_runner = Runner(model, TokenizerAdapter(tokenizer), labels)
    comparison_rows = rows[:8]
    comparison = TeacherCache(
        root / "runs" / f"teacher-full-base-diagnostic-{len(comparison_rows)}.sqlite",
        {
            "format": 2,
            "source_sha256": MODEL_SHA256,
            "data_commitment": data_commitment(root),
            "examples": len(comparison_rows),
            "example_ids": object_hash([r["id"] for r in comparison_rows]),
            "teacher": "unmodified-full-base-training-diagnostic",
            "temperature": 2,
            "topk": 64,
            "probability_dtype": "float32",
        },
    )
    seen_comparison = set(comparison.ids())
    for row in comparison_rows:
        if row["id"] not in seen_comparison:
            comparison.put(
                row["id"], _teacher_record(model, base_runner.tokenizer, row["text"], row["label"])
            )
    comparison.close()
    adaptation_path = root / "runs" / f"adaptation-{variants}-{limit or 'all'}.jsonl"
    if augmented:
        augmented = pseudo_label_variants(adaptation_path, augmented, base_runner)
    vocabulary_path = root / "runs" / "vocabulary.json"
    vocabulary = Vocabulary.load(vocabulary_path)
    trim_model(model, vocabulary)
    adapter = CompactTokenizer(tokenizer, vocabulary)
    all_rows = rows + augmented
    parent_labels = {r["id"]: r["label"] for r in rows}
    identity = {
        "format": 2,
        "source_sha256": MODEL_SHA256,
        "vocabulary_sha256": sha256_file(vocabulary_path),
        "data_commitment": data_commitment(root),
        "example_ids": object_hash([r["id"] for r in all_rows]),
        "adaptation_sha256": object_hash(all_rows),
        "examples": len(all_rows),
        "temperature": 2,
        "topk": 64,
        "probability_dtype": "float32",
        "top_ids_space": "compact; original IDs also stored per position",
        "teacher": "dense-body-retained-vocabulary",
        "smoke_only": limit is not None,
        "augmentation": "train-only deletion/swap; no unpinned synonym resource",
        "augmented_examples": len(augmented),
        "pseudo_label_changes": sum(
            row["label"] != parent_labels[row["parent_id"]] for row in augmented
        ),
    }
    cache_path = (
        root / "runs" / ("teacher-v2-smoke.sqlite" if limit is not None else "teacher.sqlite")
    )
    cache = TeacherCache(cache_path, identity)
    seen = set(cache.ids())
    for index, row in enumerate(all_rows):
        if row["id"] not in seen:
            cache.put(row["id"], _teacher_record(model, adapter, row["text"], row["label"]))
        if (index + 1) % 100 == 0:
            print(
                json.dumps(
                    {"event": "teacher_cache", "completed": index + 1, "total": len(all_rows)}
                ),
                flush=True,
            )
    cached_ids = cache.ids()
    count = len(cached_ids)
    cache.close()
    if set(cached_ids) != {r["id"] for r in all_rows}:
        raise ValueError("Teacher cache IDs differ from the full adaptation set")
    if limit is None:
        commit_teacher(root, cache_path, identity)
    return {**identity, "cached": count, "path": str(cache_path.relative_to(root))}


def distillation_loss(logits, top_ids, top_probabilities, tail_probability, targets):
    logq = (logits.float() / 2).log_softmax(-1)
    top_logq = logq.gather(-1, top_ids)
    selected = torch.zeros_like(logq, dtype=torch.bool).scatter_(-1, top_ids, True)
    tail_logq = torch.logsumexp(logq.masked_fill(selected, -torch.inf), dim=-1)
    tail_logq = tail_logq.clamp_min(torch.finfo(torch.float32).min)
    p = top_probabilities.float()
    tail = tail_probability.float()
    # xlogy handles an exact zero teacher probability.
    kl = (torch.special.xlogy(p, p) - p * top_logq).sum(-1)
    kl += torch.special.xlogy(tail, tail) - tail * tail_logq
    ce = torch.nn.functional.cross_entropy(logits.float(), targets)
    return 4 * kl.mean() + 0.3 * ce
