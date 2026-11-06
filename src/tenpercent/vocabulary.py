from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn

from .constants import SYSTEM_PROMPT
from .data import load_split
from .io import object_hash, read_json, write_json
from .model import TokenizerAdapter, load_tokenizer


def byte_symbols() -> set[str]:
    visible = list(range(ord("!"), ord("~") + 1))
    visible += list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    codepoints = visible[:]
    unseen = 0
    for value in range(256):
        if value not in visible:
            visible.append(value)
            codepoints.append(256 + unseen)
            unseen += 1
    return {chr(value) for value in codepoints}


class Vocabulary:
    def __init__(self, retained: list[int], parents: dict[int, tuple[int, int]]):
        if retained != sorted(set(retained)) or not retained:
            raise ValueError("Retained token IDs must be nonempty, sorted, and unique")
        self.retained = retained
        self.compact = {original: i for i, original in enumerate(retained)}
        self.parents = parents
        self._expansion = lru_cache(maxsize=None)(self._expand)

    def _expand(self, token: int) -> tuple[int, ...]:
        if token in self.compact:
            return (token,)
        if token not in self.parents:
            raise ValueError(f"No retained decomposition for token {token}")
        left, right = self.parents[token]
        return self._expansion(left) + self._expansion(right)

    def expand(self, tokens: list[int]) -> list[int]:
        return [piece for token in tokens for piece in self._expansion(token)]

    def encode(self, tokens: list[int]) -> list[int]:
        return [self.compact[token] for token in self.expand(tokens)]

    def save(self, path: Path):
        write_json(
            path,
            {
                "format": 1,
                "retained": self.retained,
                "parents": {str(k): list(v) for k, v in sorted(self.parents.items())},
            },
        )

    @classmethod
    def load(cls, path: Path):
        value = read_json(path)
        if value.get("format") != 1:
            raise ValueError("Unsupported vocabulary format")
        parents = {int(k): tuple(v) for k, v in value["parents"].items()}
        for token, children in parents.items():
            if len(children) != 2 or token in children:
                raise ValueError("Invalid BPE parents")
        result = cls(value["retained"], parents)
        # Detect cycles without recursing through potentially malicious input.
        done, active = set(result.retained), set()

        def visit(token):
            if token in done:
                return
            if token in active or token not in parents:
                raise ValueError("Cyclic or incomplete BPE decomposition")
            active.add(token)
            for child in parents[token]:
                visit(child)
            active.remove(token)
            done.add(token)

        for token in parents:
            visit(token)
        return result


def build(root: Path) -> dict:
    base = root / "assets" / "base"
    tokenizer = load_tokenizer(base)
    adapter = TokenizerAdapter(tokenizer)
    tokenizer_json = read_json(base / "tokenizer.json")
    if tokenizer_json["model"]["type"] != "BPE":
        raise ValueError("This fallback requires a BPE tokenizer")
    vocab = tokenizer_json["model"]["vocab"]
    reverse = {i: symbol for symbol, i in vocab.items()}
    parents = {}
    for merge in tokenizer_json["model"]["merges"]:
        left, right = merge.split(" ") if isinstance(merge, str) else merge
        output = left + right
        if output not in vocab or left not in vocab or right not in vocab:
            raise ValueError("Merge references an unknown token")
        parents.setdefault(vocab[output], (vocab[left], vocab[right]))
    keep = set(range(8192)) & set(reverse)
    keep.update(tokenizer.all_special_ids)
    keep.update(item["id"] for item in tokenizer_json.get("added_tokens", []))
    for symbol in byte_symbols():
        if symbol not in vocab:
            raise ValueError("Tokenizer lacks a byte leaf")
        keep.add(vocab[symbol])
    labels = read_json(root / "data" / "manifest.json")["labels"]
    keep.update(adapter.prompt(""))
    keep.update(tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False))
    for label in labels:
        keep.update(adapter.label("", label))
    text_count = 0
    for split in ("train", "working", "feedback_text"):
        rows = load_split(root, f"banking/{split}", purpose="vocabulary")
        for row in rows:
            keep.update(tokenizer.encode(row["text"], add_special_tokens=False))
            text_count += 1
    result = Vocabulary(sorted(keep), parents)
    for token, symbol in reverse.items():
        pieces = result._expansion(token)
        if "".join(reverse[p] for p in pieces) != symbol:
            raise ValueError("BPE expansion does not preserve token bytes")
    for text in ("Hello, world!", "💳 routing 世界 العربية", "", "\n\t café\u0301"):
        original = tokenizer.encode(text, add_special_tokens=False)
        if tokenizer.decode(result.expand(original)) != tokenizer.decode(original):
            raise ValueError("Tokenizer fallback text round-trip failed")
    path = root / "runs" / "vocabulary.json"
    if path.exists():
        previous = Vocabulary.load(path)
        if previous.retained != result.retained or previous.parents != result.parents:
            raise FileExistsError("A different vocabulary is already frozen")
    else:
        result.save(path)
    report = {
        "retained_rows": len(keep),
        "original_rows": len(reverse),
        "text_rows_scanned": text_count,
        "all_token_expansions_verified": len(reverse),
        "vocabulary_hash": object_hash(read_json(path)),
    }
    write_json(root / "runs" / "vocabulary_report.json", report)
    return report


class CompactTokenizer(TokenizerAdapter):
    def __init__(self, tokenizer, vocabulary: Vocabulary):
        super().__init__(tokenizer)
        self.vocabulary = vocabulary
        self.eos_token_id = vocabulary.compact[tokenizer.eos_token_id]

    def prompt(self, text: str) -> list[int]:
        original = TokenizerAdapter(self.tokenizer).prompt(text)
        return self.vocabulary.encode(original)

    def label(self, text: str, label: str) -> list[int]:
        original = TokenizerAdapter(self.tokenizer).label(text, label)
        return self.vocabulary.encode(original)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(
            [self.vocabulary.retained[i] for i in ids],
            skip_special_tokens=True,
        )

    def original_id(self, token: int) -> int:
        return self.vocabulary.retained[token]


def trim_model(model, vocabulary: Vocabulary):
    original = model.get_input_embeddings().weight
    rows = torch.tensor(vocabulary.retained, device=original.device)
    selected = original.detach().index_select(0, rows).contiguous()
    embedding = nn.Embedding.from_pretrained(selected, freeze=True)
    head = nn.Linear(selected.shape[1], selected.shape[0], bias=False, device="meta")
    head.weight = embedding.weight
    model.set_input_embeddings(embedding)
    model.set_output_embeddings(head)
    model.config = copy.deepcopy(model.config)
    model.config.vocab_size = len(vocabulary.retained)
    for field in ("bos_token_id", "eos_token_id", "pad_token_id"):
        value = getattr(model.config, field, None)
        if isinstance(value, int):
            setattr(model.config, field, vocabulary.compact[value])
    return model
