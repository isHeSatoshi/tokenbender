from __future__ import annotations

import hashlib
import json
import os
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .constants import (
    BANKING_ID,
    BANKING_REVISION,
    CLINC_ID,
    CLINC_REVISION,
    NEW_INTENTS,
    SEED,
)
from .io import atomic_bytes, jsonl_bytes, object_hash, read_json, sha256_file, write_json


def quotas(n: int) -> tuple[int, int, int]:
    if n < 0:
        raise ValueError("Negative split size")
    nums = (2 * n, n, n)
    q = [x // 4 for x in nums]
    order = sorted(range(3), key=lambda j: (-(nums[j] % 4), j))
    for j in order[: n - sum(q)]:
        q[j] += 1
    return tuple(q)


def stable_order(rows: list[dict], namespace: str = "") -> list[dict]:
    return sorted(
        rows,
        key=lambda r: (
            hashlib.sha256(f"{SEED}|{namespace}{r['id']}".encode()).digest(),
            r["id"],
        ),
    )


def normalize_overlap(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def split_banking(train: list[dict], test: list[dict], *, expected_clean: int | None = 3080):
    train_text = {row["text"] for row in train}
    clean = [row for row in test if row["text"] not in train_text]
    if expected_clean is not None and len(clean) != expected_clean:
        raise ValueError(f"Expected {expected_clean} clean test rows, found {len(clean)}")
    if len({r["text"] for r in clean}) != len(clean):
        raise ValueError("Within-test duplicate text requires a pre-search protocol amendment")
    by_label = defaultdict(list)
    for row in clean:
        by_label[row["label"]].append(row)
    splits = {
        "train": sorted(train, key=lambda r: r["id"]),
        "exam": [],
        "working": [],
        "feedback": [],
    }
    per_label = {}
    for label, rows in sorted(by_label.items()):
        rows = stable_order(rows)
        exam, working, feedback = quotas(len(rows))
        splits["exam"].extend(rows[:exam])
        splits["working"].extend(rows[exam : exam + working])
        splits["feedback"].extend(rows[exam + working :])
        per_label[label] = dict(exam=exam, working=working, feedback=feedback)
    for rows in splits.values():
        rows.sort(key=lambda r: r["id"])
    normalized_train = {normalize_overlap(text) for text in train_text}
    report = {
        "official_train": len(train),
        "official_test": len(test),
        "clean_test": len(clean),
        "removed_exact_train_overlap": len(test) - len(clean),
        "remaining_normalized_train_overlap": sum(
            normalize_overlap(r["text"]) in normalized_train for r in clean
        ),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "per_label": per_label,
    }
    return splits, report


def seal(payload: bytes, key: bytes, associated_data: bytes) -> bytes:
    nonce = os.urandom(12)
    return b"TP01" + nonce + AESGCM(key).encrypt(nonce, payload, associated_data)


def unseal(payload: bytes, key: bytes, associated_data: bytes) -> bytes:
    if payload[:4] != b"TP01" or len(payload) < 32:
        raise ValueError("Invalid sealed split")
    return AESGCM(key).decrypt(payload[4:16], payload[16:], associated_data)


def _records(dataset, prefix: str, label_field: str, names: list[str]) -> list[dict]:
    return [
        {"id": f"{prefix}:{i:06d}", "text": r["text"], "label": names[r[label_field]]}
        for i, r in enumerate(dataset)
    ]


def prepare(root: Path, *, include_clinc: bool = True) -> dict:
    """The sole pre-lock process allowed to read official test plaintext."""
    from datasets import load_dataset

    destination = root / "data"
    if destination.exists():
        raise FileExistsError("Data directory already exists; never regenerate a locked split")
    source = load_dataset(BANKING_ID, "default", revision=BANKING_REVISION)
    labels = source["train"].features["label"].names
    train = _records(source["train"], "banking77:train", "label", labels)
    test = _records(source["test"], "banking77:test", "label", labels)
    splits, report = split_banking(train, test)
    if len(labels) != 77 or set(NEW_INTENTS) & set(labels):
        raise ValueError("Unexpected Banking77 label schema")
    # Fetch/validate all required data before creating the one-shot destination.
    clinc_splits, clinc_report = {}, {}
    if include_clinc:
        clinc = load_dataset(CLINC_ID, "plus", revision=CLINC_REVISION)
        clinc_names = clinc["train"].features["intent"].names
        if not set(NEW_INTENTS) <= set(clinc_names):
            raise ValueError("New intent missing from pinned CLINC schema")
        for name in ("train", "validation", "test"):
            clinc_splits[name] = [
                r
                for r in _records(clinc[name], f"clinc:{name}", "intent", clinc_names)
                if r["label"] in NEW_INTENTS
            ]
        test_text = {r["text"] for r in clinc_splits["test"]}
        banking_evaluation = {r["text"] for n in ("exam", "working", "feedback") for r in splits[n]}
        for name in ("train", "validation"):
            before = len(clinc_splits[name])
            clinc_splits[name] = [
                r
                for r in clinc_splits[name]
                if r["text"] not in test_text and r["text"] not in banking_evaluation
            ]
            clinc_report[f"{name}_excluded"] = before - len(clinc_splits[name])
        for name in ("train", "validation", "test"):
            counts = Counter(r["label"] for r in clinc_splits[name])
            minimum = 10 if name == "train" else 1
            if any(counts[label] < minimum for label in NEW_INTENTS):
                raise ValueError(f"Insufficient eligible CLINC {name} examples")
        teaching = []
        for label in NEW_INTENTS:
            teaching.extend(
                stable_order(
                    [r for r in clinc_splits["train"] if r["label"] == label],
                    "clinc|",
                )[:10]
            )
        clinc_splits["train"] = teaching
        clinc_report["counts"] = {n: len(r) for n, r in clinc_splits.items()}

    destination.mkdir()
    secrets = root / ".secrets"
    secrets.mkdir(exist_ok=True, mode=0o700)
    key_path = secrets / "exam.key"
    key = AESGCM.generate_key(bit_length=256)
    fd = os.open(key_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(key)
    manifest = {
        "format": 1,
        "seed": SEED,
        "banking_revision": BANKING_REVISION,
        "clinc_revision": CLINC_REVISION if include_clinc else None,
        "labels": labels,
        "new_intents": list(NEW_INTENTS),
        "report": report,
        "clinc_report": clinc_report,
        "files": {},
        "blinding": "Local encryption and procedural guard; local owner can access key",
        "splitter_sha256": sha256_file(Path(__file__)),
    }
    files = {
        "banking/train": splits["train"],
        "banking/working": splits["working"],
        "banking/exam": splits["exam"],
        "banking/feedback": splits["feedback"],
        "banking/feedback_text": [{"id": r["id"], "text": r["text"]} for r in splits["feedback"]],
        **{f"clinc/{name}": rows for name, rows in clinc_splits.items()},
    }
    for name, rows in files.items():
        plaintext = jsonl_bytes(rows)
        locked = name in {"banking/exam", "banking/feedback", "clinc/test"}
        relative = f"{name}.jsonl" + (".enc" if locked else "")
        payload = seal(plaintext, key, name.encode()) if locked else plaintext
        atomic_bytes(destination / relative, payload, exclusive=True)
        manifest["files"][name] = {
            "path": relative,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "plaintext_sha256": hashlib.sha256(plaintext).hexdigest(),
            "count": len(rows),
            "sealed": locked,
        }
    write_json(destination / "manifest.json", manifest, exclusive=True)
    return manifest


def load_split(root: Path, name: str, *, purpose: str) -> list[dict]:
    allowed = {
        "baseline": {"banking/train", "banking/working"},
        "vocabulary": {"banking/train", "banking/working", "banking/feedback_text"},
        "teacher": {"banking/train"},
        "search": {"banking/train", "banking/working"},
        "memory": {"banking/train", "banking/working", "clinc/train", "clinc/validation"},
    }
    if name not in allowed.get(purpose, set()):
        raise PermissionError(f"Split {name!r} is forbidden for {purpose!r}")
    manifest = read_json(root / "data" / "manifest.json")
    info = manifest["files"][name]
    if info["sealed"]:
        raise PermissionError("Development loader cannot decrypt sealed data")
    path = (root / "data" / info["path"]).resolve()
    if not path.is_relative_to((root / "data").resolve()) or sha256_file(path) != info["sha256"]:
        raise ValueError("Split path or hash mismatch")
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    if len(rows) != info["count"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Split count or IDs mismatch")
    return rows


def load_feedback(root: Path, state_a: Path) -> list[dict]:
    from .bundle import bundle_manifest

    state = bundle_manifest(state_a)
    if state.get("state") != "A" or state.get("feedback_events", 0):
        raise PermissionError("Corrections require a verified frozen pre-feedback A bundle")
    manifest = read_json(root / "data" / "manifest.json")
    info = manifest["files"]["banking/feedback"]
    path = root / "data" / info["path"]
    if sha256_file(path) != info["sha256"]:
        raise ValueError("Feedback hash mismatch")
    plaintext = unseal(
        path.read_bytes(), (root / ".secrets" / "exam.key").read_bytes(), b"banking/feedback"
    )
    if hashlib.sha256(plaintext).hexdigest() != info["plaintext_sha256"]:
        raise ValueError("Feedback plaintext hash mismatch")
    return [json.loads(line) for line in plaintext.splitlines()]


def data_commitment(root: Path) -> str:
    return object_hash(read_json(root / "data" / "manifest.json"))
