import hashlib

import pytest

from tenpercent.data import load_split, quotas, seal, split_banking, stable_order, unseal
from tenpercent.io import atomic_bytes, jsonl_bytes, write_json


def rows(n, split):
    return [
        {"id": f"{split}:{i:06d}", "text": f"{split} message {i}", "label": str(i % 3)}
        for i in range(n)
    ]


def test_quotas_rounding_and_worst_cases():
    for n in range(10001):
        q = quotas(n)
        assert sum(q) == n and min(q) >= 0
        assert all(abs(a - b) <= 1 for a, b in zip(q, (n / 2, n / 4, n / 4), strict=True))
    assert quotas(39) == (19, 10, 10)
    assert quotas(40) == (20, 10, 10)
    with pytest.raises(ValueError):
        quotas(-1)


def test_exact_decontamination_and_deterministic_partition():
    train = rows(5, "train")
    test = rows(21, "test") + [{"id": "test:999", "text": train[0]["text"], "label": "other"}]
    a, report = split_banking(train, test, expected_clean=21)
    b, _ = split_banking(train, list(reversed(test)), expected_clean=21)
    assert a == b
    assert report["removed_exact_train_overlap"] == 1
    all_ids = [r["id"] for name in ("exam", "working", "feedback") for r in a[name]]
    assert len(all_ids) == len(set(all_ids)) == 21
    assert stable_order(test) == stable_order(list(reversed(test)))


def test_count_and_duplicate_fail_closed():
    with pytest.raises(ValueError, match="Expected"):
        split_banking([], rows(2, "test"), expected_clean=3)
    duplicate = [
        {"id": "1", "text": "same", "label": "a"},
        {"id": "2", "text": "same", "label": "a"},
    ]
    with pytest.raises(ValueError, match="duplicate"):
        split_banking([], duplicate, expected_clean=2)


def test_sealing_authenticates_payload_and_split():
    key = bytes(range(32))
    payload = jsonl_bytes(rows(2, "test"))
    encrypted = seal(payload, key, b"banking/exam")
    assert unseal(encrypted, key, b"banking/exam") == payload
    with pytest.raises(Exception):
        unseal(encrypted, key, b"clinc/test")
    with pytest.raises(Exception):
        unseal(encrypted[:-1] + bytes([encrypted[-1] ^ 1]), key, b"banking/exam")


def test_loader_refuses_exam_alias_and_modified_data(tmp_path):
    payload = jsonl_bytes(rows(2, "train"))
    path = tmp_path / "data" / "banking" / "train.jsonl"
    atomic_bytes(path, payload)
    write_json(
        tmp_path / "data" / "manifest.json",
        {
            "files": {
                "banking/train": {
                    "path": "banking/train.jsonl",
                    "sealed": False,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "count": 2,
                }
            },
        },
    )
    assert len(load_split(tmp_path, "banking/train", purpose="teacher")) == 2
    for split in ("banking/exam", "clinc/test", "../banking/train", "banking/feedback"):
        with pytest.raises(PermissionError):
            load_split(tmp_path, split, purpose="teacher")
    path.write_bytes(payload + b"\n")
    with pytest.raises(ValueError, match="hash"):
        load_split(tmp_path, "banking/train", purpose="teacher")
