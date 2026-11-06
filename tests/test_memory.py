import numpy as np
import pytest

from tenpercent.io import read_json, write_json
from tenpercent.memory import FeedbackMemory, MemoryConfig, splitmix64


def memory(capacity=2, radius=0.1):
    return FeedbackMemory(
        ["old", "new"],
        MemoryConfig(
            input_dim=4,
            dimensions=256,
            seed=0,
            capacity=capacity,
            radius=radius,
            blend=1,
            neighbors=1,
        ),
    )


def test_prng_against_published_splitmix64_zero_seed_vector():
    assert int(splitmix64(0, 1)[0]) == 0xE220A8397B1DCDAF
    assert np.array_equal(splitmix64(13, 100), splitmix64(13, 101)[:100])


def test_empty_memory_duplicate_conflict_and_capacity():
    m = memory(capacity=1)
    hidden = np.array([1, 2, 3, 4])
    assert m.distribution(hidden) is None
    assert m.add("1", "same", "new", hidden) == "added"
    assert m.add("1", "same", "new", hidden) == "duplicate"
    assert m.add("2", "same", "old", hidden) == "quarantined"
    assert m.add("2", "same", "old", hidden) == "quarantined"
    assert m.quarantined == ["2"]
    assert len(m.keys) == 1
    with pytest.raises(ValueError):
        m.add("1", "different", "new", hidden)
    with pytest.raises(OverflowError):
        m.add("3", "third", "old", -hidden)


def test_scaled_int8_query_and_roundtrip(tmp_path):
    m = memory()
    hidden = np.array([1, 2, 3, 4])
    m.add("1", "example", "new", hidden)
    probabilities = m.blend(hidden, np.array([1.0, 0.0]))
    assert probabilities.argmax() == 1
    assert m.distribution(-hidden) is None
    before = m.clone()
    m.save(tmp_path / "memory.json")
    loaded = FeedbackMemory.load(tmp_path / "memory.json")
    assert np.array_equal(loaded.blend(hidden, np.array([1.0, 0.0])), probabilities)
    m.add("2", "another", "old", -hidden)
    assert len(before.keys) == len(loaded.keys) == 1
    assert loaded.numeric_entries == 258 + 9


def test_zero_vector_and_radius_boundary():
    m = memory(radius=2)
    m.add("1", "zero", "new", np.zeros(4))
    assert np.isfinite(m.distribution(np.zeros(4))).all()
    assert m.distribution(np.ones(4)).argmax() == 1


def test_registered_projection_golden_hash():
    assert FeedbackMemory(["old"]).projection_sha256 == (
        "02105e89cf91d54412ad444131827464c20f3296cbc8f40a78894c94e81a9fe9"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"temperature": float("nan")},
        {"input_dim": 4.5},
        {"neighbors": True},
        {"seed": -1},
        {"blend": float("inf")},
    ],
)
def test_invalid_memory_config(changes):
    with pytest.raises(ValueError):
        MemoryConfig(**changes)


@pytest.mark.parametrize(
    "field,value",
    [
        ("label_ids", [1.5]),
        ("keys", [[0.5] * 256]),
        ("scales", [0.1]),
        ("projection_sha256", "wrong"),
    ],
)
def test_corrupt_memory_never_silently_casts_to_valid_payload(tmp_path, field, value):
    m = memory()
    m.add("one", "example", "new", np.ones(4))
    path = tmp_path / "memory.json"
    m.save(path)
    payload = read_json(path)
    payload[field] = value
    write_json(path, payload)
    with pytest.raises(ValueError):
        FeedbackMemory.load(path)
