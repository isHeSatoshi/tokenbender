from dataclasses import asdict, replace
from fractions import Fraction

import numpy as np

from tenpercent.calibration import (
    NeighborTable,
    batch_predict,
    choose_config,
    config_grid,
    macro_fraction,
    neighbor_table,
)
from tenpercent.constants import PARAMETER_CAP
from tenpercent.io import object_hash
from tenpercent.memory import FeedbackMemory, MemoryConfig


def test_full_registered_grid_is_finite_and_unique():
    configs = config_grid()
    assert len(configs) == 288
    assert len({object_hash(asdict(c)) for c in configs}) == 288


def test_cached_grid_predictions_match_actual_memory_rule_and_outside_fallback():
    rng = np.random.default_rng(4)
    config = MemoryConfig(input_dim=4, dimensions=16, neighbors=5, radius=0.3)
    memory = FeedbackMemory(["old", "new", "other"], config)
    for i in range(9):
        memory.add(str(i), f"example {i}", memory.labels[i % 3], rng.normal(size=4))
    features = np.concatenate(
        [np.stack([rng.normal(size=4) for _ in range(10)]), np.array([[0.0, 0.0, 0.0, 0.0]])]
    )
    table = neighbor_table(memory, features)
    model = np.tile([1.0, 0.0, 0.0], (len(features), 1))
    fallback = np.full(len(features), -1, dtype=np.int64)
    for k in (1, 3, 5):
        for radius in (0.05, 0.1, 0.2, 0.3):
            for blend in (0.5, 0.75, 1.0):
                for tau in (0.05, 0.1):
                    candidate = replace(
                        config, neighbors=k, radius=radius, blend=blend, temperature=tau
                    )
                    memory.config = candidate
                    expected = []
                    for feature, probabilities in zip(features, model, strict=True):
                        mixed = memory.blend(feature, probabilities)
                        expected.append(-1 if mixed is None else mixed.argmax())
                    assert np.array_equal(
                        batch_predict(table, candidate, model, fallback), expected
                    )


def test_grid_selection_enforces_old_retention_size_and_deterministic_ties():
    base = MemoryConfig(input_dim=4, dimensions=16, layer=7, neighbors=1, blend=1)
    configs = [
        replace(base, radius=r, temperature=t) for r in (0.3, 0.1, 0.05) for t in (0.1, 0.05)
    ]
    table = NeighborTable(
        np.array([[1], [2], [1], [2]]), np.array([[0.2], [0.2], [0.02], [0.02]], dtype=np.float32)
    )
    model = np.tile([1.0, 0.0, 0.0], (4, 1))
    result = choose_config(
        configs,
        {7: table},
        model,
        np.zeros(4, dtype=np.int64),
        np.array([0, 0, 1, 2]),
        2,
        [1, 2],
        {7: 100},
        source_entries=1000,
        base_bar=2,
    )
    assert result["g3_calibration_pass"]
    assert result["selected"]["old_correct"] == 2
    assert result["selected"]["new_correct"] == 2
    assert result["selected"]["config"]["radius"] == 0.05
    tied = [v["config_hash"] for v in result["grid"] if v["config"]["radius"] == 0.05]
    assert result["selected"]["config_hash"] == min(tied)
    assert all(not v["eligible"] for v in result["grid"] if v["config"]["radius"] == 0.3)
    rejected = choose_config(
        configs,
        {7: table},
        model,
        np.zeros(4, dtype=np.int64),
        np.array([0, 0, 1, 2]),
        2,
        [1, 2],
        {7: 100},
        source_entries=PARAMETER_CAP,
        base_bar=2,
    )
    assert rejected["selected"] is None
    assert not rejected["g3_calibration_pass"]


def test_macro_objective_is_not_micro_accuracy():
    assert macro_fraction(np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 2]), [1, 2]) == Fraction(
        1, 2
    )
