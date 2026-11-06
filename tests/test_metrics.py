import pytest

from tenpercent.metrics import aligned_correctness, paired_comparison, summarize


def records(correct):
    return [
        {"example_id": str(i), "true_label": "a", "predicted_label": "a" if value else "wrong"}
        for i, value in enumerate(correct)
    ]


def test_strict_claim_and_ci_are_separate():
    base = records([True] * 99 + [False])
    candidate = records([True] * 98 + [False] * 2)
    result = paired_comparison(candidate, base, base, draws=100)
    assert not result["strict_pass"]
    assert result["difference_pp"] == -1
    assert result["bar_correct"] == 99


def test_equal_states_have_zero_ci():
    same = records([True, False, True, True])
    result = paired_comparison(same, same, same, draws=100)
    assert result["strict_pass"] and result["paired_ci_95_pp"] == [0, 0]
    assert summarize(same, ["a"])["invalid"] == 1


def test_missing_ids_and_changed_truth_are_errors():
    a = records([True, False])
    with pytest.raises(ValueError):
        aligned_correctness(a, a[:1])
    b = records([True, False])
    b[1]["true_label"] = "different"
    with pytest.raises(ValueError):
        aligned_correctness(a, b)
