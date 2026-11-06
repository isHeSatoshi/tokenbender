import pytest
import torch
from torch import nn

from tenpercent.accounting import Budget, cubic_group_count
from tenpercent.masking import MaskedLinear, global_topk, replace_linears


def reference_mask(scores, keep):
    order = sorted(
        [
            (float(v), name, i)
            for name, tensor in sorted(scores.items())
            for i, v in enumerate(tensor.flatten())
        ],
        key=lambda item: (-item[0], item[1], item[2]),
    )
    result = {name: torch.zeros_like(value, dtype=torch.bool) for name, value in scores.items()}
    for _, name, index in order[:keep]:
        result[name].flatten()[index] = True
    return result


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_exact_global_topk_matches_independent_stable_sort(seed):
    torch.manual_seed(seed)
    scores = {
        "z": torch.randn(3, 8),
        "a": torch.randint(-2, 3, (4, 3)).float(),
        "signed_zeros": torch.tensor([-0.0, 0.0]),
    }
    for keep in range(sum(t.numel() for t in scores.values()) + 1):
        expected = reference_mask(scores, keep)
        actual = global_topk(scores, keep)
        assert all(torch.equal(expected[name], actual[name]) for name in scores)


def test_topk_extremes_and_invalid_scores():
    values = torch.tensor(
        [-torch.finfo(torch.float32).max, -1e-35, -0.0, 0.0, 1e-35, torch.finfo(torch.float32).max]
    )
    for keep in range(7):
        assert torch.equal(
            global_topk({"x": values}, keep)["x"], reference_mask({"x": values}, keep)["x"]
        )
    with pytest.raises(ValueError, match="Non-finite"):
        global_topk({"a": torch.tensor([float("nan")])}, 1)


def test_masked_linear_forward_and_surrogate_gradient():
    torch.manual_seed(4)
    linear = nn.Linear(8, 5, bias=False)
    module = MaskedLinear(linear)
    module.hard = torch.rand(5, 2) > 0.5
    x = torch.randn(2, 3, 8, requires_grad=True)
    grad_output = torch.randn(2, 3, 5)
    before = module.weight.clone()
    output = module(x)
    output.backward(grad_output)
    expanded = module.hard.repeat_interleave(4, 1)
    assert torch.equal(
        output, torch.nn.functional.linear(x.detach(), torch.where(expanded, before, 0))
    )
    expected_w_grad = grad_output.reshape(-1, 5).T @ x.detach().reshape(-1, 8)
    expected_scores = (expected_w_grad * before).reshape(5, 2, 4).sum(-1)
    assert torch.allclose(module.scores.grad, expected_scores, atol=1e-6)
    assert torch.allclose(x.grad, grad_output @ torch.where(expanded, before, 0), atol=1e-6)
    assert module.weight.grad is None
    torch.optim.Adam([module.scores], lr=0.01).step()
    assert torch.equal(module.weight, before)


def test_replace_linears_and_all_one_bias_identity():
    model = nn.Module()
    model.self_attn = nn.Module()
    model.self_attn.q_proj = nn.Linear(8, 8, bias=True)
    x = torch.randn(2, 8)
    reference = model.self_attn.q_proj(x).detach()
    modules = replace_linears(model)
    assert len(modules) == 1
    assert torch.equal(model.self_attn.q_proj(x), reference)


def test_budget_and_schedule():
    budget = Budget(16384)
    assert budget.keep_groups == 31872320
    assert budget.source_entries == 152800000
    assert budget.total_groups == 327548928
    with pytest.raises(ValueError):
        Budget(151936)
    counts = [cubic_group_count(step, 1000, 327548928, 31872320) for step in range(1001)]
    assert counts[0] == 327548928 and counts[-1] == 31872320
    assert all(a >= b for a, b in zip(counts, counts[1:]))
