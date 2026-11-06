from __future__ import annotations

import hashlib

import torch
from torch import nn
from torch.nn import functional as F

from .constants import BODY_SUFFIXES


def ordered_float_keys(tensor: torch.Tensor) -> torch.Tensor:
    """Monotonic uint32 keys stored in int64; equal signed zeros are tied."""
    if tensor.dtype != torch.float32:
        raise TypeError("Mask scores must be FP32")
    values = torch.where(tensor == 0, torch.zeros_like(tensor), tensor)
    bits = values.contiguous().view(torch.int32).to(torch.int64)
    return torch.where(bits < 0, (~bits) & 0xFFFFFFFF, bits ^ 0x80000000)


@torch.no_grad()
def global_topk(scores: dict[str, torch.Tensor], keep: int) -> dict[str, torch.Tensor]:
    """Exact two-pass radix threshold, no full-size global concatenation or sort."""
    items = sorted(scores.items())
    total = sum(value.numel() for _, value in items)
    if not items or not 0 <= keep <= total:
        raise ValueError("Invalid global top-k budget")
    devices = {value.device for _, value in items}
    if len(devices) != 1:
        raise ValueError("Scores must be on one device")
    device = items[0][1].device
    finite = torch.ones((), dtype=torch.bool, device=device)
    for _, value in items:
        finite.logical_and_(torch.isfinite(value).all())
    if not bool(finite):
        raise ValueError("Non-finite mask scores")
    if keep in (0, total):
        return {
            name: torch.full_like(value, keep == total, dtype=torch.bool) for name, value in items
        }
    histogram = torch.zeros(65536, dtype=torch.int64, device=device)
    for _, value in items:
        high = ordered_float_keys(value.flatten()) >> 16
        histogram.add_(torch.bincount(high, minlength=65536))
    reverse_cumulative = histogram.flip(0).cumsum(0)
    high_bucket = 65535 - int(
        torch.searchsorted(
            reverse_cumulative,
            torch.tensor(keep, device=device),
        )
    )
    higher = int(histogram[high_bucket + 1 :].sum())
    lower_histogram = torch.zeros_like(histogram)
    for _, value in items:
        keys = ordered_float_keys(value.flatten())
        in_bucket = keys[(keys >> 16) == high_bucket]
        lower_histogram.add_(torch.bincount(in_bucket & 0xFFFF, minlength=65536))
    lower_bucket = 65535 - int(
        torch.searchsorted(
            lower_histogram.flip(0).cumsum(0),
            torch.tensor(keep - higher, device=device),
        )
    )
    threshold = (high_bucket << 16) | lower_bucket
    strictly_higher = higher + int(lower_histogram[lower_bucket + 1 :].sum())
    ties_needed = keep - strictly_higher
    result = {}
    for name, value in items:
        keys = ordered_float_keys(value.flatten())
        mask = keys > threshold
        if ties_needed:
            tied = torch.nonzero(keys == threshold, as_tuple=False).flatten()
            take = min(ties_needed, tied.numel())
            mask[tied[:take]] = True
            ties_needed -= take
        result[name] = mask.reshape(value.shape)
    if ties_needed:
        raise RuntimeError("Exact global mask tie allocation failed")
    return result


class MaskedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, scores, hard, group_size, bias):
        ctx.save_for_backward(x, weight, hard)
        ctx.group_size = group_size
        expanded = hard.repeat_interleave(group_size, dim=1)
        masked = torch.where(
            expanded, weight, torch.zeros((), dtype=weight.dtype, device=weight.device)
        )
        return F.linear(x, masked, bias)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, hard = ctx.saved_tensors
        g = ctx.group_size
        expanded = hard.repeat_interleave(g, dim=1)
        masked = torch.where(
            expanded, weight, torch.zeros((), dtype=weight.dtype, device=weight.device)
        )
        grad_x = grad_output @ masked if ctx.needs_input_grad[0] else None
        del masked, expanded
        x2 = x.reshape(-1, x.shape[-1])
        dy = grad_output.reshape(-1, grad_output.shape[-1])
        grad_scores = torch.empty(hard.shape, device=x.device, dtype=torch.float32)
        # Score gradients are accumulated in FP32; tile output rows to bound memory.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x2.float()
            for begin in range(0, weight.shape[0], 128):
                end = min(begin + 128, weight.shape[0])
                grad_w = dy[:, begin:end].float().T @ xf
                grad_scores[begin:end] = (
                    (grad_w * weight[begin:end].float()).reshape(end - begin, -1, g).sum(-1)
                )
        return grad_x, None, grad_scores, None, None, None


class MaskedLinear(nn.Module):
    def __init__(self, linear: nn.Linear, group_size: int = 4):
        super().__init__()
        if linear.weight.shape[1] % group_size:
            raise ValueError("Group crosses a row boundary")
        self.register_buffer("weight", linear.weight.detach())
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach())
        self.group_size = group_size
        self.scores = nn.Parameter(
            torch.zeros(
                (self.weight.shape[0], self.weight.shape[1] // group_size),
                dtype=torch.float32,
                device=self.weight.device,
            )
        )
        self.register_buffer("hard", torch.ones_like(self.scores, dtype=torch.bool))
        self.in_features = linear.in_features
        self.out_features = linear.out_features

    def forward(self, x):
        return MaskedLinearFunction.apply(
            x, self.weight, self.scores, self.hard, self.group_size, self.bias
        )

    @torch.no_grad()
    def initialize(self, input_norm: torch.Tensor | None = None):
        importance = self.weight.float().abs()
        if input_norm is not None:
            importance.mul_(input_norm.to(importance.device).unsqueeze(0))
        scores = importance.reshape(*self.scores.shape, self.group_size).sum(-1)
        self.scores.copy_((scores - scores.mean()) / scores.std(unbiased=False).clamp_min(1e-8))


def replace_linears(model, group_size=4) -> dict[str, MaskedLinear]:
    model.requires_grad_(False)
    result = {}
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and (name + ".weight").endswith(BODY_SUFFIXES):
            parent_name, child = name.rsplit(".", 1)
            parent = model.get_submodule(parent_name)
            replacement = MaskedLinear(module, group_size)
            setattr(parent, child, replacement)
            result[name] = replacement
    if not result:
        raise ValueError("No supported transformer matrices found")
    return dict(sorted(result.items()))


def set_global_mask(modules: dict[str, MaskedLinear], keep: int):
    masks = global_topk({name: module.scores for name, module in modules.items()}, keep)
    for name, module in modules.items():
        # Never mutate a mask saved by an outstanding autograd/checkpoint graph.
        module.hard = masks[name]


@torch.no_grad()
def frozen_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(
        [(n, p) for n, p in model.named_parameters() if not p.requires_grad]
        + [(n, b) for n, b in model.named_buffers() if n.endswith(".weight") or n.endswith(".bias")]
    ):
        digest.update(name.encode())
        data = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
        digest.update(data.tobytes())
    return digest.hexdigest()
