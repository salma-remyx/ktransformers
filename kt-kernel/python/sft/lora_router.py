# Mixture-of-LoRA routing gate and load-balancing loss for LoRA Experts
# SPDX-License-Identifier: Apache-2.0

"""
Mixture-of-LoRA (MoL) routing for the LoRA Experts side path.

LoRAExperts (kt_kernel.sft.lora) composes its specialist LoRA experts with a
hardcoded uniform average: every token gets a fixed 1/N contribution from every
specialist, so the experts cannot specialize and adding specialists dilutes the
ones that matter. This module replaces that fixed average with a learned,
token-level router: a small linear gate scores each specialist from the hidden
state, the top-k specialists are selected per token, and their normalized gate
scores weight their outputs.

A load-balancing auxiliary loss (Switch-Transformer form, adapted from the
Macaron-V1 Mixture-of-LoRA setup which relies on balanced expert utilization to
keep its per-domain specialists extensible) is exposed via
``collect_kt_mol_load_balance_loss`` / ``clear_kt_mol_load_balance_loss`` so a
training loop can add it to the task loss without holding a reference to the
model internals. Statistics are accumulated at forward time in a module-level
buffer and read back by the training loop, which keeps the router usable under
gradient checkpointing where a Python-side sum over per-layer losses would be
recomputed and double-counted.

The router is opt-in: LoRAExperts falls back to its uniform-average behavior
when it has no ``router`` attribute set, so existing checkpoints and configs
are unaffected.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Per-layer router statistics accumulated during forward:
# (importance_sum, load_sum, num_tokens). One entry per routing event.
_lora_router_stats: list[tuple[torch.Tensor, torch.Tensor, int]] = []


class LoRARouter(nn.Module):
    """Learned token-level gate over the LoRA Experts specialists.

    Scores each specialist from the hidden state, keeps the top-k per token,
    and returns scores that already sum to 1 over the selected specialists.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        top_k: int = 1,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if not 1 <= top_k <= num_experts:
            raise ValueError(f"top_k must be in [1, num_experts={num_experts}], got {top_k}")
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (gate_scores, topk_idx), both with leading dims of hidden_states.

        gate_scores has a final dim of ``top_k`` and sums to 1 over that dim;
        topk_idx holds the selected expert indices per token.
        """
        logits = self.gate(hidden_states).float()
        # Top-k first so softmax normalizes over only the selected experts.
        topk_scores, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        gate_scores = F.softmax(topk_scores, dim=-1).to(hidden_states.dtype)
        return gate_scores, topk_idx


def _record_router_stats(logits: torch.Tensor, topk_idx: torch.Tensor) -> None:
    """Accumulate Switch-Transformer importance/load statistics for one routing event.

    ``importance`` keeps its autograd graph so the load-balancing loss can
    train the gate; ``load`` comes from argmax selection and is a constant.
    """
    num_experts = logits.shape[-1]
    importance = logits.softmax(dim=-1).sum(dim=tuple(range(logits.dim() - 1)))
    load = F.one_hot(topk_idx, num_classes=num_experts).sum(dim=tuple(range(topk_idx.dim() - 1)))
    load = load.to(importance.dtype)
    num_tokens = logits.numel() // num_experts
    _lora_router_stats.append((importance, load.detach(), num_tokens))


def reset_kt_mol_router_stats() -> None:
    """Drop any accumulated router statistics (e.g. at the start of a step)."""
    _lora_router_stats.clear()


def collect_kt_mol_load_balance_loss(reduction: str = "mean") -> torch.Tensor | None:
    """Return the accumulated load-balancing loss over routed forward passes.

    Switch-Transformer form: num_experts * sum_e(importance_e * load_e), where
    importance is the softmax mass and load the top-k selection count, both
    normalized by the token count of their forward pass. The loss is 1.0 when
    routing is perfectly balanced and grows toward num_experts as it collapses
    onto a single specialist. Returns None when no router ran (uniform-average
    LoRA Experts, or router layers unused), so callers can unconditionally add
    the result to their task loss.

    ``reduction="mean"`` averages across routing events; ``"sum"`` adds them,
    which matches how a per-layer auxiliary loss would enter the total.
    """
    if not _lora_router_stats:
        return None
    per_event = []
    for importance, load, num_tokens in _lora_router_stats:
        if num_tokens == 0:
            continue
        per_event.append(
            importance.numel() * torch.sum((importance / num_tokens) * (load / num_tokens))
        )
    if not per_event:
        return None
    stacked = torch.stack(per_event)
    return stacked.mean() if reduction == "mean" else stacked.sum()


def clear_kt_mol_load_balance_loss() -> None:
    """Alias of reset, named for the training-loop read-then-clear idiom."""
    _lora_router_stats.clear()


def route_lora_experts(lora_experts: nn.Module, hidden_states: torch.Tensor) -> torch.Tensor:
    """Route hidden_states through lora_experts' specialists with its learned gate.

    This is the routed counterpart of LoRAExperts' uniform average: top-k
    specialists per token, weighted by their normalized gate scores. Specialists
    not selected for a token contribute nothing, so adding a specialist no
    longer dilutes the others.
    """
    router = getattr(lora_experts, "router", None)
    if router is None:
        return lora_experts(hidden_states)

    logits = router.gate(hidden_states).float()
    gate_scores, topk_idx = torch.topk(logits, router.top_k, dim=-1)
    gate_scores = F.softmax(gate_scores, dim=-1).to(hidden_states.dtype)
    _record_router_stats(logits, topk_idx)

    routed = torch.zeros_like(hidden_states)
    for slot in range(router.top_k):
        expert_idx = topk_idx[..., slot]
        weights = gate_scores[..., slot].unsqueeze(-1).to(hidden_states.dtype)
        for idx, expert in enumerate(lora_experts.experts):
            mask = expert_idx == idx
            if not bool(mask.any()):
                continue
            routed = routed + torch.where(
                mask.unsqueeze(-1), expert(hidden_states) * weights, torch.zeros_like(routed)
            )
    return routed
