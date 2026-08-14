# Gated router for GPU-side LoRA Experts
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight gated router for LoRA Expert MLPs.

Adapted from MixLoRA (Cai et al., "MixLoRA: Enhancing Large Language Models
Fine-Tuning with LoRA-based Mixture of Experts", arXiv:2404.15159): instead of
averaging every LoRA expert over all tokens, a small linear gate scores each
token per expert, softmax-normalizes, and only the top-k experts contribute
with renormalized weights. This concentrates each expert's gradient on the
tokens it actually serves and scales to more experts without paying full
dense-combination cost.

Scoped out (target-native simplification): MixLoRA's load-balancing
regularizer on the gate and its multi-task benchmark suite. The gate here is
trained purely end-to-end through the base task loss; load balancing can be
added later as an auxiliary loss if experts collapse.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GatedLoRARouter(nn.Module):
    """Linear gate producing per-token expert logits for LoRA experts.

    Zero-initialized so training starts from the uniform-average behavior of
    the un-gated :class:`~kt_kernel.sft.lora.LoRAExperts` path (softmax over
    equal logits yields uniform top-k weights); the gate then learns to
    specialize from there.

    Args:
        hidden_size: dimension of the tokens routed (input hidden dim).
        num_experts: number of LoRA expert MLPs to route between.
        top_k: how many experts each token uses (clamped to ``num_experts``).
        gate_bias: optional learnable per-expert bias added to gate logits.
    """

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int,
        gate_bias: bool = False,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = max(1, min(top_k, num_experts))
        self.gate = nn.Linear(hidden_size, num_experts, bias=gate_bias, device=device, dtype=dtype)
        nn.init.zeros_(self.gate.weight)
        if gate_bias:
            nn.init.zeros_(self.gate.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return float32 expert logits of shape ``(..., num_experts)``."""
        return self.gate(hidden_states).float()


def route_experts(
    gate: nn.Module,
    experts: nn.ModuleList,
    hidden_states: torch.Tensor,
    return_stats: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Gated top-k dispatch of ``hidden_states`` over ``experts``.

    Each token is scored per expert by ``gate``, the top-k experts are
    softmax-normalized (renormalized over the selected k), and expert outputs
    are blended per token. Each expert only computes on the tokens routed to
    it, so cost scales with top-k rather than the expert count.

    Args:
        gate: module mapping ``(..., hidden_size)`` to ``(..., num_experts)``
            logits; its ``top_k`` attribute selects experts per token
            (defaults to all experts when absent).
        experts: expert MLPs to dispatch between.
        hidden_states: tokens of shape ``(..., hidden_size)``.
        return_stats: when True, also return per-expert routing statistics.

    Returns:
        The combined output with the shape/dtype of ``hidden_states`` and,
        when ``return_stats``, a tuple with a float32 tensor of shape
        ``(num_experts,)`` holding each expert's mean routing weight
        (diagnostic for expert collapse).
    """
    orig_shape = hidden_states.shape
    flat = hidden_states.reshape(-1, orig_shape[-1])
    num_tokens = flat.shape[0]
    top_k = max(1, min(int(getattr(gate, "top_k", len(experts))), len(experts)))

    logits = gate(flat)
    topk_weights, topk_idx = torch.topk(logits, top_k, dim=-1)
    topk_weights = F.softmax(topk_weights.float(), dim=-1)

    expert_out = torch.zeros_like(flat)
    routing_stats = torch.zeros(len(experts), device=flat.device, dtype=torch.float32)
    for expert_idx, expert in enumerate(experts):
        mask = topk_idx == expert_idx
        selected = mask.any(dim=-1)
        if not selected.any():
            continue
        token_weights = (topk_weights * mask).sum(dim=-1, keepdim=True)
        out = expert(flat[selected]).to(flat.dtype)
        expert_out[selected] += out * token_weights[selected].to(out.dtype)
        routing_stats[expert_idx] = token_weights[selected].sum() / num_tokens

    output = expert_out.reshape(orig_shape).to(hidden_states.dtype)
    if return_stats:
        return output, routing_stats
    return output
