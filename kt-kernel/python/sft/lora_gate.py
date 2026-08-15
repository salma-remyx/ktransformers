# Adaptive gating for LoRA expert mixtures
# SPDX-License-Identifier: Apache-2.0

"""
Adaptive gating for GPU-side LoRA expert mixtures.

``LoRAExperts`` (lora.py) mixes its experts by uniform averaging — every expert
contributes ``1/num_experts`` on every token, so the mixture cannot specialize
and its cost grows linearly with the expert count. This module replaces that
uniform mix with a learned router whose activation threshold is itself learned
from the input, so the number of experts active per token varies with the
difficulty of the token instead of being a fixed hyperparameter.

Two pieces, matching the paper's decomposition:

- ``LoRAGate`` — the router. A linear gate produces per-expert logits; an
  expert fires when its softmax probability clears the uniform bar ``1/E``, and
  the survivors are renormalized to carry the full mixture mass.
- ``ThresholdNetwork`` — a tiny per-token MLP that emits the activation
  threshold π, added to the gate logits before the softmax. ``tanh(π - π₀)``
  acts as a residual around the paper's default threshold, so an untrained net
  stays at the default and no expert is starved early in training.

Adapted from "AdaMoLE: Fine-Tuning Large Language Models with Adaptive Mixture
of Low-Rank Adaptation Experts" (Liu & Luo, 2024, arXiv:2405.00361). The
auxiliary bits that paper carries — its separate training recipe and benchmark
suite — are intentionally not ported; the router plugs into the existing KT SFT
LoRA-experts path and trains with the repo's existing optimizer.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LoRAGate", "ThresholdNetwork", "DEFAULT_THRESHOLD_PI"]


# The paper's threshold-π default: a gate bias of 0.1 sits just above the
# uniform-softmax weight (1/E), so by default every expert stays active and the
# gate degrades gracefully toward the uniform mix it replaces.
DEFAULT_THRESHOLD_PI = 0.1


class ThresholdNetwork(nn.Module):
    """Per-token threshold π that decides how many LoRA experts may fire.

    Two hidden layers over the hidden state — too small to memorize the input,
    just enough to tell an easy token from a hard one. π is computed per token
    rather than pooled over the sequence: the KT layer path receives a flat
    ``(qlen, hidden)`` batch, so a pooled π would make one token's routing
    depend on every other token in the batch.
    """

    def __init__(self, hidden_size: int, default_pi: float = DEFAULT_THRESHOLD_PI):
        super().__init__()
        self.default_pi = default_pi
        self.w_up = nn.Linear(hidden_size, hidden_size // 2, bias=False)
        self.w_hidden = nn.Linear(hidden_size // 2, hidden_size // 4, bias=False)
        self.w_pi = nn.Linear(hidden_size // 4, 1, bias=False)
        # Tiny-but-nonzero so π starts essentially at the default while
        # gradients still reach the hidden layers from step one.
        nn.init.normal_(self.w_pi.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return π of shape [..., 1], broadcastable over the expert dim."""
        h = F.relu(self.w_hidden(F.relu(self.w_up(hidden_states))))
        return self.default_pi + torch.tanh(self.w_pi(h))


class LoRAGate(nn.Module):
    """Router that adaptively selects a variable number of LoRA experts.

    Unlike a static top-k router, the activation threshold is produced by a
    :class:`ThresholdNetwork` from the token itself, so the expert count
    adapts per token. Weights are renormalized over the experts that cleared
    the threshold, so the surviving experts carry the full mixture mass.
    """

    def __init__(
        self,
        num_experts: int,
        hidden_size: int,
        default_pi: float = DEFAULT_THRESHOLD_PI,
        gate_init_std: float = 0.01,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(hidden_size, num_experts, bias=False, device=device, dtype=dtype)
        self.threshold_net = ThresholdNetwork(hidden_size, default_pi=default_pi).to(device=device, dtype=dtype)
        nn.init.normal_(self.gate.weight, mean=0.0, std=gate_init_std)
        self.last_num_active: torch.Tensor | None = None

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(gate_weights, active_mask)`` over the trailing expert dim.

        ``gate_weights`` sums to 1 over the active experts and is 0 elsewhere;
        ``active_mask`` is a float mask of which experts fired. Both are
        differentiable with respect to the gate and the threshold network.
        """
        gate_logits = self.gate(hidden_states).float()
        pi = self.threshold_net(hidden_states).float()
        gate_probs = torch.softmax(gate_logits + pi, dim=-1)

        # An expert fires when its probability clears the uniform bar, capped so
        # at least one and at most all experts are active per token.
        threshold = 1.0 / self.num_experts
        active = (gate_probs >= threshold).to(gate_probs.dtype)
        active = self._clamp_at_least_one(active, gate_probs)

        self.last_num_active = active.sum(dim=-1)

        weights = gate_probs * active
        denom = weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return weights / denom, active

    @staticmethod
    def _clamp_at_least_one(active: torch.Tensor, gate_probs: torch.Tensor) -> torch.Tensor:
        """Force the argmax expert active when the threshold filters out all of them."""
        any_active = active.sum(dim=-1, keepdim=True) > 0
        top = F.one_hot(gate_probs.argmax(dim=-1), num_classes=gate_probs.shape[-1]).to(active.dtype)
        forced = torch.where(any_active, active, top)
        return forced
