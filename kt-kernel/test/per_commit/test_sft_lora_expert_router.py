# SPDX-License-Identifier: Apache-2.0

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

import pytest
import torch
import torch.nn as nn

from kt_kernel.sft import LoRAExperts
from kt_kernel.sft.lora import (
    _collect_kt_lora_params,
    load_lora_experts_from_adapter,
    save_lora_experts_to_adapter,
)
from kt_kernel.sft.lora_expert_router import GatedLoRARouter, route_experts

HIDDEN = 8
INTERMEDIATE = 16


def _make_experts(num_experts=4, top_k=None, device="cpu"):
    torch.manual_seed(0)
    experts = LoRAExperts(
        num_experts=num_experts,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        device=device,
        dtype=torch.float32,
        top_k=top_k,
    )
    # LoRA init leaves le_down at zero, so expert outputs start at zero; give
    # the experts nonzero outputs so routing has a signal to train against.
    with torch.no_grad():
        for expert in experts.experts:
            expert.le_down.weight.normal_()
    return experts


def test_default_lora_experts_stay_uniform_average():
    experts = _make_experts(num_experts=4)
    assert experts.gate is None
    assert experts.top_k == 4
    hidden_states = torch.randn(2, 5, HIDDEN)

    manual = torch.zeros_like(hidden_states)
    for expert in experts.experts:
        manual = manual + expert(hidden_states)
    assert torch.allclose(experts(hidden_states), manual / 4)


def test_top_k_enables_zero_initialized_gate():
    experts = _make_experts(num_experts=4, top_k=2)
    assert isinstance(experts.gate, GatedLoRARouter)
    assert experts.top_k == 2
    assert torch.count_nonzero(experts.gate.gate.weight) == 0


def test_gated_forward_matches_functional_router():
    experts = _make_experts(num_experts=4, top_k=2)
    with torch.no_grad():
        experts.gate.gate.weight.normal_()
    hidden_states = torch.randn(3, 7, HIDDEN)

    modular, stats = route_experts(experts.gate, experts.experts, hidden_states, return_stats=True)
    assert torch.allclose(experts(hidden_states), modular)
    # Softmax over the selected top-k renormalizes to 1 per token.
    assert stats.shape == (4,)
    assert stats.sum().item() == pytest.approx(1.0, abs=1e-5)


def test_gate_receives_gradients_end_to_end():
    experts = _make_experts(num_experts=4, top_k=2)
    hidden_states = torch.randn(6, HIDDEN)
    experts(hidden_states).float().square().sum().backward()

    assert experts.gate.gate.weight.grad is not None
    # With zero-init gate logits every expert gets weight 1/2, so the
    # gradient distinguishes experts by their (nonzero) outputs.
    assert experts.gate.gate.weight.grad.abs().sum() > 0
    routed = [e for e in experts.experts if e.le_gate.weight.grad is not None]
    assert routed, "no expert received gradient"


def test_route_experts_computes_each_expert_only_on_routed_tokens():
    calls = []

    class _CountingExpert(nn.Module):
        def __init__(self, idx):
            super().__init__()
            self.idx = idx
            self.linear = nn.Linear(HIDDEN, HIDDEN, bias=False)

        def forward(self, x):
            calls.append((self.idx, x.shape[0]))
            return self.linear(x)

    gate = GatedLoRARouter(HIDDEN, num_experts=3, top_k=1, device="cpu", dtype=torch.float32)
    with torch.no_grad():
        gate.gate.weight.normal_()
    counted = nn.ModuleList([_CountingExpert(i) for i in range(3)])
    hidden_states = torch.randn(10, HIDDEN)

    out, stats = route_experts(gate, counted, hidden_states, return_stats=True)
    assert out.shape == hidden_states.shape

    # Every token routed exactly once, across whichever experts won it.
    assert sum(n for _, n in calls) == 10
    assert stats.sum().item() == 1


def _make_wrapper(experts, layer_idx=3):
    return SimpleNamespace(layer_idx=layer_idx, lora_experts=experts)


def test_gate_params_are_collected_and_checkpointed(tmp_path):
    experts = _make_experts(num_experts=4, top_k=2)
    wrapper = _make_wrapper(experts)
    model = SimpleNamespace(_kt_wrappers=[wrapper])

    params = _collect_kt_lora_params([wrapper])
    gate_param_ids = {id(p) for p in experts.gate.parameters()}
    assert gate_param_ids.issubset({id(p) for p in params})

    with torch.no_grad():
        experts.gate.gate.weight.normal_()
    expected_gate = experts.gate.gate.weight.detach().clone()

    save_lora_experts_to_adapter(model, str(tmp_path))

    # Reset gate to zero, then confirm the adapter round-trips it back.
    with torch.no_grad():
        experts.gate.gate.weight.zero_()
    load_lora_experts_from_adapter(model, str(tmp_path))
    assert torch.allclose(experts.gate.gate.weight.data.cpu(), expected_gate)


def test_uniform_checkpoint_still_round_trips_without_gate(tmp_path):
    experts = _make_experts(num_experts=2)
    wrapper = _make_wrapper(experts, layer_idx=0)
    model = SimpleNamespace(_kt_wrappers=[wrapper])

    with torch.no_grad():
        experts.experts[0].le_gate.weight.normal_()
    expected = experts.experts[0].le_gate.weight.detach().clone()

    save_lora_experts_to_adapter(model, str(tmp_path))
    with torch.no_grad():
        experts.experts[0].le_gate.weight.zero_()
    load_lora_experts_from_adapter(model, str(tmp_path))
    assert torch.allclose(experts.experts[0].le_gate.weight.data.cpu(), expected)
