# SPDX-License-Identifier: Apache-2.0

"""Adaptive gating for the GPU-side LoRA expert mixture.

Exercises the LoRAExperts call site in kt_kernel.sft.lora: with a gate attached
the uniform 1/E mix is replaced by a learned, threshold-based routing whose
active expert count varies per token. Default (no gate) behavior is unchanged.
"""

import os
import tempfile
from types import SimpleNamespace

import pytest
import torch

from kt_kernel.sft import LoRAExperts, LoRAGate
from kt_kernel.sft.lora import load_lora_experts_from_adapter, save_lora_experts_to_adapter


def _make_experts(num_experts=3, hidden=8, intermediate=16, with_gate=False, seed=0):
    torch.manual_seed(seed)
    gate = (
        LoRAGate(
            num_experts,
            hidden,
            default_pi=0.1,
            device="cpu",
            dtype=torch.float32,
        )
        if with_gate
        else None
    )
    module = LoRAExperts(
        num_experts=num_experts,
        hidden_size=hidden,
        intermediate_size=intermediate,
        device="cpu",
        dtype=torch.float32,
        gate=gate,
    )
    # Non-zero down-projections so experts are distinguishable.
    for expert in module.experts:
        torch.nn.init.normal_(expert.le_down.weight, std=0.1)
    return module


class _GateWrapper(SimpleNamespace):
    """Stand-in for a KTMoELayerWrapper, carrying just what save/load touch."""

    def __init__(self, layer_idx, lora_experts):
        super().__init__(layer_idx=layer_idx, lora_experts=lora_experts)


def test_ungated_mix_is_uniform_average():
    module = _make_experts(num_experts=4, seed=1)
    hidden_states = torch.randn(5, 8)

    expected = sum(expert(hidden_states) for expert in module.experts) / 4

    assert module.gate is None
    assert torch.allclose(module(hidden_states), expected, atol=1e-6)


def test_gate_mixes_by_thresholded_weights():
    module = _make_experts(num_experts=3, with_gate=True, seed=2)
    hidden_states = torch.randn(6, 8)

    gate_weights, active = module.gate(hidden_states)

    # Active experts' weights renormalize to 1; inactive are exactly zero.
    assert torch.allclose(gate_weights.sum(dim=-1), torch.ones(6), atol=1e-5)
    assert torch.equal(gate_weights * (1 - active), torch.zeros_like(gate_weights))
    # At least one expert fires per token.
    assert (active.sum(dim=-1) >= 1).all()

    output = module(hidden_states)
    manual = torch.zeros_like(output)
    for idx, expert in enumerate(module.experts):
        manual = manual + expert(hidden_states) * gate_weights[:, idx : idx + 1]
    assert torch.allclose(output, manual, atol=1e-5)


def test_gate_reduces_to_uniform_mix_when_untrained():
    """A zero-initialized router must degrade to the uniform mix it replaces."""
    hidden_states = torch.randn(4, 8)
    gated = _make_experts(num_experts=3, with_gate=True, seed=3)
    ungated = _make_experts(num_experts=3, with_gate=False, seed=3)

    # Same experts, same seed -> identical weights; only routing differs.
    for g_expert, u_expert in zip(gated.experts, ungated.experts):
        for name in ("le_gate", "le_up", "le_down"):
            getattr(g_expert, name).weight.data.copy_(getattr(u_expert, name).weight.data)

    # Zero gate weights + zero threshold head -> uniform probabilities, and
    # with num_experts=3 every prob (1/3) clears the 1/3 bar, so all experts
    # stay on and the mix is exactly the uniform average.
    with torch.no_grad():
        gated.gate.gate.weight.zero_()
        gated.gate.threshold_net.w_pi.weight.zero_()

    assert torch.allclose(gated(hidden_states), ungated(hidden_states), atol=1e-5)


def test_gate_gradients_reach_router_and_threshold():
    module = _make_experts(num_experts=3, with_gate=True, seed=4)
    hidden_states = torch.randn(7, 8, requires_grad=True)

    module(hidden_states).sum().backward()

    for name, param in module.gate.named_parameters():
        assert param.grad is not None and param.grad.abs().sum() > 0, f"no gradient reached {name}"
    assert hidden_states.grad is not None


def test_gate_round_trips_through_adapter_checkpoint():
    module = _make_experts(num_experts=3, with_gate=True, seed=5)
    wrapper = _GateWrapper(layer_idx=3, lora_experts=module)

    class _Model:
        _kt_wrappers = [wrapper]

    hidden_states = torch.randn(2, 8)
    before = module(hidden_states)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_lora_experts_to_adapter(_Model(), tmpdir)
        # Re-init to different values, then restore.
        for param in module.parameters():
            with torch.no_grad():
                param.normal_(std=0.5)
        assert not torch.allclose(module(hidden_states), before, atol=1e-3)
        load_lora_experts_from_adapter(_Model(), tmpdir)

    assert torch.allclose(module(hidden_states), before, atol=1e-5)


def test_config_env_flag_enables_gate():
    from kt_kernel.sft.config import KTConfig

    cfg = KTConfig(kt_use_lora_experts=True, kt_lora_expert_num=4, kt_lora_expert_gate=True)
    assert cfg.kt_lora_expert_gate is True

    cfg_off = KTConfig(kt_use_lora_experts=True, kt_lora_expert_num=4)
    assert cfg_off.kt_lora_expert_gate is False


def test_wrapper_wiring_shape_matches_gate_constructor():
    """The gate wrapper.py builds must accept the kwargs it is given.

    Mirrors the wrap_moe_layers_with_kt_wrapper construction (device/dtype
    kwargs, then LoRAExperts(gate=...)) on CPU, so a signature drift between
    lora_gate.py and wrapper.py fails here rather than at training time.
    """
    hidden_size, num_experts = 16, 3
    hidden_states = torch.randn(4, hidden_size)

    gate = LoRAGate(num_experts, hidden_size, device="cpu", dtype=torch.float32)
    module = LoRAExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=8,
        device="cpu",
        dtype=torch.float32,
        gate=gate,
    )

    assert module.gate is gate
    assert gate.gate.weight.shape == (num_experts, hidden_size)
    assert gate.gate.weight.device.type == "cpu"

    out = module(hidden_states)
    assert out.shape == (4, hidden_size)
    assert torch.isfinite(out).all()


@pytest.mark.parametrize("default_pi", [0.1, 0.5, 2.0])
def test_threshold_net_starts_near_default_pi(default_pi):
    from kt_kernel.sft import ThresholdNetwork

    torch.manual_seed(0)
    net = ThresholdNetwork(hidden_size=8, default_pi=default_pi)
    hidden_states = torch.randn(3, 5, 8)

    pi = net(hidden_states)

    # Small-but-nonzero init: π starts essentially at the default while
    # gradients still reach the hidden layers from step one.
    assert pi.shape == (3, 5, 1)
    assert torch.allclose(pi, torch.full_like(pi, default_pi), atol=1e-2)
    assert pi.grad_fn is not None
