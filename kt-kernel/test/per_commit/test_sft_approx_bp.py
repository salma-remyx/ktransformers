# SPDX-License-Identifier: Apache-2.0

from collections import Counter

import pytest
import torch

from kt_kernel.sft.approx_bp import (
    BACKWARD_BITS_PER_ELEMENT,
    _SILU_SLOPES,
    _pack_segments,
    _segment_index,
    _unpack_slopes,
    approx_silu,
    estimate_lora_expert_savings,
)
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import LoRAExpertMLP, LoRAExperts


def test_approx_bp_forward_matches_silu_exactly():
    x = torch.linspace(-8.0, 8.0, 401, dtype=torch.float64)

    torch.testing.assert_close(approx_silu(x), torch.nn.functional.silu(x))


def test_approx_bp_backward_returns_step_function_derivative():
    x = torch.tensor([-7.0, -3.0, -0.5, 0.5, 3.0, 7.0], requires_grad=True)
    (grad,) = torch.autograd.grad(approx_silu(x).sum(), x)

    # Four segments, 2 bits each: the surrogate is constant inside a segment
    # and jumps at the fitted boundaries c = (-6.305, -0.0008685, 6.326).
    assert BACKWARD_BITS_PER_ELEMENT == 2
    assert grad[0] == pytest.approx(0.0, abs=1e-3)
    assert grad[1] == grad[2] == pytest.approx(-0.0406, abs=1e-3)
    assert grad[3] == grad[4] == pytest.approx(1.0404, abs=1e-3)
    assert grad[5] == pytest.approx(1.0, abs=1e-3)


def test_approx_bp_saves_two_bits_per_element_of_backward_state():
    torch.manual_seed(3)
    x = torch.randn(1001, dtype=torch.float32)

    packed = _pack_segments(_segment_index(x))
    slopes = _unpack_slopes(packed, x.numel(), x)

    # Four 2-bit fields per byte, rounded up for the tail.
    assert packed.numel() == (1001 + 3) // 4
    assert packed.dtype == torch.uint8
    assert torch.equal(slopes, torch.tensor(_SILU_SLOPES, dtype=x.dtype)[_segment_index(x)])


def test_lora_expert_mlp_forward_is_unchanged_by_approx_bp():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5, 8)
    exact = LoRAExpertMLP(8, 16, device="cpu", dtype=torch.float32)
    approx = LoRAExpertMLP(8, 16, device="cpu", dtype=torch.float32, approx_bp=True)
    approx.load_state_dict(exact.state_dict())

    torch.testing.assert_close(approx(hidden_states), exact(hidden_states))


def test_lora_expert_mlp_approx_bp_backward_stays_finite_and_nonzero():
    torch.manual_seed(1)
    expert = LoRAExpertMLP(8, 16, device="cpu", dtype=torch.float32, approx_bp=True)
    # le_down starts at zero (LoRA convention), so randomize it to give the
    # backward a nonzero path to flow through.
    with torch.no_grad():
        expert.le_down.weight.normal_(0.0, 0.3)
    x = torch.randn(3, 5, 8, requires_grad=True)
    expert(x).square().mean().backward()

    assert torch.isfinite(x.grad).all()
    for name, param in expert.named_parameters():
        assert param.grad is not None, name
        assert torch.isfinite(param.grad).all(), name
        assert torch.count_nonzero(param.grad) > 0, name


def test_lora_expert_mlp_approx_bp_gradients_track_exact_direction():
    torch.manual_seed(2)
    exact = LoRAExpertMLP(8, 16, device="cpu", dtype=torch.float32)
    approx = LoRAExpertMLP(8, 16, device="cpu", dtype=torch.float32, approx_bp=True)
    approx.load_state_dict(exact.state_dict())
    with torch.no_grad():
        exact.le_down.weight.normal_(0.0, 0.3)
    approx.load_state_dict(exact.state_dict())
    x = torch.randn(64, 8)
    target = torch.randn(64, 8)

    exact(x).sub(target).square().mean().backward()
    approx(x).sub(target).square().mean().backward()

    for name, param in exact.named_parameters():
        reference = param.grad
        surrogate = approx.get_parameter(f"{name}").grad
        correlation = torch.corrcoef(torch.stack([reference.flatten(), surrogate.flatten()]))[0, 1]
        # Approx-BP decouples forward and backward, so gradients are aligned
        # in direction rather than equal in magnitude.
        assert correlation.item() > 0.5, name


def test_lora_experts_plumbs_approx_bp_into_every_expert():
    experts = LoRAExperts(3, 8, 16, device="cpu", dtype=torch.float32, approx_bp=True)

    assert experts.approx_bp is True
    assert all(expert.approx_bp for expert in experts.experts)


def _retained_for_backward(module, hidden_states):
    saved = []

    def pack(tensor):
        saved.append((str(tensor.dtype), tensor.numel() * tensor.element_size()))
        return tensor

    def unpack(tensor):
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
        module(hidden_states).square().mean().backward()
    return saved


def test_approx_bp_shrinks_what_autograd_retains_for_backward():
    hidden_states = torch.randn(64, 512)

    def build(approx_bp):
        experts = LoRAExperts(8, 512, 1024, device="cpu", dtype=torch.float32, approx_bp=approx_bp)
        with torch.no_grad():
            for expert in experts.experts:
                expert.le_down.weight.normal_(0.0, 0.3)
        return experts

    exact = Counter(_retained_for_backward(build(False), hidden_states))
    approx = Counter(_retained_for_backward(build(True), hidden_states))

    # Per expert, exact SiLU retains TWO [tokens, intermediate] inputs (the
    # SiLU input and the le_up input it multiplies); ReSiLU2 retains only the
    # le_up one, because the SiLU input becomes a 2-bit segment index.
    intermediate_bytes = 64 * 1024 * 4
    assert exact[("torch.float32", intermediate_bytes)] == 32
    assert approx[("torch.float32", intermediate_bytes)] == 24

    # 8 experts' activation inputs collapse into packed indices, four per byte.
    assert approx[("torch.uint8", 16384)] == 8
    assert all(dtype != "torch.uint8" for dtype, _ in exact)


def test_kt_config_reads_approx_bp_env_default(monkeypatch):
    monkeypatch.setenv("ACCELERATE_KT_APPROX_BP", "1")
    assert KTConfig().kt_approx_bp is True

    monkeypatch.delenv("ACCELERATE_KT_APPROX_BP")
    assert KTConfig().kt_approx_bp is False

    assert KTConfig(kt_approx_bp=True).kt_approx_bp is True


def test_estimate_lora_expert_savings_matches_two_bits_per_element():
    # Per activation: 8 bf16 elements (16 bytes) exact vs a 2-byte segment
    # index, so 14 bytes saved per expert per token row.
    assert estimate_lora_expert_savings(2, 4, 8, 1, dtype=torch.bfloat16) == 2 * 14
    assert estimate_lora_expert_savings(2, 4, 8, 0, dtype=torch.bfloat16) == 0


def test_estimate_lora_expert_savings_is_nontrivial_for_a_realistic_layer():
    saved = estimate_lora_expert_savings(
        num_experts=8, hidden_size=4096, intermediate_size=1024, num_tokens=4096
    )

    # 8 experts * 4096 tokens * 1024 elements * (2 bytes - 1/4 byte) ~= 56 MiB/layer.
    elements = 8 * 4096 * 1024
    assert saved == elements * 2 - (elements * 2 + 7) // 8
    assert saved > 50 * 1024 * 1024
