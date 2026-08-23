# SPDX-License-Identifier: Apache-2.0

"""Weight-decomposed (magnitude/direction) LoRA normalization for KT SFT.

Adapted from BiDoRA (arXiv:2410.09758): the direction buffers KT already
owns are re-normalized post-step and rescaled by a decoupled magnitude row,
with the magnitude advanced from the accumulated direction gradient in place
of BiDoRA's held-out-data upper loop.
"""

from types import SimpleNamespace

import torch

from kt_kernel.sft import lora as sft_lora
from kt_kernel.sft.arch import MOEArchConfig
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import kt_adapt_peft_lora, update_kt_lora_pointers
from kt_kernel.sft.magnitude import (
    apply_lora_magnitude,
    init_lora_magnitude,
    step_lora_magnitude,
)


def _moe_config(num_experts=2, hidden=4, intermediate=3):
    return MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=num_experts,
        intermediate_size=intermediate,
        num_experts_per_tok=2,
    )


def _lora_buffers(num_experts=2, hidden=4, intermediate=3, rank=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    buffers = {
        "gate_lora_a": torch.randn(num_experts, rank, hidden, generator=generator),
        "gate_lora_b": torch.randn(num_experts, intermediate, rank, generator=generator),
        "up_lora_a": torch.randn(num_experts, rank, hidden, generator=generator),
        "up_lora_b": torch.randn(num_experts, intermediate, rank, generator=generator),
        "down_lora_a": torch.randn(num_experts, rank, intermediate, generator=generator),
        "down_lora_b": torch.randn(num_experts, hidden, rank, generator=generator),
    }
    grad_buffers = {f"grad_{name}": torch.randn_like(tensor) for name, tensor in buffers.items()}
    return buffers, grad_buffers


def test_recomposition_sets_direction_norm_to_magnitude():
    buffers, _ = _lora_buffers()
    state = init_lora_magnitude(buffers, _moe_config(), scaling=1.0)

    apply_lora_magnitude(buffers, state)

    for proj in ("gate", "up", "down"):
        product = torch.matmul(buffers[f"{proj}_lora_b"], buffers[f"{proj}_lora_a"])
        norms = product.float().norm(dim=-1)
        expected = state.magnitudes[f"{proj}_lora_magnitude"].float()
        torch.testing.assert_close(norms, expected, rtol=1e-3, atol=1e-4)


def test_first_step_matches_vanilla_lora_when_b_starts_at_zero():
    buffers, _ = _lora_buffers()
    for key in ("gate_lora_b", "up_lora_b", "down_lora_b"):
        buffers[key].zero_()

    state = init_lora_magnitude(buffers, _moe_config(), scaling=1.0)
    product_before = torch.matmul(buffers["gate_lora_b"], buffers["gate_lora_a"]).clone()

    apply_lora_magnitude(buffers, state)

    product_after = torch.matmul(buffers["gate_lora_b"], buffers["gate_lora_a"])
    # B is zero, so the recomposition is a zero-times-eps rescale: the first
    # forward stays bit-comparable to plain LoRA instead of exploding.
    assert product_after.abs().max().item() == 0.0
    torch.testing.assert_close(product_after, product_before)


def test_magnitude_step_uses_direction_gradient_mass():
    buffers, grad_buffers = _lora_buffers()
    state = init_lora_magnitude(buffers, _moe_config(), scaling=1.0)
    before = state.magnitudes["gate_lora_magnitude"].clone()
    grad_buffers["grad_gate_lora_b"].fill_(2.0)

    step_lora_magnitude(state, grad_buffers, lr=0.1)

    signal = grad_buffers["grad_gate_lora_b"].float().abs().mean(dim=-1)
    torch.testing.assert_close(
        state.magnitudes["gate_lora_magnitude"].float(),
        before.float() + 0.1 * signal,
        rtol=1e-3,
        atol=1e-4,
    )


def _fused_layer(num_experts=2, hidden=4, intermediate=3, rank=2, magnitude=True):
    buffers, grad_buffers = _lora_buffers(num_experts, hidden, intermediate, rank)
    layer = SimpleNamespace(
        layer_idx=0,
        hidden_size=hidden,
        moe_config=_moe_config(num_experts, hidden, intermediate),
        wrapper=None,
        experts=object(),  # non-None so kt_adapt_peft_lora reaches the fused branch
        _experts_attr="experts",
        _fused_experts=True,
        _lora_rank=rank,
        _kt_managed_lora_enabled=True,
        _kt_lora_magnitude=magnitude,
        _full_weight_grad=False,
        _lora_pointers_dirty=False,
        _fused_expert_lora_params=[],
        _peft_lora_modules=None,
    )
    return layer, buffers, grad_buffers


def test_kt_adapt_peft_lora_seeds_magnitude_state_only_when_enabled(monkeypatch):
    layer, buffers, grad_buffers = _fused_layer(magnitude=True)

    def _fake_create(*args, **kwargs):
        return buffers, grad_buffers, [torch.nn.Parameter(t) for t in buffers.values()]

    monkeypatch.setattr(sft_lora, "_create_fused_expert_lora_buffers", _fake_create)
    kt_adapt_peft_lora(SimpleNamespace(_kt_wrappers=[layer]))

    assert layer._lora_magnitude_state is not None
    # gate/up magnitudes are [E, I] and down is [E, H]: (3 + 3 + 4) per expert.
    assert layer._lora_magnitude_state.num_columns() == 2 * (3 + 3 + 4)

    disabled, disabled_buffers, disabled_grads = _fused_layer(magnitude=False)
    monkeypatch.setattr(
        sft_lora,
        "_create_fused_expert_lora_buffers",
        lambda *args, **kwargs: (disabled_buffers, disabled_grads, []),
    )
    kt_adapt_peft_lora(SimpleNamespace(_kt_wrappers=[disabled]))
    assert disabled._lora_magnitude_state is None


def test_update_kt_lora_pointers_applies_decomposition_to_kernel_buffers(monkeypatch):
    layer, buffers, grad_buffers = _fused_layer(magnitude=True)

    def _fake_create(*args, **kwargs):
        return buffers, grad_buffers, [torch.nn.Parameter(t) for t in buffers.values()]

    monkeypatch.setattr(sft_lora, "_create_fused_expert_lora_buffers", _fake_create)
    kt_adapt_peft_lora(SimpleNamespace(_kt_wrappers=[layer]))
    data_ptr_before = buffers["gate_lora_b"].data_ptr()
    magnitude = layer._lora_magnitude_state.magnitudes["gate_lora_magnitude"]

    update_kt_lora_pointers(SimpleNamespace(_kt_wrappers=[layer]), learning_rate=0.05)

    # The C++ kernel keeps reading the same storage, so the decomposition must
    # land in place rather than reallocating the buffers it points at.
    assert buffers["gate_lora_b"].data_ptr() == data_ptr_before
    # Recomposition holds exactly: the direction's column norm is the magnitude.
    product = torch.matmul(buffers["gate_lora_b"], buffers["gate_lora_a"])
    norms = product.float().norm(dim=-1)
    torch.testing.assert_close(norms, magnitude.float(), rtol=1e-2, atol=1e-3)
    assert layer._lora_pointers_dirty


def test_update_kt_lora_pointers_without_magnitude_leaves_buffers_untouched(monkeypatch):
    layer, buffers, grad_buffers = _fused_layer(magnitude=False)

    def _fake_create(*args, **kwargs):
        return buffers, grad_buffers, []

    monkeypatch.setattr(sft_lora, "_create_fused_expert_lora_buffers", _fake_create)
    kt_adapt_peft_lora(SimpleNamespace(_kt_wrappers=[layer]))
    snapshot = {name: tensor.clone() for name, tensor in buffers.items()}

    update_kt_lora_pointers(SimpleNamespace(_kt_wrappers=[layer]), learning_rate=0.05)

    for name, tensor in buffers.items():
        torch.testing.assert_close(tensor, snapshot[name])
    assert layer._lora_pointers_dirty


class _PeftLoraLinear(torch.nn.Module):
    """Smallest PEFT-shaped adapter: a plain ``weight`` Parameter."""

    def __init__(self, rows: int, cols: int):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(rows, cols))


class _PeftExpert(torch.nn.Module):
    """Expert exposing gate/up/down projections carrying PEFT lora_A/lora_B."""

    def __init__(self, hidden: int, intermediate: int):
        super().__init__()
        for name, rows, cols in (
            ("gate_proj", intermediate, hidden),
            ("up_proj", intermediate, hidden),
            ("down_proj", hidden, intermediate),
        ):
            projection = torch.nn.Module()
            projection.lora_A = _PeftLoraLinear(2, cols)
            projection.lora_B = _PeftLoraLinear(rows, 2)
            setattr(self, name, projection)


def test_peft_view_path_also_seeds_magnitude_state(monkeypatch):
    num_experts, hidden, intermediate = 2, 4, 3
    moe_config = _moe_config(num_experts, hidden, intermediate)
    buffers, grad_buffers = _lora_buffers(num_experts, hidden, intermediate)

    layer = SimpleNamespace(
        layer_idx=0,
        moe_config=moe_config,
        wrapper=None,
        experts=torch.nn.ModuleList([_PeftExpert(hidden, intermediate) for _ in range(num_experts)]),
        _experts_attr="experts",
        _fused_experts=False,
        _lora_rank=2,
        _kt_managed_lora_enabled=True,
        _kt_lora_magnitude=True,
        _full_weight_grad=False,
        _lora_pointers_dirty=False,
    )

    monkeypatch.setattr(sft_lora, "_create_lora_view_buffers", lambda *args, **kwargs: buffers)
    monkeypatch.setattr(sft_lora, "_create_lora_grad_buffers", lambda *args, **kwargs: grad_buffers)
    monkeypatch.setattr(sft_lora, "_replace_peft_weights_with_views", lambda *args, **kwargs: None)
    kt_adapt_peft_lora(SimpleNamespace(_kt_wrappers=[layer]))

    # The PEFT modules were collected for the wrapper and the same magnitude
    # state is seeded over the contiguous view buffers.
    assert len(layer._peft_lora_modules) == num_experts
    assert layer._lora_magnitude_state is not None
    assert layer._lora_magnitude_state.num_columns() == 2 * (3 + 3 + 4)


def test_config_flag_defaults_off_and_reads_env_override(monkeypatch):
    monkeypatch.delenv("ACCELERATE_KT_LORA_MAGNITUDE", raising=False)
    assert KTConfig().kt_lora_magnitude is False

    monkeypatch.setenv("ACCELERATE_KT_LORA_MAGNITUDE", "true")
    assert KTConfig(kt_backend="AMXBF16").kt_lora_magnitude is True
