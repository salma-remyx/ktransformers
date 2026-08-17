# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from types import SimpleNamespace

from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.lora import sync_kt_lora_gradients
from kt_kernel.sft.riemannian_lora_grads import apply_riemannian_lora_precondition


def _lora_pair(rank=2, in_features=3, out_features=4, seed=0):
    """One PEFT-style (lora_A, lora_B) module pair with distinct factors."""
    generator = torch.Generator().manual_seed(seed)
    lora_a = torch.nn.Linear(in_features, rank, bias=False)
    lora_b = torch.nn.Linear(rank, out_features, bias=False)
    with torch.no_grad():
        lora_a.weight.copy_(torch.randn(lora_a.weight.shape, generator=generator))
        lora_b.weight.copy_(torch.randn(lora_b.weight.shape, generator=generator))
    lora_a.weight.grad = torch.randn_like(lora_a.weight)
    lora_b.weight.grad = torch.randn_like(lora_b.weight)
    return lora_a, lora_b


def _peft_model(num_experts=2, num_layers=1):
    """Model carrying PEFT-injected KT LoRA, mirroring kt_adapt_peft_lora's layout."""
    wrappers = []
    for layer_idx in range(num_layers):
        peft_lora_modules = {}
        for expert_idx in range(num_experts):
            peft_lora_modules[expert_idx] = {
                f"{proj}_proj": _lora_pair(seed=expert_idx) for proj in ("gate", "up", "down")
            }
        wrappers.append(SimpleNamespace(layer_idx=layer_idx, _peft_lora_modules=peft_lora_modules))
    return SimpleNamespace(_kt_wrappers=wrappers)


def _expected_preconditioned(pair, grads, reg):
    """The paper's update direction, computed independently of the module.

    ``grads`` is the ``(grad_a, grad_b)`` snapshot taken *before* the call.
    """
    lora_a, lora_b = pair
    a = lora_a.weight.detach().float()
    b = lora_b.weight.detach().float()
    pre_a = torch.inverse(b.t() @ b + reg * torch.eye(b.shape[1]))
    pre_b = torch.inverse(a @ a.t() + reg * torch.eye(a.shape[0]))
    return pre_a @ grads[0].float(), grads[1].float() @ pre_b


def test_sync_helper_applies_preconditioner_when_config_opts_in():
    model = _peft_model()
    reg = 0.5
    # wrap_moe_layers_with_kt_wrapper copies the KTConfig knob onto each
    # wrapper alongside the other per-layer LoRA training flags.
    for wrapper in model._kt_wrappers:
        wrapper._lora_riemannian_reg = reg
    gate_pair = model._kt_wrappers[0]._peft_lora_modules[0]["gate_proj"]
    grads = (gate_pair[0].weight.grad.clone(), gate_pair[1].weight.grad.clone())

    sync_kt_lora_gradients(model)

    expected_a, expected_b = _expected_preconditioned(gate_pair, grads, reg)
    torch.testing.assert_close(gate_pair[0].weight.grad, expected_a)
    torch.testing.assert_close(gate_pair[1].weight.grad, expected_b)
    assert not torch.equal(gate_pair[0].weight.grad, grads[0])


def test_sync_helper_is_a_noop_without_the_knob():
    model = _peft_model()  # wrappers carry no _lora_riemannian_reg flag
    gate_pair = model._kt_wrappers[0]._peft_lora_modules[0]["gate_proj"]
    before_a = gate_pair[0].weight.grad.clone()
    before_b = gate_pair[1].weight.grad.clone()

    sync_kt_lora_gradients(model)

    torch.testing.assert_close(gate_pair[0].weight.grad, before_a)
    torch.testing.assert_close(gate_pair[1].weight.grad, before_b)


def test_preconditioner_leaves_parameters_and_missing_grads_alone():
    model = _peft_model(num_experts=2)
    weights_before = [
        (pair[0].weight.detach().clone(), pair[1].weight.detach().clone())
        for pair in model._kt_wrappers[0]._peft_lora_modules[0].values()
    ]
    # Expert 1 never received a gradient (closed optimizer window).
    for pair in model._kt_wrappers[0]._peft_lora_modules[1].values():
        pair[0].weight.grad = None
        pair[1].weight.grad = None

    conditioned = apply_riemannian_lora_precondition(model, reg=1e-3)

    assert conditioned == 3  # expert 0's three projections only
    for (weight_a, weight_b), pair in zip(
        weights_before, model._kt_wrappers[0]._peft_lora_modules[0].values()
    ):
        torch.testing.assert_close(pair[0].weight, weight_a)
        torch.testing.assert_close(pair[1].weight, weight_b)


def test_preconditioned_direction_uses_the_partner_factors_gram():
    """dA must be scaled by B's gram and dB by A's — not by each factor's own."""
    model = _peft_model(num_experts=1, num_layers=1)
    model._kt_wrappers[0]._peft_lora_modules = {0: {"gate_proj": _lora_pair(seed=3)}}
    lora_a, lora_b = model._kt_wrappers[0]._peft_lora_modules[0]["gate_proj"]
    grad_a_before = lora_a.weight.grad.clone()
    grad_b_before = lora_b.weight.grad.clone()
    rank = lora_a.weight.shape[0]

    apply_riemannian_lora_precondition(model, reg=1e-6)

    # dA <- (B^T B)^-1 dA: B's gram is [r, r], so it mixes A's rank rows.
    pre_a = torch.inverse(
        lora_b.weight.detach().float().t() @ lora_b.weight.detach().float() + 1e-6 * torch.eye(rank)
    )
    torch.testing.assert_close(
        lora_a.weight.grad.float(), pre_a @ grad_a_before.float(), rtol=1e-4, atol=1e-5
    )
    # dB <- dB (A A^T)^-1: A's gram is [r, r], so it mixes B's rank columns.
    pre_b = torch.inverse(
        lora_a.weight.detach().float() @ lora_a.weight.detach().float().t() + 1e-6 * torch.eye(rank)
    )
    torch.testing.assert_close(
        lora_b.weight.grad.float(), grad_b_before.float() @ pre_b, rtol=1e-4, atol=1e-5
    )


def test_fused_lora_buffers_are_preconditioned_per_expert():
    rank, hidden, inter, experts = 2, 3, 4, 2
    order = ("gate", "up", "down")
    params = []
    for proj in order:
        for side in ("a", "b"):
            shape = (experts, rank, hidden) if side == "a" else (experts, inter, rank)
            param = torch.nn.Parameter(torch.randn(shape, dtype=torch.bfloat16))
            param.grad = torch.randn(shape, dtype=torch.bfloat16)
            params.append(param)
    wrapper = SimpleNamespace(layer_idx=0, _fused_expert_lora_params=params)
    model = SimpleNamespace(_kt_wrappers=[wrapper])

    gate_a, gate_b = params[0], params[1]
    grad_b_before = gate_b.grad.clone()

    conditioned = apply_riemannian_lora_precondition(model, reg=1e-2)

    assert conditioned == 3
    # Each expert's dB row block is scaled by that expert's own A A^T inverse,
    # so the two experts move by different amounts.
    for expert_idx in range(experts):
        a = gate_a.detach()[expert_idx].float()
        pre = torch.inverse(a @ a.t() + 1e-2 * torch.eye(rank))
        expected = grad_b_before[expert_idx].float() @ pre
        torch.testing.assert_close(
            gate_b.grad[expert_idx].float(), expected, rtol=2e-2, atol=2e-2
        )


def test_non_positive_reg_is_rejected():
    model = _peft_model()
    with pytest.raises(ValueError, match="positive"):
        apply_riemannian_lora_precondition(model, reg=0.0)


def test_config_knob_reads_environment_default():
    import os

    key = "ACCELERATE_KT_LORA_RIEMANNIAN_REG"
    previous = os.environ.get(key)
    try:
        os.environ[key] = "0.25"
        config = KTConfig()
        assert config.kt_lora_riemannian_reg == 0.25
    finally:
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous

    assert KTConfig().kt_lora_riemannian_reg is None


def test_kt_config_field_survives_from_object_round_trip():
    """The knob must keep working through the plugin-to-KTConfig path."""
    plugin = SimpleNamespace(kt_lora_rank=8, kt_lora_riemannian_reg=1e-2)
    config = KTConfig.from_object(plugin)
    assert config.kt_lora_riemannian_reg == 1e-2
    assert config.kt_lora_rank == 8
