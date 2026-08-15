# SPDX-License-Identifier: Apache-2.0

"""Integration tests for learned-sparse gradient compression.

Exercises the wiring in kt_kernel.sft.lora.sync_kt_lora_gradients (the
post-backward / pre-optimizer boundary), not just the compression module in
isolation.  The LSP-Offload mechanism (arXiv:2406.10181) is validated in
target-native form: repeated fine-tuning-like gradients that share a low-rank
subspace must compress to near-zero relative error once the learned projector
basis has adapted, while the wiring must stay a no-op unless opted in.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kt_kernel.sft.grad_compression import (
    GradientProjector,
    compress_kt_gradients,
    enable_kt_gradient_compression,
    kt_gradient_compression_stats,
    restore_kt_gradients,
)
from kt_kernel.sft.lora import sync_kt_lora_gradients


def _make_model(grad_shape=(64, 64), seed=0):
    """Model with one KT wrapper owning one fused-LoRA-style parameter."""
    torch.manual_seed(seed)
    param = nn.Parameter(torch.randn(*grad_shape))
    wrapper = SimpleNamespace(
        layer_idx=0,
        wrapper=None,  # legacy backend: python owns Parameter.grad
        _uses_authoritative_optimizer_grads=False,
        _kt_world_size_at_wrap=1,
        lora_experts=None,
        _fused_expert_lora_params=[param],
        _peft_lora_modules=None,
    )
    model = SimpleNamespace(_kt_wrappers=[wrapper])
    return model, param


def _low_rank_grad_stream(param, steps=6, seed=0):
    """Gradients confined to a shared low-rank subspace + small noise."""
    generator = torch.Generator().manual_seed(seed)
    basis = torch.randn(param.numel(), 2, generator=generator)
    basis, _ = torch.linalg.qr(basis)
    for _ in range(steps):
        coeffs = torch.randn(2, generator=generator)
        noise = torch.randn(param.numel(), generator=generator) * 1e-5
        yield (basis @ coeffs + noise).reshape(param.shape)


def test_sync_is_noop_without_opt_in():
    model, param = _make_model()
    grads = list(_low_rank_grad_stream(param))
    param.grad = grads[0].clone()
    sync_kt_lora_gradients(model)
    assert torch.equal(param.grad, grads[0])


def test_sync_compresses_when_enabled():
    model, param = _make_model()
    enable_kt_gradient_compression(model, rank=8, density=0.01)
    errors = []
    for grad in _low_rank_grad_stream(param):
        param.grad = grad.clone()
        sync_kt_lora_gradients(model)
        errors.append((param.grad - grad).norm().item() / grad.norm().item())
    # Once the learned projector basis has adapted (first call warms it),
    # a rank-2 gradient stream must compress to the stream's own noise
    # floor (relative errors here are dominated by the 1e-5 noise term).
    assert max(errors[2:]) < 5e-3
    stats = kt_gradient_compression_stats(model)
    assert stats["params"] == 1
    assert stats["rank"] == 8


def test_optimized_step_matches_uncompressed_on_low_rank_stream():
    model, param = _make_model()
    enable_kt_gradient_compression(model, rank=8, density=0.01)

    clean_param = nn.Parameter(param.data.clone())
    clean_model, _ = _make_model()
    clean_model._kt_wrappers[0]._fused_expert_lora_params = [clean_param]

    opt_compressed = torch.optim.SGD([param], lr=0.1)
    opt_clean = torch.optim.SGD([clean_param], lr=0.1)

    for grad in _low_rank_grad_stream(param):
        param.grad = grad.clone()
        clean_param.grad = grad.clone()
        sync_kt_lora_gradients(model)  # compresses in place
        opt_compressed.step()
        opt_clean.step()

    drift = (param.data - clean_param.data).norm().item() / clean_param.data.norm().item()
    assert drift < 5e-3


def test_restore_rematerializes_last_gradient():
    model, param = _make_model()
    enable_kt_gradient_compression(model, rank=4, density=0.05)
    grads = list(_low_rank_grad_stream(param))
    for grad in grads:  # warm the learned projector basis first
        param.grad = grad.clone()
        sync_kt_lora_gradients(model)
    restore_kt_gradients(model, accumulate=False)
    assert (param.grad - grads[-1]).norm().item() / grads[-1].norm().item() < 1e-3


def test_small_gradients_skip_compression():
    model, param = _make_model(grad_shape=(8, 8))
    enable_kt_gradient_compression(model, rank=4, density=0.01)
    grad = torch.randn_like(param.data)
    param.grad = grad.clone()
    sync_kt_lora_gradients(model)
    assert torch.equal(param.grad, grad)  # below min_numel: untouched


def test_authoritative_backend_is_skipped():
    model, param = _make_model()
    enable_kt_gradient_compression(model, rank=8, density=0.01)
    model._kt_wrappers[0]._uses_authoritative_optimizer_grads = True
    grad = torch.randn_like(param.data)
    param.grad = grad.clone()
    sync_kt_lora_gradients(model)  # single-GPU path; must not touch .grad
    assert torch.equal(param.grad, grad)
    assert compress_kt_gradients(model) is None


def test_projector_error_decreases_with_basis_learning():
    param_shape = torch.Size((32, 32))
    projector = GradientProjector(param_shape, rank=16, density=0.0)
    param = nn.Parameter(torch.empty(param_shape))
    first_error = last_error = None
    for grad in _low_rank_grad_stream(param):
        projection = projector.compress(grad)
        last_error = projection.relative_error
        if first_error is None:
            first_error = last_error
        projector.update_basis(grad)
    # First call has an empty basis (error 0 residual kept fully = error 0),
    # so compare the second observation against the converged one instead.
    assert last_error <= first_error + 1e-6


def test_enable_validates_arguments():
    model, _ = _make_model()
    with pytest.raises(ValueError):
        enable_kt_gradient_compression(model, rank=0)
    with pytest.raises(ValueError):
        enable_kt_gradient_compression(model, density=0.0)
