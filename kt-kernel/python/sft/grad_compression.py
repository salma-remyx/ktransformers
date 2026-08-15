# Low-rank + sparse gradient compression for KT SFT offloading
# SPDX-License-Identifier: Apache-2.0
"""Opt-in gradient compression for KT-managed trainable parameters.

Implements the core mechanism of LSP-Offload (arXiv:2406.10181, "Practical
offloading for fine-tuning LLM on commodity GPU via learned sparse
projectors"): instead of communicating full LoRA/base-weight gradients
across the CPU↔GPU boundary, represent each gradient as

    g ~= B @ c + scatter(top-k residual)

where ``B`` is a *learned* projector basis (rank ``r``) maintained across
steps, and the component of the gradient outside the learned subspace is
kept sparsely (top-``k`` magnitudes). Fine-tuning gradients are known to be
dominated by a small set of directions, so once the basis captures the
gradient subspace the sparse residual — and the representation error —
collapses; that is what lets the paper offload at near-native accuracy.

Mode 2 (adapted port) relative to the paper:

* kept at fidelity — learned low-rank projector + sparse residual + basis
  update from the observed gradient stream (power-iteration style QR);
* substituted — the paper's dedicated offload runtime, per-layer RPC
  protocol and GPU-side projector serving are replaced by the repo's own
  optimizer-boundary sync point (``sync_kt_lora_gradients``, which runs
  after ``KTMoEFunction.backward`` and before ``optimizer.step()``);
  compressed gradients are written through the ordinary ``Parameter.grad``
  contract so any optimizer works unchanged;
* skipped — the paper's evaluation/benchmark harness.

Authoritative C++-owned gradient backends
(``_uses_authoritative_optimizer_grads``) publish optimizer gradients
directly from C++ buffers and are deliberately skipped here.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Gradients smaller than this are cheap to move whole; compressing them
# costs more than it saves.
DEFAULT_MIN_NUMEL = 1024


@dataclass
class GradientProjection:
    """Compressed representation of one parameter's gradient."""

    coeffs: torch.Tensor  # [rank] basis coefficients (float32)
    indices: torch.Tensor  # [k] flat indices of kept residual entries
    values: torch.Tensor  # [k] float32 values of kept residual entries
    shape: torch.Size
    grad_norm: float
    relative_error: float


class GradientProjector:
    """Learned low-rank projector with sparse residual for one parameter.

    The basis is updated *after* each compression, so every fresh gradient
    must first be represented by the previously learned subspace plus a
    sparse residual — the regime in which the learned projector earns its
    keep. Basis update orthonormalizes ``[basis | g]`` and truncates back to
    ``rank`` columns.
    """

    def __init__(self, shape, rank: int, density: float):
        self.shape = torch.Size(shape)
        self.rank = int(rank)
        self.density = float(density)
        self.numel = int(math.prod(self.shape))
        self.basis = torch.empty(self.numel, 0, dtype=torch.float32)

    def residual_k(self) -> int:
        return max(1, min(self.numel, int(self.density * self.numel)))

    def compress(self, grad: torch.Tensor) -> GradientProjection:
        """Compress ``grad`` against the current basis (basis unchanged)."""
        g = grad.detach().reshape(-1).to(torch.float32)
        grad_norm = float(g.norm())
        if grad_norm == 0.0:
            return GradientProjection(
                coeffs=self.basis.new_zeros(self.basis.shape[1]),
                indices=torch.empty(0, dtype=torch.long),
                values=torch.empty(0, dtype=torch.float32),
                shape=self.shape,
                grad_norm=0.0,
                relative_error=0.0,
            )

        if self.basis.shape[1] > 0:
            coeffs = self.basis.T @ g
            residual = g - self.basis @ coeffs
        else:
            coeffs = torch.zeros(0, dtype=torch.float32)
            residual = g

        residual_norm = float(residual.norm())
        k = self.residual_k()
        if residual.numel() > k:
            top_vals, top_idx = torch.topk(residual.abs(), k)
            indices = top_idx
            values = residual[top_idx]
            kept_sq = float((top_vals**2).sum())
            dropped = math.sqrt(max(residual_norm**2 - kept_sq, 0.0))
        else:
            indices = torch.arange(residual.numel(), dtype=torch.long)
            values = residual
            dropped = 0.0

        return GradientProjection(
            coeffs=coeffs.detach(),
            indices=indices.detach(),
            values=values.detach(),
            shape=self.shape,
            grad_norm=grad_norm,
            relative_error=dropped / grad_norm,
        )

    def update_basis(self, grad: torch.Tensor) -> None:
        """Fold ``grad`` into the projector basis (one power-iteration step)."""
        g = grad.detach().reshape(-1).to(torch.float32)
        if self.rank <= 0 or self.basis.shape[1] >= min(self.rank, self.numel):
            return
        stacked = torch.cat([self.basis, g.unsqueeze(1)], dim=1)
        q, _ = torch.linalg.qr(stacked)
        self.basis = q[:, : min(self.rank, q.shape[1])].contiguous()

    def reconstruct(self, projection: GradientProjection) -> torch.Tensor:
        """Materialize the approximate gradient from its projection (float32).

        ``projection`` was taken against a prefix of the current basis (the
        basis may have grown since), so project the coefficients onto the
        matching leading columns rather than requiring an exact width match.
        """
        out = torch.zeros(self.numel, dtype=torch.float32)
        n = min(self.basis.shape[1], projection.coeffs.numel())
        if n > 0:
            out += self.basis[:, :n] @ projection.coeffs[:n]
        if projection.values.numel() > 0:
            out[projection.indices] += projection.values
        return out.reshape(self.shape)


@dataclass
class KTGradientCompression:
    """Per-model state for low-rank + sparse gradient compression."""

    rank: int = 64
    density: float = 0.01  # fraction of gradient elements kept in the residual
    min_numel: int = DEFAULT_MIN_NUMEL
    projectors: dict = field(default_factory=dict)  # id(param) -> GradientProjector
    params: dict = field(default_factory=dict)  # id(param) -> nn.Parameter
    projections: dict = field(default_factory=dict)  # id(param) -> GradientProjection
    last_errors: dict = field(default_factory=dict)  # id(param) -> float

    def summary(self) -> dict:
        """Aggregate statistics about the most recent compression pass."""
        errors = list(self.last_errors.values())
        if not errors:
            return {"params": 0}
        return {
            "params": len(errors),
            "mean_relative_error": sum(errors) / len(errors),
            "max_relative_error": max(errors),
            "rank": self.rank,
            "density": self.density,
        }


def enable_kt_gradient_compression(
    model: nn.Module,
    rank: int = 64,
    density: float = 0.01,
    min_numel: int = DEFAULT_MIN_NUMEL,
) -> KTGradientCompression:
    """Attach gradient-compression state to ``model`` (opt-in).

    Must be called after the KT wrappers exist (i.e. after
    ``wrap_moe_layers_with_kt_wrapper``). Compression then runs inside
    ``sync_kt_lora_gradients`` — after backward, before the optimizer.
    """
    if rank <= 0:
        raise ValueError(f"rank must be positive, got {rank}")
    if not (0.0 < density <= 1.0):
        raise ValueError(f"density must be in (0, 1], got {density}")
    state = KTGradientCompression(rank=int(rank), density=float(density), min_numel=int(min_numel))
    setattr(model, "_kt_grad_compression", state)
    logger.info("KT gradient compression enabled: rank=%d density=%g min_numel=%d", rank, density, min_numel)
    return state


def disable_kt_gradient_compression(model: nn.Module) -> None:
    """Detach compression state; gradients pass through untouched again."""
    if hasattr(model, "_kt_grad_compression"):
        delattr(model, "_kt_grad_compression")


def compress_kt_gradients(model: nn.Module) -> KTGradientCompression | None:
    """Compress KT trainable gradients in place; no-op unless enabled.

    Called from ``sync_kt_lora_gradients`` (single- and multi-GPU). For each
    KT-managed trainable parameter the current ``.grad`` is projected onto
    the learned basis plus a sparse residual, the basis is updated from the
    gradient, and ``.grad`` is replaced by the reconstruction so the
    optimizer consumes the compressed representation unchanged.
    """
    state = getattr(model, "_kt_grad_compression", None)
    if state is None:
        return None

    from .lora import _find_kt_wrappers, get_kt_trainable_params  # imported here: avoids cycle

    wrappers = _find_kt_wrappers(model)
    if wrappers and any(getattr(w, "_uses_authoritative_optimizer_grads", False) for w in wrappers):
        # C++-authoritative backends publish optimizer gradients directly
        # from their own buffers; overwriting .grad here would desynchronize
        # that lifecycle. Compression is a CPU→optimizer-boundary feature.
        logger.debug("KT gradient compression skipped: authoritative C++ gradient backend active")
        return None

    for param in get_kt_trainable_params(model):
        grad = param.grad
        if grad is None or grad.numel() < state.min_numel:
            continue
        key = id(param)
        projector = state.projectors.get(key)
        if projector is None:
            projector = GradientProjector(grad.shape, rank=state.rank, density=state.density)
            state.projectors[key] = projector
            state.params[key] = param
        projection = projector.compress(grad)
        projector.update_basis(grad)
        state.projections[key] = projection
        state.last_errors[key] = projection.relative_error
        param.grad = projector.reconstruct(projection).to(dtype=grad.dtype, device=grad.device)
    return state


def restore_kt_gradients(model: nn.Module, accumulate: bool = False) -> None:
    """Re-materialize compressed gradients onto ``Parameter.grad``.

    The inverse of ``compress_kt_gradients``: useful when a consumer on the
    far side of an offload boundary needs the reconstructed dense gradient
    from the stored projections.
    """
    state = getattr(model, "_kt_grad_compression", None)
    if state is None:
        return
    for key, projection in state.projections.items():
        param = state.params.get(key)
        projector = state.projectors.get(key)
        if param is None or projector is None:
            continue
        recon = projector.reconstruct(projection).to(dtype=param.dtype, device=param.device)
        param.grad = recon if not accumulate or param.grad is None else param.grad + recon


def kt_gradient_compression_stats(model: nn.Module) -> dict | None:
    """Return aggregate stats for the last compression pass, if enabled."""
    state = getattr(model, "_kt_grad_compression", None)
    if state is None:
        return None
    return state.summary()
