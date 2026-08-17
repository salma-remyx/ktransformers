# Riemannian preconditioning for KT LoRA gradients
# SPDX-License-Identifier: Apache-2.0

"""
Riemannian preconditioning of KT-managed LoRA gradients, applied in place on
the per-expert C++ gradient buffers just before the optimizer step.

Adapted from "Riemannian Preconditioned LoRA for Fine-Tuning Foundation
Models" (Zhang & Pilanci, arXiv:2402.02347).  The paper derives a Riemannian
metric on the low-rank matrix field whose descent direction replaces the flat
gradients of the two LoRA factors with

    dA <- (B^T B + lambda I)^-1  dA          # left-multiplied,  r x r
    dB <-  dB (A A^T + lambda I)^-1          # right-multiplied, r x r

so each step descends the same loss under a closed-form, self-scaling
preconditioner: the factor that has grown large is automatically damped.
Because it is a plain gradient transform, it composes with the repo's existing
AdamW/SGD optimizer rather than replacing it.

Target-native adaptation
------------------------
The paper ships the metric inside a bespoke fused optimizer that owns the LoRA
factors directly.  KT SFT instead keeps the factors in contiguous per-expert
buffers that the C++ MoE kernel fills during backward and the optimizer then
reads through views, so the same math is expressed as a pre-step transform
over those buffers:

*   ``gate``/``up``/``down`` projections are preconditioned independently —
    the paper gives each LoRA pair its own preconditioner.
*   Each expert owns an independent pair, matching the buffer layout
    ``[expert, r, in]`` / ``[expert, out, r]``; experts never mix.
*   Gram matrices are accumulated in float32 whatever the bf16 buffer dtype,
    and the damped inverse is obtained with :func:`torch.linalg.solve` rather
    than by forming an explicit inverse.
*   Rescaled gradients are written back into the C++ buffer so the optimizer
    — which may be reading that exact buffer as the parameter's grad view —
    observes the preconditioned direction.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Projection -> factor pair.  ``a`` contracts the module input (shape
# ``[r, in]``), ``b`` expands back out (shape ``[out, r]``).
_LORA_PROJECTIONS = ("gate", "up", "down")

_FUSED_PARAM_ORDER = tuple(f"{proj}_lora_{side}" for proj in _LORA_PROJECTIONS for side in ("a", "b"))

DEFAULT_REG = 1e-6


def _damped_inverse(gram: torch.Tensor, reg: float) -> torch.Tensor:
    """Return ``(gram + reg * I)^-1`` for one or a batch of ``r x r`` matrices."""
    r = gram.shape[-1]
    identity = torch.eye(r, dtype=gram.dtype, device=gram.device).expand_as(gram)
    # linalg.solve needs the RHS batch dims spelled out rather than broadcast.
    return torch.linalg.solve(gram + reg * identity, identity)


def _precondition_batched(
    grad_a: torch.Tensor,
    grad_b: torch.Tensor,
    weight_a: torch.Tensor,
    weight_b: torch.Tensor,
    reg: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precondition independent LoRA pairs stacked along dim 0.

    ``grad_a``/``weight_a`` are ``[N, r, in]`` and ``grad_b``/``weight_b`` are
    ``[N, out, r]``, where every index along ``N`` is a separate pair.  All
    arithmetic runs in float32; the inputs are not modified.
    """
    a32 = weight_a.detach().float()
    b32 = weight_b.detach().float()

    # dA <- (B^T B + reg I)^-1 dA ; dB <- dB (A A^T + reg I)^-1
    pre_a = _damped_inverse(torch.bmm(b32.transpose(1, 2), b32), reg)
    pre_b = _damped_inverse(torch.bmm(a32, a32.transpose(1, 2)), reg)

    return torch.bmm(pre_a, grad_a.detach().float()), torch.bmm(grad_b.detach().float(), pre_b)


def _write_back(param: torch.Tensor, updated: torch.Tensor) -> None:
    """Copy a preconditioned gradient into the parameter's existing grad view."""
    param.grad.copy_(updated.reshape_as(param.grad).to(param.grad.dtype))


def _apply_to_fused(wrapper, reg: float) -> int:
    """Precondition KT-managed fused-expert LoRA buffers."""
    fused = getattr(wrapper, "_fused_expert_lora_params", None)
    if not fused:
        return 0

    by_name = dict(zip(_FUSED_PARAM_ORDER, fused))
    conditioned = 0
    for proj in _LORA_PROJECTIONS:
        param_a = by_name.get(f"{proj}_lora_a")
        param_b = by_name.get(f"{proj}_lora_b")
        if param_a is None or param_b is None:
            continue
        if param_a.grad is None or param_b.grad is None:
            # Closed optimizer window or a rank that owns no parameters.
            continue
        new_a, new_b = _precondition_batched(
            param_a.grad, param_b.grad, param_a.detach(), param_b.detach(), reg
        )
        _write_back(param_a, new_a)
        _write_back(param_b, new_b)
        conditioned += 1
    return conditioned


def _apply_to_peft(wrapper, reg: float) -> int:
    """Precondition PEFT-injected expert LoRA views."""
    peft_modules = getattr(wrapper, "_peft_lora_modules", None)
    if not peft_modules:
        return 0

    # Rebatch per projection so each expert is one entry along dim 0.
    by_proj: dict[str, list] = {proj: [] for proj in _LORA_PROJECTIONS}
    for expert_idx in sorted(peft_modules):
        for proj_name, (lora_a, lora_b) in peft_modules[expert_idx].items():
            # arch weight_names carry the projection suffix, e.g. "gate_proj".
            proj = proj_name.rsplit("_proj", 1)[0]
            if proj not in by_proj or lora_a.weight.grad is None or lora_b.weight.grad is None:
                continue
            by_proj[proj].append((lora_a, lora_b))

    conditioned = 0
    for proj, pairs in by_proj.items():
        if not pairs:
            continue
        new_a, new_b = _precondition_batched(
            torch.stack([p.weight.grad for p, _ in pairs]),
            torch.stack([q.weight.grad for _, q in pairs]),
            torch.stack([p.weight.detach() for p, _ in pairs]),
            torch.stack([q.weight.detach() for _, q in pairs]),
            reg,
        )
        for idx, (lora_a, lora_b) in enumerate(pairs):
            _write_back(lora_a.weight, new_a[idx])
            _write_back(lora_b.weight, new_b[idx])
        conditioned += len(pairs)
    return conditioned


def apply_riemannian_lora_precondition(model, reg: float = DEFAULT_REG) -> int:
    """Precondition every KT LoRA gradient in ``model`` before the step.

    Walks the model's KT wrappers and rewrites, in place, the gradients of all
    C++-managed expert LoRA factors using the Riemannian preconditioner of
    arXiv:2402.02347.  Parameters without a gradient are left untouched, and
    ordinary (non-KT) parameters are never visited.

    Returns the number of ``(expert, projection)`` pairs preconditioned; ``0``
    means the model has no KT LoRA gradients to act on.
    """
    if not reg > 0.0:
        raise ValueError(f"Riemannian preconditioner regularization must be positive, got {reg}")

    wrappers = getattr(model, "_kt_wrappers", None)
    if not wrappers:
        return 0

    conditioned = 0
    for wrapper in wrappers:
        conditioned += _apply_to_fused(wrapper, reg)
        conditioned += _apply_to_peft(wrapper, reg)

    if conditioned:
        logger.debug(
            "Riemannian-preconditioned %d KT LoRA gradient pairs (reg=%g)", conditioned, reg
        )
    return conditioned
