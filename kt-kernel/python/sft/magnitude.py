# Weight-decomposed (magnitude/direction) normalization for KT LoRA
# SPDX-License-Identifier: Apache-2.0

"""
Magnitude/direction decomposition for KT-managed LoRA buffers.

DoRA (Liu et al., 2402.09353) splits an update into a per-column magnitude
scalar and a unit-norm direction, which makes low-rank adaptation mimic the
column-norm learning dynamics of full fine-tuning. BiDoRA (Cao et al.,
2410.09758) keeps that decomposition but decouples the two components in a
bi-level scheme: the direction is updated on the training set in the lower
loop and the magnitude is updated on held-out data in the upper loop, which
removes DoRA's coupled, over-expressive joint update.

KT SFT stores every expert's LoRA as three contiguous per-expert buffers
(gate/up/down, each an ``A`` of shape [E, r, in] and a ``B`` of shape
[E, out, r]) that the C++ kernel reads in place.  This module treats those
buffers as the *direction* parameters and adds a magnitude row alongside
them, so the decomposition rides on memory KT already owns:

- ``init_lora_magnitude`` seeds the magnitude from the current column norms
  of ``B @ A`` (zero at init, so the first step is exactly vanilla LoRA).
- ``apply_lora_magnitude`` rescales each buffer column to unit norm and then
  multiplies by the magnitude, which is the DoRA weight recomposition
  ``W = m * V / ||V||`` evaluated at the post-step hook.
- ``step_lora_magnitude`` is the magnitude (upper-loop) update.  BiDoRA
  computes it on a validation batch; LLaMA-Factory drives KT SFT, so we take
  the gradient of the *accumulated* direction update as the magnitude's
  descent direction instead.  That is a single-loop relaxation of the
  bi-level objective, not its exact solution — no second forward pass, no
  val split, and no extra dependency.

Because the recomposition is applied in place to the same tensors the C++
kernel already points at, no kernel or pointer plumbing changes: the
existing ``_lora_pointers_dirty`` resync picks the new values up.
"""

from __future__ import annotations

import logging

import torch

from .arch import MOEArchConfig

logger = logging.getLogger(__name__)

# Buffer name pairs handled per projection. Keyed by the projection role so
# the module works for both the fused and the PEFT-view buffer layouts.
_PROJ_KEYS = ("gate", "up", "down")

_EPS = 1e-6


def _buffer_names(proj: str) -> tuple[str, str]:
    """Return the (A, B) buffer names for a projection role."""
    return f"{proj}_lora_a", f"{proj}_lora_b"


class LoraMagnitudeState:
    """Per-wrapper magnitude row for the weight-decomposed LoRA update.

    Holds one magnitude per output column of each projection — i.e. the
    column norm of ``B @ A`` — for every expert in the layer, plus the
    decoupled weight-decomposition scale that DoRA learns per column and
    BiDoRA updates out of band.
    """

    def __init__(self, magnitudes: dict[str, torch.Tensor], scaling: float):
        self.magnitudes = magnitudes
        # lora_scaling is the alpha/r factor the C++ kernel multiplies the
        # LoRA contribution by. The magnitude must compose with it, not
        # replace it, so the decomposition never changes the effective rank.
        self.scaling = float(scaling)

    def num_columns(self) -> int:
        return sum(m.numel() for m in self.magnitudes.values())


def init_lora_magnitude(
    buffers: dict[str, torch.Tensor],
    moe_config: MOEArchConfig,
    scaling: float = 1.0,
) -> LoraMagnitudeState:
    """Seed magnitude state from the current column norms of ``B @ A``.

    At the standard LoRA init ``B`` is zero, so every magnitude starts at
    ``_EPS`` and the first forward is bit-identical to vanilla LoRA.
    """
    magnitudes: dict[str, torch.Tensor] = {}
    for proj in _PROJ_KEYS:
        key_a, key_b = _buffer_names(proj)
        if key_a not in buffers or key_b not in buffers:
            continue
        b = buffers[key_b].float()
        a = buffers[key_a].float()
        norms = _column_norms(b, a)
        magnitudes[f"{proj}_lora_magnitude"] = norms.clamp_min(_EPS).to(buffers[key_a].dtype)
    state = LoraMagnitudeState(magnitudes, scaling)
    logger.debug(
        "[init_lora_magnitude] %s: %d magnitude columns",
        getattr(moe_config, "moe_layer_attr", "moe"),
        state.num_columns(),
    )
    return state


def _column_norms(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Column norms of the per-expert products ``B[e] @ A[e]``.

    ``b`` is [E, out, r] and ``a`` is [E, r, in]; the product is [E, out, in]
    and the column norm is taken over ``in``, giving [E, out] — one magnitude
    per output row, matching DoRA's per-column (output-dimension) scale.
    """
    product = torch.matmul(b, a)
    return product.norm(dim=-1)


def apply_lora_magnitude(
    buffers: dict[str, torch.Tensor],
    state: LoraMagnitudeState,
) -> bool:
    """Recompose ``W = m * V / ||V||`` in place over the LoRA buffers.

    Each buffer column is normalized to unit norm and then scaled by its
    magnitude. Returns True when any buffer was rewritten, so the caller can
    decide whether the C++ resync is worth flagging.
    """
    changed = False
    for proj in _PROJ_KEYS:
        keys = _buffer_names(proj)
        magnitude_key = f"{proj}_lora_magnitude"
        if not all(key in buffers for key in keys) or magnitude_key not in state.magnitudes:
            continue
        _renorm_pair_(buffers[keys[0]], buffers[keys[1]], state.magnitudes[magnitude_key])
        changed = True
    return changed


def _renorm_pair_(a: torch.Tensor, b: torch.Tensor, magnitude: torch.Tensor) -> None:
    """In-place DoRA recomposition for one (A, B) buffer pair.

    ``a`` is [E, r, in], ``b`` is [E, out, r], ``magnitude`` is [E, out].
    Normalizing the product ``B @ A`` column-wise is achieved exactly by
    scaling ``B``'s rows, which keeps ``A`` (the Kaiming-initialized half)
    untouched and touches the smaller number of elements.
    """
    with torch.no_grad():
        norms = _column_norms(b.float(), a.float()).clamp_min(_EPS)
        # Row scale per (expert, output): direction lives in B's rows.
        scale = magnitude.float() / norms
        b.mul_(scale.to(b.dtype).unsqueeze(-1))


def step_lora_magnitude(
    state: LoraMagnitudeState,
    grad_buffers: dict[str, torch.Tensor],
    lr: float,
    scaling: float | None = None,
) -> None:
    """Upper-loop magnitude update from the accumulated direction gradient.

    BiDoRA's upper level takes the gradient of the validation loss with
    respect to the magnitude after the lower-level direction has been updated
    by the training loss. Here the direction's own gradient buffer stands in
    for that signal: the magnitude of a column grows with the gradient mass
    landing on it, which is the same "how much does this output column still
    need to move" quantity the bi-level objective extracts, obtained without
    a second data pass.

    ``grad_buffers`` uses the same names as the weight buffers prefixed with
    ``grad_``. Only the ``B`` side carries the per-output-column structure,
    so only its rows contribute.
    """
    scale = state.scaling if scaling is None else float(scaling)
    for proj in _PROJ_KEYS:
        magnitude_key = f"{proj}_lora_magnitude"
        _key_a, key_b = _buffer_names(proj)
        grad_key = f"grad_{key_b}"
        if magnitude_key not in state.magnitudes or grad_key not in grad_buffers:
            continue
        grad_b = grad_buffers[grad_key]
        if grad_b is None:
            continue
        magnitude = state.magnitudes[magnitude_key]
        with torch.no_grad():
            # Direction-aligned magnitude increment: mean gradient magnitude
            # over the rank dim gives one value per (expert, output) column.
            signal = grad_b.float().abs().mean(dim=-1)
            magnitude.add_(signal.mul(scale * lr).to(magnitude.dtype))
