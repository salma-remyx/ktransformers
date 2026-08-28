"""Fixed-grid discrete refinement for AMX weight quantization.

Adapted from "ReQuant: Fixed-Grid Discrete Refinement for Post-Training
Quantization" (arXiv:2608.07019). ReQuant's core insight: the integer
assignments a PTQ initializer produces need not be final — iteratively
revisiting them on the same discrete grid, accepting only updates that
strictly reduce reconstruction error, monotonically improves the quantized
model at zero deployment cost.

This module applies that loop to the per-row symmetric grids used by the
AMXINT4 / AMXINT8 weight quantizers in kt-kernel (see
``operators/amx/la/amx_buffers.hpp::BufferBInt4Impl`` and
``amx_kernels.hpp::GemmKernel224Int8::BufferB``): row step ``s = amax / qmax``
with ``qmax = 7`` (int4) or ``127`` (int8), round-to-nearest assignments,
dequant ``s * q``.

Two target-native substitutions relative to the paper, made necessary by the
fact that this converter is a weight-only offline tool:

* Objective: the paper refines against layer-output reconstruction error,
  which requires calibration activations this pipeline does not have. We
  refine against per-row weight MSE instead.
* Grid: with a weight-MSE objective and a frozen scale, the initial
  round-to-nearest solution is already optimal, so the scale must move for
  refinement to do anything. Each sweep therefore re-fits the row scale by
  closed-form least squares (``s = <w, q> / <q, q>``) while the *discrete*
  grid — the integer levels ``{-qmax, ..., qmax}`` — stays fixed exactly as
  in the paper. Both the assignment update and the scale update are
  MSE-monotone, so the loop strictly reduces weight MSE per accepted sweep.

Pass-through property (why this needs no C++ changes): the refined solution
``(q*, s*)`` is emitted as a plain BF16 tensor ``W' = s* * q*`` such that
re-running the existing AMX quantizer on ``W'`` recovers ``(q*, s*)``
element-for-element:

* The row scale is snapped to ``bf16(s* * qmax) / qmax`` during refinement,
  so the boundary value ``s* * qmax`` is exactly representable in BF16. The
  AMX quantizer recomputes its step from ``amax(W')`` and therefore recovers
  exactly ``s*``.
* One element per row is pinned to the grid boundary in assignment space
  (the cheapest such element, i.e. the one already closest to it), so
  ``amax(W') = s* * qmax`` holds even when refinement moved every
  assignment off the boundary.
* Every other emitted value lies within BF16 rounding of an exact grid
  point, far inside the half-step decision margin, so the quantizer's
  round-to-nearest pass reproduces ``q*``.

Feeding ``W'`` to ``KTMoEWrapper`` therefore materializes the refined
assignments in the saved ``.kt`` tensors.
"""

from typing import Dict

import torch

# Max magnitude of the integer grid per AMX quantization method. Both AMXINT4
# and MOE_INT4 pack to the same 4-bit per-row grid; likewise for int8.
METHOD_QMAX: Dict[str, int] = {
    "int4": 7,
    "moe_int4": 7,
    "int8": 127,
    "moe_int8": 127,
}


def _rtn_assign(w: torch.Tensor, s: torch.Tensor, qmax: int) -> torch.Tensor:
    """Round-to-nearest assignment on the per-row grid with step ``s``."""
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    return torch.clamp(torch.round(w * inv), -qmax, qmax)


def refine_weight_grid_(
    weight: torch.Tensor,
    qmax: int,
    num_sweeps: int = 2,
    row_chunk: int = 65536,
) -> Dict[str, float]:
    """Refine AMX grid assignments of a weight tensor in place.

    Args:
        weight: ``[..., K]`` BF16/FP16/FP32 weight tensor; one quantization
            row per last-dim slice of length ``K`` (matches the AMX BufferB
            per-row layout for gate/up/down projections). Modified in place.
        qmax: Grid boundary (7 for AMXINT4, 127 for AMXINT8).
        num_sweeps: Maximum refinement sweeps; each accepted sweep strictly
            reduces per-row weight MSE. 0 leaves the tensor untouched.
        row_chunk: Rows refined per chunk, bounding the FP32 working set.

    Returns:
        Stats dict: ``initial_mse`` (plain RTN), ``final_mse`` (after
        refinement, i.e. what the AMX-quantized output will achieve),
        ``mse_history`` per sweep, and ``changed`` (fraction of rows whose
        solution moved).
    """
    if weight.shape[-1] == 0:
        raise ValueError("weight's last dim (K) must be non-zero")
    if qmax <= 0:
        raise ValueError(f"qmax must be positive, got {qmax}")
    if num_sweeps <= 0:
        return {"initial_mse": float("nan"), "final_mse": float("nan"), "mse_history": [], "changed": 0.0}

    orig = weight
    if not weight.is_contiguous():
        weight = weight.contiguous()
    rows = weight.reshape(-1, weight.shape[-1])
    orig_dtype = weight.dtype
    total_initial = 0.0
    total_final = 0.0
    total_changed = 0
    sweep_sums: Dict[int, float] = {}

    for start in range(0, rows.shape[0], row_chunk):
        chunk = rows[start : start + row_chunk]
        w = chunk.float()

        # RTN initialization: the assignment the AMX quantizer would produce.
        s = w.abs().amax(dim=1, keepdim=True) / qmax
        q = _rtn_assign(w, s, qmax)
        mse = ((w - s * q) ** 2).sum(dim=1)
        initial = mse.clone()

        for sweep in range(num_sweeps):
            # Scale re-fit: closed-form least squares given fixed assignments,
            # snapped so s * qmax is exactly representable in the storage
            # dtype (the AMX quantizer recovers its step from the BF16 max).
            denom = (q * q).sum(dim=1, keepdim=True)
            s_new = torch.where(denom > 0, (w * q).sum(dim=1, keepdim=True) / denom, s)
            s_new = (s_new * qmax).to(orig_dtype).float() / qmax
            # Assignment revisit on the fixed integer grid given the new scale.
            q_new = _rtn_assign(w, s_new, qmax)
            mse_new = ((w - s_new * q_new) ** 2).sum(dim=1)
            # Accept only strictly-improving rows (ReQuant's monotone rule).
            better = mse_new < mse - 1e-12 * (mse + 1.0)
            if not bool(better.any()):
                break
            s = torch.where(better.unsqueeze(1), s_new, s)
            q = torch.where(better.unsqueeze(1), q_new, q)
            mse = torch.where(better, mse_new, mse)
            sweep_sums[sweep] = sweep_sums.get(sweep, 0.0) + float(mse.sum())

        # Boundary pin: keep one element per row at +-qmax so the AMX
        # quantizer recovers step s from amax(W'). Cheapest element = the
        # one already closest to the boundary.
        pin_col = q.abs().argmax(dim=1, keepdim=True)
        q_pin = torch.gather(q, 1, pin_col)
        need_pin = q_pin.abs() < qmax
        q.scatter_(1, pin_col, torch.where(need_pin, q_pin.sign() * qmax, q_pin))

        # Emission: W' = s * q in the storage dtype. The final MSE is the
        # error the AMX-quantized output will actually achieve, since the
        # quantizer recovers (s, q) from W' exactly (see module docstring).
        out = (s * q).to(orig_dtype)
        final = ((w - s * q) ** 2).sum(dim=1)

        total_initial += float(initial.sum())
        total_final += float(final.sum())
        total_changed += int((final < initial - 1e-12 * (initial + 1.0)).sum())
        chunk.copy_(out)

    if weight is not orig:
        orig.copy_(weight)

    n = max(rows.shape[0], 1)
    history = [sweep_sums[i] / n for i in sorted(sweep_sums)]
    return {
        "initial_mse": total_initial / n,
        "final_mse": total_final / n,
        "mse_history": history,
        "changed": total_changed / n,
    }
