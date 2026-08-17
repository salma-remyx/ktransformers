#!/usr/bin/env python
# coding=utf-8
"""Reconstruction-guided rounding for calibration-free INT4/INT8 expert quantization.

Adapted from "ReRound: Reconstructive Rounding to Resolve Midpoint Ambiguity in
Calibration-Free LLM Quantization" (arXiv:2608.11045).

Round-to-nearest is ambiguous for weights that fall on or near the midpoint
between two quantization levels: either rounding direction is locally equally
wrong, so the choice is arbitrary and the arbitrary choices accumulate.
ReRound resolves the ambiguity with three pieces:

1. A *tolerance region*: only weights within ``tolerance`` of a midpoint are
   re-decided; everything else keeps its RTN value.
2. A *reconstruction guidance signal* that picks the rounding direction for
   ambiguous weights.
3. A *candidate sweep + selection*: sweeping the tolerance produces several
   candidate quantized matrices, and the one whose de-quantized leading
   singular values best match the original's is kept. RTN (``tolerance=0``)
   is always in the sweep, so selection can never do worse than RTN on the
   selection metric.

The paper trains a conditional diffusion model to produce the guidance
signal. That estimator is substituted here with a parameter-free
error-feedback proxy: an ambiguous weight is rounded in the direction that
cancels the accumulated signed rounding error of the row's already-decided
weights. This is the same signal diffusion reconstructions provide -- a
context-informed second opinion for ties only -- at zero training cost.

Runs entirely offline on the converter's already-materialized BF16 expert
tensors, so it adds no inference-time overhead.
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional

import torch

# Effective symmetric grid used by the AMX online quantizer (BufferBInt4Impl):
# per-row scale amax/7 with signed levels [-8, 7]. 112 = 7 * 16 is the packed
# form of the same grid (nibble * 16 * amax/112 == nibble * amax/7).
_DEFAULT_TOLERANCES = (0.0, 0.02, 0.05, 0.1, 0.15, 0.25)


def _row_scales(weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Per-row symmetric scales ``amax / qmax`` for a [..., rows, cols] tensor."""
    qmax = float(2 ** (num_bits - 1) - 1)
    amax = weight.abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    return amax / qmax


@dataclass
class RoundingReport:
    """Summary of one guided-rounding pass."""

    num_bits: int
    tolerances: List[float]
    chosen_tolerance: float
    ambiguous_fraction: float
    changed_fraction: float
    relative_gap: float
    num_matrices: int


@torch.no_grad()
def leading_singular_values(matrix: torch.Tensor, q_top: int = 16, n_iter: int = 4, seed: int = 0) -> torch.Tensor:
    """Deterministic estimate of the ``q_top`` largest singular values.

    ``torch.svd_lowrank`` draws a random probe matrix with no way to seed it,
    which makes the ReRound selection metric nondeterministic: the tolerance
    chosen in the selection pass would not necessarily still win when
    re-measured. Block power iteration with a fixed-seed start gives the same
    answer every call, at roughly the accuracy svd_lowrank itself provides.

    Args:
        matrix: a single [rows, cols] matrix
        q_top: number of leading singular values
        n_iter: power-iteration steps
        seed: probe seed; identical for every candidate so comparisons are fair
    """
    q = min(q_top, min(matrix.shape[-2:]))
    gen = torch.Generator(device=matrix.device).manual_seed(seed)
    probe = torch.randn(matrix.shape[-1], q, generator=gen, device=matrix.device, dtype=matrix.dtype)
    orthogonalized, _ = torch.linalg.qr(probe)
    for _ in range(n_iter):
        left, _ = torch.linalg.qr(matrix @ orthogonalized)
        orthogonalized, _ = torch.linalg.qr(matrix.T @ left)
    return torch.linalg.svdvals(matrix @ orthogonalized)


@torch.no_grad()
def singular_value_gap(original: torch.Tensor, dequantized: torch.Tensor, q_top: int = 16) -> torch.Tensor:
    """L1 distance between leading singular values, per leading matrix dim.

    Args:
        original: full-precision weights [..., rows, cols]
        dequantized: candidate weights, same shape
        q_top: number of leading singular values to compare

    Returns:
        Tensor of shape original.shape[:-2], the selection metric per matrix.
    """
    flat_o = original.reshape(-1, *original.shape[-2:]).float()
    flat_d = dequantized.reshape(-1, *dequantized.shape[-2:]).float()
    q = min(q_top, min(flat_o.shape[-2:]))
    if q < 1:
        return torch.zeros(original.shape[:-2])
    gaps = [
        (leading_singular_values(flat_o[i], q) - leading_singular_values(flat_d[i], q)).abs().sum()
        for i in range(flat_o.shape[0])
    ]
    return torch.stack(gaps).reshape(original.shape[:-2])


def _candidate(weight: torch.Tensor, scale: torch.Tensor, tolerance: float, num_bits: int) -> torch.Tensor:
    """One tolerance's candidate quantized matrix (integer levels, same shape)."""
    qmax = 2 ** (num_bits - 1) - 1
    qmin = -(2 ** (num_bits - 1))
    levels = weight / scale
    rtn = torch.round(levels).clamp(qmin, qmax)
    if tolerance <= 0.0:
        return rtn.to(torch.int8)

    lower = torch.floor(levels)
    # Distance of the *weight* from the interval midpoint, in level units.
    distance_to_midpoint = torch.abs(levels - lower - 0.5)
    ambiguous = distance_to_midpoint < tolerance
    if not bool(ambiguous.any()):
        return rtn.to(torch.int8)

    # Error feedback: defer ambiguous weights, carry the signed rounding error
    # of the already-decided ones, and break each tie toward cancelling it.
    decided_error = torch.where(ambiguous, torch.zeros_like(levels), rtn - levels)
    prefix = torch.cumsum(decided_error, dim=-1)
    carried = torch.cat([torch.zeros_like(prefix[..., :1]), prefix[..., :-1]], dim=-1)
    toward_higher = (carried + (lower + 1 - levels)).abs() < (carried + (lower - levels)).abs()
    guided = torch.where(ambiguous, torch.where(toward_higher, lower + 1, lower), rtn)
    return guided.clamp(qmin, qmax).to(torch.int8)


@torch.no_grad()
def guided_round(
    weight: torch.Tensor,
    num_bits: int = 4,
    tolerances: Optional[List[float]] = None,
    q_top: int = 16,
    max_matrix_dim: int = 8192,
) -> tuple[torch.Tensor, RoundingReport]:
    """Quantize ``weight`` with tolerance-swept, reconstruction-guided rounding.

    Args:
        weight: full-precision expert weights [..., rows, cols]
        num_bits: target weight bit-width (4 or 8)
        tolerances: midpoint tolerance sweep; plain RTN (0.0) is added if the
            caller omitted it, so selection is never worse than RTN
        q_top: leading singular values used by the selection metric
        max_matrix_dim: rows/cols cap above which matrices are chunked to bound
            the power-iteration working set

    Returns:
        (dequantized candidate weights in the input dtype, report)
    """
    if num_bits not in (4, 8):
        raise ValueError(f"num_bits must be 4 or 8, got {num_bits}")
    if tolerances is None:
        tolerances = list(_DEFAULT_TOLERANCES)
    # Keep RTN in the sweep so selection is never worse than RTN.
    tolerances = sorted(set(tolerances) | {0.0})
    widest = max(tolerances)

    weight = weight.contiguous()
    in_dtype = weight.dtype
    work = weight.float()
    rows, cols = work.shape[-2], work.shape[-1]
    scale = _row_scales(work, num_bits)
    levels = work / scale
    ambiguous_fraction = float((((levels - torch.floor(levels)) - 0.5).abs() < widest).float().mean())
    changed_fraction = 0.0

    if rows < 2 or cols < 2:
        # A vector has no singular-value spectrum to match; RTN is the answer.
        rtn = _candidate(work, scale, 0.0, num_bits)
        report = RoundingReport(
            num_bits,
            list(tolerances),
            0.0,
            ambiguous_fraction,
            0.0,
            0.0,
            int(torch.numel(work) / max(rows * cols, 1)),
        )
        return (rtn.float() * scale).to(in_dtype), report

    # Chunk oversized matrices: power-iteration memory scales with rows*cols*q.
    row_step = min(rows, max_matrix_dim)
    col_step = min(cols, max_matrix_dim)

    # One tolerance decision per leading-dim index (per expert), accumulated
    # across the row/col blocks of that expert. RTN seeds the sweep at index 0,
    # so a candidate only wins by strictly beating it on the summed metric.
    gap_totals = torch.zeros((len(tolerances),) + work.shape[:-2])

    for row_begin in range(0, rows, row_step):
        for col_begin in range(0, cols, col_step):
            block = work[..., row_begin : row_begin + row_step, col_begin : col_begin + col_step]
            block_scale = _row_scales(block, num_bits)
            for tol_idx, tol in enumerate(tolerances):
                q = _candidate(block, block_scale, tol, num_bits)
                gap_totals[tol_idx] += singular_value_gap(block, q.float() * block_scale, q_top)

    winner = gap_totals.argmin(dim=0)
    out = torch.empty_like(work)
    for tol_idx, tol in enumerate(tolerances):
        mask = winner == tol_idx
        if not bool(mask.any()):
            continue
        picked = mask.nonzero()
        lead = tuple(picked[:, i] for i in range(picked.shape[1]))
        sub = work[lead]
        sub_scale = _row_scales(sub, num_bits)
        sub_q = _candidate(sub, sub_scale, tol, num_bits)
        sub_rtn = _candidate(sub, sub_scale, 0.0, num_bits)
        changed_fraction = max(changed_fraction, float((sub_q != sub_rtn).float().mean()))
        out[lead] = sub_q.float() * sub_scale

    report = RoundingReport(
        num_bits=num_bits,
        tolerances=list(tolerances),
        chosen_tolerance=tolerances[int(winner.mode().values.item())] if winner.numel() else 0.0,
        ambiguous_fraction=ambiguous_fraction,
        changed_fraction=changed_fraction,
        relative_gap=float(
            (gap_totals[winner, ...].sum() / work.abs().sum().clamp(min=1e-12)).item()
        ),
        num_matrices=int(torch.numel(work) / (rows * cols)),
    )
    return out.to(in_dtype), report


def apply_to_experts(
    gate_proj: torch.Tensor,
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    num_bits: int = 4,
    tolerances: Optional[List[float]] = None,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply guided rounding to one layer's stacked expert projections.

    Takes and returns the exact tensor layout ``convert_cpu_weights.py``
    materializes for ``KTMoEWrapper.load_weights_from_tensors``:
    [num_experts, rows, cols] BF16. Quantization parameters (per-row amax/7
    scales) are unchanged, so the AMX online-quantization path sees the same
    grid as before -- only the rounding direction of midpoint-ambiguous
    weights can differ.
    """
    rounded = []
    for name, tensor in (("gate", gate_proj), ("up", up_proj), ("down", down_proj)):
        out, report = guided_round(tensor, num_bits=num_bits, tolerances=tolerances)
        rounded.append(out)
        if verbose:
            print(
                f"  [midpoint:{name}] tol={report.chosen_tolerance:.2f} "
                f"ambiguous={report.ambiguous_fraction:.4f} changed={report.changed_fraction:.4f} "
                f"rel_gap={report.relative_gap:.6f}"
            )
    return rounded[0], rounded[1], rounded[2]


def parse_tolerance_spec(spec: Optional[str]) -> Optional[List[float]]:
    """Parse a comma-separated tolerance sweep like ``0.0,0.05,0.1``.

    Plain RTN (0.0) is always included so the selection sweep retains its
    no-worse-than-RTN guarantee even if the user omits it.
    """
    if spec is None:
        return None
    values = [float(v) for v in spec.split(",") if v.strip()]
    if not values:
        raise ValueError(f"Empty tolerance sweep: {spec!r}")
    bad = [v for v in values if v < 0.0 or v > 0.5]
    if bad:
        raise ValueError(f"Tolerances must be in [0, 0.5], got {bad}")
    return sorted(set(values) | {0.0})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bits", type=int, default=4, choices=(4, 8), help="Target weight bit-width")
    parser.add_argument("--tolerances", type=str, default=None, help="Comma-separated midpoint tolerance sweep")
    parser.add_argument("--experts", type=int, default=4, help="Synthetic experts for the self-check")
    parser.add_argument("--rows", type=int, default=256, help="Rows per expert matrix")
    parser.add_argument("--cols", type=int, default=256, help="Cols per expert matrix")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    gen = torch.Generator().manual_seed(args.seed)
    weight = (torch.randn(args.experts, args.rows, args.cols, generator=gen) / 20).to(torch.bfloat16)
    out, report = guided_round(weight, num_bits=args.bits, tolerances=parse_tolerance_spec(args.tolerances))
    scale = _row_scales(weight.float(), args.bits)
    rtn = torch.round(weight.float() / scale).clamp(-(2 ** (args.bits - 1)), 2 ** (args.bits - 1) - 1) * scale
    gap_rtn = singular_value_gap(weight.float(), rtn.to(weight.dtype))
    gap_out = singular_value_gap(weight.float(), out)
    print(f"matrices={report.num_matrices} bits={report.num_bits} chosen_tol={report.chosen_tolerance:.2f}")
    print(f"ambiguous={report.ambiguous_fraction:.4f} changed={report.changed_fraction:.4f}")
    print(f"singular-value gap: rtn={gap_rtn.sum():.4f} guided={gap_out.sum():.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
