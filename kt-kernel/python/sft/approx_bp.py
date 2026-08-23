# Approximate backpropagation for the LoRA-expert activation
# SPDX-License-Identifier: Apache-2.0

"""
Memory-cheap activation backward for LoRA-expert SFT.

Adapted from "Reducing Fine-Tuning Memory Overhead by Approximate and
Memory-Sharing Backpropagation" (Approx-BP, arXiv:2406.16282).

Approx-BP observes that the forward and backward passes of a
non-linearity can be *decoupled*: the gradient estimator stays unbiased
enough for fine-tuning as long as the substitute primitive stays close to
the original in function space. Applying that to SiLU gives ReSiLU2, which
keeps the exact SiLU primitive (so a fine-tune starts from the pretrained
model, bit-for-bit) but takes the derivative from a 4-segment step function
fit to SiLU. The step index of each element is one of four values, so the
backward needs 2 bits per element instead of the full activation input that
autograd's SiLU saves for ``x * sigmoid(x)``.

This module ports the activation half of the paper. The paper's
memory-sharing half (MS-LN / MS-RMSNorm, folding norm affine parameters
into the following linear layer) is intentionally out of scope: norms in
the SFT path are moved to the GPU by ``move_non_experts_to_gpu`` and are
not owned by the LoRA expert modules here.
"""

from __future__ import annotations

import torch

# Piecewise-linear primitive h~(x) = sum_i a_i * ReLU(x - c_i) + a_last * ReLU(x - c_last),
# fit to SiLU over the real line by simulated annealing in the paper
# (Eqn. 13-14). The a_last term keeps the constraint sum_i a_i * c_i = 0
# implicit, i.e. h~(0) = 0.
_SILU_A = (-0.04060, 1.081)
_SILU_C = (-6.305, -0.0008685, 6.326)

# Derivative of the primitive: a step function whose slope on segment s is
# sum(a_i for c_i below the segment). Four segments => 2 bits of activation
# memory per element for the backward.
_SILU_SLOPES = (
    0.0,
    _SILU_A[0],
    _SILU_A[0] + _SILU_A[1],
    1.0,
)

# Bits of activation memory the backward needs per element, versus a full
# copy of the activation input for autograd's SiLU.
BACKWARD_BITS_PER_ELEMENT = 2

# How many 2-bit segment indices pack into one uint8.
_SEGMENTS_PER_BYTE = 8 // BACKWARD_BITS_PER_ELEMENT

# Little-endian bit weights for packing four 2-bit fields into a byte.
_SEGMENT_BIT_WEIGHTS = (1, 4, 16, 64)


def _segment_index(x: torch.Tensor) -> torch.Tensor:
    """Map each element to one of the four derivative segments."""
    index = torch.zeros_like(x, dtype=torch.long)
    for bound in _SILU_C:
        index = index + (x > bound).to(torch.long)
    return index.reshape(-1)


def _pack_segments(index: torch.Tensor) -> torch.Tensor:
    """Pack 2-bit segment indices four to a byte, as the paper's kernel does."""
    padding = (-index.numel()) % _SEGMENTS_PER_BYTE
    if padding:
        index = torch.nn.functional.pad(index, (0, padding))
    groups = index.reshape(-1, _SEGMENTS_PER_BYTE)
    weights = torch.tensor(_SEGMENT_BIT_WEIGHTS, dtype=torch.long, device=groups.device)
    return (groups * weights).sum(dim=1).to(torch.uint8)


def _unpack_slopes(packed: torch.Tensor, numel: int, like: torch.Tensor) -> torch.Tensor:
    """Recover one slope per element from the packed 2-bit fields."""
    shifts = torch.arange(0, 8, BACKWARD_BITS_PER_ELEMENT, device=packed.device)
    segments = ((packed.to(torch.uint8).unsqueeze(1) >> shifts) & 0b11).long().reshape(-1)
    return like.new_tensor(_SILU_SLOPES)[segments[:numel]]


class ReSiLU2(torch.autograd.Function):
    """SiLU forward, step-function derivative backward (paper's ReSiLU2).

    The forward output is identical to ``torch.nn.functional.silu``, so
    enabling this changes no numerics of the pretrained path — only what the
    autograd graph keeps alive until backward (a packed 2-bit segment index
    rather than the activation input), and the surrogate derivative it
    applies there.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        ctx.save_for_backward(_pack_segments(_segment_index(x)))
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):  # type: ignore[override]
        (packed,) = ctx.saved_tensors
        flat = grad_output.reshape(-1)
        return (flat * _unpack_slopes(packed, flat.numel(), grad_output)).reshape(grad_output.shape)


def approx_silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU whose backward keeps a 2-bit segment index instead of ``x``."""
    return ReSiLU2.apply(x)


def backward_saved_bytes(numel: int, dtype: torch.dtype) -> int:
    """Bytes autograd saves for one activation, exact SiLU vs ReSiLU2.

    Exact SiLU (``x * sigmoid(x)``) saves the input tensor so it can
    recompute ``sigmoid(x)`` and ``1 + x * (1 - sigmoid(x))`` in backward;
    ReSiLU2 saves ``numel`` 2-bit segment indices instead.
    """
    if numel < 0:
        raise ValueError(f"activation element count must be non-negative, got {numel}")
    exact = numel * torch.tensor([], dtype=dtype).element_size()
    approx = (numel * BACKWARD_BITS_PER_ELEMENT + 7) // 8
    return exact - approx


def estimate_lora_expert_savings(
    num_experts: int,
    hidden_size: int,
    intermediate_size: int,
    num_tokens: int,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    """Bytes saved per MoE layer when LoRA experts run ReSiLU2 backwards.

    ``LoRAExpertMLP`` activates ``num_experts`` experts of
    ``[num_tokens, intermediate_size]`` elements each, and each activation
    input is what autograd's SiLU would otherwise retain until backward.
    """
    _ = hidden_size  # the activation footprint is set by the intermediate dim
    if min(num_experts, hidden_size, intermediate_size, num_tokens) < 0:
        raise ValueError("expert/tensor dimensions must be non-negative")
    return num_experts * backward_saved_bytes(num_tokens * intermediate_size, dtype)
