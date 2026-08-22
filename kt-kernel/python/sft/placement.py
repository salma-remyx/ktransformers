# Memory-capped expert placement advisor
# SPDX-License-Identifier: Apache-2.0

"""
Solve ``kt_num_gpu_experts`` from device memory instead of asking the user.

The device-map builders in :mod:`kt_kernel.sft.wrapper` split each MoE layer's
routed experts between GPU residents and CPU residents using a single global
knob. Leaving that knob at ``0`` wastes the free VRAM the trainer could have
used for expert weights; picking it by hand is guesswork that has to be redone
per model and per GPU.

This module treats the choice as a small integer allocation problem under a
memory cap: with a uniform per-layer budget, all layers can host the same
number of GPU experts, so the whole placement collapses to one integer program
with a closed-form solution -- fill the largest feasible uniform level. That is
the same "solve the placement first, then execute the solved policy" framing as
LazyTrain (arXiv:2608.11919), reduced to the placement knob this repo actually
exposes. LazyTrain's mixed-integer schedule over checkpointing, activation
placement, recomputation and communication overlap lives in a layer-streaming
CPU-master trainer that kt-kernel does not have; the parts that survive the
translation are the memory cap, the integer feasibility constraint, and the
remainder rule.

Deliberately parameter-free: no solver dependency, no calibration run. The
overhead term is a safety fraction rather than a learned estimator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch

from .arch import MOEArchConfig

logger = logging.getLogger(__name__)

# Fraction of the budget held back for activations, gradients, optimizer state
# and fragmentation. Safety margin rather than a modelled cost: a full
# activation/optimizer memory model is part of the scheduling layer this port
# intentionally does not bring over.
DEFAULT_RESERVE_FRACTION = 0.15

_BYTES_PER_PARAM = {
    "AMXBF16": 2,
    "AMXBF16_SFT": 2,
    "AMXINT8": 1,
    "AMXINT8_SFT": 1,
    "FP8": 1,
    "RAWINT4": 0.5,
}


def _bytes_per_expert_param(method: str | None) -> float:
    """Bytes per expert weight element for a KT backend method.

    Unknown methods fall back to 2 bytes (BF16) so the advisor stays useful on
    backends it has not been taught about without silently over-filling VRAM.
    """
    if method is None:
        return 2.0
    return _BYTES_PER_PARAM.get(str(method).upper(), 2.0)


@dataclass
class PlacementAdvice:
    """Result of advising a GPU-expert placement under a memory cap."""

    num_gpu_experts_per_layer: int
    total_gpu_experts: int
    total_experts: int
    usable_budget_bytes: int
    expert_cost_bytes: int
    total_expert_bytes: int
    budget_fully_used: bool

    @property
    def placement_ratio(self) -> float:
        """Fraction of all routed experts resident on GPU."""
        if self.total_experts <= 0:
            return 0.0
        return self.total_gpu_experts / self.total_experts


def estimate_expert_bytes(
    moe_config: MOEArchConfig,
    hidden_size: int,
    num_layers: int,
    method: str | None = None,
) -> int:
    """Bytes for every routed expert of a model, at the backend's precision.

    Each expert holds three projections (gate, up, down); ``weight_names``
    differs by architecture but the shapes are always
    ``[intermediate, hidden] + [intermediate, hidden] + [hidden, intermediate]``.
    """
    intermediate = moe_config.intermediate_size
    params_per_expert = 3 * intermediate * hidden_size
    bytes_per_expert = params_per_expert * _bytes_per_expert_param(method)
    return int(round(moe_config.expert_num * num_layers * bytes_per_expert))


def solve_num_gpu_experts(
    moe_config: MOEArchConfig,
    hidden_size: int,
    num_layers: int,
    budget_bytes: int,
    method: str | None = None,
    reserve_fraction: float = DEFAULT_RESERVE_FRACTION,
) -> PlacementAdvice:
    """Maximally fill ``budget_bytes`` with GPU-resident routed experts.

    The placement is uniform across layers, matching what the device-map
    builders can express: with an equal per-layer cost, one integer program --
    maximize ``k`` subject to ``num_layers * k <= floor(budget / cost)`` --
    describes every feasible placement, and the largest feasible ``k`` is its
    optimum. Layer counts that do not divide the affordable expert total strand
    a remainder, reported via ``budget_fully_used`` rather than rounded up,
    because rounding up would exceed the cap and OOM at load time.
    """
    num_experts = moe_config.expert_num
    num_layers = max(1, int(num_layers))
    budget_bytes = max(0, int(budget_bytes))

    total_expert_bytes = estimate_expert_bytes(moe_config, hidden_size, num_layers, method)
    if num_experts <= 0 or total_expert_bytes <= 0:
        return PlacementAdvice(0, 0, 0, 0, 0, total_expert_bytes, True)

    usable = int(budget_bytes * (1.0 - max(0.0, min(reserve_fraction, 0.9))))
    per_expert = total_expert_bytes // (num_experts * num_layers)
    affordable = usable // per_expert if per_expert > 0 else 0

    num_gpu_experts = min(num_experts, affordable // num_layers)
    # The uniform-level constraint strands a remainder smaller than one full
    # layer of experts; flag it so callers can see the budget was not exhausted.
    budget_fully_used = (
        num_gpu_experts * num_layers == affordable or num_gpu_experts == num_experts
    )

    advice = PlacementAdvice(
        num_gpu_experts_per_layer=num_gpu_experts,
        total_gpu_experts=num_gpu_experts * num_layers,
        total_experts=num_experts * num_layers,
        usable_budget_bytes=usable,
        expert_cost_bytes=per_expert,
        total_expert_bytes=total_expert_bytes,
        budget_fully_used=budget_fully_used,
    )
    logger.info(
        "Placement advisor: %d/%d experts per layer on GPU "
        "(%d of %d bytes, reserve_fraction=%.2f, method=%s)",
        num_gpu_experts,
        num_experts,
        num_gpu_experts * num_layers * per_expert,
        total_expert_bytes,
        reserve_fraction,
        method,
    )
    return advice


def total_gpu_memory_bytes(device: str = "cuda:0") -> int | None:
    """Free-plus-reserved VRAM for ``device`` via torch, or ``None`` offline.

    Returns the *currently free* memory rather than total capacity: whatever
    the framework already allocated is not available to expert weights.
    """
    if not torch.cuda.is_available():
        return None
    try:
        index = int(device.split(":")[1]) if ":" in device else torch.cuda.current_device()
        free_bytes, _total = torch.cuda.mem_get_info(index)
    except (ValueError, IndexError, RuntimeError, AssertionError):
        return None
    return int(free_bytes)
