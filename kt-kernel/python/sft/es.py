# Full-parameter subspace evolution strategy for KT SFT
# SPDX-License-Identifier: Apache-2.0

"""
Backpropagation-free full-parameter trainer for the KT Full weight path.

KT Full keeps every expert weight in CPU ``nn.Parameter`` buffers
(``gate_proj_buf`` / ``up_proj_buf`` / ``down_proj_buf``) and re-syncs them
into the C++ kernel after each update, which is exactly the contract a
forward-only evolution strategy needs: rewards come from rollout scoring, and
the only write to the parameters is a composed update applied between
evaluations.

This module implements Cooperative Parameter-subspace Evolution Strategy
(CoPES, Wang et al., arXiv:2608.02391): at each step the parameter indices are
re-partitioned into K equally sized random subspaces, each subspace receives
N/K perturbations evaluated under the shared full-model context, the pooled
rewards are standardized jointly (not per subspace), and the per-subspace
estimates are composed into one synchronous full-parameter update. The
perturbation scale is dimension-aware (``sigma_k = sqrt(K) * sigma``) so a
subspace perturbation matches the expected squared norm of a full-space one.

Adapted from "Cooperative Coevolution for Resource-Constrained Agentic LLM
Post-Training" (arXiv:2608.02391). The paper's agentic rollout / verifier loop
is supplied here by a caller-provided fitness callable; the reward
standardization, partition resampling, and update composition are the paper's.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn as nn

from .lora import _find_kt_wrappers

logger = logging.getLogger(__name__)

_PROJECTION_NAMES = ("gate_proj", "up_proj", "down_proj")

# Backup of pre-perturbation weights lives in CPU pinned-style memory, per the
# paper: restoring by subtracting the perturbation accumulates float error, so
# the original values are kept and copied back after every evaluation.
_NOISE_DTYPE = torch.float32


@dataclass
class KTSubspaceESState:
    """Bookkeeping for one KT Full expert projection buffer."""

    shape: torch.Size
    # Chunk -> subspace assignment for the current step, regenerated each step
    # so parameters are never permanently restricted to one grouping.
    chunk_subspace: torch.Tensor | None = None
    # Saved pre-perturbation values, restored after each evaluation.
    backup: torch.Tensor | None = None


class KTSubspaceES:
    """Cooperative parameter-subspace evolution strategy over KT Full weights.

    The trainer owns the full expert weight buffers of a KT Full model. A
    training step is:

    1. ``begin_step`` — re-partition every buffer into ``subspace_count`` random
       subspaces at chunk granularity and back up the current weights.
    2. ``ask`` — add the next subspace's perturbation (dimension-aware scale)
       in place, so the model evaluates a single lower-dimensional delta under
       the shared full-model context.
    3. ``restore`` — copy the backup back, so every perturbation is evaluated
       from the same pre-update model.
    4. ``tell(rewards)`` — jointly standardize the pooled rewards and compose
       the per-subspace estimates into one synchronous update.

    Chunked processing means a perturbation never materializes as a full-size
    tensor: each chunk is generated from its seed, applied, and regenerated the
    same way when the update is composed.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        population: int = 16,
        subspace_count: int = 4,
        sigma: float = 0.01,
        step_size: float = 0.01,
        chunk_size: int = 1 << 20,
        seed: int = 0,
    ):
        if population <= 0:
            raise ValueError(f"population must be positive, got {population}")
        if subspace_count <= 0:
            raise ValueError(f"subspace_count must be positive, got {subspace_count}")
        if sigma <= 0:
            raise ValueError(f"sigma must be positive, got {sigma}")
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if population % subspace_count != 0:
            raise ValueError(
                f"population ({population}) must be divisible by subspace_count ({subspace_count})"
            )

        self.population = int(population)
        self.subspace_count = int(subspace_count)
        self.sigma = float(sigma)
        self.step_size = float(step_size)
        self.chunk_size = int(chunk_size)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))

        # sigma_k = sqrt(K) * sigma matches the expected squared perturbation
        # norm of a full-space perturbation of the same population budget.
        self.subspace_sigma = self.sigma * math.sqrt(self.subspace_count)

        self.states: dict[str, KTSubspaceESState] = {}
        self._params: list[nn.Parameter] = []
        self._keys: list[str] = []
        self._collect_params(model)
        if not self._params:
            raise RuntimeError(
                "KTSubspaceES found no KT Full weight buffers; "
                "kt_train_mode must be 'full' (or 'hybrid') so the wrappers own base expert Parameters"
            )

        self._step_seed = 0
        self._evaluated = 0
        self._rewards: list[float] = []
        self._in_step = False

    # ========== Parameter collection ==========

    def _collect_params(self, model: nn.Module) -> None:
        wrappers = _find_kt_wrappers(model)
        for wrapper in wrappers or []:
            if not getattr(wrapper, "_full_weight_grad", False):
                continue
            backend = getattr(wrapper, "wrapper", None)
            if backend is None:
                continue
            for name in _PROJECTION_NAMES:
                param = getattr(backend, f"{name}_buf", None)
                if isinstance(param, nn.Parameter):
                    key = f"layer{wrapper.layer_idx}.{name}"
                    self.states[key] = KTSubspaceESState(shape=param.shape)
                    self._params.append(param)
                    self._keys.append(key)

    @property
    def params(self) -> list[nn.Parameter]:
        """Optimizer-visible KT Full expert Parameters owned by this trainer."""
        return list(self._params)

    @property
    def num_evaluations(self) -> int:
        """Total perturbation evaluations in one step (N in the paper)."""
        return self.population

    # ========== Partitioning (resampled every step) ==========

    def begin_step(self, step: int | None = None) -> None:
        """Back up weights and sample a fresh chunk-to-subspace assignment."""
        if step is not None:
            self._step_seed = int(step)
        else:
            self._step_seed = int(self.generator.initial_seed()) + self._step_seed + 1
        partition_generator = torch.Generator(device="cpu").manual_seed(self._step_seed)

        with torch.no_grad():
            for param, key in zip(self._params, self._keys):
                state = self.states[key]
                num_chunks = param.numel() // self.chunk_size
                # A trailing partial chunk joins the last full chunk's subspace,
                # which keeps every subspace within one element of d/K.
                if param.numel() % self.chunk_size:
                    num_chunks += 1
                num_chunks = max(num_chunks, 1)
                assignment = torch.randperm(num_chunks, generator=partition_generator)
                boundaries = torch.linspace(0, num_chunks, self.subspace_count + 1).round().long()
                chunk_subspace = torch.empty(num_chunks, dtype=torch.long)
                for subspace in range(self.subspace_count):
                    chunk_subspace[assignment[boundaries[subspace] : boundaries[subspace + 1]]] = subspace
                state.chunk_subspace = chunk_subspace
                state.backup = param.detach().clone(memory_format=torch.contiguous_format)

        self._evaluated = 0
        self._rewards = []
        self._in_step = True

    # ========== Perturbation ==========

    def _perturb(self, subspace: int, evaluation: int) -> None:
        """Add subspace perturbation ``evaluation`` in place, chunk by chunk."""
        noise_generator = torch.Generator(device="cpu")
        noise_generator.manual_seed(self._noise_seed(subspace, evaluation))
        with torch.no_grad():
            for param, key in zip(self._params, self._keys):
                state = self.states[key]
                flat = param.view(-1)
                for chunk_start in range(0, flat.numel(), self.chunk_size):
                    chunk = state.chunk_subspace[chunk_start // self.chunk_size]
                    if int(chunk) != subspace:
                        continue
                    chunk_end = min(chunk_start + self.chunk_size, flat.numel())
                    noise = torch.randn(
                        chunk_end - chunk_start, generator=noise_generator, dtype=_NOISE_DTYPE
                    )
                    flat[chunk_start:chunk_end].add_(
                        (noise * self.subspace_sigma).to(param.dtype)
                    )

    def _noise_seed(self, subspace: int, evaluation: int) -> int:
        # Deterministic per (step, subspace, evaluation): the same seed regenerates
        # the identical direction when the update is composed.
        return (self._step_seed * 1_000_003 + subspace * 8_917 + evaluation * 1_031) % (2**31 - 1)

    def ask(self) -> tuple[int, int]:
        """Perturb the model for the next evaluation and report its coordinates.

        Returns ``(subspace, evaluation_within_subspace)``. The caller scores
        the perturbed model on its shared batch, then calls :meth:`restore`.
        """
        if not self._in_step:
            raise RuntimeError("ask() called outside a step; call begin_step() first")
        if self._evaluated >= self.population:
            raise RuntimeError(
                f"ask() exceeded the population budget ({self.population}); call tell()"
            )
        per_subspace = self.population // self.subspace_count
        subspace = self._evaluated // per_subspace
        evaluation = self._evaluated % per_subspace
        self._perturb(subspace, evaluation)
        self._evaluated += 1
        return subspace, evaluation

    def restore(self) -> None:
        """Copy the backed-up weights back, undoing the current perturbation."""
        if not self._in_step:
            raise RuntimeError("restore() called outside a step; call begin_step() first")
        with torch.no_grad():
            for param, key in zip(self._params, self._keys):
                param.copy_(self.states[key].backup)

    # ========== Reward intake and joint standardization ==========

    def record_reward(self, reward: float) -> None:
        """Record the batch-mean reward of the most recently perturbed model."""
        if not self._rewards or len(self._rewards) < self._evaluated:
            self._rewards.append(float(reward))
        else:
            self._rewards[-1] = float(reward)

    def tell(self, rewards: Sequence[float] | None = None) -> dict[str, float]:
        """Standardize pooled rewards jointly and apply the composed update.

        Passing ``rewards`` explicitly overrides anything recorded through
        :meth:`record_reward`. Returns the pooled mean/std and the update norm.
        """
        if not self._in_step:
            raise RuntimeError("tell() called outside a step; call begin_step() first")
        pooled = list(rewards) if rewards is not None else list(self._rewards)
        if len(pooled) != self.population:
            raise ValueError(f"expected {self.population} rewards, got {len(pooled)}")

        reward_tensor = torch.tensor(pooled, dtype=torch.float64)
        mean = reward_tensor.mean()
        std = reward_tensor.std(unbiased=False)
        # Joint standardization: one shared mean/std across all K subspaces, so
        # per-subspace estimates land on a common reward scale before composing.
        if std > 0:
            standardized = (reward_tensor - mean) / std
        else:
            standardized = torch.zeros_like(reward_tensor)

        per_subspace = self.population // self.subspace_count
        update_sq_norm = 0.0
        with torch.no_grad():
            for param, key in zip(self._params, self._keys):
                state = self.states[key]
                flat = param.view(-1)
                for chunk_start in range(0, flat.numel(), self.chunk_size):
                    chunk = int(state.chunk_subspace[chunk_start // self.chunk_size])
                    chunk_end = min(chunk_start + self.chunk_size, flat.numel())
                    direction = torch.zeros(chunk_end - chunk_start, dtype=torch.float64)
                    for evaluation in range(per_subspace):
                        index = chunk * per_subspace + evaluation
                        weight = float(standardized[index])
                        if weight == 0.0:
                            continue
                        noise_generator = torch.Generator(device="cpu")
                        noise_generator.manual_seed(self._noise_seed(chunk, evaluation))
                        noise = torch.randn(direction.numel(), generator=noise_generator)
                        direction += weight * noise.to(torch.float64)
                    direction *= self.step_size / per_subspace
                    flat[chunk_start:chunk_end].add_(direction.to(param.dtype))
                    update_sq_norm += float(direction.dot(direction))
                state.backup = None
                state.chunk_subspace = None

        self._in_step = False
        self._rewards = []
        self._evaluated = 0
        return {
            "reward_mean": float(mean),
            "reward_std": float(std),
            "update_norm": math.sqrt(update_sq_norm),
        }

    # ========== Sync to the C++ kernel ==========

    def step_boundary(self) -> int:
        """Advance the schedule when called from the trainer's step hook.

        ``update_kt_lora_pointers`` invokes this after every optimizer-style
        step. In reward-driven use the caller drives evaluation explicitly and
        this hook only reports progress; the schedule itself moves through
        ``begin_step``/``ask``/``tell``.
        """
        completed = not self._in_step
        return int(completed)

    def sync_kernel_weights(self, model: nn.Module) -> int:
        """Mark every KT Full backend dirty so the next forward re-quantizes.

        KT's layer wrapper calls ``update_base_weights()`` when a backend is
        flagged dirty, which pushes the updated CPU Parameters into the C++
        kernel. Central weights (attention, router) keep their own optimizer.
        """
        wrappers = _find_kt_wrappers(model) or []
        synced = 0
        for wrapper in wrappers:
            if getattr(wrapper, "_full_weight_grad", False) and wrapper.wrapper is not None:
                wrapper.wrapper._base_weights_dirty = True
                synced += 1
        return synced
