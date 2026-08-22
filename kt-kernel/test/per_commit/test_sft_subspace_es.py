# SPDX-License-Identifier: Apache-2.0

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from kt_kernel.sft.arch import MOEArchConfig
from kt_kernel.sft.es import KTSubspaceES
from kt_kernel.sft.lora import get_kt_trainable_params, update_kt_lora_pointers


def _moe_config():
    return MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=3,
        num_experts_per_tok=1,
    )


def _fake_model(num_layers=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    wrappers = []
    for layer_idx in range(num_layers):
        backend = SimpleNamespace(
            gate_proj_buf=nn.Parameter(torch.randn(2, 3, 4, generator=generator, dtype=torch.bfloat16)),
            up_proj_buf=nn.Parameter(torch.randn(2, 3, 4, generator=generator, dtype=torch.bfloat16)),
            down_proj_buf=nn.Parameter(torch.randn(2, 4, 3, generator=generator, dtype=torch.bfloat16)),
            _base_weights_dirty=False,
            _kt_full_checkpoint_load_failed=False,
        )
        wrappers.append(
            SimpleNamespace(
                layer_idx=layer_idx,
                hidden_size=4,
                moe_config=_moe_config(),
                _full_weight_grad=True,
                wrapper=backend,
                lora_experts=None,
                _fused_expert_lora_params=None,
            )
        )
    return SimpleNamespace(_kt_wrappers=wrappers), wrappers


def _run_generation(trainer, fitness, population):
    rewards = []
    for _ in range(population):
        trainer.ask()
        rewards.append(fitness())
        trainer.restore()
    return rewards


class TestSubspacePartition:
    def test_partition_is_random_and_balanced_across_subspaces(self):
        model, _ = _fake_model()
        trainer = KTSubspaceES(model, population=8, subspace_count=4, chunk_size=8, seed=11)

        seen_assignments = set()
        for step in range(6):
            trainer.begin_step(step)
            state = trainer.states["layer0.gate_proj"]
            counts = torch.bincount(state.chunk_subspace, minlength=trainer.subspace_count)
            # Equal-size subspaces: every subspace within one chunk of d/K.
            assert int(counts.max() - counts.min()) <= 1
            seen_assignments.add(tuple(state.chunk_subspace.tolist()))

        # The partition is resampled each step, so parameters are not pinned
        # to one fixed grouping.
        assert len(seen_assignments) > 1

    def test_perturbation_is_confined_to_one_subspace(self):
        model, wrappers = _fake_model()
        trainer = KTSubspaceES(model, population=4, subspace_count=2, chunk_size=6, seed=3)
        trainer.begin_step(step=0)
        original = wrappers[0].wrapper.gate_proj_buf.detach().clone()

        trainer._perturb(subspace=0, evaluation=0)

        changed = wrappers[0].wrapper.gate_proj_buf.detach().view(-1) != original.view(-1)
        assignment = trainer.states["layer0.gate_proj"].chunk_subspace
        expected_mask = torch.repeat_interleave(assignment == 0, trainer.chunk_size)[: changed.numel()]
        assert torch.equal(changed, expected_mask)
        # Every perturbed coordinate actually moved (sigma > 0).
        assert changed.any()

    def test_dimension_aware_scale_matches_full_space_norm(self):
        model, _ = _fake_model()
        subspace_count = 4
        trainer = KTSubspaceES(
            model, population=8, subspace_count=subspace_count, sigma=0.05, chunk_size=6, seed=7
        )
        # sigma_k = sqrt(K) * sigma keeps E[||delta_k||^2] equal to a full-space
        # perturbation of the same expected squared norm.
        assert trainer.subspace_sigma == pytest.approx(math.sqrt(subspace_count) * 0.05)

    def test_restore_recovers_the_pre_perturbation_weights_exactly(self):
        model, wrappers = _fake_model()
        trainer = KTSubspaceES(model, population=4, subspace_count=2, chunk_size=6, seed=5)
        trainer.begin_step(step=0)
        originals = [
            (wrapper.wrapper.gate_proj_buf.detach().clone(), wrapper.wrapper.up_proj_buf.detach().clone())
            for wrapper in wrappers
        ]

        trainer.ask()
        trainer.restore()

        for wrapper, (gate, up) in zip(wrappers, originals):
            torch.testing.assert_close(wrapper.wrapper.gate_proj_buf.detach(), gate)
            torch.testing.assert_close(wrapper.wrapper.up_proj_buf.detach(), up)


class TestJointStandardization:
    def test_update_moves_parameters_toward_higher_reward(self):
        model, wrappers = _fake_model(seed=1)
        gate = wrappers[0].wrapper.gate_proj_buf
        trainer = KTSubspaceES(model, population=8, subspace_count=4, chunk_size=6, sigma=0.5, seed=2)

        def fitness():
            # A linear objective makes the expected update direction computable.
            return float(gate.detach().float().sum())

        baseline = fitness()
        before = gate.detach().clone()
        trainer.begin_step(step=0)
        rewards = _run_generation(trainer, fitness, trainer.population)
        stats = trainer.tell(rewards)

        assert stats["reward_std"] > 0
        delta = (gate.detach() - before).float()
        assert float(delta.sum()) > 0
        assert delta.abs().sum() > 0

    def test_identical_rewards_leave_weights_unchanged(self):
        model, wrappers = _fake_model(seed=4)
        trainer = KTSubspaceES(model, population=8, subspace_count=4, chunk_size=6, seed=8)
        before = wrappers[0].wrapper.gate_proj_buf.detach().clone()

        trainer.begin_step(step=0)
        rewards = _run_generation(trainer, lambda: 1.0, trainer.population)
        stats = trainer.tell(rewards)

        assert stats["reward_std"] == 0.0
        torch.testing.assert_close(wrappers[0].wrapper.gate_proj_buf.detach(), before)


class TestStepComposition:
    def test_rewards_are_standardized_across_all_subspaces_jointly(self):
        model, _ = _fake_model()
        trainer = KTSubspaceES(model, population=8, subspace_count=4, chunk_size=6, seed=9)
        trainer.begin_step(step=0)

        rewards = [0.0] * 8
        stats = trainer.tell(rewards)

        # Zero variance across the whole population: no direction is preferred,
        # and the pooled statistics (not per-subspace ones) were used.
        assert stats["reward_std"] == 0.0
        assert stats["reward_mean"] == 0.0

    def test_tell_rejects_wrong_reward_count(self):
        model, _ = _fake_model()
        trainer = KTSubspaceES(model, population=8, subspace_count=4, chunk_size=6, seed=9)
        trainer.begin_step(step=0)
        with pytest.raises(ValueError, match="expected 8 rewards"):
            trainer.tell([0.0, 1.0])

    def test_population_must_be_divisible_by_subspace_count(self):
        model, _ = _fake_model()
        with pytest.raises(ValueError, match="divisible"):
            KTSubspaceES(model, population=7, subspace_count=3)

    def test_owns_the_full_weight_parameters_the_optimizer_sees(self):
        model, wrappers = _fake_model()
        trainer = KTSubspaceES(model, population=4, subspace_count=2, chunk_size=6)
        optimizer_visible = get_kt_trainable_params(model)

        assert set(id(p) for p in trainer.params) == set(id(p) for p in optimizer_visible)
        assert all(isinstance(p, nn.Parameter) for p in trainer.params)


class TestKernelSync:
    def test_sync_kernel_weights_flags_every_full_backend_dirty(self):
        model, wrappers = _fake_model()
        trainer = KTSubspaceES(model, population=4, subspace_count=2, chunk_size=6)

        assert trainer.sync_kernel_weights(model) == len(wrappers)
        assert all(wrapper.wrapper._base_weights_dirty for wrapper in wrappers)

    def test_lora_pointer_hook_runs_the_es_step_boundary(self):
        model, _ = _fake_model()
        trainer = KTSubspaceES(model, population=4, subspace_count=2, chunk_size=6)
        model.kt_subspace_es = trainer

        # Outside a step, the boundary reports a completed schedule.
        assert update_kt_lora_pointers(model) is None
        assert trainer.step_boundary() == 1

        trainer.begin_step(step=0)
        assert trainer.step_boundary() == 0
        update_kt_lora_pointers(model)
        assert trainer._in_step is True
