# SPDX-License-Identifier: Apache-2.0

"""Auto GPU-expert placement: memory-capped advice feeding build_kt_device_map."""

from types import SimpleNamespace
from unittest.mock import patch

from kt_kernel.sft.arch import get_moe_arch_config
from kt_kernel.sft.config import KTConfig
from kt_kernel.sft.placement import (
    PlacementAdvice,
    estimate_expert_bytes,
    solve_num_gpu_experts,
)
from kt_kernel.sft.wrapper import build_kt_device_map, build_kt_device_map_simplified


GB = 1 << 30
SLACK = 1 << 10


def _budget_for(experts_on_gpu: int, per_expert: int, num_layers: int) -> int:
    """Budget whose post-reserve floor still covers ``experts_on_gpu`` experts."""
    needed = experts_on_gpu * num_layers * per_expert
    return int(needed / 0.85) + SLACK



def _config(num_layers=4, num_experts=8, hidden=128, intermediate=64):
    return SimpleNamespace(
        architectures=["Qwen3MoeForCausalLM"],
        hidden_size=hidden,
        num_hidden_layers=num_layers,
        num_experts=num_experts,
        moe_intermediate_size=intermediate,
        num_experts_per_tok=2,
        max_position_embeddings=64,
    )


def _gpu_expert_keys(device_map, config):
    return [key for key, value in device_map.items() if str(value).startswith("cuda") and ".experts." in key]


def test_estimate_expert_bytes_counts_three_projections_per_expert():
    moe_config = get_moe_arch_config(_config())
    # 3 projections * intermediate(64) * hidden(128) * 2 bytes (BF16) per expert.
    assert estimate_expert_bytes(moe_config, 128, 4, method="AMXBF16") == 8 * 4 * 3 * 64 * 128 * 2


def test_solve_num_gpu_experts_fills_largest_uniform_level():
    moe_config = get_moe_arch_config(_config())
    per_expert = 3 * 64 * 128 * 2
    total = 8 * 4 * per_expert
    budget = _budget_for(2, per_expert, num_layers=4)
    advice = solve_num_gpu_experts(moe_config, 128, 4, budget, method="AMXBF16")

    assert isinstance(advice, PlacementAdvice)
    assert advice.num_gpu_experts_per_layer == 2
    assert advice.total_gpu_experts == 8
    assert advice.total_expert_bytes == total
    assert advice.placement_ratio == 8 / 32


def test_solve_num_gpu_experts_zero_budget_keeps_all_experts_on_cpu():
    moe_config = get_moe_arch_config(_config())
    advice = solve_num_gpu_experts(moe_config, 128, 4, 0, method="AMXBF16")
    assert advice.num_gpu_experts_per_layer == 0
    assert advice.total_gpu_experts == 0


def test_build_kt_device_map_uses_advisor_in_auto_mode():
    config = _config(num_layers=2, num_experts=4, hidden=128, intermediate=64)
    per_expert = 3 * 64 * 128 * 2
    budget = _budget_for(2, per_expert, num_layers=2)  # 2 experts/layer over 2 layers
    cfg = KTConfig(kt_backend="AMXBF16", kt_auto_gpu_experts=True, kt_gpu_memory_gb=1)

    with patch(
        "kt_kernel.sft.placement.total_gpu_memory_bytes", return_value=budget
    ) as probe:
        device_map = build_kt_device_map(config, cfg, device="cuda:0")

    probe.assert_called_once_with("cuda:0")
    gpu_keys = _gpu_expert_keys(device_map, config)
    assert len(gpu_keys) == 4  # 2 layers * 2 GPU experts
    assert device_map["model.layers.0.mlp.experts.0"].startswith("cuda")
    assert device_map["model.layers.0.mlp.experts.1"].startswith("cuda")
    assert device_map["model.layers.0.mlp.experts.2"] == "cpu"
    assert device_map["model.layers.1.mlp.experts.0"].startswith("cuda")
    # Non-expert modules stay on GPU regardless of the advisor's answer.
    assert str(device_map["lm_head"]).startswith("cuda")
    assert str(device_map["model.layers.0"]).startswith("cuda")


def test_build_kt_device_map_explicit_knob_beats_auto_mode():
    config = _config(num_layers=2, num_experts=4, hidden=128, intermediate=64)
    cfg = KTConfig(kt_backend="AMXBF16", kt_num_gpu_experts=1, kt_auto_gpu_experts=True)

    with patch(
        "kt_kernel.sft.placement.total_gpu_memory_bytes", return_value=64 * GB
    ) as probe:
        device_map = build_kt_device_map(config, cfg, device="cuda:0")

    probe.assert_not_called()
    assert len(_gpu_expert_keys(device_map, config)) == 2  # 1 expert per layer


def test_build_kt_device_map_offline_falls_back_to_gpu_memory_override():
    config = _config(num_layers=2, num_experts=4, hidden=128, intermediate=64)
    per_expert = 3 * 64 * 128 * 2
    cfg = KTConfig(kt_backend="AMXBF16", kt_auto_gpu_experts=True, kt_gpu_memory_gb=1)

    # No GPU visible -> kt_gpu_memory_gb drives the budget instead. 1 GiB is
    # enough for all 8 experts at these toy dimensions.
    with patch(
        "kt_kernel.sft.placement.total_gpu_memory_bytes", return_value=None
    ), patch(
        "kt_kernel.sft.placement.solve_num_gpu_experts", wraps=solve_num_gpu_experts
    ) as solve:
        device_map = build_kt_device_map(config, cfg, device="cuda:7")

    budget_arg = solve.call_args.args[3]
    assert budget_arg == GB
    # 1 GiB holds every expert at these toy dimensions, so the override is
    # honored by placing all of them rather than falling back to CPU-only.
    assert len(_gpu_expert_keys(device_map, config)) == 8


def test_build_kt_device_map_simplified_delegates_to_advisor_in_auto_mode():
    config = _config(num_layers=2, num_experts=4, hidden=128, intermediate=64)
    per_expert = 3 * 64 * 128 * 2
    budget = _budget_for(2, per_expert, num_layers=2)
    cfg = KTConfig(kt_backend="AMXBF16", kt_auto_gpu_experts=True)

    with patch(
        "kt_kernel.sft.placement.total_gpu_memory_bytes", return_value=budget
    ):
        device_map = build_kt_device_map_simplified(config, cfg, device="cuda:0")

    assert len(_gpu_expert_keys(device_map, config)) == 4
