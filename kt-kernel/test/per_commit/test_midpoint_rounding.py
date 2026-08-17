"""Tests for reconstruction-guided midpoint rounding in the CPU weight converter.

``scripts/midpoint_rounding.py`` implements the tolerance-swept rounding
scheme of ReRound (arXiv:2608.11045) with a parameter-free error-feedback
guidance proxy, and ``scripts/convert_cpu_weights.py`` exposes it through the
opt-in ``--midpoint-tolerances`` flag.

These tests exercise the wiring through the existing converter module (not
just the new module in isolation) and check the property the scheme
guarantees: because plain RTN is always tolerance index 0 of the sweep, the
selected candidate can never score worse than RTN on the selection metric.
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


midpoint_rounding = _load_script("midpoint_rounding")
convert_cpu_weights = _load_script("convert_cpu_weights")


def _rtn_dequant(weight: torch.Tensor, num_bits: int) -> torch.Tensor:
    """Reference round-to-nearest quantization on the same per-row grid."""
    qmax = 2 ** (num_bits - 1) - 1
    scale = midpoint_rounding._row_scales(weight.float(), num_bits)
    q = torch.round(weight.float() / scale).clamp(-qmax - 1, qmax)
    return (q * scale).to(weight.dtype)


@pytest.mark.cpu
def test_selected_candidate_never_worse_than_rtn():
    """The spectral selection metric must not regress relative to plain RTN."""
    for seed in range(4):
        gen = torch.Generator().manual_seed(seed)
        weight = (torch.randn(3, 64, 128, generator=gen) / 20).to(torch.bfloat16)
        out, report = midpoint_rounding.guided_round(weight, num_bits=4)
        assert report.tolerances[0] == 0.0

        gap_rtn = midpoint_rounding.singular_value_gap(weight.float(), _rtn_dequant(weight, 4))
        gap_out = midpoint_rounding.singular_value_gap(weight.float(), out.float())
        assert bool((gap_out <= gap_rtn + 1e-4).all()), (
            f"seed={seed}: guided {gap_out.tolist()} vs RTN {gap_rtn.tolist()}"
        )


@pytest.mark.cpu
def test_guided_round_is_deterministic_and_stays_on_grid():
    gen = torch.Generator().manual_seed(7)
    weight = (torch.randn(2, 48, 96, generator=gen) / 20).to(torch.bfloat16)
    first, _ = midpoint_rounding.guided_round(weight, num_bits=4)
    second, _ = midpoint_rounding.guided_round(weight, num_bits=4)
    assert torch.equal(first, second)

    # Output must land on the int4 grid: (value / row scale) is an integer level.
    # bf16 storage quantizes the grid points themselves, so allow ~half a
    # level-1 bf16 ulp of slack rather than an exact-integer comparison.
    scale = midpoint_rounding._row_scales(first.float(), 4)
    levels = first.float() / scale
    assert bool(((levels - torch.round(levels)).abs() < 0.05).all())
    assert float(levels.abs().max()) <= 8.0 + 1e-3


@pytest.mark.cpu
def test_apply_to_experts_preserves_layout_and_dtype():
    gen = torch.Generator().manual_seed(11)
    gate = (torch.randn(2, 32, 64, generator=gen) / 20).to(torch.bfloat16)
    up = (torch.randn(2, 32, 64, generator=gen) / 20).to(torch.bfloat16)
    down = (torch.randn(2, 64, 32, generator=gen) / 20).to(torch.bfloat16)

    g2, u2, d2 = midpoint_rounding.apply_to_experts(gate, up, down, num_bits=4)
    for original, rounded in ((gate, g2), (up, u2), (down, d2)):
        assert rounded.shape == original.shape
        assert rounded.dtype == original.dtype
        assert rounded.is_contiguous()
        # Rounding only moves weights onto grid points; it cannot invent new magnitude.
        assert bool((rounded.abs() <= original.abs().amax(dim=-1, keepdim=True) + 1e-6).all())


@pytest.mark.cpu
def test_converter_accepts_midpoint_tolerances_option(tmp_path):
    """The converter wiring: the flag must reach OnlineQuantConverter."""
    # OnlineQuantConverter indexes the input directory in __init__, so give it
    # one real (minimal) safetensors file.
    from safetensors.torch import save_file

    input_dir = tmp_path / "in"
    input_dir.mkdir()
    save_file({"model.layers.0.mlp.experts.0.gate_proj.weight": torch.zeros(8, 8)}, str(input_dir / "w.safetensors"))

    model_config = {
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "hidden_size": 8,
        "moe_intermediate_size": 8,
    }

    tolerances = midpoint_rounding.parse_tolerance_spec("0.05,0.1")
    assert tolerances == [0.0, 0.05, 0.1]

    converter = convert_cpu_weights.OnlineQuantConverter(
        input_path=str(input_dir),
        output_path=str(tmp_path / "out"),
        model_config=model_config,
        cpuinfer_threads=2,
        threadpool_count=1,
        input_type="bf16",
        quant_method="int4",
        merge_to_safetensor=False,
        midpoint_tolerances=tolerances,
    )
    assert converter.midpoint_tolerances == [0.0, 0.05, 0.1]
    converter.close()

    # Omitted flag means plain RTN: the guided path stays dormant.
    default_converter = convert_cpu_weights.OnlineQuantConverter(
        input_path=str(input_dir),
        output_path=str(tmp_path / "out2"),
        model_config=model_config,
        cpuinfer_threads=2,
        threadpool_count=1,
        input_type="bf16",
        quant_method="int4",
        merge_to_safetensor=False,
    )
    assert default_converter.midpoint_tolerances is None
    default_converter.close()


@pytest.mark.cpu
def test_zero_tolerance_matches_round_to_nearest():
    """A 0.0-only sweep is exactly RTN, so the two must agree elementwise."""
    gen = torch.Generator().manual_seed(3)
    weight = (torch.randn(2, 64, 64, generator=gen) / 20).to(torch.bfloat16)
    out, report = midpoint_rounding.guided_round(weight, num_bits=4, tolerances=[0.0])
    assert report.chosen_tolerance == 0.0
    assert torch.equal(out, _rtn_dequant(weight, 4))


@pytest.mark.cpu
def test_parse_tolerance_spec_rejects_out_of_range():
    with pytest.raises(ValueError):
        midpoint_rounding.parse_tolerance_spec("0.9")
    with pytest.raises(ValueError):
        midpoint_rounding.parse_tolerance_spec("-0.1")
    with pytest.raises(ValueError):
        midpoint_rounding.parse_tolerance_spec(",")
    # RTN is always included, even when the user's sweep omits it.
    assert midpoint_rounding.parse_tolerance_spec("0.05") == [0.0, 0.05]
