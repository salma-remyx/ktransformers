"""Tests for fixed-grid refinement of AMX weight quantization.

Covers both the refinement module itself and its wiring into
``scripts/convert_cpu_weights.py`` (OnlineQuantConverter): with
``refine_sweeps > 0`` the converter must refine expert projections in place
before handing them to the AMX quantizer, and the refined tensor must
re-quantize to the refined assignments under the AMX per-row grid (the
pass-through property that lets the existing C++ quantizer materialize the
refined solution unchanged).
"""

import os
import sys
import unittest

import torch

# Register this test for CPU CI.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="default")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts"))
import convert_cpu_weights
import fixed_grid_refinement

refine_weight_grid_ = fixed_grid_refinement.refine_weight_grid_
METHOD_QMAX = fixed_grid_refinement.METHOD_QMAX


def _amx_rtn_dequant(w: torch.Tensor, qmax: int):
    """Simulate the AMX per-row RTN quantizer: s = amax/qmax, round-to-nearest.

    Mirrors BufferBInt4Impl (int4) and GemmKernel224Int8::BufferB (int8):
    per-row scale from the absolute max, nearest-integer assignments clamped
    to the grid, dequant s * q.
    """
    wf = w.float()
    s = wf.abs().amax(dim=1, keepdim=True) / qmax
    inv = torch.where(s > 0, 1.0 / s, torch.zeros_like(s))
    q = torch.clamp(torch.round(wf * inv), -qmax, qmax)
    return s, q


def _mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(((a.float() - b.float()) ** 2).mean())


class TestFixedGridRefinement(unittest.TestCase):
    def test_refinement_reduces_mse_int4(self):
        torch.manual_seed(0)
        w = (torch.randn(64, 512) * 0.05).to(torch.bfloat16)
        s0, q0 = _amx_rtn_dequant(w, qmax=7)
        baseline = _mse(w, s0 * q0)

        w_ref = w.clone()
        stats = refine_weight_grid_(w_ref, qmax=7, num_sweeps=4)

        self.assertLess(stats["final_mse"], stats["initial_mse"])
        self.assertAlmostEqual(stats["initial_mse"], baseline * w.numel() / w.shape[0], places=4)
        # Non-increasing sweep trajectory (ReQuant's monotone acceptance rule).
        hist = stats["mse_history"]
        self.assertTrue(all(b <= a + 1e-9 for a, b in zip(hist, hist[1:])))

    def test_refinement_never_worsens_int8(self):
        # At 8 bits RTN is already near the weight-MSE optimum (the paper's
        # large gains are at low bit-width), so refinement may legitimately
        # change nothing — but it must never do worse.
        torch.manual_seed(1)
        w = (torch.randn(32, 256) * 0.02).to(torch.bfloat16)
        w_ref = w.clone()
        stats = refine_weight_grid_(w_ref, qmax=127, num_sweeps=3)
        self.assertLessEqual(stats["final_mse"], stats["initial_mse"])

    def test_amx_requant_reproduces_refined_grid(self):
        """Re-running the AMX RTN quantizer on the refined tensor must recover
        the refined solution (same MSE), proving the C++ pass-through."""
        torch.manual_seed(2)
        for qmax in (7, 127):
            w = (torch.randn(48, 384) * 0.03).to(torch.bfloat16)
            w_ref = w.clone()
            stats = refine_weight_grid_(w_ref, qmax=qmax, num_sweeps=4)

            s, q = _amx_rtn_dequant(w_ref, qmax=qmax)
            requant_mse = float(((w.float() - s * q) ** 2).sum() / w.shape[0])
            # BF16 emission shifts the recovered scale by ~2^-9 relative.
            self.assertAlmostEqual(requant_mse, stats["final_mse"], delta=0.05 * stats["final_mse"] + 1e-12)
            # End-to-end: re-quantized refined tensor never worse than RTN.
            self.assertLessEqual(requant_mse, stats["initial_mse"])

    def test_zero_rows_and_zero_sweeps_are_safe(self):
        w = torch.zeros(4, 128, dtype=torch.bfloat16)
        stats = refine_weight_grid_(w, qmax=7, num_sweeps=2)
        self.assertEqual(stats["final_mse"], 0.0)
        self.assertTrue(bool((w == 0).all()))

        w2 = torch.randn(8, 64).to(torch.bfloat16)
        snapshot = w2.clone()
        refine_weight_grid_(w2, qmax=7, num_sweeps=0)
        self.assertTrue(torch.equal(w2, snapshot))

    def test_method_qmax_covers_online_methods(self):
        for method in ("int4", "int8", "moe_int4", "moe_int8"):
            self.assertIn(method, METHOD_QMAX)


class TestConverterWiring(unittest.TestCase):
    """Exercise the OnlineQuantConverter hook added at the quantization call site."""

    def _make_converter(self, refine_sweeps: int, quant_method: str = "int4"):
        # Bypass the heavy filesystem/config __init__; the hook only needs
        # quant_method and refine_sweeps.
        conv = convert_cpu_weights.OnlineQuantConverter.__new__(convert_cpu_weights.OnlineQuantConverter)
        conv.quant_method = quant_method
        conv.refine_sweeps = refine_sweeps
        return conv

    def test_hook_refines_in_place_when_enabled(self):
        torch.manual_seed(3)
        gate = (torch.randn(2, 16, 128) * 0.05).to(torch.bfloat16)
        up = (torch.randn(2, 16, 128) * 0.05).to(torch.bfloat16)
        down = (torch.randn(2, 128, 16) * 0.05).to(torch.bfloat16)
        snapshots = [t.clone() for t in (gate, up, down)]

        s0, q0 = _amx_rtn_dequant(gate.reshape(-1, gate.shape[-1]), qmax=7)
        baseline = _mse(gate, (s0 * q0).reshape(gate.shape))

        conv = self._make_converter(refine_sweeps=4)
        conv._maybe_refine_expert_weights(gate, up, down)

        self.assertFalse(torch.equal(gate, snapshots[0]))
        s1, q1 = _amx_rtn_dequant(gate.reshape(-1, gate.shape[-1]), qmax=7)
        refined = _mse(gate, (s1 * q1).reshape(gate.shape))
        self.assertLess(refined, baseline)

    def test_hook_is_noop_when_disabled(self):
        torch.manual_seed(4)
        tensors = [(torch.randn(2, 8, 64) * 0.05).to(torch.bfloat16) for _ in range(3)]
        snapshots = [t.clone() for t in tensors]

        conv = self._make_converter(refine_sweeps=0)
        conv._maybe_refine_expert_weights(*tensors)

        for t, snap in zip(tensors, snapshots):
            self.assertTrue(torch.equal(t, snap))

    def test_constructor_accepts_refine_sweeps_default_off(self):
        import inspect

        sig = inspect.signature(convert_cpu_weights.OnlineQuantConverter.__init__)
        self.assertIn("refine_sweeps", sig.parameters)
        self.assertEqual(sig.parameters["refine_sweeps"].default, 0)


if __name__ == "__main__":
    unittest.main()
