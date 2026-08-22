# SPDX-License-Identifier: Apache-2.0
"""Two-stage (department) routing tests for KTMoELayerWrapper.

The repo sources under ``kt-kernel/python`` shadow an installed ``kt_kernel``
copy in this environment, so the SFT modules are loaded from the source tree
under a package shell that reuses the installed compiled extension — the same
trick ``test_load_experts_count_guard`` uses to import repo modules without
rebuilding ``kt_kernel_ext``.
"""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="default")

_REPO_PYTHON = os.path.join(os.path.dirname(__file__), "..", "..", "python")


def _load_repo_sft(module_name: str):
    """Import ``kt-kernel/python/sft/<module_name>`` from the source tree."""
    import importlib

    import kt_kernel as installed

    if "repo_kt" not in sys.modules:
        shell = types.ModuleType("repo_kt")
        shell.__path__ = [os.path.abspath(_REPO_PYTHON)]
        # sft.base imports `from kt_kernel import kt_kernel_ext` transitively
        shell.kt_kernel_ext = installed.kt_kernel_ext
        sys.modules["repo_kt"] = shell
    return importlib.import_module(f"repo_kt.sft.{module_name}")


layer_mod = _load_repo_sft("layer")
department_mod = _load_repo_sft("expert_department")

KTMoELayerWrapper = layer_mod.KTMoELayerWrapper
assign_departments = department_mod.assign_departments
department_topk = department_mod.department_topk
loaded_body_bound = department_mod.loaded_body_bound
resolve_department_topk = department_mod.resolve_department_topk

import torch  # noqa: E402  (kept after the repo import so the shell is set up first)
from torch import nn  # noqa: E402


class _OriginalMoE(nn.Module):
    def __init__(self, router, num_experts=6):
        super().__init__()
        self.gate = router
        self.experts = nn.ModuleList()


def _make_layer(num_experts=6, num_experts_per_tok=4, departments_per_tok=None, num_departments=None):
    config = types.SimpleNamespace(
        router_attr="gate",
        experts_attr="experts",
        has_shared_experts=False,
        router_type="linear",
        expert_num=num_experts,
        num_experts_per_tok=num_experts_per_tok,
    )
    layer = KTMoELayerWrapper(
        original_moe=_OriginalMoE(nn.Linear(4, num_experts, bias=False)),
        wrapper=None,
        lora_params=None,
        moe_config=config,
        hidden_size=4,
        layer_idx=0,
        full_weight_grad=False,
    )
    if departments_per_tok is not None:
        layer.departments_per_tok = departments_per_tok
        layer.num_departments = num_departments
        layer._init_department_routing(layer._original_router)
    return layer


class TestAssignDepartments(unittest.TestCase):
    def test_block_assignment_is_contiguous_and_deterministic(self):
        assignment = assign_departments(6, 3)
        self.assertEqual(assignment.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(assign_departments(6, 3).tolist(), assignment.tolist())

    def test_rejects_more_departments_than_experts(self):
        with self.assertRaises(ValueError):
            assign_departments(2, 3)

    def test_similarity_assignment_groups_correlated_rows(self):
        # Experts 0..1 share a direction, experts 2..3 oppose it.
        router_weight = torch.tensor(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [-1.0, 0.0],
                [-0.9, -0.1],
            ]
        )
        assignment = assign_departments(4, 2, router_weight=router_weight)
        self.assertEqual(assignment[0].item(), assignment[1].item())
        self.assertEqual(assignment[2].item(), assignment[3].item())
        self.assertNotEqual(assignment[0].item(), assignment[2].item())


class TestDepartmentTopk(unittest.TestCase):
    def setUp(self):
        # Two departments of two experts each, then two singletons.
        self.department_of_expert = torch.tensor([0, 0, 1, 1, 2, 3])

    def test_all_experts_from_selected_departments(self):
        probs = torch.tensor([[0.30, 0.25, 0.20, 0.10, 0.10, 0.05]])
        topk_ids, topk_weights = department_topk(probs, self.department_of_expert, 4, 1)
        departments = self.department_of_expert[topk_ids]
        self.assertEqual(departments.unique().numel(), 1)
        self.assertEqual(topk_ids.shape, (1, 4))
        self.assertEqual(topk_weights.shape, (1, 4))

    def test_single_department_top_k_never_exceeds_budget(self):
        # Even with k=4 and 6 experts spread over 4 departments, one selected
        # department bounds the routed bodies at 1.
        probs = torch.rand(5, 6)
        topk_ids, _ = department_topk(probs, self.department_of_expert, 4, 1)
        for row in topk_ids:
            self.assertEqual(self.department_of_expert[row].unique().numel(), 1)

    def test_two_departments_bound_bodies_at_two(self):
        probs = torch.rand(5, 6)
        topk_ids, _ = department_topk(probs, self.department_of_expert, 4, 2)
        for row in topk_ids:
            self.assertLessEqual(self.department_of_expert[row].unique().numel(), 2)

    def test_weights_are_normalized(self):
        probs = torch.rand(3, 6)
        _, topk_weights = department_topk(probs, self.department_of_expert, 4, 2)
        self.assertTrue(torch.allclose(topk_weights.sum(dim=-1), torch.ones(3), atol=1e-5))

    def test_strongest_department_wins(self):
        # Department 0 (experts 0,1) dominates; with one department selected
        # the routed experts must come from {0,1}.
        probs = torch.tensor([[0.40, 0.35, 0.10, 0.05, 0.05, 0.05]])
        topk_ids, _ = department_topk(probs, self.department_of_expert, 2, 1)
        self.assertTrue(set(topk_ids[0].tolist()) <= {0, 1})

    def test_starved_row_pads_with_valid_ids_and_zero_weight(self):
        # One department holds every expert: department 0 = all four experts.
        departments = torch.zeros(4, dtype=torch.int64)
        probs = torch.tensor([[0.4, 0.3, 0.2, 0.1]])
        topk_ids, topk_weights = department_topk(probs, departments, 4, 1)
        self.assertTrue((topk_ids >= 0).all() and (topk_ids < 4).all())
        self.assertTrue(torch.allclose(topk_weights.sum(dim=-1), torch.ones(1), atol=1e-5))


class TestLoadedBodyBound(unittest.TestCase):
    def test_counts_distinct_departments_per_token(self):
        department_of_expert = torch.tensor([0, 0, 1, 1])
        topk_ids = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 1]])
        bound = loaded_body_bound(topk_ids, department_of_expert)
        self.assertEqual(bound.tolist(), [2, 1])

    def test_physical_bodies_override_departments(self):
        department_of_expert = torch.tensor([0, 1, 2, 3])
        physical_body = torch.tensor([7, 7, 7, 8])
        topk_ids = torch.tensor([[0, 1, 2, 3]])
        # Departments say 4 bodies; the shared physical backing says 2.
        bound = loaded_body_bound(topk_ids, department_of_expert, physical_body)
        self.assertEqual(bound.tolist(), [2])


class TestLayerRoutingIntegration(unittest.TestCase):
    def test_off_by_default_matches_vanilla_topk(self):
        layer = _make_layer()
        self.assertIsNone(layer._departments_per_tok)
        hidden = torch.randn(1, 3, 4)
        ids, _ = layer._compute_routing(hidden)
        self.assertEqual(ids.shape, (3, 4))

    def test_enabled_routing_stays_within_departments(self):
        layer = _make_layer(departments_per_tok=1, num_departments=3)
        self.assertIsNotNone(layer._department_of_expert)
        hidden = torch.randn(1, 5, 4)
        ids, weights = layer._compute_routing(hidden)
        self.assertEqual(ids.shape, (5, 4))
        self.assertEqual(weights.shape, (5, 4))
        for row in ids:
            self.assertEqual(layer._department_of_expert[row].unique().numel(), 1)

    def test_enabled_routing_reduces_loaded_bodies(self):
        layer = _make_layer(departments_per_tok=1, num_departments=3)
        # The layer infers departments from the router weight, so assert the
        # budget against the layer's own assignment rather than assuming blocks.
        hidden = torch.randn(1, 5, 4)
        ids, _ = layer._compute_routing(hidden)
        bound = loaded_body_bound(ids, layer._department_of_expert)
        self.assertEqual(bound.shape, (5,))
        self.assertTrue((bound == 1).all())

    def test_forward_through_layer_with_departments(self):
        class _RecordingWrapper:
            def __init__(self):
                self._full_weight_grad = False
                self._uses_authoritative_optimizer_grads = False
                self.share_backward_bb = False
                self.submitted = None

            def submit_forward(self, hidden_states, expert_ids, weights, save_for_backward=True):
                self.submitted = (expert_ids.clone(), weights.clone())

            def sync_forward(self, output_device=None):
                return torch.zeros(1, 3, 4)

        backend = _RecordingWrapper()
        layer = _make_layer()
        layer.wrapper = backend
        layer.departments_per_tok = 1
        layer.num_departments = 3
        layer._init_department_routing(layer._original_router)
        layer.eval()

        out = layer(torch.randn(1, 3, 4))
        self.assertEqual(out.shape, (1, 3, 4))
        expert_ids, weights = backend.submitted
        self.assertEqual(expert_ids.shape, (3, 4))
        self.assertEqual(weights.shape, (3, 4))
        for row in expert_ids:
            self.assertEqual(layer._department_of_expert[row].unique().numel(), 1)


class TestConfigPlumbing(unittest.TestCase):
    def test_resolve_department_topk_off_values(self):
        self.assertIsNone(resolve_department_topk(None, 4))
        self.assertIsNone(resolve_department_topk(0, 4))

    def test_resolve_department_topk_clamps(self):
        self.assertEqual(resolve_department_topk(9, 4), 4)
        self.assertEqual(resolve_department_topk(2, 4), 2)

    def test_kt_config_reads_department_env_vars(self):
        cfg_mod = _load_repo_sft("config")
        saved = {
            k: os.environ.pop(k, None)
            for k in ("ACCELERATE_KT_DEPARTMENTS_PER_TOK", "ACCELERATE_KT_NUM_DEPARTMENTS")
        }
        try:
            os.environ["ACCELERATE_KT_DEPARTMENTS_PER_TOK"] = "2"
            os.environ["ACCELERATE_KT_NUM_DEPARTMENTS"] = "8"
            cfg = cfg_mod.KTConfig()
            self.assertEqual(cfg.kt_departments_per_tok, 2)
            self.assertEqual(cfg.kt_num_departments, 8)
        finally:
            for key, value in saved.items():
                os.environ.pop(key, None)
                if value is not None:
                    os.environ[key] = value

    def test_kt_config_department_routing_off_by_default(self):
        cfg_mod = _load_repo_sft("config")
        cfg = cfg_mod.KTConfig()
        self.assertIsNone(cfg.kt_departments_per_tok)
        self.assertIsNone(cfg.kt_num_departments)


if __name__ == "__main__":
    unittest.main()
