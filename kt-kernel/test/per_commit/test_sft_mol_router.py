# SPDX-License-Identifier: Apache-2.0

"""Mixture-of-LoRA routing over LoRA Experts (Macaron-V1 style, arXiv:2608.09819).

LoRAExperts in kt_kernel.sft.lora composes its specialists with a hardcoded
uniform average. These tests cover the routed path that replaces it when a
LoRARouter is attached, plus the wiring through KTMoELayerWrapper.
"""

import torch
from torch import nn

from kt_kernel.sft.arch import MOEArchConfig
from kt_kernel.sft.layer import KTMoELayerWrapper
from kt_kernel.sft.lora import LoRAExperts
from kt_kernel.sft.lora_router import (
    LoRARouter,
    clear_kt_mol_load_balance_loss,
    collect_kt_mol_load_balance_loss,
    reset_kt_mol_router_stats,
)


def _lora_experts(num_experts=3, hidden_size=4, intermediate_size=5, device="cpu"):
    torch.manual_seed(0)
    return LoRAExperts(
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        device=device,
        dtype=torch.float32,
    )


def test_uniform_average_without_router_is_unchanged():
    experts = _lora_experts()
    hidden_states = torch.randn(2, 3, 4)
    with torch.no_grad():
        expected = sum(expert(hidden_states) for expert in experts.experts) / experts.num_experts
        actual = experts(hidden_states)
    torch.testing.assert_close(actual, expected)
    assert collect_kt_mol_load_balance_loss() is None


def test_router_selects_and_weights_top_k_experts():
    experts = _lora_experts(num_experts=3)
    experts.router = LoRARouter(num_experts=3, hidden_size=4, top_k=1, device="cpu", dtype=torch.float32)
    hidden_states = torch.randn(2, 3, 4)

    gate_scores, topk_idx = experts.router(hidden_states)
    assert gate_scores.shape == (2, 3, 1)
    assert torch.allclose(gate_scores.sum(dim=-1), torch.ones(2, 3), atol=1e-5)

    with torch.no_grad():
        routed = experts(hidden_states)
        per_expert = torch.stack([e(hidden_states) for e in experts.experts])  # [E, T, H]
        expected = torch.zeros_like(routed)
        flat_idx = topk_idx.reshape(-1)
        flat_out = routed.reshape(-1, 4)
        flat_expert = per_expert.reshape(3, -1, 4)
        expected_flat = expected.reshape(-1, 4)
        for t in range(flat_idx.numel()):
            expected_flat[t] = flat_expert[flat_idx[t], t]
        torch.testing.assert_close(routed, expected)


def test_router_top_k_two_blends_two_experts():
    experts = _lora_experts(num_experts=3)
    experts.router = LoRARouter(num_experts=3, hidden_size=4, top_k=2, device="cpu", dtype=torch.float32)
    hidden_states = torch.randn(2, 2, 4)
    with torch.no_grad():
        routed = experts(hidden_states)
        gate_scores, topk_idx = experts.router(hidden_states)
        per_expert = torch.stack([e(hidden_states) for e in experts.experts])  # [E, T, H]
        flat_scores = gate_scores.reshape(-1, 2)
        flat_idx = topk_idx.reshape(-1, 2)
        flat_out = per_expert.reshape(3, -1, 4)
        expected = torch.zeros_like(routed).reshape(-1, 4)
        for t in range(flat_idx.shape[0]):
            acc = torch.zeros(4)
            for slot in range(2):
                acc = acc + flat_scores[t, slot] * flat_out[flat_idx[t, slot], t]
            expected[t] = acc
        torch.testing.assert_close(routed, expected.reshape_as(routed))


def test_router_gradients_flow_to_gate_and_experts():
    reset_kt_mol_router_stats()
    experts = _lora_experts(num_experts=2)
    experts.router = LoRARouter(num_experts=2, hidden_size=4, top_k=1, device="cpu", dtype=torch.float32)
    hidden_states = torch.randn(4, 4)

    loss = experts(hidden_states).square().sum()
    aux = collect_kt_mol_load_balance_loss()
    clear_kt_mol_load_balance_loss()
    assert aux is not None
    total = loss + aux
    total.backward()

    assert experts.router.gate.weight.grad is not None
    assert torch.count_nonzero(experts.router.gate.weight.grad) > 0
    for expert in experts.experts:
        assert expert.le_gate.weight.grad is not None
        assert torch.isfinite(expert.le_gate.weight.grad).all()


def test_load_balance_loss_balanced_routing_is_near_zero():
    reset_kt_mol_router_stats()
    experts = _lora_experts(num_experts=2)
    experts.router = LoRARouter(num_experts=2, hidden_size=4, top_k=1, device="cpu", dtype=torch.float32)
    # Equal gate weights -> every specialist receives identical logits, so the
    # softmax importance and top-k load are uniform across experts. The
    # Switch-Transformer loss is exactly 1.0 for perfectly balanced routing.
    with torch.no_grad():
        experts.router.gate.weight.zero_()
    experts(torch.randn(6, 4))
    loss = collect_kt_mol_load_balance_loss()
    clear_kt_mol_load_balance_loss()
    assert loss is not None
    assert abs(float(loss.detach()) - 1.0) < 1e-4


def test_load_balance_loss_skewed_routing_is_large():
    reset_kt_mol_router_stats()
    experts = _lora_experts(num_experts=2)
    experts.router = LoRARouter(num_experts=2, hidden_size=4, top_k=1, device="cpu", dtype=torch.float32)
    # One dominant column -> all tokens route to one specialist.
    with torch.no_grad():
        experts.router.gate.weight.zero_()
        experts.router.gate.weight[0, 0] = 50.0
    experts(torch.randn(6, 4))
    loss = collect_kt_mol_load_balance_loss()
    clear_kt_mol_load_balance_loss()
    assert loss is not None
    # Collapsed routing scores strictly above the balanced value of 1.0; with a
    # large but finite logit gap softmax mass is ~1, not exactly 1.
    assert float(loss.detach()) > 1.1


def test_router_rejects_out_of_range_top_k():
    try:
        LoRARouter(num_experts=2, hidden_size=4, top_k=3, device="cpu", dtype=torch.float32)
    except ValueError as exc:
        assert "top_k" in str(exc)
    else:
        raise AssertionError("expected ValueError for top_k > num_experts")


class _FakeWrapper:
    def __init__(self):
        self._full_weight_grad = False
        self._uses_authoritative_optimizer_grads = False
        self.share_backward_bb = False
        self._next_backward_wrapper = None
        self.output = None
        self.weights_shape = None

    def submit_forward(self, hidden_states, _expert_ids, weights, save_for_backward=True):
        self.output = torch.zeros_like(hidden_states)
        self.weights_shape = weights.shape

    def sync_forward(self, output_device=None):
        output = self.output.clone()
        return output if output_device is None else output.to(output_device)

    def backward(self, grad_output, output_device=None):
        grad_input = torch.zeros_like(grad_output)
        grad_weights = torch.zeros(self.weights_shape, dtype=torch.bfloat16)
        if output_device is not None:
            grad_input = grad_input.to(output_device)
            grad_weights = grad_weights.to(output_device)
        return grad_input, grad_weights

    def clear_checkpoint_output(self):
        pass


class _MoEWithLoRAExperts(nn.Module):
    def __init__(self, hidden_size=4, expert_num=2, intermediate_size=3, num_lora_experts=3):
        super().__init__()
        self.gate = nn.Linear(hidden_size, expert_num, bias=False)
        self.experts = nn.Identity()
        self.lora_experts = LoRAExperts(
            num_experts=num_lora_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            device="cpu",
            dtype=torch.float32,
        )
        self.lora_experts.router = LoRARouter(
            num_experts=num_lora_experts,
            hidden_size=hidden_size,
            top_k=1,
            device="cpu",
            dtype=torch.float32,
        )


def _moe_config() -> MOEArchConfig:
    return MOEArchConfig(
        moe_layer_attr="mlp",
        router_attr="gate",
        experts_attr="experts",
        weight_names=("gate_proj", "up_proj", "down_proj"),
        expert_num=2,
        intermediate_size=3,
        num_experts_per_tok=1,
        has_shared_experts=False,
    )


def test_routed_lora_experts_through_layer_wrapper():
    reset_kt_mol_router_stats()
    torch.manual_seed(0)
    original_moe = _MoEWithLoRAExperts()
    layer = KTMoELayerWrapper(
        original_moe=original_moe,
        wrapper=_FakeWrapper(),
        lora_params=None,
        moe_config=_moe_config(),
        hidden_size=4,
        layer_idx=0,
        lora_experts=original_moe.lora_experts,
        full_weight_grad=False,
    )

    hidden_states = torch.randn(2, 3, 4)
    with torch.no_grad():
        expected = original_moe.lora_experts(hidden_states)
        actual = layer(hidden_states)
    torch.testing.assert_close(actual, expected)

    # The router parameter is reachable through the wrapper, so the existing
    # optimizer-parameter collection picks it up automatically.
    names = dict(layer.named_parameters())
    assert "lora_experts.router.gate.weight" in names

    train_input = hidden_states.clone().requires_grad_(True)
    layer(train_input).square().sum().backward()
    assert names["lora_experts.router.gate.weight"].grad is not None
    assert torch.isfinite(names["lora_experts.router.gate.weight"].grad).all()
    clear_kt_mol_load_balance_loss()
