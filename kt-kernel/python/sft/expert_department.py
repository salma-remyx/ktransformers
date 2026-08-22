# Expert department routing for KT SFT
# SPDX-License-Identifier: Apache-2.0

"""
Department-structured expert routing (two-stage routing).

Adapted from DeaMoE: Efficient MoE Structure for Fast Small-Batch Decoding
(arXiv:2608.14385). DeaMoE groups experts into "departments" whose members
share most of their parameters and keeps only a few private parameters per
expert, then routes in two stages (department first, expert second) so a
decoding step loads one shared department body per token instead of k
unrelated expert bodies.

KT-kernel's C++ MoE backends already accept a ``physical_to_logical_map``
(see ``AMXSFTMoEWrapper.load_weights``), which is how several routed experts
can be backed by one physical weight body. What KT lacks is the routing half:
a way to pick, per token, a department and then the experts inside it, so the
per-step set of loaded bodies collapses even when top-k fires across many
logical experts.

This module supplies that routing half:

- :func:`assign_departments` — build a logical-expert -> department
  assignment from either an explicit list or router-weight similarity.
- :func:`department_topk` — the two-stage routing primitive: pick the top
  departments, then the top experts *within* those departments, recombining
  per-expert weights into a fixed-width ``(qlen, num_experts_per_tok)`` layout
  the C++ buffer copy expects.
- :func:`loaded_body_bound` — the paper's headline metric (per-step loaded
  weight bodies) as a pure function of the routing decision.

Only the routing/budget accounting is implemented here. Department *weight
sharing* (collapsing physical bodies via ``physical_to_logical_map``) is left
to the existing load path so a shared-body experiment only has to swap the
map, not the router.
"""

from __future__ import annotations

import torch

__all__ = [
    "assign_departments",
    "department_topk",
    "loaded_body_bound",
    "resolve_department_topk",
]


def _validate_router_weight(router_weight: torch.Tensor, num_experts: int) -> torch.Tensor:
    if router_weight.dim() != 2:
        raise ValueError(f"router_weight must be 2-D [E, H], got shape {tuple(router_weight.shape)}")
    if router_weight.shape[0] != num_experts:
        raise ValueError(
            f"router_weight has {router_weight.shape[0]} rows but the MoE declares {num_experts} experts"
        )
    return router_weight.detach().to(torch.float32)


def assign_departments(
    num_experts: int,
    num_departments: int,
    router_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return an ``int64`` tensor ``[num_experts]`` mapping expert -> department id.

    With ``router_weight`` given, assignment is similarity-greedy: rows are
    cosine-similarity clustered so experts the router already treats alike
    land in the same department. Without it, experts fall back to contiguous
    blocks (``expert e -> e * num_departments // num_experts``), which is
    deterministic and needs no weights.

    Both strategies are parameter-free proxies for DeaMoE's *trained*
    department heads; the paper learns the grouping, we infer it.
    """
    if num_departments < 1:
        raise ValueError(f"num_departments must be >= 1, got {num_departments}")
    if num_departments > num_experts:
        raise ValueError(f"num_departments ({num_departments}) cannot exceed num_experts ({num_experts})")
    if num_departments == 1:
        return torch.zeros(num_experts, dtype=torch.int64)

    if router_weight is None:
        # Contiguous blocks: expert e lives in department floor(e * D / E).
        scale = (torch.arange(num_experts, dtype=torch.float64) * num_departments) / num_experts
        return scale.to(torch.int64).clamp_(max=num_departments - 1)

    weight = _validate_router_weight(router_weight, num_experts)
    normed = torch.nn.functional.normalize(weight, dim=1, eps=1e-8)
    # Seed one department center per contiguous block, then a single
    # assignment pass. Deterministic, no random init, no iterations.
    block = assign_departments(num_experts, num_departments)
    centers = torch.stack([normed[block == d].mean(dim=0) for d in range(num_departments)])
    centers = torch.nn.functional.normalize(centers, dim=1, eps=1e-8)
    return (normed @ centers.T).argmax(dim=1).to(torch.int64)


def department_topk(
    router_probs: torch.Tensor,
    department_of_expert: torch.Tensor,
    num_experts_per_tok: int,
    num_departments_per_tok: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Two-stage top-k over departments, then over experts inside them.

    Args:
        router_probs: ``[qlen, E]`` non-negative routing scores (softmax or
            sigmoid probabilities). Not required to be normalized.
        department_of_expert: ``[E]`` int64 expert -> department id.
        num_experts_per_tok: final top-k width (``num_experts_per_tok``).
        num_departments_per_tok: how many departments a token may draw from.
            Must satisfy ``1 <= num_departments_per_tok <= num_experts_per_tok``.

    Returns:
        ``(topk_ids, topk_weights)`` with shapes ``[qlen, num_experts_per_tok]``,
        matching what ``submit_forward`` validates. Stage-1 department scores
        are the sum of member probabilities, so a department is strong when its
        members collectively are. Rows that cannot fill ``num_experts_per_tok``
        slots from the selected departments are padded by repeating the
        strongest in-department expert with zero weight — ids stay valid for
        the C++ gather, the padding contributes nothing to the output.
    """
    if router_probs.dim() != 2:
        raise ValueError(f"router_probs must be 2-D [qlen, E], got shape {tuple(router_probs.shape)}")
    if not 1 <= num_departments_per_tok <= num_experts_per_tok:
        raise ValueError(
            f"num_departments_per_tok must be in [1, num_experts_per_tok={num_experts_per_tok}], "
            f"got {num_departments_per_tok}"
        )

    qlen, num_experts = router_probs.shape
    if tuple(department_of_expert.shape) != (num_experts,):
        raise ValueError(
            f"department_of_expert has shape {tuple(department_of_expert.shape)}, expected [{num_experts}]"
        )
    probs = router_probs.to(torch.float32)
    department_of_expert = department_of_expert.to(torch.int64)
    num_departments = int(department_of_expert.max().item()) + 1

    # Stage 1: department score = summed member probability.
    dept_scores = torch.zeros(qlen, num_departments, dtype=probs.dtype, device=probs.device)
    dept_scores.scatter_add_(1, department_of_expert.expand(qlen, -1), probs)
    top_dept_ids = torch.topk(dept_scores, k=num_departments_per_tok, dim=-1, sorted=False).indices
    dept_mask = torch.zeros_like(dept_scores, dtype=torch.bool)
    dept_mask.scatter_(1, top_dept_ids, True)

    # Stage 2: keep only in-department experts, then take the global top-k.
    # Taking top-k over the masked scores (rather than k per department)
    # reproduces DeaMoE's "no redundant loading" budget: at most
    # num_departments_per_tok bodies are touched, and the k slots are spent on
    # the strongest experts those bodies actually contain.
    allowed = dept_mask[:, department_of_expert]
    masked = probs.masked_fill(~allowed, float("-inf"))
    topk_weights, topk_ids = torch.topk(masked, k=num_experts_per_tok, dim=-1, sorted=False)

    # Slots whose only candidates were masked out come back from topk with an
    # arbitrary index and -inf weight. Substituting the row's strongest allowed
    # expert at weight 0 keeps every id valid for the C++ gather while the pad
    # contributes nothing, so a starved department set cannot leak an expert
    # from outside the selected departments.
    strongest_allowed = torch.where(allowed, probs, torch.full_like(probs, float("-inf"))).argmax(dim=-1)
    invalid = ~torch.isfinite(topk_weights)
    topk_ids = torch.where(invalid, strongest_allowed.unsqueeze(-1).expand_as(topk_ids), topk_ids)
    topk_weights = torch.where(invalid, torch.zeros_like(topk_weights), topk_weights)

    total = topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = torch.where(total > 0, topk_weights / total.clamp_min(1e-20), topk_weights)
    return topk_ids, topk_weights


def loaded_body_bound(
    topk_ids: torch.Tensor,
    department_of_expert: torch.Tensor,
    physical_body_of_expert: torch.Tensor | None = None,
) -> torch.Tensor:
    """Distinct weight bodies touched per token by one routing decision.

    This is DeaMoE's per-step loaded-weight metric, computed without touching
    the backend: for each token it counts the distinct bodies behind its routed
    experts, where a body is a department by default or an entry of
    ``physical_body_of_expert`` when experts share physical weights through
    ``physical_to_logical_map``.

    Returns an ``int64`` tensor of shape ``[qlen]``; small-batch decoding cost
    scales with its mean, and its max is the per-token worst case.
    """
    if topk_ids.dim() != 2:
        raise ValueError(f"topk_ids must be 2-D, got shape {tuple(topk_ids.shape)}")
    body_of_expert = (
        department_of_expert.to(torch.int64)
        if physical_body_of_expert is None
        else physical_body_of_expert.to(torch.int64)
    )
    routed_bodies = body_of_expert[topk_ids.to(torch.int64)]
    qlen, k = routed_bodies.shape
    # Count distinct bodies per row without a python loop: offset each row into
    # its own value range so a single bincount splits cleanly per row.
    num_bodies = int(body_of_expert.max().item()) + 1
    row_offsets = torch.arange(qlen, device=routed_bodies.device).unsqueeze(1) * num_bodies
    flat = (routed_bodies + row_offsets).reshape(-1)
    counts = torch.bincount(flat, minlength=qlen * num_bodies)
    return (counts.reshape(qlen, num_bodies) > 0).sum(dim=1).to(torch.int64)


def resolve_department_topk(
    num_departments_per_tok: int | None,
    num_experts_per_tok: int,
) -> int | None:
    """Normalize a ``kt_departments_per_tok`` config value.

    Returns ``None`` when department routing is off, so callers can branch on
    a single falsy check. Values above ``num_experts_per_tok`` are clamped:
    asking for more departments than experts would silently disable stage 1.
    """
    if num_departments_per_tok is None or num_departments_per_tok <= 0:
        return None
    if num_departments_per_tok > num_experts_per_tok:
        return num_experts_per_tok
    return num_departments_per_tok
