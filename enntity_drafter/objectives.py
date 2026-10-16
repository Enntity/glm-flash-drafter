# SPDX-License-Identifier: MIT AND Apache-2.0
# Portions are derived from vllm-project/speculators (Apache-2.0, LICENSES/Apache-2.0.txt); see NOTICE.
"""Training objectives for the scratch drafter.

``upstream``
    Unmodified ``DFlash2DraftModel.forward``: unary loss (``--unary-loss``,
    default KL to the target distribution) plus K-way selector CE, weighted per
    block position by ``fixed-exp-decay`` (gamma 4) or ``dpace``.  Used for
    every published round.
``selector_prefix`` (new)
    Acceptance-aligned: the same unary and selector terms, but each block slot k
    is weighted by the (detached) probability that the *served path* -- the
    selector's greedy choice among the unary top-k -- survives slots 1..k-1:
    ``w_k = floor + (1 - floor) * prod_{j<k} q_j`` with
    ``q_j = contains_target_j * softmax(selector_scores_j)[target]``.
    Optional ``eal_weight`` adds ``-E[accepted length]/(B-1)`` computed with
    the *non-detached* q, i.e. a direct surrogate of the chain proxy.  Unlike
    upstream D-PACE (which weights by unary confidence), this weights by the
    selector path, which earlier experiments found to be the bottleneck
    (oracle-in-top-16 accepted length 6.04 vs realised 4.40).
``selector_inset``
    ``upstream`` with the selector CE restricted to slots whose target is already in the unary top-K: upstream
    injects an absent target as the K-th candidate, a choice serving can never make.  Unary term unchanged.

``selector_prefix`` and ``selector_inset`` are experimental: no published round used them.
"""

from __future__ import annotations

from functools import partial
from typing import Any

import torch
from torch.nn import functional as F

OBJECTIVES = ("upstream", "selector_prefix", "selector_inset")


def loss_kwargs(device_type: str, unary_loss: str, per_position: str, gamma: float,
                selector_alpha: float, dpace_alpha: float) -> dict[str, Any]:
    """Speculators loss configuration; the fused Triton kernels need CUDA."""

    from speculators.losses import resolve_loss_config

    implementation = "fused" if device_type == "cuda" else "eager"
    return {
        "loss_config": resolve_loss_config(unary_loss, implementation),
        "tv_loss_fn": resolve_loss_config("tv", implementation)["tv"][0],
        "gamma": gamma,
        "per_position_loss_weight": per_position,
        "dpace_alpha": dpace_alpha,
        "selector_loss_alpha": selector_alpha,
    }


def upstream_loss_light(model: Any, inputs: dict[str, torch.Tensor], *, max_anchors: int,
                        loss_config: dict[str, Any], gamma: float = 4.0,
                        per_position_loss_weight: str = "fixed-exp-decay", dpace_alpha: float = 0.5,
                        selector_loss_alpha: float = 1.0, selector_inset: bool = False,
                        **_ignored: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """``DFlash2DraftModel.forward``'s loss through the same upstream calls, minus the
    metric-only full-vocabulary passes (``--train-metrics light``, training steps only).

    The loss is built exactly as ``dflash2.metrics.compute_metrics`` builds it
    (``compound_loss`` over the unary logits + alpha x ``compute_selector_loss``),
    so value and gradients are bit-identical.  Skipped (they never reach the loss):
    the TV acceptance estimate (a second fused [T, V] kernel), the greedy unary
    accuracy argmax, a repeated target argmax and the fp32 target logsumexp for the
    candidate-mass metric -- ~11 reads/writes of a [T, V] tensor per micro-step.
    Returns the selector-path ``eal`` (what the trainer logs) and ``loss``.
    """

    from speculators.losses import compound_loss, dflash_loss_decay, dpace_loss_decay
    from speculators.models.dflash2.metrics import compute_selector_loss, selector_training_candidates
    from speculators.models.metrics import compute_accepted_length_counts

    hidden, unary, targets, mask, indices = model._backbone_forward(
        inputs["hidden_states"], inputs["input_ids"], inputs["loss_mask"],
        inputs["verifier_last_hidden_states"], inputs["document_ids"], inputs.get("position_ids"),
        max_anchors=max_anchors,
    )
    block, from_anchor = model.block_size, model.config.sample_from_anchor
    predecessors = model._predecessor_ids(inputs["input_ids"], indices)
    target_ids = targets.argmax(dim=-1)
    candidates = unary.topk(model.candidate_selector.top_k, dim=-1).indices
    training_ids, target_pos, contains = selector_training_candidates(candidates, target_ids)
    scores = model.candidate_selector.score_candidates(unary, hidden, predecessors.reshape(1, -1), training_ids)

    pos_idx = (torch.arange(unary.shape[1], device=unary.device) % block).unsqueeze(0)
    if per_position_loss_weight == "dpace":
        decay = partial(dpace_loss_decay, loss_mask=mask, block_size=block, dpace_alpha=dpace_alpha)
    else:
        decay = partial(dflash_loss_decay, gamma=gamma, sample_from_anchor=from_anchor)
    unary_loss, _ = compound_loss(unary, targets, mask, pos_idx, loss_config=loss_config, decay_fn=decay)
    selector_mask = mask * contains.to(mask.dtype) if selector_inset else mask   # objective selector_inset
    selector_loss = compute_selector_loss(scores, target_pos, selector_mask, block, gamma=gamma,
                                          per_position_loss_weight=per_position_loss_weight,
                                          dpace_alpha=dpace_alpha, sample_from_anchor=from_anchor)
    loss = unary_loss + selector_loss_alpha * selector_loss

    with torch.no_grad():
        chosen = training_ids.gather(-1, scores.detach().argmax(dim=-1, keepdim=True)).squeeze(-1)
        correct = chosen.eq(target_ids) & contains
        blocks = unary.shape[1] // block
        start = 0 if from_anchor else 1
        eal_sum, eal_total = compute_accepted_length_counts(correct.view(blocks, block)[:, start:],
                                                            mask.to(torch.bool).view(blocks, block)[:, start:])
    one = torch.ones((), device=unary.device)
    return loss, {"eal_sum": eal_sum, "eal_total": eal_total, "loss_sum": loss.detach().clone(), "loss_total": one}


def selector_prefix_loss(model: Any, inputs: dict[str, torch.Tensor], *, max_anchors: int,
                         loss_config: dict[str, Any], selector_alpha: float = 1.0,
                         floor: float = 0.1, eal_weight: float = 0.0,
                         **_ignored: Any) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from speculators.models.dflash2.metrics import selector_training_candidates

    hidden, unary, targets, mask, indices = model._backbone_forward(
        inputs["hidden_states"], inputs["input_ids"], inputs["loss_mask"],
        inputs["verifier_last_hidden_states"], inputs["document_ids"], inputs.get("position_ids"),
        max_anchors=max_anchors,
    )
    block = model.block_size
    top_k = model.candidate_selector.top_k
    predecessors = model._predecessor_ids(inputs["input_ids"], indices).reshape(1, -1)
    target_ids = targets.argmax(-1)
    candidates = unary.topk(top_k, dim=-1).indices
    training_ids, target_pos, contains = selector_training_candidates(candidates, target_ids)
    scores = model.candidate_selector.score_candidates(unary, hidden, predecessors, training_ids)

    # Per-position unary term from the configured Speculators loss(es).
    unary_terms = sum(weight * fn(unary, targets) for fn, weight in loss_config.values())  # [1,T]
    selector_terms = F.cross_entropy(scores.float().flatten(0, 1), target_pos.flatten(),
                                     reduction="none").view_as(target_pos)
    valid = mask.to(torch.float32)
    num_blocks = unary.shape[1] // block

    survive = contains.float() * scores.float().softmax(-1).gather(-1, target_pos.unsqueeze(-1)).squeeze(-1)
    survive = torch.where(valid > 0, survive, torch.ones_like(survive)).view(num_blocks, block)
    # Exclusive prefix product: weight of slot k uses slots 1..k-1 (slot 0 is the anchor).
    # survive[:, 0] is 1 (anchor slot is never valid), so reach[:, 1] == 1.
    held = survive.detach()
    reach = torch.cumprod(torch.cat([torch.ones_like(held[:, :1]), held[:, :-1]], dim=1), dim=1)
    weights = (floor + (1.0 - floor) * reach).reshape(1, -1) * valid
    denominator = weights.sum().clamp_min(1e-6)
    loss = ((unary_terms.float() + selector_alpha * selector_terms) * weights).sum() / denominator

    metrics: dict[str, torch.Tensor] = {}
    if eal_weight:
        draft = survive[:, 1:]
        block_valid = valid.view(num_blocks, block)[:, 1:].any(-1).float()
        expected = (torch.cumprod(draft, dim=1).sum(-1) * block_valid).sum() / block_valid.sum().clamp_min(1)
        loss = loss - eal_weight * expected / (block - 1)
        metrics["soft_eal"] = expected.detach() + 1.0
    metrics["mean_reach_last_slot"] = reach[:, -1].mean().detach()
    return loss, metrics


def training_loss(objective: str, model: Any, inputs: dict[str, torch.Tensor], *, max_anchors: int,
                  on_policy: bool, kwargs: dict[str, Any], selector_prefix: dict[str, Any],
                  light_metrics: bool = False):
    if objective == "upstream" and light_metrics:
        return upstream_loss_light(model, inputs, max_anchors=max_anchors, **kwargs)
    if objective == "selector_inset":
        return upstream_loss_light(model, inputs, max_anchors=max_anchors, selector_inset=True, **kwargs)
    if objective == "upstream":
        _, loss, metrics = model(**inputs, max_anchors=max_anchors, **kwargs)
        return loss, metrics
    if objective == "selector_prefix":
        return selector_prefix_loss(model, inputs, max_anchors=max_anchors,
                                    loss_config=kwargs["loss_config"],
                                    selector_alpha=kwargs["selector_loss_alpha"], **selector_prefix)
    raise ValueError(f"unknown objective {objective!r}; choose from {OBJECTIVES}")


__all__ = ["OBJECTIVES", "loss_kwargs", "selector_prefix_loss", "training_loss", "upstream_loss_light"]
