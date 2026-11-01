# SPDX-License-Identifier: Apache-2.0
"""What the serving engine does with a drafter, measured offline (``--eval-served``).

The plain chain proxy (``eal``) differs from serving in three ways, and each one has hidden real changes:

* **Edge weight.** Serving may score a candidate as ``unary + edge * transition`` with ``edge < 1``; the upstream
  selector, and so ``eal``, always uses 1.
* **Candidates.** Serving picks among the unary top-k only. Training metrics add the target as the k-th candidate when
  it is missing, which serving can never do.
* **Verify width.** Serving verifies only the drafts worth their row: a table turns the pick's confidence into the
  chance the target accepts it, the running product is the draft's survival, and the draft is cut at the first position
  whose survival falls below the price of a row. A draft that was cut is never accepted, however right it was.

``summarize`` replays that rule. Its table is fitted on the blocks it scores, so it is the rule at its best: it shows
what a drafter's confidence can buy, not one serving session.

The confidence bin edges are the upper ten of knapcio's measured table (MIT,
https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4, ``overlay/glm_bav_table_seg.json``).
"""

from __future__ import annotations

from typing import Any, Iterable, NamedTuple

import torch

#: Edge weights ``--eval-served`` scores: the two the engine ships besides the upstream 1.
EDGES = (0.6, 0.8, 1.0)
#: Bin ``b`` holds log-confidences in ``[CONF_EDGES[b - 1], CONF_EDGES[b])``.
CONF_EDGES = (-1.5, -1.0, -0.7, -0.5, -0.35, -0.22, -0.12, -0.06, -0.03, -0.01)
#: Tokens a verified row must be worth.
ROW_PRICE = 0.3


class Outcome(NamedTuple):
    """Selector chains, one per block: pick confidence [B, D], accepted drafts [B], drafted slots [B]."""

    confidence: torch.Tensor
    accepted: torch.Tensor
    drafted: torch.Tensor


def edge_scores(selector: Any, unary: torch.Tensor, hidden: torch.Tensor, predecessors: torch.Tensor,
                candidates: torch.Tensor, edge: float) -> torch.Tensor:
    """Candidate scores with the selector's transition term scaled by ``edge``; 1 is the upstream scorer."""

    if edge == 1.0:
        return selector.score_candidates(unary, hidden, predecessors, candidates)
    picked = unary.gather(-1, candidates)
    return picked + edge * selector.transition_scores(hidden, predecessors, candidates).to(picked.dtype)


def chain_outcomes(scores: torch.Tensor, candidates: torch.Tensor, targets: torch.Tensor, valid: torch.Tensor,
                   block: int, start: int) -> Outcome:
    """Greedy selector chains from candidate ``scores`` [1, T, K]; a chain ends at its first miss or invalid slot."""

    picks = scores.argmax(-1, keepdim=True)
    chosen = candidates.gather(-1, picks).squeeze(-1)
    confidence = scores.float().log_softmax(-1).gather(-1, picks).squeeze(-1)
    blocks = chosen.shape[1] // block

    def slots(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.view(blocks, block)[:, start:]

    usable = slots(valid.to(torch.bool))
    drafted = usable.long().cumprod(1).sum(1)
    accepted = (slots(chosen.eq(targets)) & usable).long().cumprod(1).sum(1)
    kept = drafted > 0
    return Outcome(slots(confidence)[kept].cpu(), accepted[kept].cpu(), drafted[kept].cpu())


def summarize(outcome: Outcome, price: float = ROW_PRICE) -> dict[str, float]:
    """``eal`` of the chains, then ``tok_step`` and verified draft ``rows`` under the width rule at ``price``, and
    ``value``, what that rule maximizes (tokens less the price of the rows)."""

    confidence, accepted, drafted = outcome
    bins = torch.bucketize(confidence, torch.tensor(CONF_EDGES), right=True)
    position = torch.arange(confidence.shape[1])
    reached = (accepted[:, None] >= position) & (drafted[:, None] > position)
    won = accepted[:, None] > position
    seen = torch.bincount(bins[reached], minlength=len(CONF_EDGES) + 1).to(torch.float64)
    hits = torch.bincount(bins[reached & won], minlength=len(CONF_EDGES) + 1).to(torch.float64)
    rate = torch.where(seen > 0, hits / seen.clamp_min(1), hits.sum() / seen.sum().clamp_min(1))
    keep = rate[bins].cumprod(1) >= price
    keep[:, 0] = True
    width = keep.long().cumprod(1).sum(1).minimum(drafted)
    blocks = len(accepted)
    tokens = (1 + accepted.minimum(width)).sum().item() / blocks
    rows = width.sum().item() / blocks
    return {"eal": (1 + accepted).sum().item() / blocks, "tok_step": tokens, "rows": rows,
            "value": tokens - price * rows}


class Tally:
    """Chains per edge weight; blocks stay whole until the end because the width rule's table needs all of them."""

    def __init__(self, edges: Iterable[float]):
        self.parts: dict[float, list[Outcome]] = {edge: [] for edge in edges}

    def add(self, edge: float, outcome: Outcome) -> None:
        if len(outcome.accepted):
            self.parts[edge].append(outcome)

    def score(self, model: Any, input_ids: torch.Tensor, backbone: tuple[torch.Tensor, ...]) -> None:
        """One row's chains at every edge, from a ``_backbone_forward`` result."""

        hidden, unary, targets, mask, indices = backbone
        selector = model.candidate_selector
        predecessors = model._predecessor_ids(input_ids, indices).reshape(1, -1)
        candidates = unary.topk(selector.top_k, dim=-1).indices
        target_ids = targets.argmax(dim=-1)
        start = 0 if model.config.sample_from_anchor else 1
        for edge in self.parts:
            scores = edge_scores(selector, unary, hidden, predecessors, candidates, edge)
            self.add(edge, chain_outcomes(scores, candidates, target_ids, mask, model.block_size, start))

    @classmethod
    def merged(cls, tallies: Iterable["Tally"]) -> "Tally":
        tallies = list(tallies)
        result = cls(tallies[0].parts)
        for tally in tallies:
            for edge, parts in tally.parts.items():
                result.parts[edge].extend(parts)
        return result

    def summary(self) -> dict[str, float]:
        result: dict[str, float] = {}
        for edge, parts in self.parts.items():
            if parts:
                whole = Outcome(*(torch.cat(column) for column in zip(*parts)))
                result.update({f"served_e{round(edge * 10):02d}_{k}": v for k, v in summarize(whole).items()})
        return result


__all__ = ["CONF_EDGES", "EDGES", "Outcome", "ROW_PRICE", "Tally", "chain_outcomes", "edge_scores", "summarize"]
