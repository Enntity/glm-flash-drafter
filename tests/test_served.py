"""CPU tests for the offline model of what the engine serves (no speculators or teacher needed).

    python -m pytest tests/test_served.py -v

* the edge weight scales the transition term only, and 1.0 is the upstream scorer;
* a chain stops at its first miss or invalid slot, and confidence is the pick's log softmax over the real candidates;
* the width rule cuts at the first draft whose survival falls below the row price, never below one draft, and never
  verifies undrafted slots;
* a tally gives the same answer however the blocks are split across ranks.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _common  # noqa: E402,F401

import torch  # noqa: E402

from enntity_drafter import served  # noqa: E402


class Selector:
    """The two calls of speculators' CandidateSelector that scoring uses."""

    def __init__(self, transition: torch.Tensor):
        self.transition = transition

    def transition_scores(self, hidden, predecessors, candidates):
        return self.transition

    def score_candidates(self, unary, hidden, predecessors, candidates):
        return unary.gather(-1, candidates) + self.transition


class EdgeScoreTests(unittest.TestCase):
    def setUp(self):
        self.unary = torch.tensor([[[3.0, 1.0, 0.0]]])
        self.candidates = torch.tensor([[[0, 1]]])
        self.selector = Selector(torch.tensor([[[0.0, 5.0]]]))

    def test_edge_scales_only_the_transition(self):
        scores = served.edge_scores(self.selector, self.unary, None, None, self.candidates, 0.6)
        self.assertTrue(torch.allclose(scores, torch.tensor([[[3.0, 1.0 + 0.6 * 5.0]]])))

    def test_full_edge_is_the_upstream_scorer(self):
        scores = served.edge_scores(self.selector, self.unary, None, None, self.candidates, 1.0)
        self.assertTrue(torch.equal(scores, torch.tensor([[[3.0, 6.0]]])))

    def test_edge_can_change_the_pick(self):
        picks = [served.edge_scores(self.selector, self.unary, None, None, self.candidates, edge).argmax(-1).item()
                 for edge in (0.3, 1.0)]
        self.assertEqual(picks, [0, 1])


class ChainTests(unittest.TestCase):
    block = 4

    def chain(self, picks: list[list[int]], targets: list[list[int]], valid: list[list[int]]):
        """Blocks of slots whose candidates are [pick, other], so the pick scores 2 against 0."""

        ids = torch.tensor(picks).reshape(1, -1)
        candidates = torch.stack([ids, ids + 1000], dim=-1)
        scores = torch.zeros_like(candidates, dtype=torch.float32)
        scores[..., 0] = 2.0
        return served.chain_outcomes(scores, candidates, torch.tensor(targets).reshape(1, -1),
                                     torch.tensor(valid).reshape(1, -1), self.block, start=1)

    def test_chain_stops_at_the_first_miss_and_skips_the_anchor_slot(self):
        outcome = self.chain([[9, 1, 2, 3], [9, 1, 7, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 1, 1], [0, 1, 1, 1]])
        self.assertEqual(outcome.accepted.tolist(), [3, 1])
        self.assertEqual(outcome.drafted.tolist(), [3, 3])

    def test_invalid_slot_ends_the_chain_and_a_block_with_nothing_is_dropped(self):
        outcome = self.chain([[9, 1, 2, 3], [9, 1, 2, 3]], [[0, 1, 2, 3], [0, 1, 2, 3]], [[0, 1, 0, 1], [0, 0, 0, 0]])
        self.assertEqual(outcome.accepted.tolist(), [1])
        self.assertEqual(outcome.drafted.tolist(), [1])

    def test_confidence_is_the_log_softmax_of_the_pick(self):
        outcome = self.chain([[9, 1, 2, 3]], [[0, 1, 2, 3]], [[0, 1, 1, 1]])
        expected = math.log(math.exp(2.0) / (math.exp(2.0) + 1.0))
        self.assertAlmostEqual(outcome.confidence[0, 0].item(), expected, places=5)


class WidthRuleTests(unittest.TestCase):
    @staticmethod
    def outcome(confidence: list[list[float]], accepted: list[int], drafted: list[int] | None = None):
        return served.Outcome(torch.tensor(confidence), torch.tensor(accepted),
                              torch.tensor(drafted if drafted else [len(confidence[0])] * len(confidence)))

    def test_cuts_unsure_drafts_and_keeps_sure_ones(self):
        sure, unsure = -0.001, -2.0
        blocks = [[sure, sure, sure]] * 90 + [[unsure, unsure, unsure]] * 10
        summary = served.summarize(self.outcome(blocks, [3] * 90 + [0] * 10))
        self.assertAlmostEqual(summary["rows"], 0.9 * 3 + 0.1 * 1)
        self.assertAlmostEqual(summary["eal"], 1 + 0.9 * 3)
        self.assertAlmostEqual(summary["tok_step"], 1 + 0.9 * 3)
        self.assertAlmostEqual(summary["value"], summary["tok_step"] - served.ROW_PRICE * summary["rows"])

    def test_a_cut_draft_cannot_be_accepted(self):
        # the unsure second draft is usually wrong, but this block got it right: the cut still loses it
        sure, unsure = -0.001, -2.0
        blocks = [[sure, unsure]] * 19 + [[sure, unsure]]
        summary = served.summarize(self.outcome(blocks, [1] * 19 + [2]))
        self.assertAlmostEqual(summary["eal"], 2.05)
        self.assertAlmostEqual(summary["tok_step"], 2.0)
        self.assertAlmostEqual(summary["rows"], 1.0)

    def test_verifies_one_draft_at_least_and_only_drafted_slots(self):
        unsure, sure = -2.0, -0.001
        one = served.summarize(self.outcome([[unsure, unsure]] * 4, [0] * 4))
        self.assertEqual(one["rows"], 1.0)
        short = served.summarize(self.outcome([[sure, sure, sure]] * 4, [1] * 4, [1] * 4))
        self.assertEqual(short["rows"], 1.0)

    def test_empty_bins_never_divide_by_zero(self):
        summary = served.summarize(self.outcome([[-0.001]], [1]))
        self.assertEqual(summary["tok_step"], 2.0)

    def test_a_group_with_no_blocks_reports_nothing(self):
        tally = served.Tally((0.6,))
        tally.add(0.6, served.Outcome(torch.zeros((0, 3)), torch.zeros(0, dtype=torch.long),
                                      torch.zeros(0, dtype=torch.long)))
        self.assertEqual(tally.summary(), {})

    def test_summary_is_independent_of_the_rank_split(self):
        torch.manual_seed(0)
        confidence = -torch.rand(40, 7) * 2
        accepted = torch.randint(0, 8, (40,))
        drafted = torch.full((40,), 7)
        whole, parts = served.Tally((0.6, 1.0)), [served.Tally((0.6, 1.0)) for _ in range(3)]
        for edge in (0.6, 1.0):
            whole.add(edge, served.Outcome(confidence, accepted, drafted))
            for index, part in enumerate(parts):
                pick = torch.arange(index, 40, 3)
                part.add(edge, served.Outcome(confidence[pick], accepted[pick], drafted[pick]))
        merged = served.Tally.merged(parts)
        self.assertEqual(whole.summary(), merged.summary())

    def test_summary_keys_name_the_edge(self):
        tally = served.Tally((0.6,))
        tally.add(0.6, self.outcome([[-0.001]], [1]))
        self.assertEqual(sorted(tally.summary()), [f"served_e06_{k}" for k in sorted(("eal", "rows", "tok_step", "value"))])


if __name__ == "__main__":
    unittest.main()
