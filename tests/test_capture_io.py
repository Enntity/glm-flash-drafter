"""CPU tests for ``enntity_drafter.capture_io`` and the trainer's capture path.

    python -m pytest tests/test_capture_io.py -v

Synthetic captures in both Atlas formats (pilot chunks; fast per-rank SP
segments with a per-request window) are written from known tensors, so every
read can be checked exactly:

* shard merging, tiling/overlap/id-disagreement refusals, metadata joins;
* window/anchor rules (full window or true sequence start, generated span,
  last block), crops, duplicates, held-out splits, partition copy lists;
* the packed sample equals the full row bit for bit where it is read, and the
  model's loss/metrics on the packed sample equal the full-row forward with
  the same anchors (the window-as-document packing is exact);
* the agreement mask; DataLoader workers reproduce the in-process stream;
* a 2-rank data-parallel tiny run on a capture index learns, and resume is exact.

``LoaderRealDataTests`` reads real captures from ``$DRAFTER_LOADER_TEST_ROOT``
(``index/index.json`` + the rows; skipped when unset) and checks the packed
samples against an independent whole-file reader.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from _common import REPO, TINY, TRAIN_MODULE, launcher  # noqa: E402

from enntity_drafter import capture_io as cd  # noqa: E402
from enntity_drafter import data as sd  # noqa: E402
from enntity_drafter import fp8 as fp8_convert  # noqa: E402
from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter import train as ts  # noqa: E402
from enntity_drafter.model import install_fixed_anchor_mask  # noqa: E402
from enntity_drafter.objectives import loss_kwargs  # noqa: E402

LAYERS = [5, 14, 24, 33, 42]
WINDOW = 16
REAL = Path(os.environ.get("DRAFTER_LOADER_TEST_ROOT", "/nonexistent"))


def write_bin(path: Path, tensor: torch.Tensor) -> None:
    path.write_bytes(tensor.contiguous().bfloat16().view(torch.uint16).numpy().tobytes())


def write_pilot(directory: Path, slots: torch.Tensor, ids: list[int], cuts: list[int]) -> None:
    """slots [6, T, H]; chunks between consecutive cut points (a suffix if cuts[0] > 0)."""

    directory.mkdir(parents=True)
    for start, end in zip(cuts, cuts[1:]):
        name = directory / f"p{start:09d}-n{end - start}"
        write_bin(name.with_suffix(".bin"), slots[:, start:end])
        name.with_suffix(".json").write_text(json.dumps({
            "schema": cd.PILOT_SCHEMA, "pos0": start, "rows": end - start, "hidden_size": slots.shape[-1],
            "dtype": "bf16", "layers": LAYERS, "token_ids": ids[start:end], "seq_tokens": len(ids)}))


def write_fast(directory: Path, slots: torch.Tensor, ids: list[int], start: int, chunk: int,
               sp: bool = True) -> None:
    """Window [start, T); SP chunks: rank 1 writes the lower half of L rows, rank 0 the upper and all F."""

    directory.mkdir(parents=True)
    hidden = slots.shape[-1]

    def segment(rank: int, kind: str, a: int, b: int, tokens: bool = True) -> None:
        name = directory / f"r{rank}-{kind}-p{a:09d}-n{b - a}"
        data = slots[:5, a:b] if kind == "L" else slots[5:, a:b]
        write_bin(name.with_suffix(".bin"), data)
        name.with_suffix(".json").write_text(json.dumps({
            "schema": cd.FAST_SCHEMA, "rank": rank, "kind": "layers" if kind == "L" else "final",
            "layers": LAYERS if kind == "L" else [], "pos0": a, "rows": b - a, "hidden_size": hidden,
            "dtype": "bf16", "sp": sp, "from": start, "seq_tokens": len(ids),
            "token_ids": ids[a:b] if tokens else None}))

    total = len(ids)
    for a in range(start, total, chunk):
        b = min(total, a + chunk)
        segment(0, "F", a, b)
        middle = (a + b) // 2 if sp and b - a > 4 else a
        if middle > a:
            segment(1, "L", a, middle, tokens=False)
        segment(0, "L", middle, b)


def synthetic_capture(root: Path, cfg: dict, *, train_rows: int = 24, heldout_rows: int = 4,
                      length: int = 96) -> tuple[dict[str, dict], list[dict]]:
    """Synthetic learnable rows written as pilot (even) and fast (odd) captures + metadata."""

    rows = sd.synthetic_rows(cfg, train_rows=train_rows, heldout_rows=heldout_rows, length=length)
    truth, meta = {}, []
    for index, row in enumerate(rows):
        name = row["id"]
        slots = torch.cat([row["hidden_states"].permute(1, 0, 2), row["final_hidden"][None]], 0)
        ids = row["token_ids"].tolist()
        P = 24 + (index % 3)
        if index % 2 == 0:
            write_pilot(root / "pilot" / name, slots, ids, [0, 40, 41, length])
        else:
            write_fast(root / "fast" / name, slots, ids, P - WINDOW, chunk=32)
        truth[name] = {"slots": slots.bfloat16(), "ids": ids, "P": P}
        meta.append({"name": name, "P": P, "ids": ids, "source": "synthetic",
                     "kind": "heldout" if row["split"] == "heldout" else "api"})
    return truth, meta


def tiny_window_config(directory: Path) -> Path:
    cfg = json.loads(TINY.read_text())
    cfg["sliding_window"] = WINDOW
    path = directory / "tiny-window.json"
    path.write_text(json.dumps(cfg))
    return path


class CaptureFixture:
    @classmethod
    def build(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="capture-dataset-")
        cls.root = Path(cls.tmp.name)
        cls.config = tiny_window_config(cls.root)
        cls.cfg = sm.load_config(cls.config)
        cls.truth, meta = synthetic_capture(cls.root / "data", cls.cfg)
        (cls.root / "meta.jsonl").write_text("".join(json.dumps(m) + "\n" for m in meta))
        heldout = [m for m in meta if m["kind"] == "heldout"]
        (cls.root / "heldout.jsonl").write_text("".join(json.dumps({"id": m["name"]}) + "\n" for m in heldout[:3]))
        cls.index = cd.build_index(cls.root / "data", ["pilot"], ["fast"], [str(cls.root / "meta.jsonl")],
                                   str(cls.root / "heldout.jsonl"), cls.root / "index", window=WINDOW,
                                   block=8, layers=LAYERS, pilot_as_data=True)
        sd.write_synthetic_target_dir(cls.root / "target", cls.cfg)


class IndexAndSampleTests(unittest.TestCase, CaptureFixture):
    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_index_rows_splits_and_anchor_ranges(self):
        rows = {r["name"]: r for r in self.index["rows"]}
        self.assertEqual(len(rows), 28)
        self.assertEqual(self.index["skipped"], {})
        splits = [r["split"] for r in rows.values()]
        self.assertEqual((splits.count("train"), splits.count("heldout"), splits.count("heldout_extra")), (24, 3, 1))
        for name, row in rows.items():
            P = self.truth[name]["P"]
            if row["format"] == "pilot":
                self.assertEqual((row["lo"], row["a_min"]), (0, P))   # true sequence start: no window needed
            else:
                self.assertEqual((row["lo"], row["a_min"]), (P - WINDOW, P))  # exactly the window kept
                self.assertEqual(row["ranks"], [0, 1])
            self.assertEqual(row["a_max"], 96 - 8)

    def test_partial_pilot_copy_needs_full_window(self):
        name = "train-000"
        truth = self.truth[name]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_pilot(data / "pilot" / name, truth["slots"].float(), truth["ids"], [41, 96])   # suffix only
            (data / "meta.jsonl").write_text(json.dumps({"name": name, "P": 50}) + "\n")
            index = cd.build_index(data, ["pilot"], [], [str(data / "meta.jsonl")], None, data / "idx",
                                   window=WINDOW, block=8, layers=LAYERS)
            row = index["rows"][0]
            self.assertEqual((row["lo"], row["a_min"], row["a_max"]), (41, 41 + WINDOW, 88))
            write_pilot(data / "pilot2" / name, truth["slots"].float(), truth["ids"], [41, 96])
            (data / "meta.jsonl").write_text(json.dumps({"name": name, "P": 45}) + "\n")
            index = cd.build_index(data, ["pilot2"], [], [str(data / "meta.jsonl")], None, data / "idx2",
                                   window=WINDOW, block=8, layers=LAYERS)
            self.assertEqual(index["rows"][0]["a_min"], 57)        # P=45 anchors lack the window until 57
            self.assertEqual(cd.eligible_range(41, 60, 45, 16, 8), (57, 52))   # empty: skipped, never padded

    def test_fast_shard_merge_refusals(self):
        truth = self.truth["train-001"]
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "t"
            write_fast(directory, truth["slots"].float(), truth["ids"], 10, chunk=32)
            scan = cd.scan_fast_dir(directory, "t")
            self.assertEqual((scan["lo"], scan["hi"], scan["ids"]), (10, 96, truth["ids"][10:]))
            victim = sorted(directory.glob("r1-L-*.json"))[0]
            meta = json.loads(victim.read_text())
            meta["pos0"] += 1
            victim.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "missing or size|do not tile"):
                cd.scan_fast_dir(directory, "t")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "t"
            write_fast(directory, truth["slots"].float(), truth["ids"], 10, chunk=32)
            victim = sorted(directory.glob("r0-L-*.json"))[0]
            meta = json.loads(victim.read_text())
            meta["token_ids"][0] += 1
            victim.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "disagree"):
                cd.scan_fast_dir(directory, "t")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp) / "t"
            write_fast(directory, truth["slots"].float(), truth["ids"], 10, chunk=32)
            next(directory.glob("r0-F-*.bin")).rename(directory / "x.part")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                cd.scan_fast_dir(directory, "t")

    def test_metadata_mismatch_and_duplicates(self):
        name = "train-002"
        truth = self.truth[name]
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            write_pilot(data / "a" / name, truth["slots"].float(), truth["ids"], [41, 96])     # no anchors (P=24)
            write_fast(data / "b" / name, truth["slots"].float(), truth["ids"], 24 - WINDOW, chunk=32)
            write_pilot(data / "c" / "other", truth["slots"].float(), truth["ids"], [0, 96])
            bad = list(truth["ids"])
            bad[3] += 1
            (data / "meta.jsonl").write_text(json.dumps({"name": name, "P": 24}) + "\n"
                                             + json.dumps({"name": "other", "P": 24, "ids": bad}) + "\n")
            index = cd.build_index(data, ["a", "c"], ["b"], [str(data / "meta.jsonl")], None, data / "idx",
                                   window=WINDOW, block=8, layers=LAYERS)
            self.assertEqual([r["dir"] for r in index["rows"]], [f"b/{name}"])   # more anchors wins
            self.assertEqual(index["skipped"], {"ids_mismatch": 1, "duplicate": 1})

    def test_manifest_allowlist_and_pilot_offserving(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            names = ["train-001", "train-003", "train-005"]
            for name in names:
                truth = self.truth[name]
                write_fast(data / "fast-rows" / "round1-x" / "p0" / name, truth["slots"].float(), truth["ids"],
                           truth["P"] - WINDOW, chunk=32)
            truth = self.truth["train-001"]   # an older capture of the same row with more anchors (from 0)
            write_fast(data / "fast-rows" / "round1-old" / "p0" / "train-001", truth["slots"].float(), truth["ids"],
                       0, chunk=32)
            truth = self.truth["train-000"]
            write_pilot(data / "pilot-rows" / "train-000", truth["slots"].float(), truth["ids"], [0, 96])
            manifest = [  # generated count only (P = T - generated); one stale stored count; one row not on disk
                {"name": "train-001", "generated": 96 - self.truth["train-001"]["P"],
                 "dir": "fast-rows/round1-x/p0/train-001",
                 "source": "code", "kind": "api"},
                {"tag": "train-003", "P": self.truth["train-003"]["P"], "stored": 3, "source": "agentic"},
                {"name": "train-005", "P": self.truth["train-005"]["P"], "prompt_id": "prompt-5"},
                {"name": "never-captured", "P": 10},
            ]
            (data / "manifest.json").write_text(json.dumps({"rows": manifest}))
            (data / "meta.jsonl").write_text(json.dumps({"name": "train-000", "P": 24}) + "\n")
            (data / "heldout.jsonl").write_text(json.dumps({"id": "prompt-5"}) + "\n")
            index = cd.build_index(data, ["pilot-rows"], ["fast-rows/round1-*/p*"],
                                   [str(data / "meta.jsonl")], str(data / "heldout.jsonl"), data / "idx",
                                   window=WINDOW, block=8,
                                   layers=LAYERS, manifest=[str(data / "manifest.json")])
            rows = {r["name"]: r for r in index["rows"]}
            self.assertEqual(sorted(rows), ["train-001", "train-005"])   # pilot row not in the allowlist
            self.assertEqual(rows["train-001"]["dir"], "fast-rows/round1-x/p0/train-001")   # listed dir wins
            self.assertEqual(rows["train-001"]["P"], self.truth["train-001"]["P"])
            self.assertEqual((rows["train-001"]["split"], rows["train-005"]["split"]), ("train", "heldout"))
            self.assertEqual(index["skipped"], {"manifest_mismatch": 1, "manifest_rows_not_indexed": 2})
            index = cd.build_index(data, ["pilot-rows"], [], [str(data / "meta.jsonl")], None, data / "idx2",
                                   window=WINDOW, block=8, layers=LAYERS)
            self.assertEqual([r["split"] for r in index["rows"]], ["pilot_offserving"])

    def test_meta_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = Path(tmp) / "records.jsonl"
            records.write_text(json.dumps({"name": "x_s0", "id": "x#s0", "tokens": 10, "completion_tokens": 4,
                                           "prompt_ids_used": [1, 2, 3, 4, 5]}) + "\n"
                               + json.dumps({"id": "y", "error": "boom"}) + "\n")
            items = Path(tmp) / "items.jsonl"
            items.write_text(json.dumps({"id": "x#s0", "prompt_ids": [9, 9, 9, 9, 9, 9], "completion_ids": [6, 7, 8, 9],
                                         "on_policy": False, "kind": "api"}) + "\n")
            meta = cd.load_meta([records, items])
            self.assertEqual(meta["x_s0"]["P"], 5)                     # Atlas re-tokenized prompt wins
            self.assertEqual(meta["x_s0"]["ids"], [1, 2, 3, 4, 5, 6, 7, 8, 9])
            self.assertFalse(meta["x_s0"]["on_policy"])
            self.assertNotIn("y", meta)

    def test_packed_sample_matches_rows(self):
        index = cd.CaptureIndex(self.root / "index" / "index.json", self.root / "data")
        reader = cd.SegmentReader(self.root / "data", index.hidden)
        for row in index.rows:
            truth = self.truth[row["name"]]
            rng = random.Random(row["name"])
            anchors = cd.choose_anchors(row, rng, 6, 0)
            self.assertTrue(all(row["a_min"] <= a <= row["a_max"] for a in anchors))
            sample = cd.build_sample(index, reader, row, anchors)
            inputs = sample["inputs"]
            positions = inputs["position_ids"][0]
            hidden = inputs["hidden_states"][0].view(-1, 5, index.hidden)
            self.assertTrue(torch.equal(hidden, truth["slots"][:5, positions].permute(1, 0, 2)))
            self.assertEqual(inputs["input_ids"][0].tolist(), [truth["ids"][p] for p in positions.tolist()])
            self.assertEqual(inputs["loss_mask"][0].tolist(), [p >= truth["P"] for p in positions.tolist()])
            final = inputs["verifier_last_hidden_states"][0]
            for compact, a in zip(sample["anchors"], anchors):
                self.assertEqual(int(positions[compact]), a)
                need = torch.arange(compact - 1, compact + 7)
                self.assertTrue(torch.equal(final[need], truth["slots"][5, a - 1: a + 7]))
                window = positions[(inputs["document_ids"][0] == inputs["document_ids"][0, compact])]
                self.assertLessEqual(int(window.min()), max(row["lo"], a - WINDOW))   # full window present
            documents = inputs["document_ids"][0]
            breaks = (documents[1:] != documents[:-1]).nonzero().flatten()
            for b in breaks.tolist():
                self.assertGreater(int(positions[b + 1]), int(positions[b]) + 1)   # documents never touch
            self.assertEqual(sample["supervised"], sum(min(7, 96 - a - 1) for a in anchors))

    def test_crop_bounds_anchor_draw(self):
        row = {"a_min": 100, "a_max": 10_000}
        for seed in range(20):
            anchors = cd.choose_anchors(row, random.Random(seed), 512, 1024)
            self.assertEqual(len(anchors), 512)
            self.assertLess(anchors[-1] - anchors[0], 1024)
            self.assertEqual(anchors, sorted(set(anchors)))
        self.assertEqual(len(cd.choose_anchors({"a_min": 5, "a_max": 9}, random.Random(0), 512, 0)), 5)
        self.assertEqual(cd.choose_anchors({"a_min": 9, "a_max": 5}, random.Random(0), 512, 0), [])

    def test_packing_is_exact_against_full_row_forward(self):

        torch.manual_seed(0)
        model = sm.build_model(self.cfg)
        sm.install_frozen_target(model, {k: v.float() for k, v in sd.synthetic_target(self.cfg).items()},
                                 "pre-final-norm")
        sm.scratch_init(model, self.cfg, 3)
        with torch.no_grad():   # make the selector / convolutions non-trivial
            model.candidate_selector.successor_codebook.normal_(0, 0.02)
        model.eval()
        kwargs = loss_kwargs("cpu", "kl_div", "fixed-exp-decay", 4.0, 1.0, 0.5)
        index = cd.CaptureIndex(self.root / "index" / "index.json", self.root / "data")
        reader = cd.SegmentReader(self.root / "data", index.hidden)
        compared = 0
        for row in index.rows[:6]:
            truth = self.truth[row["name"]]
            anchors = cd.choose_anchors(row, random.Random(7), 5, 0)
            sample = cd.build_sample(index, reader, row, anchors)
            lo, hi = row["lo"], row["hi"]
            full = {
                "hidden_states": truth["slots"][:5, lo:hi].permute(1, 0, 2).reshape(1, hi - lo, -1).float(),
                "input_ids": torch.tensor(truth["ids"][lo:hi]).unsqueeze(0),
                "loss_mask": (torch.arange(lo, hi) >= truth["P"]).unsqueeze(0),
                "verifier_last_hidden_states": truth["slots"][5, lo:hi].unsqueeze(0).float(),
                "document_ids": torch.zeros(1, hi - lo, dtype=torch.long),
                "position_ids": torch.arange(lo, hi).unsqueeze(0),
            }
            results = []
            for inputs, chosen in ((full, [a - lo for a in anchors]),
                                   ({k: v.float() if v.is_floating_point() else v
                                     for k, v in sample["inputs"].items()}, sample["anchors"])):
                old = install_fixed_anchor_mask(model, chosen)
                try:
                    with torch.no_grad():
                        _, loss, metrics = model(**inputs, max_anchors=len(chosen), **kwargs)
                finally:
                    model._build_attention_mask = old
                results.append((float(loss), {k: float(v.sum()) for k, v in metrics.items()}))
            self.assertAlmostEqual(results[0][0], results[1][0], places=5)
            for key, value in results[0][1].items():
                self.assertAlmostEqual(value, results[1][1][key], places=4, msg=key)
            compared += 1
        self.assertEqual(compared, 6)

    def test_bundle_metrics_are_the_sum_of_its_rows(self):
        """Several rows packed into one micro-step (round 2b ``--pack-rows``): every metric sum and count of the
        bundle's forward equals the sum over the rows' own samples (each window is its own document)."""


        torch.manual_seed(0)
        model = sm.build_model(self.cfg)
        sm.install_frozen_target(model, {k: v.float() for k, v in sd.synthetic_target(self.cfg).items()},
                                 "pre-final-norm")
        sm.scratch_init(model, self.cfg, 3)
        with torch.no_grad():
            model.candidate_selector.successor_codebook.normal_(0, 0.02)
        model.eval()
        kwargs = loss_kwargs("cpu", "kl_div", "fixed-exp-decay", 4.0, 1.0, 0.5)
        index = cd.CaptureIndex(self.root / "index" / "index.json", self.root / "data")
        reader = cd.SegmentReader(self.root / "data", index.hidden)

        def run(sample):
            inputs = {k: v.float() if v.is_floating_point() else v for k, v in sample["inputs"].items()}
            old = install_fixed_anchor_mask(model, sample["anchors"])
            try:
                with torch.no_grad():
                    _, _, metrics = model(**inputs, max_anchors=len(sample["anchors"]), **kwargs)
            finally:
                model._build_attention_mask = old
            return {k: float(v.sum()) for k, v in metrics.items()}

        parts = [(row, cd.choose_anchors(row, random.Random(k), 5, 0)) for k, row in enumerate(index.rows[1:6:2])]
        bundle = cd.build_bundle(index, reader, parts)
        alone = [cd.build_sample(index, reader, row, anchors) for row, anchors in parts]
        self.assertEqual(bundle["context_rows"], sum(a["context_rows"] for a in alone))
        self.assertEqual(bundle["supervised"], sum(a["supervised"] for a in alone))
        self.assertEqual(len(set(bundle["inputs"]["document_ids"][0].tolist())),
                         sum(len(set(a["inputs"]["document_ids"][0].tolist())) for a in alone))
        packed, separate = run(bundle), [run(a) for a in alone]
        blocks = [s["eal_total"] for s in separate]
        for key in packed:
            if packed[key.rsplit("_", 1)[0] + "_total"] == 1:   # per-forward means (losses): anchor-weighted
                expect = sum(s[key] * b for s, b in zip(separate, blocks)) / sum(blocks)
            else:
                expect = sum(s[key] for s in separate)
            self.assertAlmostEqual(packed[key], expect, places=3, msg=key)

    def test_agreement_mask(self):
        model = sm.build_model(self.cfg)
        sm.install_frozen_target(model, {k: v.float() for k, v in sd.synthetic_target(self.cfg).items()},
                                 "pre-final-norm")
        index = cd.CaptureIndex(self.root / "index" / "index.json", self.root / "data")
        reader = cd.SegmentReader(self.root / "data", index.hidden)
        row = index.rows[0]
        anchors = cd.choose_anchors(row, random.Random(1), 4, 0)
        sample = cd.build_sample(index, reader, row, anchors)
        inputs = {k: v.float() if v.is_floating_point() else v for k, v in sample["inputs"].items()}
        slots = [c + k for c in sample["anchors"] for k in range(1, 8)]
        kept = cd.agreement_keep(model, inputs, sample["anchors"], "argmax")
        self.assertTrue(bool(kept[0, slots].all()))          # synthetic final[i-1] predicts token i exactly
        corrupted = dict(inputs, input_ids=inputs["input_ids"].clone())
        corrupted["input_ids"][0, slots[3]] = (corrupted["input_ids"][0, slots[3]] + 1) % 63
        kept = cd.agreement_keep(model, corrupted, sample["anchors"], "argmax")
        self.assertEqual(int((~kept[0, slots]).sum()), 1)
        self.assertFalse(bool(kept[0, slots[3]]))
        kept = cd.agreement_keep(model, corrupted, sample["anchors"], "minprob:1e-30")
        self.assertTrue(bool(kept[0, slots].all()))

    def test_stream_is_deterministic_and_workers_match(self):
        kwargs = dict(rank=0, world=2, seed=5, max_anchors=8, crop=32)
        stream = cd.CaptureTrainStream(self.root / "index" / "index.json", self.root / "data", **kwargs)
        again = cd.CaptureTrainStream(self.root / "index" / "index.json", self.root / "data", **kwargs)
        direct = [stream[i] for i in range(6)]
        self.assertEqual([s["anchors"] for s in direct], [again[i]["anchors"] for i in range(6)])
        loaded = list(cd.train_loader(again, 0, 6, workers=2, pin_memory=False))
        for a, b in zip(direct, loaded):
            self.assertEqual((a["row"], a["anchors"]), (b["row"], b["anchors"]))
            self.assertTrue(torch.equal(a["inputs"]["hidden_states"], b["inputs"]["hidden_states"]))
        other = cd.CaptureTrainStream(self.root / "index" / "index.json", self.root / "data",
                                      **dict(kwargs, rank=1))
        self.assertTrue(all(r["split"] == "train" for r in stream.rows))
        epoch = len(stream.epoch_items)
        names = [stream.row_for(i)["name"] for i in range(epoch // 2)] + \
                [other.row_for(i)["name"] for i in range(epoch // 2)]
        self.assertEqual(sorted(names), sorted(stream.rows[i]["name"] for i in stream.epoch_items))

    def test_partition(self):
        index = cd.partition_index(json.loads((self.root / "index" / "index.json").read_text()), 2)
        ranks = [r["rank"] for r in index["rows"] if r["split"] == "train"]
        self.assertEqual(sorted(set(ranks)), [0, 1])
        self.assertLessEqual(abs(index["rank_anchor_load"][0] - index["rank_anchor_load"][1]), 80)
        heldout_files = {s[0] for r in index["rows"] if r["split"] != "train" for s in r["layer_segs"]}
        for rank in ("0", "1"):
            self.assertTrue(heldout_files <= set(index["copy_lists"][rank]))
        raw = json.loads((self.root / "index" / "index.json").read_text())
        heldout_gb = sum(r["bytes"] for r in raw["rows"] if r["split"] != "train") / 1e9
        train_gb = sum(r["bytes"] for r in raw["rows"] if r["split"] == "train") / 1e9
        capped = cd.partition_index(raw, 2, {0: heldout_gb + 0.25 * train_gb})
        self.assertLessEqual(capped["rank_gb_upper_bound"][0], heldout_gb + 0.25 * train_gb + 1e-9)
        self.assertGreater(capped["rank_gb_upper_bound"][1], capped["rank_gb_upper_bound"][0])
        with self.assertRaises(ValueError):
            cd.partition_index(json.loads((self.root / "index" / "index.json").read_text()), 2,
                               {0: heldout_gb, 1: heldout_gb})
        path = self.root / "index" / "index-w2.json"
        path.write_text(json.dumps(index))
        streams = [cd.CaptureTrainStream(path, self.root / "data", rank=r, world=2, seed=1, max_anchors=8, crop=0)
                   for r in (0, 1)]
        self.assertFalse({r["name"] for r in streams[0].rows} & {r["name"] for r in streams[1].rows})
        for r, stream in enumerate(streams):
            needed = {s[0] for row in stream.rows for s in row["layer_segs"]}
            self.assertTrue(needed <= set(index["copy_lists"][str(r)]))


def two_source_index(root: Path, name: str = "index-2src.json") -> Path:
    """The fixture index with train rows split into sources ``a`` (even rows) and ``b`` (odd rows)."""

    raw = json.loads((root / "index" / "index.json").read_text())
    train = [r for r in raw["rows"] if r["split"] == "train"]
    for k, row in enumerate(train):
        row["source"] = "a" if k % 2 == 0 else "b"
    path = root / "index" / name
    path.write_text(json.dumps(raw))
    return path


class BalancedStreamTests(unittest.TestCase, CaptureFixture):
    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_one_source_epoch_covers_every_position_once(self):
        path = two_source_index(self.root)
        stream = cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=8, crop=20,
                                       source_weights="a=0.7,b=0.3")
        plan = stream.balanced
        for source in ("a", "b"):
            per_epoch = len(plan["items"][source])
            seen: dict[str, list[int]] = {}
            k = 0
            item = 0
            while k < per_epoch:   # walk the schedule until this source finished one epoch
                stream._schedule_to(item)
                if plan["sources"][plan["schedule"][item][0]] == source:
                    sample = stream[item]
                    self.assertLessEqual(len(sample["anchors"]), 8)
                    self.assertLessEqual(sample["context_rows"], 20 + WINDOW + 8)
                    (row, anchors), = stream._balanced_item(item)
                    seen.setdefault(row["name"], []).extend(anchors)
                    k += 1
                item += 1
            rows = [r for r in stream.rows if r["source"] == source]
            self.assertEqual(set(seen), {r["name"] for r in rows})
            for row in rows:   # every eligible position exactly once
                self.assertEqual(sorted(seen[row["name"]]), list(range(row["a_min"], row["a_max"] + 1)))
        report = stream.source_report(400)
        self.assertAlmostEqual(report["a"]["anchor_share"], 0.7, delta=0.03)
        self.assertAlmostEqual(report["b"]["anchor_share"], 0.3, delta=0.03)
        again = cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=8, crop=20,
                                      source_weights="a=0.7,b=0.3")
        loaded = list(cd.train_loader(again, 5, 15, workers=3, pin_memory=False))
        for i, sample in zip(range(5, 15), loaded):
            self.assertEqual((sample["row"], sample["anchors"]), (stream[i]["row"], stream[i]["anchors"]))
        with self.assertRaises(ValueError):   # a source without weight is refused
            cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=8, crop=20,
                                  source_weights="a=1")
        groups = cd.parse_source_weights("a+b=0.5,c=0.5")
        self.assertEqual(groups, [(("a", "b"), 0.5), (("c",), 0.5)])

    def test_packed_bundles_cover_every_position_once_and_caps_hold(self):
        path = two_source_index(self.root)
        stream = cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=200, crop=256,
                                       source_weights="a=0.7,b=0.3", pack=True)
        plain = cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=200, crop=256,
                                      source_weights="a=0.7,b=0.3")
        plan = stream.balanced
        for source in ("a", "b"):
            self.assertLess(len(plan["items"][source]), len(plain.balanced["items"][source]))   # packed fewer items
            per_epoch = len(plan["items"][source])
            seen: dict[str, list[int]] = {}
            k = item = 0
            while k < per_epoch:
                stream._schedule_to(item)
                if plan["sources"][plan["schedule"][item][0]] == source:
                    sample = stream[item]
                    self.assertLessEqual(len(sample["anchors"]), 200)
                    self.assertLessEqual(sample["context_rows"], 256 + WINDOW + 8)
                    for row, anchors in stream._balanced_item(item):
                        seen.setdefault(row["name"], []).extend(anchors)
                    k += 1
                item += 1
            for row in (r for r in stream.rows if r["source"] == source):   # every eligible position exactly once
                self.assertEqual(sorted(seen[row["name"]]), list(range(row["a_min"], row["a_max"] + 1)))
        # per-source epoch cap from a switch item on; the freed share goes to the other source
        def make():
            return cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=8, crop=20,
                                         source_weights="a=0.9,b=0.1", pack=True)

        before = make().source_report(400)
        self.assertGreater(before["a"]["epochs"], 3)
        late = make()
        late._schedule_to(29)
        with self.assertRaises(RuntimeError):   # the schedule already passed the switch
            late.cap_epochs(400, 2.5, 10)
        late.cap_epochs(400, 2.5, 40)
        self.assertEqual(late.source_report(40)["a"]["items"], before_items := make().source_report(40)["a"]["items"])
        self.assertGreater(before_items, 0)
        capped = make()
        report = capped.cap_epochs(400, 2.5, 0)
        self.assertEqual(report["capped"], ["a"])
        self.assertAlmostEqual(report["planned_epochs"]["a"], 2.5, delta=0.01)
        after = capped.source_report(400)
        self.assertLessEqual(after["a"]["epochs"], 2.5 * 1.05)
        self.assertGreater(after["b"]["anchor_share"], before["b"]["anchor_share"] + 0.1)

    def test_cap_share_goes_to_fill_sources_only(self):
        raw = json.loads(two_source_index(self.root).read_text())
        for k, row in enumerate(r for r in raw["rows"] if r["split"] == "train"):
            row["source"] = "abc"[k % 3]
        path = self.root / "index" / "index-3src.json"
        path.write_text(json.dumps(raw))

        def make():
            return cd.CaptureTrainStream(path, self.root / "data", rank=0, world=1, seed=3, max_anchors=8, crop=20,
                                         source_weights="a=0.8,b=0.1,c=0.1", pack=True)

        before = make().source_report(300)
        self.assertGreater(before["a"]["epochs"], 3)
        report = make().cap_epochs(300, 3.0, 0, fill={"b"})
        self.assertEqual(report["capped"], ["a"])
        self.assertAlmostEqual(report["share"]["c"], 0.1, delta=1e-3)   # not a fill source: keeps its target share
        self.assertGreater(report["share"]["b"], 0.15)                 # b takes what a gave up
        self.assertAlmostEqual(sum(report["share"].values()), 1.0, delta=2e-3)
        self.assertAlmostEqual(report["planned_epochs"]["a"], 3.0, delta=0.01)

    def test_stratified_resident_partition(self):
        raw = json.loads(two_source_index(self.root).read_text())
        train = [r for r in raw["rows"] if r["split"] == "train"]
        index = cd.partition_index(json.loads(json.dumps(raw)), 2, stratify="source")
        for source, loads in index["rank_anchor_load_by"].items():
            self.assertLessEqual(abs(loads[0] - loads[1]), 40, source)
        held = {s for r in raw["rows"] for s in cd._needed_files(r, raw["window"])}
        resident_rows = {r["name"] for r in train[::3]}
        resident = {f for r in raw["rows"] if r["split"] != "train" or r["name"] in resident_rows
                    for f in cd._needed_files(r, raw["window"])}
        self.assertTrue(resident < held)
        index = cd.partition_index(json.loads(json.dumps(raw)), 2, stratify="source", resident={0: resident})
        rank0 = {r["name"] for r in index["rows"] if r.get("rank") == 0}
        self.assertTrue(rank0 and rank0 <= resident_rows)
        self.assertTrue(set(index["copy_lists"]["0"]) <= resident)


class CaptureTrainingTests(unittest.TestCase, CaptureFixture):
    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def cmd(self, output: Path, *extra: str, ranks: int = 2) -> list[str]:
        return launcher(ranks) + [
            *TRAIN_MODULE, "--config", str(self.config), "--teacher", str(self.root / "target"),
            "--capture-index", str(self.root / "index" / "index.json"), "--capture-root", str(self.root / "data"),
            "--output", str(output), "--device", "cpu", "--lr", "2e-3", "--warmup", "30", "--anchors", "32",
            "--crop", "64", "--eval-anchors", "48", "--weight-decay", "0.0", "--log-every", "50",
            "--loader-workers", "1", *extra,
        ]

    @staticmethod
    def run_cmd(cmd: list[str]) -> None:
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        done = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        if done.returncode:
            raise AssertionError(f"failed:\n{done.stdout[-3000:]}\n{done.stderr[-6000:]}")

    def test_two_rank_capture_run_learns_and_resumes(self):
        full = self.root / "full"
        self.run_cmd(self.cmd(full, "--steps", "300", "--eval-every", "100", "--checkpoint-every", "100"))
        records = [json.loads(line) for line in (full / "train.jsonl").read_text().splitlines()]
        evals = [r for r in records if r["phase"] == "eval"]
        train = [r for r in records if r["phase"] == "train"]
        for record in evals:
            print(f"  capture eval step {record['step']:3d}: tokens_seen {record['tokens_seen']:8.0f} "
                  f"loss {record['loss']:.4f} eal {record['eal']:.4f} recall@16 "
                  f"{record['unary_candidate_recall_at_16']:.3f} extra eal {record.get('heldout_extra/eal', 0):.3f}")
        self.assertLess(evals[-1]["loss"], 0.5 * evals[0]["loss"])
        self.assertGreater(evals[-1]["eal"], evals[0]["eal"] + 2.0)
        self.assertIn("heldout_extra/eal", evals[-1])
        self.assertEqual(evals[-1]["rows"], 3)
        self.assertTrue(all(b["tokens_seen"] > a["tokens_seen"] for a, b in zip(train, train[1:])))
        resumed = self.root / "resumed"
        self.run_cmd(self.cmd(resumed, "--steps", "300", "--eval-every", "100", "--checkpoint-every", "100",
                              "--stop-after", "100"))
        self.run_cmd(self.cmd(resumed, "--steps", "300", "--eval-every", "100", "--checkpoint-every", "100"))
        final = [json.loads(line) for line in (resumed / "train.jsonl").read_text().splitlines()]
        self.assertEqual([r for r in final if r["phase"] == "eval"][-1], evals[-1])
        print(f"  resume 100 -> 300 reproduces the final capture eval exactly (eal {evals[-1]['eal']:.4f})")

    def test_smoke_probe_and_time_budget_plan(self):
        output = self.root / "budget"
        self.run_cmd(self.cmd(output, "--steps", "1000000", "--smoke-steps", "6", "--probe", "--min-free-gb", "1",
                              "--time-budget-h", "0.03", "--eval-points", "4", "--warmup", "30",
                              "--checkpoint-s-estimate", "0", "--reserve-s", "0", ranks=2))
        records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        phases = [r["phase"] for r in records]
        self.assertEqual(phases.count("smoke_step"), 6)
        self.assertEqual(phases.count("probe"), 1)
        smoke = json.loads((output / "smoke.json").read_text())
        self.assertGreater(smoke["planned_steps"], 6)
        evals = [r for r in records if r["phase"] == "eval"]
        self.assertEqual(evals[-1]["step"], smoke["planned_steps"])
        self.assertGreaterEqual(len(evals), 5)
        print(f"  budget plan: {smoke['planned_steps']} steps ({smoke['step_s_median']} s/step), "
              f"{len(evals)} evals, wall {records[-1].get('elapsed_s')}")

    def test_low_memory_abort_is_clean(self):
        output = self.root / "lowmem"
        command = self.cmd(output, "--steps", "50", "--min-free-gb", "8", "--eval-every", "50")
        env = {**os.environ, "OMP_NUM_THREADS": "1", "DRAFTER_FAKE_MEMAVAIL_GB": "5"}
        done = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        self.assertNotEqual(done.returncode, 0)
        last = json.loads((output / "train.jsonl").read_text().splitlines()[-1])
        self.assertEqual((last["phase"], last["reason"], last["step"]), ("abort", "low_memory", 1))

    def test_balanced_init_from_run_reports_sources_and_depths(self):
        parent = self.root / "parent"
        self.run_cmd(self.cmd(parent, "--steps", "60", "--eval-every", "60", "--checkpoint-every", "60"))
        parent_eval = [json.loads(line) for line in (parent / "train.jsonl").read_text().splitlines()][-1]
        path = two_source_index(self.root, "index-2src-run.json")
        child = self.root / "child"
        command = self.cmd(child, "--steps", "40", "--eval-every", "40", "--checkpoint-every", "40",
                           "--source-weights", "a=0.25,b=0.75", "--init-from",
                           str(parent / "checkpoints" / "step-0000060"), "--warmup", "5", "--lr", "5e-4",
                           "--smoke-steps", "4")
        command[command.index(str(self.root / "index" / "index.json"))] = str(path)
        self.run_cmd(command)
        records = [json.loads(line) for line in (child / "train.jsonl").read_text().splitlines()]
        evals = [r for r in records if r["phase"] == "eval"]
        # step 0 of the child evaluates the parent's weights exactly
        for key in ("loss", "eal"):
            self.assertAlmostEqual(evals[0][key], parent_eval[key], places=5)
        self.assertEqual(len(evals[0]["accept_by_depth"]), 7)
        survive = evals[-1]["survive_by_depth"]
        self.assertTrue(all(a >= b for a, b in zip(survive, survive[1:])))
        self.assertAlmostEqual(1.0 + sum(survive), evals[-1]["eal"], delta=1e-3)   # rounded to 4 places
        sources = [r for r in records if r["phase"] == "sources"][-1]
        self.assertEqual(set(sources["total"]), {"a", "b"})
        self.assertAlmostEqual(sources["total"]["a"]["anchor_share"], 0.25, delta=0.08)
        receipt = json.loads((child / "receipt.json").read_text())
        stamp = receipt["provenance"]
        self.assertEqual((stamp["warm_start"], stamp["parent"]["step"]), ("none", 60))
        self.assertTrue((child / "checkpoints" / "step-0000040" / "PROVENANCE.json").is_file())
        print(f"  init-from: parent eal {parent_eval['eal']:.4f} == child step-0 eal {evals[0]['eal']:.4f}; "
              f"accept_by_depth {evals[-1]['accept_by_depth']}; sources {sources['total']}")
        # a checkpoint without the scratch stamp is refused
        bad = self.root / "bad-ckpt"
        bad.mkdir()
        (bad / "rank0.pt").write_bytes((parent / "checkpoints" / "step-0000060" / "rank0.pt").read_bytes())
        (bad / "PROVENANCE.json").write_text(json.dumps({"lineage_root": "elsewhere"}))
        command = self.cmd(self.root / "child-bad", "--steps", "2", "--init-from", str(bad), ranks=1)
        env = {**os.environ, "OMP_NUM_THREADS": "1"}
        done = subprocess.run(command, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
        self.assertNotEqual(done.returncode, 0)
        self.assertIn("scratch provenance", done.stderr)

    def test_round2b_run_packs_caps_groups_and_prunes(self):
        """Round 2b flags in one budgeted 2-rank run: packed rows, per-source epoch cap applied at plan time,
        held-out eval groups by source with a per-category breakdown, checkpoint pruning, gc freeze, diagnostics."""

        raw = json.loads(two_source_index(self.root, "index-2b.json").read_text())
        for k, row in enumerate(r for r in raw["rows"] if r["split"] == "heldout"):
            row["source"], row["category"] = ("a" if k % 2 == 0 else "b"), ("x" if k < 2 else "y")
        path = self.root / "index" / "index-2b.json"
        path.write_text(json.dumps(raw))
        output = self.root / "r2b"
        command = self.cmd(output, "--steps", "1000000", "--smoke-steps", "4", "--time-budget-h", "0.02",
                           "--eval-points", "3", "--warmup", "10", "--checkpoint-points", "3", "--keep-checkpoints", "1",
                           "--checkpoint-s-estimate", "0", "--reserve-s", "0", "--source-weights", "a=0.9,b=0.1",
                           "--pack-rows", "--source-max-epochs", "1.0", "--eval-groups", "ga=a,gb=b:5",
                           "--eval-breakdown", "category", "--gc-freeze", "--fp8-device", "--deferred-metrics",
                           "--log-every", "2", "--anchors", "200", "--crop", "256")
        command[command.index(str(self.root / "index" / "index.json"))] = str(path)
        self.run_cmd(command)
        records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        caps = [r for r in records if r["phase"] == "source_caps"]
        self.assertEqual(len(caps), 1)
        self.assertEqual(caps[0]["switch_item"], 4 * 1 + 2 * 1 + 1)   # smoke items + prefetch depth + 1
        self.assertTrue(all("a" in p["capped"] for p in caps[0]["per_rank"]))   # (b too: tiny pools, long plan)
        sources = [r for r in records if r["phase"] == "sources"][-1]["total"]
        evals = [r for r in records if r["phase"] == "eval"]
        for key in ("eal", "rows", "gb/eal", "ga.x/eal", "ga.y/eal", "heldout_extra/eal"):
            self.assertIn(key, evals[-1], key)
        self.assertEqual((evals[-1]["rows"], evals[-1]["gb/rows"], evals[-1]["ga.x/rows"]), (2, 1, 1))
        self.assertNotIn("gb.x/eal", evals[-1])   # a one-category group has no breakdown
        train = [r for r in records if r["phase"] == "train"]
        for key in ("proc_cpu_cores", "main_thread_busy", "loader_ready_mean", "step_enqueue_s_rank0"):
            self.assertIn(key, train[-1], key)
        self.assertIn("step_gpu_tail_s", train[-1]["rank1"])   # the other rank's telemetry
        steps = sorted(p.name for p in (output / "checkpoints").glob("step-*"))
        self.assertEqual(len(steps), 1)
        self.assertEqual(int(steps[0].split("-")[1]), evals[-1]["step"])
        print(f"  round 2b run: {evals[-1]['step']} steps, caps {caps[0]['per_rank'][0]['planned_epochs']}, "
              f"sources {[(k, v['epochs']) for k, v in sources.items()]}, groups eal {evals[-1]['eal']:.3f} / "
              f"{evals[-1]['gb/eal']:.3f}")

    def test_partitioned_index_and_agreement_mask_run(self):
        index = cd.partition_index(json.loads((self.root / "index" / "index.json").read_text()), 2)
        path = self.root / "index" / "index-w2.json"
        path.write_text(json.dumps(index))
        output = self.root / "partitioned"
        command = self.cmd(output, "--steps", "20", "--eval-every", "20", "--checkpoint-every", "20",
                           "--agreement-mask", "argmax")
        command[command.index(str(self.root / "index" / "index.json"))] = str(path)
        self.run_cmd(command)
        records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        self.assertEqual(records[-1]["phase"], "eval")
        self.assertEqual(records[-1]["step"], 20)

    def test_byo_index_excludes_private_heldout_and_trains(self):
        """drafter-join's bring-your-own index: an existing capture index (whose held-out split may be someone
        else's private set) minus excluded ids; the check split is carved from train rows only; the trainer reads
        it in place."""

        from enntity_drafter import join

        source = json.loads((self.root / "index" / "index.json").read_text())
        train = [r["name"] for r in source["rows"] if r["split"] == "train"]
        for k, row in enumerate(r for r in source["rows"] if r["split"] == "train"):
            row["prompt_id"] = f"prompt-{k}" if k == 1 else None   # a regenerated row listed by its prompt
            row["id"] = row["name"] + "#s0" if k == 2 else row["id"]   # an older index: sample-suffixed id only
        (self.root / "index" / "index-ids.json").write_text(json.dumps(source))
        private = self.root / "private-heldout.jsonl"   # ids as an operator's held-out list spells them
        private.write_text("".join(json.dumps({"id": i}) + "\n" for i in
                                   [train[0].replace("_", "/"), "prompt-1", train[2], train[3], train[4]]))
        out = self.root / "byo-index"
        args = join.byo_capture_index(self.root / "index" / "index-ids.json", out, [private])
        site = cd.CaptureIndex(out / "index.json", self.root / "data")
        names = {r["name"] for r in site.rows}
        self.assertEqual(names, set(train[5:]))   # no excluded row, none of the source's held-out rows
        self.assertEqual(site.index["byo"]["dropped"], {"not_train": len(source["rows"]) - len(train),
                                                        "excluded": 5})
        self.assertGreater(len(site.split("heldout")), 0)
        self.assertGreater(len(site.split("train")), len(site.split("heldout")))
        self.assertEqual(site.index["byo"]["check_rows"], len(site.split("heldout")))
        row = site.rows[0]
        self.assertEqual(site.row_tokens(row, row["lo"], row["hi"]).tolist(), self.truth[row["name"]]["ids"])
        output = self.root / "byo-run"
        command = self.cmd(output, "--steps", "10", "--eval-every", "10", "--checkpoint-every", "10", ranks=1)
        command[command.index("--capture-index") + 1] = args[1]
        self.run_cmd(command)
        records = [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]
        self.assertEqual(records[-1]["phase"], "eval")
        self.assertEqual(records[-1]["rows"], len(site.split("heldout")))


class RecipePipelineIndexTests(unittest.TestCase):
    def test_recipe_pipeline_indexes_its_captured_rows(self):
        """drafter-join's recipe pipeline indexes what scripts/capture.py wrote (``<shard>/rows/<tag>/``); it passed
        ``--fast rows/*``, which made every row directory a root and indexed nothing."""

        import argparse
        from unittest import mock

        from enntity_drafter import join

        cfg = sm.load_config(TINY)
        rows = sd.synthetic_rows(cfg, train_rows=6, heldout_rows=0, length=48)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "shard"

            def fake_run(cmd: list[str], log: Path, env=None) -> None:
                if "scripts/generate_prompts.py" in cmd:
                    (out / "prompts").mkdir(parents=True)
                    (out / "prompts" / "train.jsonl").write_text("".join(
                        json.dumps({"id": r["id"], "source": "synthetic"}) + "\n" for r in rows))
                elif "scripts/capture.py" in cmd:   # rows land in <dest>/<tag>/, records in <out>/capture-records
                    (out / "capture-logs").mkdir(parents=True)
                    with (out / "capture-logs" / "capture-records.jsonl").open("w") as records:
                        for r in rows:
                            slots = torch.cat([r["hidden_states"].permute(1, 0, 2), r["final_hidden"][None]], 0)
                            write_fast(out / "rows" / r["id"], slots, r["token_ids"].tolist(), 0, chunk=16)
                            records.write(json.dumps({"name": r["id"], "P": 8, "ids": r["token_ids"].tolist(),
                                                      "source": "synthetic"}) + "\n")
                elif "enntity_drafter.capture_io" in cmd:
                    cd.main(cmd[cmd.index("index"):])

            opts = argparse.Namespace(gen_minutes=1, spend_cap=0.0, teacher_url=None, tokenizer_dir=Path(tmp),
                                      before_capture=None, after_capture=None, atlas_server="http://127.0.0.1:1",
                                      capture_dir=["/capture"], capture_minutes=1)
            with mock.patch.object(join, "_run", fake_run):
                args = join.pipeline_store({"builder": "templates", "rows": [0, len(rows)]}, out, opts,
                                           lambda phase: None)
            index = json.loads(Path(args[1]).read_text())
        self.assertEqual(len(index["rows"]), len(rows))


class Fp8StorageTests(unittest.TestCase, CaptureFixture):
    """FP8 e4m3 storage (round 2): codec, in-place conversion, mixed BF16/FP8 index and reads."""

    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @staticmethod
    def fp8_truth(slots: torch.Tensor) -> torch.Tensor:
        """Expected reads of an FP8 row: each slot's rows quantized with the codec, as BF16."""

        bits = slots.bfloat16().view(torch.int16).numpy().view(np.uint16)
        q, scale = cd.fp8_quantize(bits)
        return torch.from_numpy(cd.fp8_dequantize(q, scale).view(np.int16)).view(torch.bfloat16)

    def test_codec(self):
        e4m3 = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn).float()
        grid = e4m3[torch.isfinite(e4m3)]
        x = torch.cat([grid, grid.flip(0)])[None, :448].bfloat16()   # amax 448 -> scale 1: exact
        q, scale = cd.fp8_quantize(x.view(torch.int16).numpy().view(np.uint16))
        self.assertEqual(float(scale[0]), 1.0)
        back = torch.from_numpy(cd.fp8_dequantize(q, scale).view(np.int16)).view(torch.bfloat16)
        self.assertTrue(torch.equal(back, x))
        gen = torch.Generator().manual_seed(0)
        h = torch.randn(64, 4096, generator=gen)
        h[:, 7] *= 40   # a massive-activation channel, as in real hidden states
        bits = h.bfloat16().view(torch.int16).numpy().view(np.uint16)
        q, scale = cd.fp8_quantize(bits)
        self.assertEqual((q.dtype, q.shape, scale.dtype, scale.shape), (np.uint8, (64, 4096), np.float32, (64,)))
        back = torch.from_numpy(cd.fp8_dequantize(q, scale).view(np.int16)).view(torch.bfloat16).float()
        rel = float((back - h.bfloat16().float()).norm() / h.bfloat16().float().norm())
        self.assertLess(rel, 0.04)
        zero = np.zeros((2, 8), np.uint16)
        q, scale = cd.fp8_quantize(zero)
        self.assertEqual(scale.tolist(), [1.0, 1.0])
        self.assertTrue((cd.fp8_dequantize(q, scale) == 0).all())
        bad = torch.ones(1, 8)
        bad[0, 3] = float("inf")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            cd.fp8_quantize(bad.bfloat16().view(torch.int16).numpy().view(np.uint16))

    def test_convert_index_and_read(self):

        FP8 = cd.FP8
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp)
            names = ["train-001", "train-003", "train-005", "train-007"]
            for k, name in enumerate(names):
                truth = self.truth[name]
                run = "round1-rows/r1/p0" if k < 2 else "round2-rows/r2/p0"
                write_fast(data / run / name, truth["slots"].float(), truth["ids"], truth["P"] - WINDOW, chunk=32)
            r2 = data / "round2-rows" / "r2" / "p0"
            measured = fp8_convert.convert_dir(r2 / "train-005", write=False)
            self.assertEqual(sorted(p.suffix for p in (r2 / "train-005").iterdir()).count(".fp8"), 0)
            # interrupted: one segment already converted (sidecar rewritten, .bin not yet deleted)
            first = sorted((r2 / "train-005").glob("r0-L-*.json"))[0]
            stash = first.with_suffix(".keep")
            stash.write_bytes(first.with_suffix(".bin").read_bytes())
            fp8_convert._segment(first, True, 0.1)
            stash.rename(first.with_suffix(".bin"))
            stats = fp8_convert.convert_dir(r2 / "train-005")
            self.assertEqual(stats["converted"], stats["segments"] - 1)
            self.assertFalse(list((r2 / "train-005").glob("*.bin")))
            self.assertLess(max(stats["rel_l2"][:5]), 0.06)
            self.assertLess(abs(measured["rel_l2"][0] - stats["rel_l2"][0]), 0.01)
            # a mixed row: only rank 1's layer segments converted
            for side in sorted((r2 / "train-007").glob("r1-L-*.json")):
                fp8_convert._segment(side, True, 0.1)
            self.assertEqual(cd.scan_fast_dir(r2 / "train-007", "x")["dtype"], "mixed")
            scan = cd.scan_fast_dir(r2 / "train-005", "x")
            self.assertEqual((scan["dtype"], scan["lo"], scan["hi"]), (FP8, self.truth["train-005"]["P"] - WINDOW, 96))
            self.assertTrue(all(s[0].endswith(".fp8") for s in scan["layer_segs"] + scan["final_segs"]))
            side = json.loads(first.read_text())
            self.assertEqual((side["dtype"], side["source_dtype"], side["scale"]["per"]), (FP8, "bf16", "row_slot"))
            # round-1 (BF16) + round-2 (FP8) manifests in one index
            m1 = [{"name": n, "P": self.truth[n]["P"], "path": f"round1-rows/r1/p0/{n}", "source": "agentic",
                   "dtype": "bf16"} for n in names[:2]]
            m2 = [{"name": n, "P": self.truth[n]["P"], "path": f"round2-rows/r2/p0/{n}", "source": "structured",
                   "dtype": FP8} for n in names[2:]]
            (data / "m1.jsonl").write_text("".join(json.dumps(m) + "\n" for m in m1))
            (data / "m2.jsonl").write_text("".join(json.dumps(m) + "\n" for m in m2))
            index = cd.build_index(data, [], ["round1-rows/*/p*", "round2-rows/*/p*"], [], None, data / "idx",
                                   window=WINDOW, block=8, layers=LAYERS,
                                   manifest=[str(data / "m1.jsonl"), str(data / "m2.jsonl")])
            self.assertEqual(index["skipped"], {"manifest_rows_not_indexed": 0})
            rows = {r["name"]: r for r in index["rows"]}
            self.assertEqual({n: r["dtype"] for n, r in rows.items()},
                             {"train-001": "bf16", "train-003": "bf16", "train-005": FP8, "train-007": "mixed"})
            self.assertEqual(index["summary"]["train"]["rows_bf16"], 2)
            fp8_bytes = rows["train-005"]["bytes"]
            stored = rows["train-005"]["hi"] - rows["train-005"]["lo"]
            self.assertEqual(fp8_bytes, stored * 6 * (self.cfg["hidden_size"] + 4))
            loaded = cd.CaptureIndex(data / "idx" / "index.json", data)
            reader = cd.SegmentReader(data, loaded.hidden)
            for name, row in rows.items():
                truth = self.truth[name]
                expect = truth["slots"]
                if row["dtype"] == FP8:
                    expect = self.fp8_truth(truth["slots"])
                elif row["dtype"] == "mixed":
                    expect = truth["slots"].clone()
                anchors = cd.choose_anchors(row, random.Random(name), 6, 0)
                sample = cd.build_sample(loaded, reader, row, anchors)
                positions = sample["inputs"]["position_ids"][0]
                got = sample["inputs"]["hidden_states"][0].view(-1, 5, loaded.hidden)
                if row["dtype"] == "mixed":   # rank 1's (lower-half) layer rows are FP8, the rest BF16
                    exact = torch.equal(got, expect[:5, positions].permute(1, 0, 2))
                    self.assertFalse(exact)
                    fp8 = self.fp8_truth(truth["slots"])
                    for i, p in enumerate(positions.tolist()):
                        self.assertTrue(torch.equal(got[i], expect[:5, p]) or torch.equal(got[i], fp8[:5, p]))
                else:
                    self.assertTrue(torch.equal(got, expect[:5, positions].permute(1, 0, 2)), name)
                final = sample["inputs"]["verifier_last_hidden_states"][0]
                for compact, a in zip(sample["anchors"], anchors):
                    need = torch.arange(compact - 1, compact + 7)
                    self.assertTrue(torch.equal(final[need], expect[5, a - 1: a + 7]), name)
            # the stream reads FP8 rows like any other
            stream = cd.CaptureTrainStream(data / "idx" / "index.json", data, rank=0, world=1, seed=1,
                                           max_anchors=8, crop=0)
            self.assertTrue(all(stream[i]["inputs"]["hidden_states"].dtype == torch.bfloat16 for i in range(4)))
            # --fp8-device: FP8 rows (alone or bundled) reach the device as e4m3 + scales; dequantized there they
            # equal the CPU codec bit for bit; BF16 and mixed rows (and bundles with them) keep the BF16 path
            fp8_row = rows["train-005"]
            parts = [(fp8_row, cd.choose_anchors(fp8_row, random.Random(1), 6, 0)),
                     (fp8_row, cd.choose_anchors(fp8_row, random.Random(2), 3, 0))]
            on_cpu = cd.build_bundle(loaded, reader, parts)
            kept = cd.build_bundle(loaded, reader, parts, fp8_device=True)
            self.assertEqual(kept["inputs"]["hidden_states_fp8"].dtype, torch.uint8)
            self.assertNotIn("hidden_states", kept["inputs"])
            got = cd.materialize_fp8(dict(kept["inputs"]))
            self.assertEqual(set(got), set(on_cpu["inputs"]))
            for key, value in on_cpu["inputs"].items():
                self.assertTrue(torch.equal(got[key], value), key)
            moved = ts.to_device(dict(kept["inputs"]), torch.device("cpu"), torch.float32)
            self.assertTrue(torch.equal(moved["hidden_states"], on_cpu["inputs"]["hidden_states"].float()))
            for other in ("train-001", "train-007"):
                mixed = cd.build_bundle(loaded, reader, [parts[0], (rows[other], cd.choose_anchors(
                    rows[other], random.Random(3), 4, 0))], fp8_device=True)
                self.assertIn("hidden_states", mixed["inputs"])

    def test_watch_converts_marked_rows_and_stops(self):

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "round2-rows"
            for k, name in enumerate(["train-009", "train-011"]):
                truth = self.truth[name]
                d = root / "fast-x" / "p0" / name
                write_fast(d, truth["slots"].float(), truth["ids"], truth["P"] - WINDOW, chunk=32)
                if k == 0:
                    (d / fp8_convert.MARKER).write_text("")
            (Path(tmp) / "STOP").write_text("")
            fp8_convert.watch(root, Path(tmp) / "log.jsonl", Path(tmp) / "STOP", 0.1, 0.01)
            log = [json.loads(line) for line in (Path(tmp) / "log.jsonl").read_text().splitlines()]
            self.assertEqual([(r["name"], r["converted"], r["segments"]) for r in log],
                             [("train-009", log[0]["segments"], log[0]["segments"])])
            self.assertFalse(list(root.glob("*/p0/train-009/*.bin")))
            self.assertFalse((root / "fast-x" / "p0" / "train-009" / fp8_convert.MARKER).exists())
            self.assertTrue(list(root.glob("*/p0/train-011/*.bin")))   # unmarked rows are left alone


def _reference_rows(directory: Path, lo: int, hi: int) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    """Independent whole-file reader: layers [5, hi-lo, H], final [hi-lo, H], ids."""

    layers, finals, ids = {}, {}, {}
    for sidecar in sorted(directory.glob("*.json")):
        meta = json.loads(sidecar.read_text())
        data = torch.from_numpy(np.fromfile(sidecar.with_suffix(".bin"), dtype=np.uint16)).view(torch.bfloat16)
        hidden, rows = meta["hidden_size"], meta["rows"]
        if meta["schema"] == cd.PILOT_SCHEMA or meta["kind"] == "layers":
            block = data.view(-1, rows, hidden)
            for i in range(rows):
                layers[meta["pos0"] + i] = block[:5, i]
        if meta["schema"] == cd.PILOT_SCHEMA or meta["kind"] == "final":
            block = data.view(-1, rows, hidden)
            for i in range(rows):
                finals[meta["pos0"] + i] = block[-1, i]
        for i, token in enumerate(meta.get("token_ids") or []):
            ids[meta["pos0"] + i] = token
    positions = range(lo, hi)
    return (torch.stack([layers[p] for p in positions], 1), torch.stack([finals[p] for p in positions]),
            [ids[p] for p in positions])


@unittest.skipUnless((REAL / "index" / "index.json").is_file(), "DRAFTER_LOADER_TEST_ROOT not set")
class LoaderRealDataTests(unittest.TestCase):
    def test_real_rows_pack_exactly(self):
        index = cd.CaptureIndex(REAL / "index" / "index.json", REAL)
        reader = cd.SegmentReader(REAL, index.hidden)
        formats = set()
        for row in index.rows:
            if not row["anchors"]:
                continue
            anchors = cd.choose_anchors(row, random.Random(row["name"]), 64, 512)
            sample = cd.build_sample(index, reader, row, anchors)
            positions = sample["inputs"]["position_ids"][0]
            lo, hi = int(positions.min()), int(positions.max()) + 1
            layers, final, ids = _reference_rows(REAL / row["dir"], lo, hi)
            local = positions - lo
            got = sample["inputs"]["hidden_states"][0].view(-1, 5, index.hidden)
            self.assertTrue(torch.equal(got, layers[:, local].permute(1, 0, 2)), row["name"])
            self.assertEqual(sample["inputs"]["input_ids"][0].tolist(), [ids[i] for i in local.tolist()])
            for compact in sample["anchors"]:
                need = torch.arange(compact - 1, compact + 7)
                self.assertTrue(torch.equal(sample["inputs"]["verifier_last_hidden_states"][0, need],
                                            final[local[need]]), row["name"])
                self.assertGreaterEqual(int(positions[compact]), row["P"])
            formats.add((row["format"], tuple(row.get("ranks", []))))
        print(f"\n  real rows packed exactly: {sorted(formats)}")
        self.assertIn(("fast", (0, 1)), formats)
        self.assertIn(("pilot", ()), formats)


if __name__ == "__main__":
    unittest.main()
