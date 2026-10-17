"""CPU tests for the scratch drafter trainer.

    python -m pytest tests/test_train.py -v

* full config is ~1.17B body parameters / 81 keys with the Atlas key names;
* the incoai guard refuses forbidden paths, metadata, symlinks, unstamped
  drafter files and hashes, and accepts this trainer's own stamped artifacts;
* no trainer module loads tensors outside the guard;
* a 2-rank data-parallel (gloo) run of the tiny config on a deterministic synthetic
  pattern improves both loss and the held-out chain proxy, exports a verified
  Atlas BF16 checkpoint, and an interrupted+resumed run matches it exactly;
* the selector_prefix objective learns; Atlas capture chunks pack into rows;
  FP8 feature rows round-trip.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import FULL, REPO, TINY, TRAIN_MODULE, launcher  # noqa: E402

import torch  # noqa: E402

from enntity_drafter import data as sd  # noqa: E402
from enntity_drafter import guard as incoai_guard  # noqa: E402
from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter.guard import (  # noqa: E402
    SCRATCH_STAMP_KEY, ForbiddenCheckpointError, check_path, check_target_dir, guarded_load_file,
    guarded_torch_load, scratch_stamp,
)



def atlas_expected_keys(layers: int) -> set[str]:
    """Bare key names Atlas's dflash_loader.rs probes for a DFlash2 drafter."""

    keys = {"fc.weight", "hidden_norm.weight", "norm.weight",
            "candidate_selector.hidden_projection.weight",
            "candidate_selector.predecessor_codebook", "candidate_selector.successor_codebook"}
    per_layer = ["input_layernorm.weight", "post_attention_layernorm.weight",
                 "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                 "self_attn.o_proj.weight", "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                 "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
                 "attention_conv.base_kernel", "attention_conv.kernel_projection.weight",
                 "mlp_conv.base_kernel", "mlp_conv.kernel_projection.weight"]
    keys |= {f"layers.{i}.{name}" for i in range(layers) for name in per_layer}
    return keys


def last_eval(path: Path) -> dict:
    return [r for r in map(json.loads, path.read_text().splitlines()) if r["phase"] == "eval"][-1]


def first_eval(path: Path) -> dict:
    return [r for r in map(json.loads, path.read_text().splitlines()) if r["phase"] == "eval"][0]


class Fixture:
    root: Path

    @classmethod
    def build(cls) -> None:
        cls.tmp = tempfile.TemporaryDirectory(prefix="scratch-drafter-")
        cls.root = Path(cls.tmp.name)
        cfg = sm.load_config(TINY)
        sd.write_synthetic_target_dir(cls.root / "target", cfg)
        rows = sd.synthetic_rows(cfg, train_rows=24, heldout_rows=4, length=96)
        sd.write_row_store(cls.root / "features", rows, capture_layers=cfg["target_layer_ids"],
                           hidden_size=cfg["hidden_size"], final_hidden="pre-final-norm",
                           target={"name": "synthetic-cpu-fixture"})

    @classmethod
    def train_cmd(cls, output: Path, *extra: str, ranks: int = 2) -> list[str]:
        return launcher(ranks) + [
            *TRAIN_MODULE, "--config", str(TINY), "--teacher", str(cls.root / "target"),
            "--features", str(cls.root / "features"), "--output", str(output), "--device", "cpu",
            "--lr", "2e-3", "--warmup", "30", "--anchors", "32", "--eval-anchors", "64",
            "--weight-decay", "0.0", "--log-every", "50", *extra,
        ]

    @staticmethod
    def run_cmd(cmd: list[str]) -> str:
        # One intra-op thread: CPU reductions are then bitwise reproducible, which
        # the exact resume comparison needs (2 threads drift by ~1e-4 per 100 steps).
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        done = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=1800)
        if done.returncode:
            raise AssertionError(f"{cmd[:3]} failed:\n{done.stdout[-3000:]}\n{done.stderr[-6000:]}")
        return done.stdout


class ParameterCountTests(unittest.TestCase):
    def test_full_config_is_1_17b_with_atlas_keys(self):
        cfg = sm.load_config(FULL)
        model = sm.build_model(cfg, device="meta")
        total, keys = sm.count_body_parameters(model)
        print(f"\nfull config body parameters: {total:,} in {keys} tensors")
        self.assertEqual(total, sm.FULL_BODY_PARAMETERS)
        self.assertEqual(keys, sm.FULL_BODY_KEYS)
        self.assertEqual(set(sm.body_parameter_names(model)), atlas_expected_keys(5))
        shapes = dict(model.named_parameters())
        self.assertEqual(tuple(shapes["fc.weight"].shape), (4096, 5 * 4096))
        self.assertEqual(tuple(shapes["candidate_selector.predecessor_codebook"].shape), (154880, 256))
        self.assertEqual(tuple(shapes["layers.0.attention_conv.kernel_projection.weight"].shape), (1024, 4096))


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _save(self, path: Path, metadata: dict | None = None) -> Path:
        from safetensors.torch import save_file

        path.parent.mkdir(parents=True, exist_ok=True)
        save_file({"w": torch.zeros(2)}, str(path), metadata=metadata)
        return path

    def test_forbidden_path_names(self):
        for name in ("incoai--GLM-5.3-Flash-DFlash2/model.safetensors",
                     "models/local-inference-lab--GLM-5.3-Flash-DFlash2-MXFP8/model.safetensors",
                     "work/base-dc77/model.safetensors"):
            path = self._save(self.root / name)
            for role in ("drafter", "target", "features"):
                with self.assertRaises(ForbiddenCheckpointError, msg=f"{name} {role}"):
                    guarded_load_file(path, role)

    def test_symlink_to_forbidden_target(self):
        real = self._save(self.root / "incoai" / "weights.safetensors")
        link = self.root / "innocent" / "draft.safetensors"
        link.parent.mkdir()
        link.symlink_to(real)
        with self.assertRaises(ForbiddenCheckpointError):
            check_path(link, "features")

    def test_sibling_provenance_and_header_metadata(self):
        path = self._save(self.root / "a" / "model.safetensors")
        (path.parent / "config.json").write_text(json.dumps({"_name_or_path": "incoai/GLM-5.3-Flash-DFlash2"}))
        with self.assertRaises(ForbiddenCheckpointError):
            check_path(path, "features")
        path = self._save(self.root / "b" / "model.safetensors",
                          {"source_revision": "dc77ff1c99eeb2df044ee3d4f0094eb033fee410"})
        with self.assertRaises(ForbiddenCheckpointError):
            check_path(path, "features")
        path = self._save(self.root / "c" / "model.safetensors")
        (path.parent / "hf_quant_config.json").write_text('{"model": "incoai/GLM-5.3-Flash-DFlash2"}')
        with self.assertRaises(ForbiddenCheckpointError):
            check_path(path, "target")

    def test_drafter_role_requires_scratch_stamp(self):
        unstamped = self._save(self.root / "renamed" / "model.safetensors")
        with self.assertRaises(ForbiddenCheckpointError):
            guarded_load_file(unstamped, "drafter")
        warm = self._save(self.root / "warm" / "model.safetensors",
                          {SCRATCH_STAMP_KEY: json.dumps({"lineage_root": "enntity-scratch-init",
                                                          "warm_start": "some-checkpoint"})})
        with self.assertRaises(ForbiddenCheckpointError):
            guarded_load_file(warm, "drafter")
        ok = self._save(self.root / "ok" / "model.safetensors",
                        {SCRATCH_STAMP_KEY: json.dumps(scratch_stamp())})
        self.assertIn("w", guarded_load_file(ok, "drafter"))

    def test_forbidden_artifact_hash(self):

        path = self._save(self.root / "h" / "model.safetensors", {SCRATCH_STAMP_KEY: json.dumps(scratch_stamp())})
        digest = incoai_guard.sha256_file(path)
        original = incoai_guard.FORBIDDEN_SHA256
        incoai_guard.FORBIDDEN_SHA256 = original | {digest}
        try:
            with self.assertRaises(ForbiddenCheckpointError):
                check_path(path, "drafter")
        finally:
            incoai_guard.FORBIDDEN_SHA256 = original

    def test_resume_file_needs_provenance(self):
        path = self.root / "ckpt" / "rank0.pt"
        path.parent.mkdir()
        torch.save({"step": 1}, path)
        with self.assertRaises(ForbiddenCheckpointError):
            guarded_torch_load(path, "drafter")
        (path.parent / "PROVENANCE.json").write_text(json.dumps(scratch_stamp()))
        self.assertEqual(guarded_torch_load(path, "drafter")["step"], 1)

    def test_target_dir_checks(self):
        cfg = sm.load_config(TINY)
        target = sd.write_synthetic_target_dir(self.root / "target", cfg)
        self.assertEqual(check_target_dir(target)["license_on_disk"], "mit")
        (target / "README.md").write_text("---\nlicense: cc-by-nc-nd-4.0\n---\n")
        with self.assertRaises(ForbiddenCheckpointError):
            check_target_dir(target)
        drafter = self.root / "drafter-as-target"
        drafter.mkdir()
        (drafter / "config.json").write_text(json.dumps({"architectures": ["DFlash2DraftModel"],
                                                         "dflash_config": {}}))
        with self.assertRaises(ForbiddenCheckpointError):
            check_target_dir(drafter)

    def test_no_unguarded_tensor_loads_in_trainer_sources(self):
        pattern = re.compile(r"\b(load_file|safe_open|torch\.load)\(")
        offenders = []
        for path in sorted(p for d in ("enntity_drafter", "scripts", "coordinator", "validator")
                           for p in (REPO / d).glob("*.py")):
            if path.name == "guard.py":
                continue
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line) and "guarded_" not in line:
                    offenders.append(f"{path.name}:{number}: {line.strip()}")
        self.assertEqual(offenders, [])


class DataTests(unittest.TestCase):
    def test_fp8_rows_round_trip(self):
        cfg = sm.load_config(TINY)
        rows = sd.synthetic_rows(cfg, train_rows=1, heldout_rows=1, length=32)
        with tempfile.TemporaryDirectory() as tmp:
            sd.write_row_store(tmp, rows, capture_layers=cfg["target_layer_ids"], hidden_size=256,
                               final_hidden="pre-final-norm", target={}, fp8=True)
            store = sd.RowStore(tmp, expect_layers=cfg["target_layer_ids"], hidden_size=256, dtype=torch.float32)
            loaded = store.load(store.split("train")[0])
            reference = rows[0]["hidden_states"].flatten(1)
            error = (loaded["hidden_states"][0] - reference).norm() / reference.norm()
            self.assertLess(float(error), 0.05)
            self.assertEqual(tuple(loaded["hidden_states"].shape), (1, 32, 5 * 256))

    def test_pack_atlas_capture(self):
        layers, hidden, n = [5, 14, 24, 33, 42], 8, 10
        slots = torch.randn(len(layers) + 1, n, hidden).bfloat16()
        ids = list(range(100, 100 + n))
        with tempfile.TemporaryDirectory() as tmp:
            chunks = Path(tmp, "chunks")
            chunks.mkdir()
            for pos0, rows in ((0, 6), (6, 4)):
                name = chunks / f"p{pos0:09d}-n{rows}"
                name.with_suffix(".bin").write_bytes(
                    slots[:, pos0:pos0 + rows].contiguous().view(torch.uint16).numpy().tobytes())
                name.with_suffix(".json").write_text(json.dumps({
                    "schema": "atlas.hidden_capture.chunk/v1", "pos0": pos0, "rows": rows,
                    "hidden_size": hidden, "dtype": "bf16", "layers": layers,
                    "token_ids": ids[pos0:pos0 + rows]}))
            row = sd.pack_atlas_capture(tmp, row_id="r0", split="train", loss_start=6)
        self.assertTrue(torch.equal(row["hidden_states"], slots[:-1].permute(1, 0, 2)))
        self.assertTrue(torch.equal(row["final_hidden"], slots[-1]))
        self.assertEqual(row["token_ids"].tolist(), ids)
        self.assertEqual(row["loss_mask"].tolist(), [False] * 6 + [True] * 4)


class TrainingTests(unittest.TestCase, Fixture):
    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_rank_learns_resumes_and_exports(self):
        steps = "300"
        full = self.root / "full"
        self.run_cmd(self.train_cmd(full, "--steps", steps, "--eval-every", "100",
                                "--checkpoint-every", "100", "--export"))
        start, end = first_eval(full / "train.jsonl"), last_eval(full / "train.jsonl")
        print(f"\n2-rank replicated tiny run, held-out: loss {start['loss']:.4f} -> {end['loss']:.4f}; "
              f"chain proxy (EAL) {start['eal']:.4f} -> {end['eal']:.4f}; "
              f"top-16 oracle {start['unary_top_16_oracle_accepted_length']:.3f} -> "
              f"{end['unary_top_16_oracle_accepted_length']:.3f}")
        for record in map(json.loads, (full / "train.jsonl").read_text().splitlines()):
            if record["phase"] == "eval":
                print(f"  eval step {record['step']:3d}: loss {record['loss']:.4f} eal {record['eal']:.4f}")
        self.assertLess(end["loss"], 0.5 * start["loss"])
        self.assertGreater(end["eal"], start["eal"] + 2.0)

        exported = [r for r in map(json.loads, (full / "train.jsonl").read_text().splitlines())
                    if r["phase"] == "export"][0]
        self.assertEqual(exported["export"]["verify"]["keys"], len(atlas_expected_keys(2)))
        self.assertEqual(exported["export"]["format"], "atlas-dflash2-bf16")
        tensors = guarded_load_file(full / "export-atlas-bf16" / "model.safetensors", "drafter")
        self.assertEqual(set(tensors), atlas_expected_keys(2))
        self.assertTrue(all(t.dtype == torch.bfloat16 for t in tensors.values()))

        resumed = self.root / "resumed"
        self.run_cmd(self.train_cmd(resumed, "--steps", steps, "--eval-every", "100",
                                "--checkpoint-every", "100", "--stop-after", "100"))
        self.run_cmd(self.train_cmd(resumed, "--steps", steps, "--eval-every", "100", "--checkpoint-every", "100"))
        self.assertEqual(last_eval(resumed / "train.jsonl"), end)
        print(f"  resume at 100 -> 300 reproduces the uninterrupted final eval exactly: eal {end['eal']:.4f}")

    def test_two_rank_accum_matches_one_process_large_batch_and_resumes(self):
        """2 ranks x accum 4 == 1 process x accum 8 (same rows, anchors, loss scale);
        replicas stay identical; an interrupted accum run resumes exactly."""

        def params(output: Path, step: int, rank: int = 0) -> dict[str, torch.Tensor]:
            path = output / "checkpoints" / f"step-{step:07d}" / f"rank{rank}.pt"
            return guarded_torch_load(path, "drafter")["params"]

        def rel_error(a: dict, b: dict, init: dict) -> float:
            num = sum(float((a[k] - b[k]).double().norm() ** 2) for k in a) ** 0.5
            den = sum(float((b[k] - init[k]).double().norm() ** 2) for k in b) ** 0.5
            return num / den

        common = ("--steps", "6", "--warmup", "2", "--eval-every", "3", "--checkpoint-every", "3")
        init_dir, one, two32, two16, resumed = (self.root / f"accum-{n}" for n in
                                                ("init", "one", "two-fp32", "two-bf16", "resumed"))
        self.run_cmd(self.train_cmd(init_dir, *common, "--lr", "0", "--stop-after", "3", ranks=1))
        self.run_cmd(self.train_cmd(one, *common, "--grad-accum", "8", ranks=1))
        self.run_cmd(self.train_cmd(two32, *common, "--grad-accum", "4", "--grad-reduce-dtype", "fp32"))
        self.run_cmd(self.train_cmd(two16, *common, "--grad-accum", "4"))
        init, reference = params(init_dir, 3), params(one, 6)
        for run in (two32, two16):
            self.assertTrue(all(torch.equal(v, params(run, 6, 1)[k]) for k, v in params(run, 6).items()),
                            f"{run.name}: replicas diverged")
        err32, err16 = rel_error(params(two32, 6), reference, init), rel_error(params(two16, 6), reference, init)
        print(f"\n2 ranks x accum 4 vs 1 process x accum 8 after 6 steps, |diff|/|update|: "
              f"fp32 wire {err32:.2e}, bf16 wire {err16:.2e}")
        self.assertLess(err32, 1e-4)
        self.assertLess(err16, 3e-2)

        self.run_cmd(self.train_cmd(resumed, *common, "--grad-accum", "4", "--stop-after", "3"))
        self.run_cmd(self.train_cmd(resumed, *common, "--grad-accum", "4"))
        self.assertTrue(all(torch.equal(v, params(resumed, 6)[k]) for k, v in params(two16, 6).items()))
        self.assertEqual(last_eval(resumed / "train.jsonl"), last_eval(two16 / "train.jsonl"))
        print("  accum-4 resume at 3 -> 6 reproduces the uninterrupted params exactly")

    def test_global_anchors_picks_the_micro_batch_and_probes_on_resume(self):   # dogfood: memory guard kills
        """--global-anchors: the largest micro-batch the probe finds memory for, grad_accum derived; the probe runs
        again on resume (it was skipped, and memory shared with other services had dropped meanwhile)."""

        from enntity_drafter.train import batch_candidates, choose_batch

        self.assertEqual(batch_candidates(512, 1, 2048, 1),
                         [(512, 4), (256, 8), (128, 16), (64, 32), (32, 64), (16, 128), (8, 256)])
        self.assertEqual(batch_candidates(512, 4, 2048, 2)[:2], [(512, 2), (256, 4)])   # x ranks
        self.assertEqual(batch_candidates(64, 4, None, 2), [(64, 4)])   # today's anchors + grad_accum
        with self.assertRaises(SystemExit):
            batch_candidates(512, 1, 1000, 3)
        probed = []
        chosen, records = choose_batch(batch_candidates(512, 1, 2048, 1),
                                       lambda a: (probed.append(a) or a <= 128, {"anchors": a}))
        self.assertEqual((chosen, probed), ((128, 16), [512, 256, 128]))
        self.assertEqual([r["fits"] for r in records], [False, False, True])
        self.assertEqual(choose_batch([(8, 4)], lambda a: (False, {"anchors": a}))[0], None)

        def records_of(output: Path) -> list[dict]:
            return [json.loads(line) for line in (output / "train.jsonl").read_text().splitlines()]

        def params(output: Path, step: int) -> dict[str, torch.Tensor]:
            return guarded_torch_load(output / "checkpoints" / f"step-{step:07d}" / "rank0.pt", "drafter")["params"]

        common = ("--steps", "4", "--eval-every", "0", "--checkpoint-every", "2", "--probe")
        reference, adaptive = self.root / "global-ref", self.root / "global-adaptive"
        self.run_cmd(self.train_cmd(reference, *common, "--grad-accum", "2", ranks=1))
        adaptive_cmd = self.train_cmd(adaptive, *common, "--global-anchors", "64", ranks=1)   # --anchors 32: cap
        self.run_cmd(adaptive_cmd + ["--stop-after", "2"])
        batch = [r for r in records_of(adaptive) if r["phase"] == "batch"][0]
        self.assertEqual((batch["anchors"], batch["grad_accum"], batch["global_anchors"]), (32, 2, 64))
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "DRAFTER_FAKE_MEMAVAIL_GB": "5"}
        done = subprocess.run(adaptive_cmd + ["--min-free-gb", "8"], cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=1800)
        self.assertEqual(done.returncode, 3, done.stderr[-2000:])
        tail = records_of(adaptive)
        self.assertEqual((tail[-1]["phase"], tail[-1]["reason"], tail[-1]["step"]), ("abort", "probe_low_memory", 2))
        self.assertEqual([r["anchors"] for r in tail if r["phase"] == "probe" and r["step"] == 2], [32, 16, 8])
        self.run_cmd(adaptive_cmd)   # memory is back: the resumed run replays the reference stream exactly
        self.assertTrue(all(torch.equal(v, params(reference, 4)[k]) for k, v in params(adaptive, 4).items()))

    def test_low_memory_pauses_with_a_checkpoint_and_resumes_exactly(self):   # dogfood: other jobs' memory spikes
        """--low-memory pause: memory checked before every micro-step; a shortfall mid-step drops the unfinished
        step, checkpoints the last completed one and exits 75; rerunning resumes and matches an undisturbed run."""

        def params(output: Path) -> dict[str, torch.Tensor]:
            return guarded_torch_load(output / "checkpoints" / "step-0000004" / "rank0.pt", "drafter")["params"]

        common = ("--steps", "4", "--eval-every", "0", "--checkpoint-every", "100", "--grad-accum", "2",
                  "--min-free-gb", "8", "--low-memory", "pause")
        reference, paused = self.root / "pause-ref", self.root / "pause-run"
        self.run_cmd(self.train_cmd(reference, *common, ranks=1))
        # 3 readings per step (2 micro-steps + the end of the step): the 8th, before step 2's second micro-step, is low
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "DRAFTER_FAKE_MEMAVAIL_GB": "100x7,5"}
        done = subprocess.run(self.train_cmd(paused, *common, ranks=1), cwd=REPO, env=env, capture_output=True,
                              text=True, timeout=1800)
        self.assertEqual(done.returncode, 75, done.stderr[-2000:])
        last = json.loads((paused / "train.jsonl").read_text().splitlines()[-1])
        self.assertEqual((last["phase"], last["reason"], last["step"]), ("pause", "low_memory", 2))
        self.assertEqual((paused / "checkpoints" / "latest-rank0").read_text(), "step-0000002")
        self.run_cmd(self.train_cmd(paused, *common, ranks=1))   # memory is back
        self.assertTrue(all(torch.equal(v, params(reference)[k]) for k, v in params(paused).items()))

    def test_checkpoint_leaves_no_host_copies_behind(self):   # dogfood: host memory 11 -> 32 GB after a checkpoint
        script = """
import json, resource, sys, tempfile
from pathlib import Path
import psutil, torch
from enntity_drafter.train import save_checkpoint
params = [torch.nn.Parameter(torch.randn(8 * 2**20)) for _ in range(4)]   # 128 MB
optimizer = torch.optim.AdamW(params)
sum(p.sum() for p in params).backward()
optimizer.step()                                                            # + 256 MB of state
model = torch.nn.Module()
for i, p in enumerate(params):
    model.register_parameter(f"p{i}", p)
scale = 1 if sys.platform == "darwin" else 1024
before, peak_before = psutil.Process().memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
with tempfile.TemporaryDirectory() as tmp:
    save_checkpoint(Path(tmp), 1, model, optimizer, 0, 1, {"stamp": "test"})
    after, peak = psutil.Process().memory_info().rss, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
print(json.dumps({"rss_growth": after - before, "peak_growth": peak - peak_before}))
"""
        done = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True, timeout=600,
                              env={**os.environ, "PYTHONPATH": str(REPO), "OMP_NUM_THREADS": "1"})
        self.assertEqual(done.returncode, 0, done.stderr[-2000:])
        growth = json.loads(done.stdout.strip().splitlines()[-1])
        print(f"\ncheckpoint of 384 MB of state: RSS {growth['rss_growth'] / 2**20:+.0f} MB after, peak "
              f"{growth['peak_growth'] / 2**20:+.0f} MB")
        self.assertLess(growth["rss_growth"], 64 * 2**20)    # back to baseline after the save
        self.assertLess(growth["peak_growth"], 128 * 2**20)  # no full host copy while saving (it was +384 MB)

    def test_sdpa_attention_matches_eager(self):
        """--attn sdpa (bool mask, fused kernel) computes the same model as eager (float mask)."""

        runs = {}
        for attn in ("eager", "sdpa"):
            output = self.root / f"attn-{attn}"
            self.run_cmd(self.train_cmd(output, "--steps", "20", "--eval-every", "10", "--checkpoint-every", "20",
                                        "--attn", attn, ranks=1))
            runs[attn] = [r for r in map(json.loads, (output / "train.jsonl").read_text().splitlines())
                          if r["phase"] == "eval"]
        for eager, sdpa in zip(runs["eager"], runs["sdpa"]):
            self.assertAlmostEqual(eager["loss"], sdpa["loss"], delta=1e-4 * abs(eager["loss"]))
            self.assertAlmostEqual(eager["eal"], sdpa["eal"], delta=1e-3)
        print(f"\nsdpa vs eager held-out loss after 20 steps: {runs['sdpa'][-1]['loss']:.6f} vs "
              f"{runs['eager'][-1]['loss']:.6f}")

    @unittest.skipUnless(torch.backends.mps.is_available(), "needs an Apple GPU (MPS)")
    def test_mps_eval_matches_cpu(self):
        """--device mps (a validator on a Mac) gives the same held-out eval as the CPU."""

        evals = {}
        for device in ("cpu", "mps"):
            output = self.root / f"eval-{device}"
            self.run_cmd(self.train_cmd(output, "--eval-only", "--eval-every", "1", "--attn", "sdpa",
                                        "--device", device, ranks=1))
            evals[device] = last_eval(output / "train.jsonl")
        self.assertAlmostEqual(evals["cpu"]["loss"], evals["mps"]["loss"], delta=1e-4 * abs(evals["cpu"]["loss"]))
        self.assertAlmostEqual(evals["cpu"]["eal"], evals["mps"]["eal"], delta=1e-3)

    def test_selector_prefix_objective_learns(self):
        output = self.root / "prefix"
        self.run_cmd(self.train_cmd(output, "--steps", "200", "--eval-every", "100", "--checkpoint-every", "200",
                                "--objective", "selector_prefix", "--eal-weight", "0.5", ranks=1))
        start, end = first_eval(output / "train.jsonl"), last_eval(output / "train.jsonl")
        print(f"\nselector_prefix (1 rank, 200 steps): eal {start['eal']:.4f} -> {end['eal']:.4f}, "
              f"held-out upstream loss {start['loss']:.4f} -> {end['loss']:.4f}")
        self.assertGreater(end["eal"], start["eal"] + 1.0)


if __name__ == "__main__":
    unittest.main()
