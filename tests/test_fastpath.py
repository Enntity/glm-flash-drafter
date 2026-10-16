"""CPU tests for the round-2 performance path (``fastpath.py`` + trainer flags).

    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python -m pytest tests/test_fastpath.py -v

Tolerances (and why):

* ``packed_sdpa`` mask == upstream ``create_anchor_block_mask_mod`` exactly (bool equality; sliding /
  full, causal / non-causal, 3 documents + pad, unsorted anchors, ragged last chunk).
* packed attention (eager and compiled layers) vs eager attention, fp32, one forward/backward:
  loss rel <= 1e-6, every gradient rel (Frobenius) <= 1e-5 (observed 4e-7 / 5e-7).
* ``--param-dtype bf16`` vs fp32 weights under bf16 autocast: bit-identical loss, gradients and
  three AdamW steps; a 300-step CLI run reproduces every logged eval of round 1 exactly.
* ``--train-metrics light`` vs upstream forward: bit-identical loss, gradients, train EAL.
* 300-step trajectories (this fixture memorises, so rounding differences grow once the loss
  nears 0): round-1 eager vs round-1 sdpa is the noise reference.  The round-2 run must stay
  within 4x that mean |log loss ratio| + 0.01, end within 0.15 EAL of round 1 and drop its
  loss 20x; in fp32 it must also agree within 1e-3 (rel loss) through step 100.
* 2 ranks x accum 2 with every round-2 flag: replicas identical; resume at 3 -> 6 reproduces
  the fp32 masters and the final eval exactly.

Single-threaded BLAS where bitwise equality is asserted: multi-threaded CPU scatter-add
(codebook gradients) and a loaded macOS BLAS are not run-to-run deterministic.
"""

from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch.nn.attention.flex_attention import create_mask  # noqa: E402

from _common import REPO, TINY  # noqa: E402
from test_train import Fixture, first_eval, last_eval  # noqa: E402

from enntity_drafter import data as sd  # noqa: E402
from enntity_drafter import fastpath as fp  # noqa: E402
from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter.guard import guarded_torch_load  # noqa: E402
from enntity_drafter.objectives import loss_kwargs, training_loss  # noqa: E402
from enntity_drafter.train import fixed_anchors_installed, trainable  # noqa: E402

ROUND1 = ("--cpu-autocast", "--attn", "sdpa")
ROUND2 = ("--cpu-autocast", "--attn", fp.PACKED_ATTN, "--anchors-per-chunk", "4", "--param-dtype", "bf16",
          "--compile-layers", "--adamw-impl", "fused", "--deferred-metrics", "--train-metrics", "light")


def tiny_model(attn: str = "eager", window: int = 24, dtype: torch.dtype = torch.float32):
    cfg = sm.load_config(TINY)
    cfg["sliding_window"] = window
    if attn == fp.PACKED_ATTN:
        fp.register_packed_attention()
    model = sm.build_model(cfg, attn_implementation=attn)
    sm.install_frozen_target(model, {k: v.to(dtype) for k, v in sd.synthetic_target(cfg).items()}, "pre-final-norm")
    sm.scratch_init(model, cfg, 3)
    torch.manual_seed(5)
    with torch.no_grad():  # move the zero-initialised branches off zero so their gradients are exercised
        for name, parameter in model.named_parameters():
            if "codebook" in name or "kernel_projection" in name:
                parameter.normal_(0, 0.05)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in sm.FROZEN_KEYS)
    if attn == fp.PACKED_ATTN:
        fp.install_packed_attention(model, anchors_per_chunk=4)
    return cfg, model


def tiny_inputs(cfg, seed: int, dtype: torch.dtype = torch.float32, length: int = 120):
    generator = torch.Generator().manual_seed(seed)
    width = len(cfg["target_layer_ids"]) * cfg["target_hidden_size"]
    docs = torch.tensor([0] * 50 + [1] * 40 + [2] * (length - 90)).unsqueeze(0)
    return {"hidden_states": torch.randn(1, length, width, generator=generator).to(dtype),
            "input_ids": torch.randint(0, cfg["vocab_size"] - 1, (1, length), generator=generator),
            "loss_mask": torch.ones(1, length, dtype=torch.bool),
            "verifier_last_hidden_states": torch.randn(1, length, cfg["target_hidden_size"],
                                                       generator=generator).to(dtype),
            "document_ids": docs, "position_ids": torch.arange(length).unsqueeze(0)}


#: Unsorted, crosses document boundaries, includes position 0, 19 anchors (ragged last chunk of 4).
ANCHORS = [70, 3, 55, 10, 100, 33, 41, 90, 61, 20, 80, 101, 51, 0, 28, 77, 12, 66, 95]
#: For a 104-row sample (second shape for the compiled model).
SHORT_ANCHORS = [70, 3, 55, 10, 33, 41, 90, 61, 20, 80, 51, 0, 28]


def loss_and_grads(model, inputs, anchors, autocast: bool = False, light: bool = False,
                   per_position: str = "fixed-exp-decay", with_metrics: bool = False):
    kwargs = loss_kwargs("cpu", "kl_div", per_position, 4.0, 1.0, 0.5)
    context = torch.autocast("cpu", dtype=torch.bfloat16) if autocast else torch.autocast("cpu", enabled=False)
    with fixed_anchors_installed(model, anchors), context:
        loss, metrics = training_loss("upstream", model, inputs, max_anchors=len(anchors), on_policy=True,
                                      kwargs=kwargs, selector_prefix={}, light_metrics=light)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in trainable(model) if p.grad is not None}
    return (loss.detach(), grads, metrics) if with_metrics else (loss.detach(), grads)


def max_rel(a: dict, b: dict) -> float:
    return max(float((a[k] - b[k]).norm() / b[k].norm().clamp_min(1e-30)) for k in b)


class PackedMaskTests(unittest.TestCase):
    def test_packed_mask_equals_upstream_mask(self):
        from speculators.models.dflash.attention import create_anchor_block_mask_mod

        T, B = 120, 8
        documents = torch.tensor([0] * 50 + [1] * 40 + [2] * 25 + [-1] * 5)
        anchors = torch.tensor(ANCHORS)
        for window, non_causal, per_chunk in ((24, True, 4), (24, False, 3), (None, True, 16), (7, True, 1)):
            mod, q_len, kv_len = create_anchor_block_mask_mod(documents, T, anchors, B, window, non_causal)
            dense = create_mask(mod, None, None, q_len, kv_len, device="cpu")[0, 0]
            packed = fp.build_packed_mask(documents.unsqueeze(0), T, anchors, B, window, non_causal, per_chunk)
            rebuilt = torch.zeros_like(dense)
            chunks, lq = packed.mask.shape[0], packed.mask.shape[2]
            for c in range(chunks):
                for i in range(lq):
                    row = int(packed.q_rows[c * lq + i])
                    rebuilt[row, packed.key_index[c]] |= packed.mask[c, 0, i]
            self.assertTrue(torch.equal(rebuilt, dense), f"window={window} non_causal={non_causal}")
            # Every original query row maps to a packed slot holding that row.
            self.assertTrue(torch.equal(packed.q_rows[packed.out_rows], torch.arange(q_len)))

    def test_attended_keys_counts_mask(self):
        from speculators.models.dflash.attention import create_anchor_block_mask_mod

        documents = torch.tensor([0] * 50 + [1] * 40 + [2] * 30)
        for window in (24, None):
            mod, q_len, kv_len = create_anchor_block_mask_mod(documents, 120, torch.tensor(ANCHORS), 8, window, True)
            dense = create_mask(mod, None, None, q_len, kv_len, device="cpu")
            self.assertEqual(fp.attended_keys(documents, ANCHORS, window, 8), int(dense.sum()))


class ModelEquivalenceTests(unittest.TestCase):
    def test_packed_attention_matches_eager_fp32(self):
        cfg, eager = tiny_model("eager")
        _, packed = tiny_model(fp.PACKED_ATTN)
        packed.load_state_dict(eager.state_dict())
        inputs = tiny_inputs(cfg, 1)
        loss0, grads0 = loss_and_grads(eager, inputs, ANCHORS)
        loss1, grads1 = loss_and_grads(packed, inputs, ANCHORS)
        print(f"\npacked vs eager (fp32): loss {float(loss1):.7f} vs {float(loss0):.7f}, "
              f"max grad rel err {max_rel(grads1, grads0):.1e}")
        self.assertLess(abs(float(loss1 - loss0)), 1e-6 * abs(float(loss0)))
        self.assertLess(max_rel(grads1, grads0), 1e-5)

        fp.compile_layers(packed)
        for parameter in packed.parameters():
            parameter.grad = None
        for seed in (1, 2):   # second call: other data, no new compile needed (dynamic shapes)
            loss2, grads2 = loss_and_grads(packed, tiny_inputs(cfg, seed, length=120 if seed == 1 else 104),
                                           ANCHORS if seed == 1 else SHORT_ANCHORS)
            for parameter in packed.parameters():
                parameter.grad = None
        for parameter in eager.parameters():
            parameter.grad = None
        loss0b, grads0b = loss_and_grads(eager, tiny_inputs(cfg, 2, length=104), SHORT_ANCHORS)
        print(f"compiled packed vs eager (fp32): loss {float(loss2):.7f} vs {float(loss0b):.7f}, "
              f"max grad rel err {max_rel(grads2, grads0b):.1e}")
        self.assertLess(abs(float(loss2 - loss0b)), 1e-6 * abs(float(loss0b)))
        self.assertLess(max_rel(grads2, grads0b), 1e-5)

    def test_flex_compiled_wiring_forward_matches_eager(self):
        """``--attn flex_compiled`` uses upstream's block mask; CPU flex has no backward, so this checks the
        forward (mask + wiring); the GPU gradients are checked by attn_equiv.py (PERF.md)."""

        cfg, eager = tiny_model("eager")
        fp.register_flex_attention()
        _, flex = tiny_model(fp.FLEX_ATTN)
        fp.install_flex_attention(flex)
        flex.load_state_dict(eager.state_dict())
        kwargs = loss_kwargs("cpu", "kl_div", "fixed-exp-decay", 4.0, 1.0, 0.5)
        losses = []
        with torch.no_grad():
            for model in (eager, flex):
                with fixed_anchors_installed(model, ANCHORS):
                    losses.append(float(training_loss("upstream", model, tiny_inputs(cfg, 1), max_anchors=len(ANCHORS),
                                                      on_policy=True, kwargs=kwargs, selector_prefix={})[0]))
        self.assertLess(abs(losses[1] - losses[0]), 1e-6 * abs(losses[0]))

    def test_light_train_metrics_same_loss_and_grads(self):
        for per_position in ("fixed-exp-decay", "dpace"):
            cfg, full = tiny_model("sdpa")
            light = copy.deepcopy(full)
            inputs = tiny_inputs(cfg, 4)
            loss0, grads0, metrics0 = loss_and_grads(full, inputs, ANCHORS, per_position=per_position,
                                                     with_metrics=True)
            loss1, grads1, metrics1 = loss_and_grads(light, inputs, ANCHORS, light=True, per_position=per_position,
                                                     with_metrics=True)
            self.assertTrue(torch.equal(loss0, loss1), per_position)
            self.assertTrue(all(torch.equal(grads0[k], grads1[k]) for k in grads0), per_position)
            for key in ("eal_sum", "eal_total", "loss_sum"):
                self.assertTrue(torch.equal(metrics0[key], metrics1[key]), (per_position, key))
        print("\n--train-metrics light: loss, gradients and train EAL bit-identical (fixed-exp-decay, dpace)")

    def test_bf16_master_weights_bit_identical_under_autocast(self):
        torch.set_num_threads(1)
        cfg, reference = tiny_model("sdpa", dtype=torch.bfloat16)
        candidate = copy.deepcopy(reference)
        master = fp.MasterWeights(trainable(candidate))
        self.assertTrue(master.low and all(p.dtype == torch.bfloat16 for p in master.low))
        self.assertTrue(all(p.dtype == torch.float32 for _, p in master.named))
        opt0 = torch.optim.AdamW([p for _, p in trainable(reference)], lr=1e-3, foreach=True)
        opt1 = torch.optim.AdamW([p for _, p in master.named], lr=1e-3, foreach=True)
        for step in range(3):
            losses = []
            for model, opt in ((reference, opt0), (candidate, opt1)):
                opt.zero_grad(set_to_none=True)
                for micro in range(2):
                    loss, _ = loss_and_grads(model, tiny_inputs(cfg, 10 * step + micro, torch.bfloat16), ANCHORS,
                                             autocast=True)
                losses.append(float(loss))
            self.assertEqual(losses[0], losses[1])
            grads1 = dict((n, p.grad) for n, p in master.named)
            for name, parameter in trainable(reference):
                self.assertTrue(torch.equal(parameter.grad, grads1[name]), f"step {step} grad {name}")
            opt0.step()
            opt1.step()
            master.sync_low()
            params1 = dict(master.named)
            for name, parameter in trainable(reference):
                self.assertTrue(torch.equal(parameter, params1[name]), f"step {step} param {name}")
        print("\nbf16 weights + fp32 masters: 3 AdamW steps x 2 micro-steps bit-identical to autocast over fp32")


class TrajectoryTests(unittest.TestCase, Fixture):
    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def evals(self, output: Path) -> list[dict]:
        return [r for r in map(json.loads, (output / "train.jsonl").read_text().splitlines()) if r["phase"] == "eval"]

    def run_parallel(self, runs: dict[str, tuple[str, ...]], common: tuple[str, ...],
                     prefix: str) -> dict[str, list[dict]]:
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
        procs = {name: subprocess.Popen(self.train_cmd(self.root / f"{prefix}-{name}", *common, *flags, ranks=1),
                                        cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                 for name, flags in runs.items()}
        for name, proc in procs.items():
            out, _ = proc.communicate(timeout=3600)
            if proc.returncode:
                raise AssertionError(f"{name} failed:\n{out[-4000:]}")
        return {name: self.evals(self.root / f"{prefix}-{name}") for name in runs}

    @staticmethod
    def divergence(run: list[dict], reference: list[dict]) -> float:
        """Mean |log(loss / reference loss)| over the evals after step 0."""

        return sum(abs(math.log(a["loss"] / b["loss"])) for a, b in zip(run[1:], reference[1:])) / (len(run) - 1)

    def check_tracks(self, runs: dict[str, list[dict]], label: str, early_tol: float | None) -> None:
        """round2 vs round1 (sdpa), with round1-eager vs round1-sdpa as the noise reference: the two
        round-1 attention kernels compute the same model and differ only in rounding, and this
        memorising fixture amplifies rounding once the loss approaches zero."""

        r1, ref, r2 = runs["round1"], runs["round1-eager"], runs["round2"]
        print(f"\n{label}: {'step':>5} {'round1':>9} {'r1 eager':>9} {'round2':>9} | eal {'round1':>7} {'eager':>7} "
              f"{'round2':>7}")
        for a, b, c in zip(r1, ref, r2):
            print(f"{'':{len(label)}}  {a['step']:5d} {a['loss']:9.5f} {b['loss']:9.5f} {c['loss']:9.5f} |     "
                  f"{a['eal']:7.4f} {b['eal']:7.4f} {c['eal']:7.4f}")
        noise, ours = self.divergence(ref, r1), self.divergence(r2, r1)
        print(f"{'':{len(label)}}  mean |log loss ratio| vs round1: eager {noise:.4f}, round2 {ours:.4f}")
        self.assertLess(ours, 4 * noise + 0.01)
        if early_tol is not None:   # before the loss collapses the runs agree closely
            for a, c in zip(r1, r2):
                if a["step"] <= 100:
                    self.assertLess(abs(c["loss"] - a["loss"]) / a["loss"], early_tol, a["step"])
        self.assertLess(abs(r2[-1]["eal"] - r1[-1]["eal"]), 0.15)
        self.assertLess(r2[-1]["loss"], 0.05 * r2[0]["loss"])

    def test_round2_tracks_round1_fp32(self):
        """Packed attention + compiled layers + fused AdamW + deferred/light metrics, fp32 CPU."""

        r2 = ("--attn", fp.PACKED_ATTN, "--anchors-per-chunk", "4", "--compile-layers", "--adamw-impl", "fused",
              "--deferred-metrics", "--train-metrics", "light")
        runs = self.run_parallel({"round1": ("--attn", "sdpa"), "round1-eager": ("--attn", "eager"), "round2": r2},
                                 ("--steps", "300", "--eval-every", "50", "--checkpoint-every", "300"), "fp32")
        self.check_tracks(runs, "fp32", early_tol=1e-3)

    def test_round2_tracks_round1_bf16_autocast(self):
        """The CUDA numerics (bf16 autocast) on CPU, all round-2 flags incl. bf16 weights."""

        common = ("--steps", "300", "--eval-every", "50", "--checkpoint-every", "300")
        # The exact pair runs alone: under heavy CPU load (inductor compile workers) the macOS BLAS
        # picks other splits and even round 1 against itself stops being bitwise reproducible.
        runs = self.run_parallel({"round1": ROUND1,
                                  "bf16-weights": ROUND1 + ("--param-dtype", "bf16", "--deferred-metrics",
                                                            "--train-metrics", "light")}, common, "bf16")
        runs.update(self.run_parallel({"round1-eager": ("--cpu-autocast", "--attn", "eager"), "round2": ROUND2},
                                      common, "bf16"))
        # bf16 weights + fp32 masters (+ deferred/light metrics) is the same arithmetic: every eval identical.
        self.assertEqual(runs["bf16-weights"], runs["round1"])
        self.assertLess(abs(runs["round2"][0]["loss"] - runs["round1"][0]["loss"]) / runs["round1"][0]["loss"], 1e-4)
        self.check_tracks(runs, "bf16", early_tol=None)

    def test_round2_two_rank_accum_resumes_exactly(self):
        def params(output: Path, step: int, rank: int = 0) -> dict[str, torch.Tensor]:
            return guarded_torch_load(output / "checkpoints" / f"step-{step:07d}" / f"rank{rank}.pt", "drafter")["params"]

        common = ("--steps", "6", "--warmup", "2", "--eval-every", "3", "--checkpoint-every", "3",
                  "--grad-accum", "2", *ROUND2)
        full, resumed = self.root / "r2-full", self.root / "r2-resumed"
        self.run_cmd(self.train_cmd(full, *common))
        self.run_cmd(self.train_cmd(resumed, *common, "--stop-after", "3"))
        self.run_cmd(self.train_cmd(resumed, *common))
        final = params(full, 6)
        self.assertTrue(all(v.dtype == torch.float32 for v in final.values()), "checkpoints hold fp32 masters")
        self.assertTrue(all(torch.equal(v, params(full, 6, 1)[k]) for k, v in final.items()), "replicas diverged")
        self.assertTrue(all(torch.equal(v, params(resumed, 6)[k]) for k, v in final.items()))
        self.assertEqual(last_eval(resumed / "train.jsonl"), last_eval(full / "train.jsonl"))
        self.assertGreater(first_eval(full / "train.jsonl")["loss"], last_eval(full / "train.jsonl")["loss"])
        print("\nround-2 flags, 2 ranks x accum 2: resume at 3 -> 6 reproduces the uninterrupted fp32 masters exactly")

    def test_perf_profile_writes_summary(self):
        output = self.root / "perf"
        self.run_cmd(self.train_cmd(output, *ROUND2, "--perf-profile", "2", "--perf-warmup", "1", "--steps", "100",
                                    "--peak-tflops", "1", ranks=1))
        summary = json.loads((output / "perf-profile.json").read_text())
        self.assertEqual(summary["steps"], 2)
        self.assertTrue(summary["categories"] and summary["top_kernels"])
        self.assertGreater(summary["flops_per_step"]["total"], summary["flops_per_step"]["gemm"])
        self.assertTrue((output / "perf-profile.md").read_text().startswith("# perf profile"))
        self.assertFalse((output / "checkpoints").exists())


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
