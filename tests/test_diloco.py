"""CPU tests for ``--sync-mode diloco`` (enntity_drafter.diloco).

    python -m pytest tests/test_diloco.py -v

* delta codecs: fp32 exact, bf16 and blockwise int8 within tolerance; named delta files check keys/shapes/dtypes;
* the outer step is PyTorch's Nesterov SGD applied to theta_global - theta_local (several fragments);
* 2 workers, H=1, outer lr 1, momentum 0, fp32: exactly synchronous SGD on the averaged gradient (directory
  transport, no process group); int8 deltas stay close; old exchange files are cleaned up;
* trainer, 2 ranks: plain averaging with H=1 and Nesterov momentum with H=20 reach the synchronous run's held-out
  chain proxy at equal tokens; replicas end identical and equal to the global weights;
* an interrupted mid-round run (int8, momentum) resumes exactly; two single-process sites exchanging through a
  directory produce bitwise the same weights as the 2-rank process-group run;
* external transport (volunteer sites): one round per run, starting from a weights file; a mid-round resume
  writes the same delta, and a resume with other start weights is refused.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _common import REPO  # noqa: E402

import torch  # noqa: E402

from enntity_drafter import diloco as dl  # noqa: E402
from enntity_drafter.guard import guarded_load_file, guarded_torch_load, scratch_stamp  # noqa: E402
from test_train import Fixture, first_eval, last_eval  # noqa: E402


def rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def checkpoint(output: Path, step: int, rank: int = 0) -> dict:
    return guarded_torch_load(output / "checkpoints" / f"step-{step:07d}" / f"rank{rank}.pt", "drafter")


def records(output: Path, phase: str) -> list[dict]:
    return [r for r in map(json.loads, (output / "train.jsonl").read_text().splitlines()) if r["phase"] == phase]


class CodecTests(unittest.TestCase):
    def test_round_trip(self):
        torch.manual_seed(0)
        flat = torch.randn(3 * dl.INT8_BLOCK + 17) * 1e-3
        flat[5] = 0.5                                       # an outlier only costs its own block
        flat[dl.INT8_BLOCK: 2 * dl.INT8_BLOCK] = 0.0        # an all-zero block decodes to zeros
        self.assertTrue(torch.equal(dl.decode(dl.encode(flat, "fp32"), flat.numel()), flat))
        self.assertLess(rel(dl.decode(dl.encode(flat, "bf16"), flat.numel()), flat), 4e-3)
        payload = dl.encode(flat, "int8")
        self.assertEqual(payload["q"].dtype, torch.int8)
        decoded = dl.decode(payload, flat.numel())
        self.assertTrue(torch.equal(decoded[dl.INT8_BLOCK: 2 * dl.INT8_BLOCK], flat[dl.INT8_BLOCK: 2 * dl.INT8_BLOCK]))
        rest = torch.cat([flat[2 * dl.INT8_BLOCK:], decoded[2 * dl.INT8_BLOCK:]]).view(2, -1)
        self.assertLess(rel(rest[1], rest[0]), 1e-2)
        self.assertLessEqual(float((decoded - flat).abs().max()), float(flat.abs().max()) / 254 + 1e-9)
        self.assertEqual(dl.payload_bytes(payload), 4 * dl.INT8_BLOCK + 4 * 4)   # 1 B/value + fp32 scale/block

    def test_named_delta_files(self):
        torch.manual_seed(3)
        deltas = {"a.weight": torch.randn(7, 300) * 1e-3, "b": torch.randn(5) * 1e-3}
        shapes = {k: tuple(v.shape) for k, v in deltas.items()}
        for dtype, tolerance in (("fp32", 0.0), ("bf16", 4e-3), ("int8", 1e-2)):
            decoded = dl.decode_named(dl.encode_named(deltas, dtype), shapes, dtype)
            for key, value in deltas.items():
                self.assertLessEqual(rel(decoded[key], value), tolerance, (dtype, key))
                self.assertEqual(decoded[key].shape, value.shape)
        encoded = dl.encode_named(deltas, "bf16")
        with self.assertRaisesRegex(ValueError, "keys differ"):
            dl.decode_named({"a.weight": encoded["a.weight"]}, shapes, "bf16")
        with self.assertRaisesRegex(ValueError, "!="):
            dl.decode_named({**encoded, "b": encoded["b"].float()}, shapes, "bf16")
        with self.assertRaisesRegex(ValueError, "!="):
            dl.decode_named({**encoded, "b": torch.zeros(6, dtype=torch.bfloat16)}, shapes, "bf16")
        with tempfile.TemporaryDirectory() as tmp:
            sha = dl.save_weights(Path(tmp) / "w.safetensors", deltas, "bf16", scratch_stamp(), {"version": 3})
            tensors, meta = dl.load_weights(Path(tmp) / "w.safetensors")
            self.assertEqual((meta["version"], meta["dtype"], len(sha)), (3, "bf16", 64))
            self.assertEqual(tensors["b"].dtype, torch.float32)


class OuterStepTests(unittest.TestCase):
    def test_outer_step_is_nesterov_sgd_on_pseudo_gradient(self):
        torch.manual_seed(1)
        local = [("a", torch.randn(200_000)), ("b", torch.randn(300, 700)), ("c", torch.randn(150_000))]
        sync = dl.Diloco(local, dl.ProcessGroupTransport(1), lr=0.7, momentum=0.9, nesterov=True,
                         delta_dtype="fp32", fragment_mb=1)
        self.assertEqual(len(sync.fragments), 3)
        reference = [t.clone().requires_grad_(True) for _, t in local]
        optimizer = torch.optim.SGD(reference, lr=0.7, momentum=0.9, nesterov=True)
        for _ in range(4):
            for _, tensor in local:   # "inner steps"
                tensor.add_(torch.randn_like(tensor) * 1e-2)
            for ref, (_, tensor) in zip(reference, local):
                ref.grad = (ref.detach() - tensor).clone()
            optimizer.step()
            record = sync.outer_step(0)
            for ref, (name, tensor) in zip(reference, local):
                self.assertLess(rel(tensor, ref.detach()), 1e-6, name)
                self.assertTrue(torch.equal(tensor, sync.views[name]))
        self.assertEqual(record["round"], 4)
        self.assertEqual(record["workers"], 1)
        self.assertEqual(record["wire_bytes_each_way"], 0)

    def test_two_sites_h1_lr1_m0_is_synchronous_sgd(self):
        """Least squares, 2 workers with their own data, inner SGD; one inner step per round, plain
        averaging == one synchronous SGD step on the mean gradient.  int8 deltas track it closely."""

        torch.manual_seed(2)
        X = [torch.randn(64, 32) for _ in range(2)]
        y = [x @ torch.randn(32) + 0.1 * torch.randn(64) for x in X]
        w0, eta, rounds = torch.randn(32), 0.05, 6

        def grad(w, k):
            return 2 * X[k].T @ (X[k] @ w - y[k]) / len(y[k])

        reference = w0.clone()
        for _ in range(rounds):
            reference -= eta * (grad(reference, 0) + grad(reference, 1)) / 2

        def run(delta_dtype: str, root: Path) -> list:
            results: list = [None, None]
            errors: list = []

            def site(k: int) -> None:
                try:
                    w = w0.clone()
                    worker = dl.Diloco([("w", w)], dl.DirectoryTransport(root, k, 2, scratch_stamp(), timeout_s=60),
                                       lr=1.0, momentum=0.0, nesterov=True, delta_dtype=delta_dtype)
                    for _ in range(rounds):
                        w -= eta * grad(w, k)
                        record = worker.outer_step(0)
                    results[k] = (w.clone(), record)
                except Exception as error:  # noqa: BLE001 - re-raised in the test thread
                    errors.append(error)

            threads = [threading.Thread(target=site, args=(k,)) for k in range(2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            if errors:
                raise errors[0]
            return results

        with tempfile.TemporaryDirectory() as tmp:
            (w_a, record), (w_b, _) = run("fp32", Path(tmp) / "fp32")
            self.assertTrue(torch.equal(w_a, w_b))
            self.assertLess(rel(w_a, reference), 1e-6)
            self.assertEqual((record["workers"], record["payload_bytes"], record["wire_bytes_each_way"]),
                             (2, 32 * 4, 32 * 4))
            root = Path(tmp) / "fp32"
            left = sorted(p.relative_to(root).as_posix() for p in root.rglob("*.safetensors"))
            self.assertEqual(left, [f"round-{rounds - 1:06d}/fragment-0000/site{k}.safetensors" for k in range(2)])
            (q_a, record), (q_b, _) = run("int8", Path(tmp) / "int8")
            self.assertTrue(torch.equal(q_a, q_b))
            self.assertLess(rel(q_a - w0, reference - w0), 2e-2)
            self.assertEqual(record["payload_bytes"], dl.INT8_BLOCK + 4)

    def test_global_weights_context_restores_local(self):
        local = [("a", torch.zeros(10))]
        sync = dl.Diloco(local, dl.ProcessGroupTransport(1), lr=1.0, momentum=0.0, nesterov=True, delta_dtype="bf16")
        local[0][1].add_(1.0)
        calls = []
        with sync.global_weights(lambda: calls.append(1)):
            self.assertTrue(torch.equal(local[0][1], torch.zeros(10)))
        self.assertTrue(torch.equal(local[0][1], torch.ones(10)))
        self.assertEqual(len(calls), 2)


class TrainerTests(unittest.TestCase, Fixture):
    @classmethod
    def setUpClass(cls):
        cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_two_rank_diloco_reaches_synchronous_at_equal_tokens(self):
        common = ("--steps", "300", "--eval-every", "100", "--checkpoint-every", "300")
        runs = {"sync": (), "diloco-h1": ("--sync-mode", "diloco", "--inner-steps", "1", "--delta-dtype", "fp32"),
                "diloco-h20": ("--sync-mode", "diloco", "--inner-steps", "20", "--outer-lr", "0.7",
                               "--outer-momentum", "0.9")}
        evals = {}
        for name, extra in runs.items():
            output = self.root / name
            self.run_cmd(self.train_cmd(output, *common, *extra))
            evals[name] = last_eval(output / "train.jsonl")
            print(f"\n{name}: held-out eal {first_eval(output / 'train.jsonl')['eal']:.3f} -> "
                  f"{evals[name]['eal']:.3f}, loss {evals[name]['loss']:.4f}", end="")
            if name != "sync":
                rank0, rank1 = checkpoint(output, 300), checkpoint(output, 300, 1)
                for key, value in rank0["params"].items():
                    self.assertTrue(torch.equal(value, rank1["params"][key]), f"{name} {key}: replicas differ")
                    self.assertTrue(torch.equal(value, rank0["diloco"]["global"][key]), f"{name} {key}: != global")
                outer = records(output, "outer")
                self.assertEqual(len(outer), 300 // int(extra[3]))
                self.assertEqual(rank0["diloco"]["round"], len(outer))
        start = first_eval(self.root / "sync" / "train.jsonl")
        for name in ("diloco-h1", "diloco-h20"):
            self.assertEqual(evals[name]["tokens_seen"], evals["sync"]["tokens_seen"])
            self.assertGreater(evals[name]["eal"], 0.97 * evals["sync"]["eal"], name)
            self.assertLess(evals[name]["loss"], 0.1 * start["loss"], name)
        print()

    def test_resume_exact_and_directory_sites_match_process_group(self):
        common = ("--steps", "12", "--eval-every", "6", "--checkpoint-every", "6", "--sync-mode", "diloco",
                  "--inner-steps", "4", "--delta-dtype", "int8", "--outer-lr", "0.7", "--outer-momentum", "0.9")
        group, resumed = self.root / "dl-group", self.root / "dl-resumed"
        self.run_cmd(self.train_cmd(group, *common))
        self.run_cmd(self.train_cmd(resumed, *common, "--stop-after", "6"))   # mid-round (rounds end at 4, 8, 12)
        self.run_cmd(self.train_cmd(resumed, *common))
        for rank in (0, 1):
            a, b = checkpoint(group, 12, rank), checkpoint(resumed, 12, rank)
            self.assertTrue(all(torch.equal(v, b["params"][k]) for k, v in a["params"].items()))
            for section in ("global", "momentum"):
                self.assertTrue(all(torch.equal(v, b["diloco"][section][k]) for k, v in a["diloco"][section].items()))
            self.assertTrue(any(v.abs().sum() > 0 for v in a["diloco"]["momentum"].values()))
        self.assertEqual(last_eval(resumed / "train.jsonl"), last_eval(group / "train.jsonl"))

        exchange = self.root / "dl-exchange"
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        procs = [subprocess.Popen(self.train_cmd(self.root / f"dl-site{k}", *common, "--diloco-transport", "dir",
                                                 "--diloco-dir", str(exchange), "--sites", "2", "--site", str(k),
                                                 ranks=1),
                                  cwd=REPO, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                 for k in range(2)]
        for proc in procs:
            _, err = proc.communicate(timeout=1800)
            self.assertEqual(proc.returncode, 0, err[-4000:])
        for k in range(2):
            site, rank = checkpoint(self.root / f"dl-site{k}", 12), checkpoint(group, 12, k)
            self.assertTrue(all(torch.equal(v, site["params"][key]) for key, v in rank["params"].items()), k)
        outer = records(self.root / "dl-site0", "outer")
        self.assertEqual([r["workers"] for r in outer], [2, 2, 2])
        self.assertEqual(outer[0]["payload_bytes"], records(group, "outer")[0]["payload_bytes"])

    def test_external_round_resumes_mid_round(self):
        """One external round per run: the delta is global - local; a run interrupted mid-round and resumed
        writes the same delta; resuming mid-round with other start weights is refused."""

        start = self.root / "ext-start"
        self.run_cmd(self.train_cmd(start, "--steps", "1", "--stop-after", "1", "--eval-every", "0", ranks=1))
        params = checkpoint(start, 1)["params"]
        base = self.root / "ext-base.safetensors"
        sha = dl.save_weights(base, params, "fp32", scratch_stamp(), {"version": 0})
        common = ("--steps", "12", "--eval-every", "0", "--checkpoint-every", "100", "--sync-mode", "diloco",
                  "--diloco-transport", "external", "--inner-steps", "4", "--delta-dtype", "fp32",
                  "--start-weights", str(base), "--delta-meta", '{"shard": "s1"}')
        whole, resumed = self.root / "ext-whole", self.root / "ext-resumed"
        self.run_cmd(self.train_cmd(whole, *common, ranks=1))
        self.run_cmd(self.train_cmd(resumed, *common, "--checkpoint-every", "2", "--stop-after", "2", ranks=1))
        self.run_cmd(self.train_cmd(resumed, *common, ranks=1))
        delta_path = whole / "deltas" / "step-0000004.safetensors"
        meta = dl.file_metadata(delta_path, "delta")
        self.assertEqual((meta["shard"], meta["step_start"], meta["step_end"], meta["base"]["sha256"]),
                         ("s1", 0, 4, sha))
        self.assertGreater(meta["tokens"], 0)
        a = guarded_load_file(delta_path, "drafter")
        b = guarded_load_file(resumed / "deltas" / "step-0000004.safetensors", "drafter")
        self.assertTrue(all(torch.equal(v, b[k]) for k, v in a.items()))
        local = checkpoint(whole, 4)["params"]
        for key, value in params.items():
            self.assertTrue(torch.allclose(a[key], value - local[key], atol=0, rtol=0))
        self.assertFalse((resumed / "deltas" / "step-0000008.safetensors").exists())   # one round per run
        other = self.root / "ext-other.safetensors"
        dl.save_weights(other, {k: v + 1e-3 for k, v in params.items()}, "fp32", scratch_stamp(), {"version": 1})
        interrupted = self.root / "ext-interrupted"
        self.run_cmd(self.train_cmd(interrupted, *common, ranks=1))   # round 0: steps 0-4
        self.run_cmd(self.train_cmd(interrupted, *common, "--checkpoint-every", "2", "--stop-after", "6", ranks=1))
        argv = self.train_cmd(interrupted, *common, ranks=1)
        argv[argv.index(str(base))] = str(other)
        with self.assertRaisesRegex(AssertionError, "not the weights this round started from"):
            self.run_cmd(argv)


if __name__ == "__main__":
    unittest.main()
