"""CPU tests for the pipeline scripts: prompt builders, held-out carve, capture collector and checker, export.

    python -m pytest tests/test_scripts.py -v
"""

from __future__ import annotations

import hashlib
import importlib.util
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

from _common import REPO, TINY, TRAIN_MODULE  # noqa: E402

import torch  # noqa: E402

from enntity_drafter import data as sd  # noqa: E402
from enntity_drafter import fp8  # noqa: E402
from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter.prompts import templates  # noqa: E402
from test_capture_io import write_fast  # noqa: E402


def script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(*argv: str, env: dict | None = None) -> str:
    done = subprocess.run([sys.executable, *argv], cwd=REPO, capture_output=True, text=True, timeout=1200,
                          env={**os.environ, "OMP_NUM_THREADS": "1", **(env or {})})
    if done.returncode:
        raise AssertionError(f"{argv[:2]} failed:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")
    return done.stdout


class PromptBuilderTests(unittest.TestCase):
    def test_templates_are_deterministic_and_generation_ready(self):
        kept, dropped = templates.build()
        again, _ = templates.build()
        self.assertEqual([r["sha256"] for r in kept], [r["sha256"] for r in again])
        self.assertEqual(len(kept), 1592)
        for row in kept[:50]:
            self.assertEqual((row["source"], row["split"]), ("template", "train"))
            self.assertEqual(row["prompt_sha"], row["sha256"][:16])
        texts = "\n".join(m["content"] for r in kept for m in r["messages"] if m.get("content"))
        for prompt in templates.EXCLUDED_EXACT:
            self.assertNotIn(prompt, texts)

    def test_coverage_is_independent_of_hash_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            digests = []
            for seed in ("0", "1"):
                out = Path(tmp) / seed
                run("-m", "enntity_drafter.prompts.coverage", "--out", str(out), "--no-tokens", "--scale", "0.05",
                    env={"PYTHONHASHSEED": seed})
                digests.append(hashlib.sha256((out / "train.jsonl").read_bytes()).hexdigest())
                rows = [json.loads(line) for line in (out / "train.jsonl").open()]
                self.assertTrue(all(r["check"]["kind"] for r in rows))
                manifest = json.loads((out / "manifest.json").read_text())
                self.assertEqual(manifest["jsonl"], "train.jsonl")
            self.assertEqual(digests[0], digests[1])

    def test_heldout_split_counts_and_thresholds(self):
        with tempfile.TemporaryDirectory() as tmp:
            prompts = Path(tmp) / "p.jsonl"
            rows = [{"id": f"t{i}", "source": "template", "category": "code" if i % 2 else "prose"} for i in range(200)]
            rows += [{"id": f"s{i}", "source": "structured", "category": "json"} for i in range(400)]
            prompts.write_text("".join(json.dumps(r) + "\n" for r in rows))
            out = Path(tmp) / "held.jsonl"
            script("heldout_split").main(["--prompts", str(prompts), "--out", str(out), "--frac", "0.1",
                                          "--count-sources", "template"])
            held = [json.loads(line) for line in out.open()]
            by = {(r["source"], r["category"]): 0 for r in held}
            for r in held:
                by[(r["source"], r["category"])] += 1
            self.assertEqual((by[("template", "code")], by[("template", "prose")]), (10, 10))
            self.assertLess(abs(by[("structured", "json")] - 40), 20)


def two_rank_segments(root: Path, tag: str, ids: list[int], start: int) -> torch.Tensor:
    """Engine-like segments of one request split over two rank capture dirs; returns slots [6, T, H]."""

    slots = torch.randn(6, len(ids), 64)
    staging = root / "staging" / tag
    write_fast(staging, slots, ids, start, chunk=16)
    for sidecar in staging.glob("*.json"):   # the engine records ids on every rank
        meta = json.loads(sidecar.read_text())
        meta["token_ids"] = ids[meta["pos0"]: meta["pos0"] + meta["rows"]]
        sidecar.write_text(json.dumps(meta))
    for path in sorted(staging.iterdir()):
        rank = path.name.split("-")[0][1:]
        target = root / f"cap{rank}" / "chunks" / tag
        target.mkdir(parents=True, exist_ok=True)
        path.rename(target / path.name)
    return slots


class CaptureScriptTests(unittest.TestCase):
    def test_collect_merges_ranks_verifies_and_marks(self):
        capture, check = script("capture"), script("check_capture")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = list(range(100, 160))
            slots = two_rank_segments(root, "row-a", ids, 8)
            dirs = [root / "cap0", root / "cap1"]
            record = capture.collect(dirs, root / "rows", "row-a", ids, 8, timeout=5)
            self.assertEqual((record["ok"], record["ids_mismatch"], record["ranks"]), (True, 0, [0, 1]))
            self.assertTrue((root / "rows" / "row-a" / capture.MARKER).exists())
            self.assertFalse((root / "cap1" / "chunks" / "row-a").exists())
            lo, hi, got_ids, layers, final = check.load_row(root / "rows" / "row-a")
            self.assertEqual((lo, hi, got_ids), (8, 60, ids[8:]))
            self.assertTrue(torch.equal(layers, slots[:5, 8:].bfloat16()))
            self.assertTrue(torch.equal(final, slots[5, 8:].bfloat16()))
            # FP8 in place: the checker reads the converted row and its error matches the codec's
            fp8.convert_dir(root / "rows" / "row-a")
            lo, hi, _, layers8, final8 = check.load_row(root / "rows" / "row-a")
            self.assertTrue(torch.equal(final8, check.roundtrip(slots[5:, 8:].bfloat16())[0]))
            rel = float((layers8.float() - layers.float()).norm() / layers.float().norm())
            self.assertLess(rel, 0.05)
            # wrong ids or a missing rank are refused
            two_rank_segments(root, "row-b", ids, 0)
            (root / "cap1" / "chunks" / "row-b").rename(root / "lost")
            record = capture.collect(dirs, root / "rows", "row-b", ids, 0, timeout=0.3)
            self.assertEqual((record["ok"], record["err"]), (False, "coverage"))
            two_rank_segments(root, "row-c", ids, 0)
            record = capture.collect(dirs, root / "rows", "row-c", [0] + ids[1:], 0, timeout=5)
            self.assertGreater(record["ids_mismatch"], 0)
            self.assertFalse((root / "rows" / "row-c" / capture.MARKER).exists())

    def test_gates_are_checked(self):
        capture, check = script("capture"), script("check_capture")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dirs = [root / "cap0", root / "cap1"]
            ids = list(range(32))
            torch.manual_seed(0)
            base = two_rank_segments(root, "gate32-a", ids, 0)
            for name in ("gate32-b", "gate32-repeat"):
                two_rank_segments(root, name, ids, 0)
            for name in ("gate32-a", "gate32-b", "gate32-repeat"):
                self.assertTrue(capture.collect(dirs, root / "rows", name, ids, 0, timeout=5)["ok"])

            def rewrite(name: str, slots: torch.Tensor) -> None:   # same tensors as a (b: suffix changed)
                for sidecar in (root / "rows" / name).glob("*.json"):
                    meta = json.loads(sidecar.read_text())
                    kind = sidecar.name.split("-")[1]
                    a, b = meta["pos0"], meta["pos0"] + meta["rows"]
                    part = slots[:5, a:b] if kind == "L" else slots[5:, a:b]
                    sidecar.with_suffix(".bin").write_bytes(part.bfloat16().view(torch.uint16).numpy().tobytes())

            rewrite("gate32-repeat", base)
            mutated = base.clone()
            mutated[:, 20:] += 1.0
            rewrite("gate32-b", mutated)
            out = root / "check.json"
            check.main(["--rows", str(root / "rows"), "--out", str(out)])
            gates = json.loads(out.read_text())["gates"]["32"]
            self.assertEqual(gates, {"repeat_exact": True, "causal_prefix_exact": True, "mutated_suffix_differs": True})


class ExportScriptTests(unittest.TestCase):
    def test_export_of_a_checkpoint_reloads(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = sm.load_config(TINY)
            sd.write_synthetic_target_dir(root / "target", cfg)
            rows = sd.synthetic_rows(cfg, train_rows=4, heldout_rows=2, length=64)
            sd.write_row_store(root / "features", rows, capture_layers=cfg["target_layer_ids"],
                               hidden_size=cfg["hidden_size"], final_hidden="pre-final-norm", target={"name": "t"})
            run(*TRAIN_MODULE, "--config", str(TINY), "--teacher", str(root / "target"),
                "--features", str(root / "features"), "--output", str(root / "run"), "--device", "cpu",
                "--steps", "4", "--warmup", "1", "--anchors", "8", "--eval-anchors", "8", "--eval-every", "4",
                "--checkpoint-every", "4", "--log-every", "4")
            out = run("scripts/export.py", "--config", str(TINY), "--checkpoint",
                      str(root / "run" / "checkpoints" / "step-0000004"), "--out", str(root / "export"))
            report = json.loads(out)
            model = sm.build_model(cfg, device="meta")
            self.assertEqual(report["verify_keys"], len(sm.body_parameter_names(model)))
            self.assertEqual(report["dtypes"], ["torch.bfloat16"])
            provenance = json.loads((root / "export" / "PROVENANCE.json").read_text())
            self.assertEqual((provenance["warm_start"], provenance["exported_from"]["step"]), ("none", 4))


if __name__ == "__main__":
    unittest.main()


class InstallScriptTests(unittest.TestCase):
    @unittest.skipUnless((REPO / "third_party" / "speculators" / ".git").is_dir(), "speculators not cloned")
    def test_speculators_install_keeps_torch_and_skips_torchaudio_torchvision(self):   # dogfood: mismatched wheels
        done = subprocess.run(["bash", str(REPO / "scripts" / "install_speculators.sh"), "--print-deps"],
                              capture_output=True, text=True, env={**os.environ, "PYTHON": sys.executable})
        self.assertEqual(done.returncode, 0, done.stderr)
        deps = done.stdout.split()
        self.assertEqual(deps[-1], "torch==" + torch.__version__.split("+")[0])   # the installed torch, pinned
        names = [re.split(r"[<>=!~;\[]", d, maxsplit=1)[0] for d in deps[:-1]]
        self.assertFalse({"torch", "torchaudio", "torchvision"} & set(names), names)
        self.assertIn("transformers>=5.0.0,<5.17.0", deps)   # speculators' tested range, kept on purpose
        script = (REPO / "scripts" / "install_speculators.sh").read_text()
        self.assertIn('pip install --no-deps "$DIR/hs_connectors" "$DIR"', script)
