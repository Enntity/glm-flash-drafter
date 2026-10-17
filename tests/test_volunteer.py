"""CPU end-to-end and failure-injection tests for volunteer training (coordinator + validator + site client).

    OMP_NUM_THREADS=1 python -m pytest tests/test_volunteer.py -v

End to end (tiny model, real trainer): an in-process coordinator with the local object store, the validator with a
private held-out set, and two sites through ``drafter-join``'s own code: one in recipe mode (synthetic shard recipes)
and one bringing its own data.  Each site first runs a canary that is fully validated; then 3 rounds of H steps.
The merged global after 3 rounds equals, bitwise, the trainer's own 2-site DiLoCo plain-averaging run
(``--diloco-transport dir``).  A sign-flipped delta of honest size is rejected by the held-out check and a large
random one by the norm bound; both stay quarantined and credited as received.

Failure injection (toy weights, a stand-in evaluator): validator crash mid-evaluation, a validator bug that wrongly
rejects and is then fixed (automatic re-validation, merge and credit), stale bases 1 and 2 versions behind (merged
at decayed weight), a coordinator restart mid-round, and a duplicate upload.  Every case ends with no completed
work lost and the right credit.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
os.environ.setdefault("OMP_NUM_THREADS", "1")   # bitwise-reproducible CPU reductions (trainer subprocesses too)
os.environ.setdefault("MKL_NUM_THREADS", "1")

from _common import REPO, TINY  # noqa: E402

import torch  # noqa: E402

from coordinator import terms  # noqa: E402
from coordinator.client import ApiClient, upload_file  # noqa: E402
from enntity_drafter import data as sd  # noqa: E402
from enntity_drafter import diloco as dl  # noqa: E402
from enntity_drafter import join  # noqa: E402
from enntity_drafter import model as sm  # noqa: E402
from enntity_drafter.guard import SCRATCH_STAMP_KEY, guarded_load_file, guarded_torch_load, scratch_stamp  # noqa: E402
from enntity_drafter.train import TRAINER_VERSION  # noqa: E402
from enntity_drafter.validation import Thresholds, evaluate, metrics_of  # noqa: E402
from test_coordinator import Net, send_delta  # noqa: E402
from coordinator.client import ApiError  # noqa: E402
from enntity_drafter.validation import judge  # noqa: E402
from validator.worker import Validator, merge_weights  # noqa: E402

H, ROUNDS = 20, 3
HPARAMS = ["--lr", "2e-3", "--warmup", "30", "--anchors", "32", "--weight-decay", "0.0"]


def run(cmd: list[str]) -> None:
    done = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=1800)
    if done.returncode:
        raise AssertionError(f"{cmd[:4]} failed:\n{done.stdout[-2000:]}\n{done.stderr[-4000:]}")


def write_delta(path: Path, deltas: dict[str, torch.Tensor], base_sha: str, meta: dict) -> Path:
    """A delta file as the trainer writes it (header + per-parameter tensors)."""

    header = {"schema": dl.DELTA_SCHEMA, "delta_dtype": "fp32", "base": {"sha256": base_sha}, "tokens": 1000.0,
              "step_start": 0, "step_end": 20, "inner_steps": 20, **meta}
    dl._save(path, dl.encode_named(deltas, "fp32"), {SCRATCH_STAMP_KEY: json.dumps(scratch_stamp()),
                                                      "delta": json.dumps(header)})
    return path


class Pump:
    """Runs ``validator.step()`` in the background until stopped (the operator's loop)."""

    def __init__(self, validator: Validator):
        self.validator, self.stop, self.errors = validator, threading.Event(), []
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self) -> None:
        while not self.stop.is_set():
            try:
                self.validator.step()
            except Exception as error:  # noqa: BLE001 - surfaced by the test
                self.errors.append(error)
                return
            time.sleep(0.2)

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=600)
        if self.errors:
            raise self.errors[0]


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix="drafter-volunteer-")
        cls.root = Path(cls.tmp.name)
        cls.cfg = sm.load_config(TINY)
        sd.write_synthetic_target_dir(cls.root / "target", cls.cfg)
        recipe = {"builder": "synthetic", "args": {"train_rows_total": 48, "length": 96}, "train_seed": 20261014}
        cls.recipe = {**recipe, "rows": [0, 24]}
        join.synthetic_store(cls.cfg, cls.recipe, cls.root / "store-recipe")    # what the recipe site builds
        join.synthetic_store(cls.cfg, {**recipe, "rows": [24, 48]}, cls.root / "store-byo")   # the BYO site's own
        heldout = sd.synthetic_rows(cls.cfg, train_rows=48, heldout_rows=8, length=96)[48:]   # validator only
        sd.write_row_store(cls.root / "heldout", heldout, capture_layers=cls.cfg["target_layer_ids"],
                           hidden_size=cls.cfg["target_hidden_size"], final_hidden="pre-final-norm",
                           target={"name": "private-heldout"})

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def base_cmd(self, output: Path, features: Path) -> list[str]:
        return [sys.executable, "-m", "enntity_drafter.train", "--config", str(TINY), "--teacher",
                str(self.root / "target"), "--features", str(features), "--output", str(output), "--device", "cpu",
                "--log-every", "50", *HPARAMS]

    def test_rounds_match_trainer_diloco_and_poison_is_rejected(self):
        root = self.root
        # 1. the reference: the trainer's own DiLoCo, two sites exchanging through a directory, plain averaging
        stores = [root / "store-recipe", root / "store-byo"]
        procs = [subprocess.Popen(self.base_cmd(root / f"ref{k}", stores[k]) + [
            "--steps", str(H * ROUNDS), "--eval-every", "0", "--checkpoint-every", str(H * ROUNDS),
            "--sync-mode", "diloco", "--inner-steps", str(H), "--delta-dtype", "fp32", "--diloco-transport", "dir",
            "--diloco-dir", str(root / "exchange"), "--sites", "2", "--site", str(k)], cwd=REPO,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for k in range(2)]
        # 2. version 0: the scratch init, as a trainer checkpoint (lr 0 leaves the weights untouched)
        run(self.base_cmd(root / "init", stores[0]) + ["--lr", "0", "--steps", "1", "--stop-after", "1",
                                                       "--eval-every", "0"])
        params = guarded_torch_load(root / "init" / "checkpoints" / "step-0000001" / "rank0.pt", "drafter")["params"]
        # 3. the network
        net = Net(root / "net", part_size=64 * 2**20, require_canary=True)
        pins = {"trainer": TRAINER_VERSION, "config_sha256": dl.config_sha256(self.cfg)}
        net.admin_call("POST", "/admin/settings/requirements", {"value": pins})
        net.admin_call("POST", "/admin/settings/training", {"value": {
            "inner_steps": H, "steps": H * ROUNDS, "canary_steps": 5, "lr": 2e-3, "warmup": 30, "anchors": 32,
            "weight_decay": 0.0, "delta_dtype": "fp32"}})
        net.admin_call("POST", "/admin/shards", {"recipes": [self.recipe] * (ROUNDS + 1)})
        eval_args = ["--config", str(TINY), "--teacher", str(root / "target"), "--features", str(root / "heldout"),
                     "--device", "cpu", "--eval-anchors", "64"]
        validator = Validator(net.validator(), root / "validator",
                              lambda path: metrics_of(evaluate(path, eval_args)), requirements=pins,
                              publish_dtype="fp32", min_deltas=2, log=lambda m: None)
        validator.bootstrap({k: v.float() for k, v in params.items()}, scratch_stamp(source="test-init"))
        sites = []
        for k, (mode, extra) in enumerate((("recipe", []), ("byo", ["--byo-features", str(stores[1]),
                                                                     "--tag", "my-notes"]))):
            invite = net.admin_call("POST", "/admin/invites", {})
            home = root / f"home{k}"
            join.redeem(invite["url"], None if k else "Recipe lab", True, home)
            opts = join.build_parser().parse_args([
                "--home", str(home), "run", "--work", str(root / f"work{k}"), "--config", str(TINY), "--teacher",
                str(root / "target"), "--mode", mode, *extra, "--eval-inprocess",
                "--trainer-arg=--device=cpu", "--trainer-arg=--sites=2", f"--trainer-arg=--site={k}"])
            sites.append(join.site_from(opts))
            sites[-1].sleep = lambda s: time.sleep(0.2)
        pump = Pump(validator)
        try:
            for site in sites:   # canaries: fully validated before any long round
                result = site.cycle()
                self.assertTrue(result["canary"])
            for round_ in range(ROUNDS):
                for site in sites:
                    result = site.cycle()
                    self.assertEqual((result["canary"], result["base_version"]), (False, round_))
                deadline = time.time() + 600
                while net.service.version_url()["version"] != round_ + 1:
                    self.assertLess(time.time(), deadline, "merge did not happen")
                    if pump.errors:
                        raise pump.errors[0]
                    time.sleep(0.2)
        finally:
            pump.close()
        for proc in procs:
            _, err = proc.communicate(timeout=1800)
            self.assertEqual(proc.returncode, 0, err[-3000:])
        # 4. the merged global equals the trainer's own DiLoCo result, bitwise
        merged, _ = dl.load_weights(validator.master(ROUNDS))
        reference = guarded_torch_load(root / "ref0" / "checkpoints" / f"step-{H * ROUNDS:07d}" / "rank0.pt",
                                       "drafter")["params"]
        for name, value in reference.items():
            self.assertTrue(torch.equal(merged[name], value), name)
        for site in sites:
            site.heartbeat()   # the client's own heartbeat passes the coordinator's whitelist
        board = net.service.dashboard()
        by_name = {s["name"]: s for s in board["sites"]}
        self.assertEqual({s["status"] for s in board["sites"]}, {"idle"})
        self.assertEqual(by_name["Recipe lab"]["deltas_accepted"], ROUNDS + 1)   # rounds + canary
        self.assertEqual(by_name["anonymous"]["deltas_accepted"], ROUNDS + 1)
        self.assertEqual(len(board["curve"]), ROUNDS + 1)
        self.assertGreater(board["curve"][-1]["metrics"]["heldout"], board["curve"][0]["metrics"]["heldout"])
        for k, site in enumerate(sites):
            site._drain_outbox()   # what the next cycle does first
            self.assertEqual(site.state["outbox"], [], "local deltas are deleted after receipt and a verdict")
            self.assertEqual(list((root / f"work{k}" / "outbox").glob("*.safetensors")), [])
            self.assertFalse(any((root / f"work{k}" / "data").glob("*/manifest.json")), "activations deleted")
        self.assertTrue(stores[1].joinpath("manifest.json").is_file(), "bring-your-own data is never deleted")
        print(f"\n{ROUNDS} rounds, 2 sites: merged global == trainer DiLoCo (dir transport) bitwise; held-out eal "
              f"{board['curve'][0]['metrics']['heldout']:.3f} -> {board['curve'][-1]['metrics']['heldout']:.3f}")

        # 5. poisoned deltas from a third site: rejected (quarantined) but received and credited as received
        current = net.service.version_url()
        v0 = {k: v.float() for k, v in params.items()}
        history = sorted(validator.state["history"])
        typical = history[len(history) // 2]
        backward = {k: merged[k] - v0[k] for k in merged}
        scale = typical / (sum(float(t.square().sum()) for t in backward.values())
                           / sum(t.numel() for t in backward.values())) ** 0.5
        meta = {"trainer": pins["trainer"], "config_sha256": pins["config_sha256"]}
        flipped = write_delta(root / "flipped.safetensors", {k: v * scale for k, v in backward.items()},
                              current["sha256"], meta)
        noise = write_delta(root / "noise.safetensors", {k: torch.randn_like(v) * typical * 100 for k, v in
                                                          merged.items()}, current["sha256"], meta)
        mallory = net.register("mallory")
        net.pass_canary(mallory)
        ids = [send_delta(mallory, path, current["version"], current["sha256"], f"mallory-{i:04d}",
                          meta={"tokens": 500.0, **pins})["delta_id"] for i, path in enumerate((flipped, noise))]
        validator.step()
        status = [mallory.call("GET", f"/api/v1/deltas/{i}") for i in ids]
        self.assertEqual([s["state"] for s in status], ["quarantined", "quarantined"])
        self.assertEqual([s["reason_code"] for s in status], ["heldout_regression", "norm_high"])
        self.assertTrue(all(s["reason"] for s in status))   # a human-readable reason for the site
        self.assertEqual(net.service.version_url()["version"], ROUNDS)   # nothing merged
        board = net.service.dashboard()
        self.assertEqual({s["name"]: s for s in board["sites"]}["mallory"]["tokens_received"], 1000.0)
        self.assertEqual(board["rejections"], {"heldout_regression": 1, "norm_high": 1})
        net.stop()


class ClientTests(unittest.TestCase):
    def test_setup_smoke_and_invite_parsing(self):
        opts = join.build_parser().parse_args(["setup", "--work", tempfile.gettempdir(), "--no-canary"])
        ok, detail = join.smoke(opts)
        self.assertTrue(ok, detail)
        self.assertIn("python >= 3.10", [check for check, _, _ in join.preflight(opts)])
        self.assertEqual(join.parse_invite("https://example.org/drafter/join#invite=inv_abc"),
                         ("https://example.org/drafter", "inv_abc"))
        with self.assertRaises(join.JoinError):
            join.parse_invite("https://example.org/drafter/join?invite=inv_abc")   # never in the query string
        with self.assertRaises(join.JoinError):
            join.parse_invite("http://example.org/drafter/join#invite=inv_abc")    # no tokens over plain HTTP

    def test_coordinator_supplied_shard_ids_cannot_escape_the_work_dir(self):   # review 11
        with tempfile.TemporaryDirectory() as tmp:
            opts = join.build_parser().parse_args(["run", "--work", tmp, "--config", str(TINY)])
            site = join.Site(ApiClient("http://127.0.0.1:9", retries=0), "token", Path(tmp) / "w", opts)
            for bad in ("../../etc", "a/b", "..", "", "x" * 65, None):
                with self.assertRaises(join.JoinError, msg=repr(bad)):
                    site.shard_id({"id": bad})
            with self.assertRaises(join.JoinError):
                site.inside(Path(tmp) / "w" / "data" / ".." / "..")
            victim = Path(tmp) / "keep"
            victim.mkdir()
            cycle = {"canary": False, "shard": {"id": "../../keep", "recipe": {"builder": "synthetic", "rows": [0, 1]}}}
            with self.assertRaises(join.JoinError):
                site._data(cycle)
            self.assertTrue(victim.is_dir())

    def test_every_server_code_has_a_client_action_that_keeps_the_delta(self):   # re-review N1
        import re

        codes = set()
        for path in ("coordinator/service.py", "coordinator/server.py", "coordinator/client.py",
                     "coordinator/storage.py"):
            codes |= set(re.findall(r'ApiError\(\w*?\d*, "(\w+)"', (REPO / path).read_text()))
        codes |= {"rate_limited", "internal"}
        self.assertGreater(len(codes), 25)
        self.assertEqual(sorted(codes - set(join.UPLOAD_ACTIONS)), [])
        with tempfile.TemporaryDirectory() as tmp:
            opts = join.build_parser().parse_args(["run", "--work", tmp, "--config", str(TINY)])
            site = join.Site(ApiClient("http://127.0.0.1:9", retries=0), "token", Path(tmp), opts)
            for code in sorted(codes):
                delta = Path(tmp) / "outbox" / f"{code}.safetensors"
                delta.parent.mkdir(exist_ok=True)
                delta.write_bytes(b"finished work")
                cycle = {"client_id": f"cid{code}", "canary": False, "stage": "upload", "delta": str(delta),
                         "delta_id": "d_1", "base": {"version": 0}}
                site.state.update(cycle=cycle, parked=[])
                error = site.upload_error(cycle, ApiError(409, code, "x"))
                action = join.UPLOAD_ACTIONS[code][0]
                kept = [p for p in Path(tmp).rglob(f"{code}.safetensors")]
                self.assertEqual([p.read_bytes() for p in kept], [b"finished work"], code)   # never discarded
                if action == "retry":
                    self.assertGreater(error.wait, 0, code)   # backs off; the cycle is untouched
                    self.assertIs(site.state["cycle"], cycle)
                elif action == "park":
                    self.assertEqual((site.state["cycle"], site.state["parked"]), (None, [cycle]))
                    self.assertEqual(error.wait, 0)
                elif action == "new_id":
                    self.assertNotEqual(cycle["client_id"], f"cid{code}")   # a different request next time
                elif action in ("aside", "aside_stop"):
                    self.assertIsNone(site.state["cycle"])
                    self.assertIn("unmergeable", str(kept[0]))
                    self.assertEqual(error.wait, 0 if action == "aside" else None)
                else:
                    self.assertEqual((action, error.wait), ("stop", None))
                    self.assertIs(site.state["cycle"], cycle)   # resumes once the user or operator acted

    def test_parked_delta_is_uploaded_after_the_canary(self):   # re-review N1
        with tempfile.TemporaryDirectory() as tmp:
            opts = join.build_parser().parse_args(["run", "--work", tmp, "--config", str(TINY), "--mode", "byo",
                                                   "--byo-features", tmp])
            uploaded = []

            class Fake:
                token = "token"

                def call(self, method, path, body=None, auth=True):
                    if path == "/api/v1/requirements":
                        return {**site.pins(), "version": 0, "canary_required": False, "open_deltas": 0,
                                "max_open_deltas": 3, "terms_sha256": terms.TERMS_SHA256}
                    if method == "POST" and path == "/api/v1/deltas":
                        uploaded.append(body["client_id"])
                        return {"delta_id": "d_parked", "state": "pending", "duplicate": False, "upload_id": "u"}
                    return {"state": "pending"}

            site = join.Site(Fake(), "token", Path(tmp), opts)
            delta = Path(tmp) / "outbox" / "parked.safetensors"
            delta.parent.mkdir()
            delta.write_bytes(b"x")
            cycle = {"client_id": "cid-parked", "canary": False, "stage": "upload", "delta": str(delta), "size": 1,
                     "sha256": "0" * 64, "base": {"version": 0, "sha256": "1" * 64}, "meta": {"tokens": 5.0}}
            site.state["cycle"] = cycle
            site.upload_error(cycle, ApiError(409, "canary_required", "canary first"))
            result = site.cycle()   # the canary has passed (canary_required is false): the parked delta goes up
            self.assertEqual((result["delta_id"], uploaded), ("d_parked", ["cid-parked"]))
            self.assertEqual(site.state["parked"], [])

    def test_int8_is_the_default_delta_format(self):
        from enntity_drafter import train

        args = train.parse_args(["--config", str(TINY), "--teacher", "t", "--features", "f", "--output", "o"])
        self.assertEqual(args.delta_dtype, "int8")

    def test_judge_fails_closed_and_weights_are_capped_per_site(self):   # review 1 and 2
        thresholds = Thresholds()
        base = {"heldout": 5.0, "agentic": 4.0, "heldout.prose": 3.0}
        for candidate in ({"heldout": float("nan"), "agentic": 4.0}, {"heldout": 5.0},
                          {"heldout": float("inf"), "agentic": 4.0}, {"heldout": "5", "agentic": 4.0}):
            self.assertEqual(judge(base, candidate, thresholds)[:2], ("reject", "heldout_invalid"), candidate)
        self.assertEqual(judge(base, {"heldout": 5.0, "agentic": 4.0}, thresholds)[0], "accept")

        def item(site, tokens):
            return {"site": site, "meta": {"tokens": tokens}, "verdict": {}, "base_version": 5}

        self.assertEqual(merge_weights([item("a", 1e12), item("b", 1000.0)], 5, thresholds), [1.0, 1.0])
        weights = merge_weights([item("a", 1e12), item("a", 1e12), item("b", 1000.0), item("c", 2000.0),
                                 item("d", 3000.0)], 5, thresholds)
        self.assertEqual(weights, [3000.0, 3000.0, 1000.0, 2000.0, 3000.0])   # a: 2 x median site, split


# --------------------------------------------------------------------------
# failure injection on toy weights
# --------------------------------------------------------------------------

PINS = {"trainer": "scratch-v1", "config_sha256": "c" * 64}


def sha256_of(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Killed(BaseException):
    pass


class Toy:
    """A 2-tensor "model": held-out score rises as weights approach a fixed target."""

    def __init__(self, seed: int = 0):
        generator = torch.Generator().manual_seed(seed)
        self.target = {"a": torch.randn(64, 32, generator=generator), "b": torch.randn(128, generator=generator)}
        self.start = {k: v + torch.randn(v.shape, generator=generator) for k, v in self.target.items()}
        self.calls = 0
        self.fail_on: set[int] = set()
        self.kill = False   # True: the failure is the process dying (not catchable by the validator loop)

    def distance(self, weights: dict[str, torch.Tensor]) -> float:
        return sum(float((weights[k] - self.target[k]).square().sum()) for k in self.target) ** 0.5

    def evaluate(self, path: Path) -> dict[str, float]:
        self.calls += 1
        if self.calls in self.fail_on:
            if self.kill:
                raise Killed("process killed during evaluation")
            raise RuntimeError("injected crash during evaluation")
        weights, _ = dl.load_weights(path)
        score = 8.0 - 6.0 * self.distance(weights) / self.distance(self.start)
        return {"heldout": score, "heldout.prose": score - 0.5}

    def honest(self, base: dict[str, torch.Tensor], step: float = 0.2, seed: int = 0) -> dict[str, torch.Tensor]:
        """delta = base - local, where local moved ``step`` toward the target plus its own noise (sites differ)."""

        generator = torch.Generator().manual_seed(1000 + seed)
        return {k: -step * (self.target[k] - base[k]) + 0.5 * step * torch.randn(base[k].shape, generator=generator)
                for k in base}


class FailureInjectionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.net = Net(self.root / "net")
        self.net.admin_call("POST", "/admin/settings/requirements", {"value": PINS})
        self.toy = Toy()
        self.vtoken = self.net.validator().token
        self.validator = self.make_validator("1")
        self.validator.bootstrap(self.toy.start, scratch_stamp(source="toy"))
        self.sites = [self.net.register(name) for name in ("ann", "ben")]
        self.count = 0

    def tearDown(self):
        self.net.stop()
        self.tmp.cleanup()

    def make_validator(self, version: str, **thresholds) -> Validator:
        return Validator(ApiClient(self.net.base, self.vtoken, retries=0), self.root / "validator",
                         self.toy.evaluate, requirements=PINS, version=version, publish_dtype="fp32", min_deltas=2,
                         thresholds=Thresholds(**thresholds), clock=self.net.clock, log=lambda m: None)

    def current(self) -> tuple[int, str, dict[str, torch.Tensor]]:
        info = self.net.service.version_url()
        return info["version"], info["sha256"], dl.load_weights(self.validator.published(info["version"]))[0]

    def upload(self, site: ApiClient, version: int | None = None, validator_version: str = "1") -> str:
        current, _, _ = self.current()
        version = current if version is None else version
        sha = self.validator.state["sha"][str(version)]
        base = dl.load_weights(self.validator.published(version))[0]
        self.count += 1
        path = write_delta(self.root / f"delta{self.count}.safetensors",
                           self.toy.honest(base, 0.15 + 0.01 * self.count, self.count), sha, PINS)
        reply = send_delta(site, path, version, sha, f"client-{self.count:04d}",
                           meta={"validator_version": validator_version})
        return reply["delta_id"]

    def received(self, site: ApiClient) -> dict:
        return site.call("GET", "/api/v1/me")

    def test_validator_crash_mid_evaluation_loses_nothing(self):
        first = self.upload(self.sites[0])
        self.toy.fail_on, self.toy.kill = {self.toy.calls + 1}, True   # the process dies in the next evaluation
        with self.assertRaises(Killed):
            self.validator.step()
        self.assertEqual(self.sites[0].call("GET", f"/api/v1/deltas/{first}")["state"], "validating")
        self.assertEqual(self.received(self.sites[0])["tokens_received"], 1000.0)   # credited while stuck
        restarted = self.make_validator("1")
        self.assertEqual(restarted.step()["verdicts"], [])   # still claimed by the crashed run
        self.net.clock.t += self.net.config.claim_timeout_s + 1
        second = self.upload(self.sites[1])
        restarted.step()
        for site, delta in ((self.sites[0], first), (self.sites[1], second)):
            self.assertEqual(site.call("GET", f"/api/v1/deltas/{delta}")["state"], "merged")
            self.assertEqual(self.received(site)["deltas_accepted"], 1)

    def test_validator_bug_then_fix_revalidates_and_merges(self):
        buggy = self.make_validator("1", reject_drop=-1.0)   # wrongly rejects everything
        ids = [self.upload(site) for site in self.sites]
        buggy.step()
        for site, delta in zip(self.sites, ids):
            status = site.call("GET", f"/api/v1/deltas/{delta}")
            self.assertEqual((status["state"], status["reason_code"]), ("quarantined", "heldout_regression"))
            self.assertEqual(self.received(site)["tokens_received"], 1000.0)   # still credited
        fixed = self.make_validator("2")
        fixed.step()   # hello with the new version re-queues the quarantined deltas
        for site, delta in zip(self.sites, ids):
            self.assertEqual(site.call("GET", f"/api/v1/deltas/{delta}")["state"], "merged")
            self.assertEqual(self.received(site)["deltas_accepted"], 1)
        self.assertEqual(self.current()[0], 1)
        events = [e["event"] for e in self.net.admin_call("GET", "/admin/audit")]
        self.assertIn("validator.version", events)

    def test_stale_bases_merge_at_decayed_weight(self):
        for site in self.sites:
            self.upload(site)
        self.validator.step()                                     # v1 from two fresh deltas
        stale1 = self.upload(self.sites[0], version=0)            # 1 version behind
        self.upload(self.sites[1])
        self.validator.step()                                     # v2
        stale2 = self.upload(self.sites[0], version=0)            # 2 versions behind
        self.upload(self.sites[1])
        self.validator.step()                                     # v3
        self.assertEqual(self.current()[0], 3)
        weights = {}
        with self.net.service.tx() as db:
            for row in db.execute("SELECT id, weight, state FROM deltas"):
                weights[row["id"]] = (row["weight"], row["state"])
        decay = Thresholds().stale_decay
        self.assertEqual(weights[stale1], (decay, "merged"))        # 2 sites: equal per site, then decayed
        self.assertEqual(weights[stale2], (decay ** 2, "merged"))
        self.assertTrue(all(state == "merged" for _, state in weights.values()))
        distances = [self.toy.distance(dl.load_weights(self.validator.published(v))[0]) for v in range(4)]
        self.assertEqual(distances, sorted(distances, reverse=True))   # every merge moved toward the target

    def upload_deltas(self, site: ApiClient, deltas: dict, name: str) -> str:
        version, sha, _ = self.current()
        path = write_delta(self.root / f"{name}.safetensors", deltas, sha, PINS)
        return send_delta(site, path, version, sha, f"client-{name}")["delta_id"]

    def states(self) -> dict[str, tuple]:
        with self.net.service.tx() as db:
            return {r["id"]: (r["state"], r["reason_code"]) for r in db.execute("SELECT * FROM deltas")}

    def test_near_duplicate_of_an_accepted_delta_is_quarantined(self):   # review 2
        _, _, base = self.current()
        honest = self.toy.honest(base, 0.2, 7)
        first = self.upload_deltas(self.sites[0], honest, "orig")
        copy = self.upload_deltas(self.sites[1], {k: v * 1.001 for k, v in honest.items()}, "copy")
        self.validator.min_deltas = 3
        self.validator.step()
        states = self.states()
        self.assertEqual(states[first], ("accepted", None))
        self.assertEqual(states[copy], ("quarantined", "near_duplicate"))

    def test_regressing_merge_is_refused_and_its_deltas_held(self):   # review 3
        self.validator.state["best"] = {"version": 0, "metrics": {"heldout": 100.0}}   # a much better global
        ids = [self.upload(site) for site in self.sites]
        self.validator.step()
        self.assertEqual(self.current()[0], 0)
        self.assertEqual({self.states()[i] for i in ids}, {("held", "merge_regression")})
        self.assertIsNone(self.validator.state["pending_merge"])
        self.assertEqual(len(self.net.admin_call("GET", "/admin/attention")["needs_review"]), 2)
        for delta in ids:   # the gate still refuses: the operator releases them with an audited override
            self.net.admin_call("POST", f"/admin/deltas/{delta}/release", {"decision": "accept", "publish": True})
        self.validator.step()
        self.assertEqual(self.current()[0], 1)
        self.assertEqual({self.states()[i][0] for i in ids}, {"merged"})
        self.assertIn("merge.gate_overridden", [e["event"] for e in self.net.admin_call("GET", "/admin/audit")])
        self.assertFalse(self.net.admin_call("GET", "/admin/attention")["force_merge_pending"])

    def test_gate_compares_one_coherent_best_version_with_a_noise_margin(self):   # re-review N2
        self.validator.state["best"] = {"version": 4, "metrics": {"a": 5.0, "b": 4.0}}
        self.validator.state["reference_metrics"] = {"a": 1.0, "b": 1.0}
        self.validator.state["noise"] = 0.0
        # a per-group maximum over versions ({"a": 5.0, "b": 4.2}) would refuse this forever
        self.assertIsNone(self.validator._gate({"a": 5.0, "b": 4.05}))
        self.assertIsNotNone(self.validator._gate({"a": 4.8, "b": 4.05}))
        self.validator.state["noise"] = 0.05   # measured eval noise widens the margin
        self.assertIsNone(self.validator._gate({"a": 4.8, "b": 4.05}))
        self.assertIn("noise", self.validator.state)

    def test_crash_between_merge_and_publish_is_deterministic(self):   # review 5
        self.validator.outer = {"lr": 1.0, "momentum": 0.9, "nesterov": True}
        ids = [self.upload(site) for site in self.sites]
        real, failed = self.validator.api.call, []

        def flaky(method, path, body=None, auth=True):
            if path.endswith("/publish") and not failed:
                failed.append(path)
                raise ApiError(503, "unavailable", "injected crash before publish")
            return real(method, path, body, auth)

        self.validator.api.call = flaky
        with self.assertRaises(ApiError):
            self.validator.step()
        plan = self.validator.state["pending_merge"]
        self.assertEqual(sorted(i["delta_id"] for i in plan["items"]), sorted(ids))
        first_master = sha256_of(self.validator.master(1))
        first_momentum = sha256_of(self.validator.workdir / "versions" / "momentum-v000001.safetensors")
        inputs = {i["delta_id"]: guarded_load_file(self.validator.workdir / "inbox" / f"{i['delta_id']}.safetensors",
                                                   "drafter") for i in plan["items"]}   # fp32 deltas
        late = self.upload(self.sites[0])   # arrives after the crash: not part of the pinned merge
        self.validator.step()
        self.assertEqual(self.current()[0], 1)
        self.assertEqual(sha256_of(self.validator.master(1)), first_master)   # same inputs, same file
        self.assertEqual(sha256_of(self.validator.workdir / "versions" / "momentum-v000001.safetensors"),
                         first_momentum)   # momentum applied once
        self.assertEqual(self.states()[late][0], "accepted")
        master0, _ = dl.load_weights(self.validator.master(0))
        master1, _ = dl.load_weights(self.validator.master(1))
        weights = [i["weight"] for i in plan["items"]]
        for name in master0:
            mean = sum(w * inputs[i["delta_id"]][name] for i, w in zip(plan["items"], weights)) / sum(weights)
            expected = master0[name] - (mean + 0.9 * mean)   # Nesterov, first step from zero momentum
            self.assertTrue(torch.allclose(master1[name], expected, atol=1e-6), name)

    def test_validator_error_is_retried_and_the_loop_continues(self):   # review 12, re-review N3
        first = self.upload(self.sites[0])
        second = self.upload(self.sites[1])
        self.toy.fail_on = {self.toy.calls + 1}   # a transient failure while evaluating the first delta
        self.validator.min_deltas = 3
        self.validator.step()                        # no exception escapes
        self.assertEqual(self.states()[first][0], "pending")   # retried with backoff, not the contributor's fault
        self.assertEqual(self.sites[0].call("GET", f"/api/v1/deltas/{first}")["review"], "retry scheduled")
        self.assertEqual(self.states()[second][0], "accepted")
        self.assertEqual(self.received(self.sites[0])["tokens_received"], 1000.0)
        self.validator.min_deltas = 2
        self.validator.step()
        self.assertEqual(self.states()[first][0], "pending")   # its backoff has not passed
        self.net.clock.t += 61
        self.validator.step()
        self.assertEqual({self.states()[d][0] for d in (first, second)}, {"merged"})

    def test_relased_shard_duplicate_merges_at_reduced_weight(self):   # re-review N4
        self.net.admin_call("POST", "/admin/shards", {"recipes": [{"builder": "synthetic", "rows": [0, 8]}]})
        version, sha, base = self.current()
        honest = self.toy.honest(base, 0.2, 11)
        first_lease = self.sites[0].call("POST", "/api/v1/shards/claim")["shard"]
        self.net.clock.t += self.net.config.lease_s + self.net.config.lease_grace_s + 1   # the lease expires
        second_lease = self.sites[1].call("POST", "/api/v1/shards/claim")["shard"]
        self.assertEqual(first_lease["id"], second_lease["id"])
        self.assertNotEqual(first_lease["lease_id"], second_lease["lease_id"])
        ids = []
        for i, (site, lease, scale) in enumerate(((self.sites[0], first_lease, 1.0),
                                                  (self.sites[1], second_lease, 1.0005))):
            path = write_delta(self.root / f"lease{i}.safetensors", {k: v * scale for k, v in honest.items()}, sha,
                               PINS)
            ids.append(send_delta(site, path, version, sha, f"lease-{i:04d}", mode="recipe", shard_id=lease["id"],
                                  lease_id=lease["lease_id"])["delta_id"])
        self.validator.min_deltas = 3
        self.validator.step()
        self.assertEqual(self.states()[ids[0]], ("accepted", None))
        self.assertEqual(self.states()[ids[1]], ("accepted", "duplicate_lease"))
        self.assertEqual(len(list((self.validator.workdir / "sketches").glob("*"))), 2)   # one stored sketch each

    def test_object_changed_after_acceptance_is_quarantined_at_merge(self):   # review 10
        self.validator.min_deltas = 3
        a, b = self.upload(self.sites[0]), self.upload(self.sites[1])
        self.validator.step()
        (self.validator.workdir / "inbox" / f"{a}.safetensors").unlink()
        with self.net.service.tx() as db:
            key = db.execute("SELECT object_key FROM deltas WHERE id=?", (a,)).fetchone()["object_key"]
        stored = self.root / "net" / "objects" / key
        stored.write_bytes(os.urandom(stored.stat().st_size))   # overwritten after acceptance
        self.validator.min_deltas = 1
        self.validator.step()
        self.assertEqual(self.states()[a], ("quarantined", "hash_mismatch"))
        self.assertEqual(self.states()[b][0], "merged")

    def test_globals_are_signed_and_sites_verify_the_signature(self):   # review LOW
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

        key, other = Ed25519PrivateKey.generate(), Ed25519PrivateKey.generate()
        self.validator.signing_key = key
        public = key.public_key().public_bytes_raw().hex()
        self.net.admin_call("POST", "/admin/settings/publish_key", {"value": public})
        for site in self.sites:
            self.upload(site)
        self.validator.step()
        opts = join.build_parser().parse_args(["run", "--work", str(self.root / "w"), "--config", str(TINY)])
        site = join.Site(self.sites[0], self.sites[0].token, self.root / "w", opts, config={})
        req, info = site.api.call("GET", "/api/v1/requirements"), site.api.call("GET", "/api/v1/global")
        self.assertEqual(info["version"], 1)
        site.verify_global(req, info)   # pins the key and checks v1
        self.assertEqual(site.config["publish_key"], public)
        with self.assertRaises(join.JoinError):
            site.verify_global(req, {**info, "signature": "00" * 64})
        with self.assertRaises(join.JoinError):
            site.verify_global(req, {**info, "sha256": "f" * 64})
        with self.assertRaises(join.JoinError):
            site.verify_global({**req, "publish_key": other.public_key().public_bytes_raw().hex()}, info)

    def test_recipe_spot_check_catches_a_fabricated_delta(self):
        self.net.admin_call("POST", "/admin/shards", {"recipes": [{"builder": "synthetic", "rows": [i, i + 8]}
                                                                  for i in (0, 8)]})

        def replay(claim: dict, base_path: Path) -> Path:   # the operator re-runs the recipe from the same base
            base = dl.load_weights(base_path)[0]
            return write_delta(self.root / f"replay-{claim['delta_id']}.safetensors", self.toy.honest(base, 0.15),
                               claim["base_sha256"], PINS)

        self.validator.replayer, self.validator.spot_check_rate = replay, 1.0
        version, sha, base = self.current()
        honest = self.toy.honest(base, 0.16)
        size = sum(float(t.square().sum()) for t in honest.values()) ** 0.5
        generator = torch.Generator().manual_seed(5)
        fake = {k: torch.randn(v.shape, generator=generator) for k, v in honest.items()}
        scale = size / sum(float(t.square().sum()) for t in fake.values()) ** 0.5
        ids = []
        for i, (site, deltas) in enumerate(((self.sites[0], honest), (self.sites[1],
                                                                      {k: v * scale for k, v in fake.items()}))):
            shard = site.call("POST", "/api/v1/shards/claim")["shard"]
            path = write_delta(self.root / f"recipe{i}.safetensors", deltas, sha, PINS)
            ids.append(send_delta(site, path, version, sha, f"recipe-{i:04d}", mode="recipe",
                                  shard_id=shard["id"])["delta_id"])
        self.validator.step()
        states = [site.call("GET", f"/api/v1/deltas/{d}") for site, d in zip(self.sites, ids)]
        self.assertEqual(states[0]["state"], "accepted")
        self.assertEqual((states[1]["state"], states[1]["reason_code"]), ("quarantined", "spot_check"))

    def test_garbage_upload_is_quarantined_without_crashing_the_validator(self):
        version, sha, _ = self.current()
        junk = self.root / "junk.safetensors"
        junk.write_bytes(os.urandom(5000))
        bad = send_delta(self.sites[0], junk, version, sha, "client-junk")["delta_id"]
        good = self.upload(self.sites[1])
        self.validator.step()
        status = self.sites[0].call("GET", f"/api/v1/deltas/{bad}")
        self.assertEqual(status["state"], "quarantined")
        self.assertTrue(status["reason"])
        self.assertEqual(self.sites[1].call("GET", f"/api/v1/deltas/{good}")["state"], "accepted")

    def test_coordinator_restart_mid_round_and_duplicate_upload(self):
        first = self.upload(self.sites[0])
        self.net.stop()
        self.net.start()   # same SQLite file and object store
        self.sites = [ApiClient(self.net.base, s.token, retries=0) for s in self.sites]
        self.validator.api = ApiClient(self.net.base, self.vtoken, retries=0)
        second = self.upload(self.sites[1])
        self.validator.step()
        self.assertEqual([self.sites[i].call("GET", f"/api/v1/deltas/{d}")["state"] for i, d in
                          ((0, first), (1, second))], ["merged", "merged"])
        path = self.root / "delta1.safetensors"   # the same file again, under a new client id
        again = send_delta(self.sites[0], path, 0, self.validator.state["sha"]["0"], "client-again")
        self.assertEqual((again["delta_id"], again["duplicate"]), (first, True))
        self.assertEqual(self.received(self.sites[0])["deltas_received"], 1)   # no double credit


if __name__ == "__main__":
    unittest.main()
