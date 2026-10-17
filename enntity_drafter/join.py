"""``drafter-join``: run a volunteer training site for the GLM-5.3-Flash drafter (see JOIN.md, docs/volunteer.md).

    drafter-join redeem '<invite link>' --accept-terms [--name NAME]   # register this site (once)
    drafter-join setup                                                  # preflight, installs, smoke, canary
    drafter-join run [--mode recipe|byo ...]                            # loop: claim, data, pull, train, upload
    drafter-join status                                                 # local state + the coordinator's view
    drafter-join update                                                 # pull the pinned code, reinstall

Every cycle: check the network's pins (trainer, config, validator version) -> claim a recipe shard (recipe mode) or
use your own data (bring-your-own mode; it never leaves this machine) -> download the current global weights
(SHA-256 checked) -> train one round with the trainer's external DiLoCo mode (checkpointed, resumable) -> check
the delta with the validator's own code (and a small public check set) -> upload only the delta and its metadata,
resumably -> delete activations.  The first cycle, and the first after any pin change, is a short canary that is
fully validated before long rounds start.  Heartbeats carry the phase and counters only.

State lives in ``--work`` (default ``~/.enntity-drafter/work``); the token in ``~/.enntity-drafter/config.json``
(mode 0600).  An interrupted run resumes where it stopped; a delta is kept locally until the coordinator has
received it and returned a verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:   # the coordinator client and terms live beside the package in the checkout
    sys.path.insert(0, str(REPO))

from coordinator import terms  # noqa: E402
from coordinator.client import ApiClient, ApiError, download, sha256_file, upload_file  # noqa: E402

CLIENT_VERSION = "0.1.0"
SHARD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")   # coordinator-supplied ids become directory names
HOME = Path(os.environ.get("DRAFTER_HOME", Path.home() / ".enntity-drafter"))
#: Published training settings a site accepts, with their types (nothing else from the network reaches argv).
TRAINING_KEYS = {"lr": float, "warmup": int, "min_lr_ratio": float, "anchors": int, "weight_decay": float,
                 "beta1": float, "beta2": float, "clip": float, "grad_accum": int, "crop": int}
FINAL = ("accepted", "merged", "quarantined")
CHECK_FRACTION = 0.05   # rows of a shard kept out of training as the public self-check set
CANARY_ROWS = 16        # prompts of the claimed shard a recipe canary generates, captures and trains on


class JoinError(RuntimeError):
    """A stop with an actionable message (printed without a traceback)."""


def say(message: str) -> None:
    print(f"[drafter-join] {message}", flush=True)


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def load_config(home: Path = HOME) -> dict[str, Any]:
    path = home / "config.json"
    if not path.is_file():
        raise JoinError(f"not registered: run `drafter-join redeem '<invite link>' --accept-terms` first ({path})")
    return json.loads(path.read_text())


def save_config(config: dict[str, Any], home: Path = HOME) -> Path:
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = home / "config.json"
    temporary = path.with_name("config.json.tmp")
    temporary.touch(mode=0o600)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(config, indent=1))
    temporary.replace(path)
    return path


def parse_invite(link: str) -> tuple[str, str]:
    """``https://<host>/<base>/join#invite=<code>`` -> (coordinator base URL, code)."""

    parts = urlsplit(link.strip())
    fragment = dict(p.split("=", 1) for p in parts.fragment.split("&") if "=" in p)
    if parts.scheme not in ("https", "http") or "invite" not in fragment:
        raise JoinError("an invite link looks like https://<host>/<path>/join#invite=<code>")
    if parts.scheme == "http" and parts.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise JoinError("refusing to send an invite code and token over plain HTTP; ask for an https:// link")
    path = parts.path[: -len("/join")] if parts.path.endswith("/join") else parts.path
    return f"{parts.scheme}://{parts.netloc}{path.rstrip('/')}", fragment["invite"]


def redeem(link: str, name: str | None, accept: bool, home: Path = HOME,
           confirm: Callable[[str], bool] | None = None) -> dict[str, Any]:
    base, code = parse_invite(link)
    api = ApiClient(base)
    info = api.call("GET", "/api/v1/terms", auth=False)
    if info["sha256"] != terms.TERMS_SHA256:
        raise JoinError("the coordinator's terms differ from this checkout's; run `drafter-join update` first")
    print(info["text"])
    print(info["attestation"]["text"])
    if not accept and not (confirm or (lambda q: input(q).strip().lower() == "y"))("Accept these terms? [y/N] "):
        raise JoinError("terms not accepted; nothing was registered")
    reply = api.call("POST", "/api/v1/redeem", {"code": code, "display_name": name, "accept_terms": info["sha256"]},
                     auth=False)
    path = save_config({"coordinator": base, "token": reply["token"], "public_id": reply["public_id"],
                        "display_name": name, "publish_key": info.get("publish_key")}, home)
    say(f"registered as {name or 'anonymous'}; token saved to {path} (mode 0600)")
    return reply


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------

def _carved(row_id: str, fraction: float = CHECK_FRACTION) -> bool:
    import hashlib

    return int(hashlib.sha256(f"check/{row_id}".encode()).hexdigest()[:8], 16) / 2**32 < fraction


def synthetic_store(cfg: dict[str, Any], recipe: dict[str, Any], out: Path) -> list[str]:
    """Synthetic recipe (CPU fixtures, smoke): rows ``[lo, hi)`` of ``data.synthetic_rows``; a hash-carved few
    become the public check split.  Returns trainer data args."""

    from . import data as sd

    args = recipe.get("args", {})
    rows = sd.synthetic_rows(cfg, train_rows=int(args.get("train_rows_total", recipe["rows"][1])), heldout_rows=0,
                             length=int(args.get("length", 96)), seed=int(args.get("data_seed", 11)))
    chosen = [dict(r, split="heldout" if _carved(r["id"]) else "train") for r in rows[recipe["rows"][0]:
                                                                                      recipe["rows"][1]]]
    sd.write_row_store(out, chosen, capture_layers=cfg["target_layer_ids"], hidden_size=cfg["target_hidden_size"],
                       final_hidden="pre-final-norm", target={"name": "synthetic-recipe"})
    return ["--features", str(out)]


def _run(cmd: list[str], log: Path, env: dict[str, str] | None = None) -> None:
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a") as stream:
        stream.write(f"$ {' '.join(shlex.quote(c) for c in cmd)}\n")
        stream.flush()
        done = subprocess.run(cmd, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT, env=env)
    if done.returncode:
        raise JoinError(f"step failed ({cmd[1] if len(cmd) > 1 else cmd[0]}); see {log}")


def pipeline_store(recipe: dict[str, Any], out: Path, opts: argparse.Namespace, phase: Callable[[str], None]
                   ) -> list[str]:
    """Recipe over the repo's prompt builders: prompts -> teacher text -> capture -> FP8 -> index.  Needs a GLM
    server (yours, or an API key of your choice) and the Atlas capture build serving the target (docs/capture.md)."""

    py, log = sys.executable, out / "pipeline.log"
    out.mkdir(parents=True, exist_ok=True)
    phase("generating")
    builder, args = recipe["builder"], recipe.get("args", {})
    cmd = [py, "scripts/generate_prompts.py", builder, "--out", str(out / "prompts")]
    if builder == "coverage":
        cmd.append("--no-tokens")
        for key in ("seed", "profile", "prefix", "scale"):
            if key in args:
                cmd += [f"--{key}", str(args[key])]
    _run(cmd, log)
    rows = (out / "prompts" / "train.jsonl").read_text().splitlines()[recipe["rows"][0]: recipe["rows"][1]]
    shard = out / "shard-prompts.jsonl"
    shard.write_text("".join(line + "\n" for line in rows))
    sources = sorted({json.loads(line)["source"] for line in rows})
    (out / "check-ids.jsonl").write_text("".join(json.dumps({"id": json.loads(line)["id"]}) + "\n" for line in rows
                                                 if _carved(json.loads(line)["id"])))
    gen = recipe.get("generation", {})
    cmd = [py, "scripts/api_generate.py", "--prompts", str(shard), "--out", str(out / "gen"), "--sources",
           ",".join(sources), "--samples", str(int(gen.get("samples", 1))), "--minutes", str(opts.gen_minutes),
           "--spend-cap", str(opts.spend_cap)]
    if opts.teacher_url:
        cmd += ["--url", opts.teacher_url, "--model", opts.teacher_model, "--key-env", opts.teacher_key_env]
    _run(cmd, log)
    _run([py, "scripts/prepare_items.py", "--prompts", str(shard), "--generations",
          str(out / "gen" / "generations.jsonl"), "--tokenizer-dir", str(opts.tokenizer_dir), "--out",
          str(out / "items.jsonl")], log)
    phase("capturing")
    if opts.before_capture:
        _run(shlex.split(opts.before_capture), log)
    try:
        _run([py, "scripts/capture.py", "--server", opts.atlas_server, "--items", str(out / "items.jsonl"),
              *[a for d in opts.capture_dir for a in ("--capture-dir", d)], "--dest", str(out / "rows"),
              "--out", str(out / "capture-logs"), "--minutes", str(opts.capture_minutes)], log)
    finally:
        if opts.after_capture:
            _run(shlex.split(opts.after_capture), log)
    stop = out / "fp8.stop"
    stop.touch()
    _run([py, "-m", "enntity_drafter.fp8", "watch", "--root", str(out / "rows"), "--log", str(out / "fp8-log.jsonl"),
          "--stop-file", str(stop)], log)
    _run([py, "-m", "enntity_drafter.capture_io", "index", "--data-root", str(out), "--fast", "rows/*", "--meta",
          str(out / "capture-logs" / "capture-records.jsonl"), "--heldout", str(out / "check-ids.jsonl"),
          "--out", str(out / "index")], log)
    return ["--capture-index", str(out / "index" / "index.json"), "--capture-root", str(out)]


# --------------------------------------------------------------------------
# the site
# --------------------------------------------------------------------------

class Site:
    def __init__(self, api: ApiClient, token: str, work: Path, opts: argparse.Namespace,
                 sleep: Callable[[float], None] = time.sleep, config: dict[str, Any] | None = None,
                 home: Path | None = None):
        from . import model as sm

        self.api, self.token, self.work, self.opts, self.sleep = api, token, Path(work).resolve(), opts, sleep
        self.config, self.home = dict(config or {}), home
        self.cfg = sm.load_config(opts.config)
        self.work.mkdir(parents=True, exist_ok=True)
        self.state_path = self.work / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {
            "rounds": 0, "tokens": 0.0, "outbox": [], "cycle": None}
        self.phase, self.beat_stop, self.live = "idle", threading.Event(), {}

    # ------------------------------------------------------------------ state
    def save(self) -> None:
        temporary = self.state_path.with_name("state.json.tmp")
        temporary.write_text(json.dumps(self.state, indent=1))
        temporary.replace(self.state_path)

    def set_phase(self, phase: str) -> None:
        self.phase = phase
        cycle = self.state.get("cycle") or {}
        say(f"{'canary' if cycle.get('canary') else 'round ' + str(self.state['rounds'] + 1)}: {phase}")

    # ------------------------------------------------------------------ pins
    def pins(self) -> dict[str, Any]:
        from . import diloco as dl
        from .train import TRAINER_VERSION
        from .validation import VALIDATOR_VERSION

        return {"trainer": TRAINER_VERSION, "config_sha256": dl.config_sha256(self.cfg),
                "validator_version": VALIDATOR_VERSION}

    def requirements(self) -> dict[str, Any]:
        req = self.api.call("GET", "/api/v1/requirements")
        mine = self.pins()
        wrong = {k: (mine[k], req[k]) for k in mine if req.get(k) is not None and req[k] != mine[k]}
        if wrong:
            detail = ", ".join(f"{k}: this site {a!r}, network {b!r}" for k, (a, b) in wrong.items())
            raise JoinError(f"version mismatch ({detail}). Run `drafter-join update`, then `drafter-join run`; "
                            "nothing was trained.")
        if req.get("version") is None:
            raise JoinError("the coordinator has no global weights yet; try again later")
        if req.get("terms_sha256") != terms.TERMS_SHA256:
            raise JoinError("the participation terms changed; run `drafter-join update`")
        return req

    # ------------------------------------------------------------------ heartbeat
    def _beat_loop(self, interval: float) -> None:
        while not self.beat_stop.wait(interval):
            self.heartbeat()

    def heartbeat(self) -> None:
        cycle = self.state.get("cycle") or {}
        body = {"phase": self.phase, "round": self.state["rounds"], "tokens": float(self.state["tokens"]),
                "client": CLIENT_VERSION, "gpus": int(self.opts.gpus)}
        if cycle.get("base"):
            body["version"] = int(cycle["base"]["version"])
        if cycle.get("shard"):
            body["shard_id"] = cycle["shard"]["id"]
        for key in ("tokens_per_s", "gpu_w", "mfu", "step"):
            if self.live.get(key) is not None:
                body[key] = self.live[key]
        try:
            self.api.call("POST", "/api/v1/heartbeat", body)
        except ApiError as error:
            say(f"heartbeat not sent ({error.code})")

    # ------------------------------------------------------------------ cycle
    def _data(self, cycle: dict[str, Any]) -> list[str]:
        if self.opts.mode == "byo":
            if self.opts.byo_features:
                return ["--features", str(self.opts.byo_features)]
            return ["--capture-index", str(self.opts.byo_capture_index), "--capture-root", str(self.opts.byo_capture_root)]
        shard, recipe = cycle["shard"], dict(cycle["shard"]["recipe"])
        synthetic = recipe["builder"] == "synthetic"
        out = self.inside(self.work / "data" / (self.shard_id(shard) + ("-canary" if cycle["canary"] else "")))
        if cycle["canary"] and not synthetic:   # the canary trains on a small slice of the shard
            recipe["rows"] = [recipe["rows"][0], min(recipe["rows"][1], recipe["rows"][0] + CANARY_ROWS)]
        cycle["data_dir"] = str(out)
        if (out / "manifest.json").is_file() if synthetic else (out / "index" / "index.json").is_file():
            return (["--features", str(out)] if synthetic else
                    ["--capture-index", str(out / "index" / "index.json"), "--capture-root", str(out)])
        shutil.rmtree(out, ignore_errors=True)   # an interrupted build starts over
        if synthetic:
            return synthetic_store(self.cfg, recipe, out)
        if self.opts.tokenizer_dir is None or self.opts.teacher is None:
            raise JoinError("recipe shards need --tokenizer-dir and --teacher (your GLM-5.3-Flash directory)")
        return pipeline_store(recipe, out, self.opts, self.set_phase)

    def _trainer(self, cycle: dict[str, Any], req: dict[str, Any]) -> Path:
        training = req.get("training", {})
        canary = cycle["canary"]
        inner = int(training.get("canary_steps", 20)) if canary else int(training.get("inner_steps", 500))
        steps = inner if canary else int(training.get("steps", 10**7))
        out = self.work / ("canary-" + cycle["client_id"] if canary else "trainer")
        delta = self.work / "outbox" / f"{cycle['client_id']}.safetensors"
        seed = int(cycle["shard"]["recipe"].get("train_seed", 20261014)) if cycle.get("shard") else self.opts.seed
        argv = [*self.launcher(), "-m", "enntity_drafter.train", "--config", str(self.opts.config), "--teacher",
                str(self.opts.teacher), *cycle["data"], "--output", str(out), "--sync-mode", "diloco",
                "--diloco-transport", "external", "--start-weights", cycle["base"]["path"], "--inner-steps", str(inner),
                "--steps", str(steps), "--eval-every", "0", "--checkpoint-every", str(max(1, inner // 5)),
                "--keep-checkpoints", "2", "--delta-out", str(delta),
                "--delta-dtype", str(training.get("delta_dtype", "int8")), "--seed", str(seed),
                "--delta-meta", json.dumps({"client_id": cycle["client_id"], "canary": canary}), "--log-every", "10"]
        for key, kind in TRAINING_KEYS.items():
            if key in training:
                argv += [f"--{key.replace('_', '-')}", str(kind(training[key]))]
        argv += self.local_trainer_args()
        if delta.is_file():   # written atomically at the end of the round (this cycle's own client id)
            return delta
        log = self.work / "logs" / f"train-{cycle['client_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a") as stream:
            proc = subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
            while proc.poll() is None:
                self._progress(out / "train.jsonl", inner)
                time.sleep(2)
        self._progress(out / "train.jsonl", inner)
        if proc.returncode or not delta.is_file():
            raise JoinError(f"training failed (exit {proc.returncode}); see {log}. Rerun `drafter-join run` to "
                            "resume from the last checkpoint.")
        return delta

    def _progress(self, log: Path, inner: int) -> None:
        if not log.is_file():
            return
        lines = log.read_text().splitlines()
        train = next((json.loads(l) for l in reversed(lines) if '"phase": "train"' in l), None)
        if train:
            self.live = {"tokens_per_s": train.get("supervised_tok_per_s"), "gpu_w": train.get("gpu_w"),
                         "mfu": train.get("mfu"), "step": int(train["step"])}
            print(f"\r  step {train['step']} (round of {inner}) - {train.get('supervised_tok_per_s', 0):,.0f} tok/s "
                  f"- loss {train.get('loss', 0):.4f}   ", end="", flush=True)

    def local_trainer_args(self) -> list[str]:
        """The site's own flags, plus memory guards on CUDA unless given: GB10 CPU and GPU share memory and
        exhausting it can hang the host, so stop cleanly below 12 GiB free and probe the worst case first."""

        local = list(self.opts.trainer_arg)
        import torch

        if torch.cuda.is_available() and not any(a.startswith("--min-free-gb") for a in local):
            local += ["--min-free-gb", "12", "--probe"]
        return local

    def launcher(self) -> list[str]:
        if self.opts.launcher:
            return shlex.split(self.opts.launcher)
        if int(self.opts.gpus) > 1:
            return [str(Path(sys.executable).with_name("torchrun")), "--nnodes=1", f"--nproc_per_node={self.opts.gpus}",
                    "--master_addr=127.0.0.1", "--master_port=29517"]
        return [sys.executable]

    def _self_check(self, cycle: dict[str, Any], delta: Path, req: dict[str, Any]) -> dict[str, Any]:
        from .validation import CheckFailed, Thresholds, candidate_weights, check_delta, evaluate, judge, metrics_of

        base = Path(cycle["base"]["path"])
        try:
            stats = check_delta(delta, base, {"base_sha256": cycle["base"]["sha256"], **{
                k: v for k, v in self.pins().items() if k != "validator_version"}}, req.get("norm_bounds"))
        except CheckFailed as failure:
            raise self._set_aside(cycle, "failed-checks", f"self-check failed before upload ({failure.code}): "
                                                          f"{failure.message}; nothing was uploaded") from failure
        if not self.opts.no_public_check:
            eval_args = ["--config", str(self.opts.config), "--teacher", str(self.opts.teacher), *cycle["data"],
                         *self.opts.trainer_arg]
            candidate = self.work / "tmp" / "candidate.safetensors"
            try:
                candidate_weights(base, delta, candidate)
                base_m = metrics_of(evaluate(base, eval_args, inprocess=self.opts.eval_inprocess))
                cand_m = metrics_of(evaluate(candidate, eval_args, inprocess=self.opts.eval_inprocess))
            except (RuntimeError, SystemExit) as error:
                say(f"public check set unavailable ({str(error)[:120]}); tensor checks passed")
                base_m = cand_m = None
            finally:
                candidate.unlink(missing_ok=True)
            if base_m is not None:
                decision, code, message, _ = judge(base_m, cand_m, Thresholds())
                if decision == "reject":
                    raise self._set_aside(cycle, "failed-checks", f"self-check on the public check set failed "
                                                                  f"({code}): {message}; nothing was uploaded")
                stats["public_check"] = {"base": base_m, "candidate": cand_m}
        return stats

    def cycle(self) -> dict[str, Any]:
        """One canary or round, resumable at every phase."""

        req = self.requirements()
        self._drain_outbox()
        cycle = self.state.get("cycle")
        if cycle is None:
            if not req["canary_required"] and req.get("open_deltas", 0) >= req.get("max_open_deltas", 10**9):
                raise JoinError("waiting: this site's earlier deltas are not merged yet (the coordinator limits "
                                "unmerged deltas per site)")
            cycle = {"client_id": uuid.uuid4().hex, "canary": bool(req["canary_required"]), "stage": "data"}
            key = f"{req['trainer']}|{req['config_sha256']}"
            if self.state.get("trainer_key") not in (None, key):   # pins changed: the old optimizer state is void
                shutil.rmtree(self.work / "trainer", ignore_errors=True)
            self.state.update(cycle=cycle, trainer_key=key)
            self.save()
        if cycle["stage"] == "data":
            self.set_phase("claiming")
            if self.opts.mode == "recipe" and not cycle.get("shard"):
                shard = self.api.call("POST", "/api/v1/shards/claim")["shard"]
                if shard is None:
                    self.state["cycle"] = None
                    self.save()
                    raise JoinError("no recipe shard is open right now; retry later or run with --mode byo")
                cycle["shard"] = shard
                self.save()
            cycle["data"] = self._data(cycle)
            cycle["stage"] = "base"
            self.save()
        if cycle["stage"] == "base":
            self.set_phase("pulling")
            info = self.api.call("GET", "/api/v1/global")
            self.verify_global(req, info)
            path = self.work / "globals" / f"v{int(info['version']):06d}.safetensors"
            if not path.is_file() or sha256_file(path) != info["sha256"]:
                download(info["url"], path, sha256=info["sha256"], size=info["size"])
            cycle["base"] = {"version": info["version"], "sha256": info["sha256"], "path": str(path)}
            cycle["stage"] = "train"
            self.save()
        if cycle["stage"] == "train":
            self.set_phase("canary" if cycle["canary"] else "training")
            delta = self._trainer(cycle, req)
            print()
            self.set_phase("checking")
            stats = self._self_check(cycle, delta, req)
            meta = stats["meta"]
            cycle.update(delta=str(delta), sha256=sha256_file(delta), size=delta.stat().st_size,
                         meta={"tokens": float(meta["tokens"]), "steps": int(meta["step_end"] - meta["step_start"]),
                               "step_start": int(meta["step_start"]), "step_end": int(meta["step_end"]),
                               "inner_steps": int(meta["inner_steps"]), "delta_dtype": meta["delta_dtype"],
                               "trainer": meta["trainer"], "config_sha256": meta["config_sha256"],
                               "validator_version": self.pins()["validator_version"], "client": CLIENT_VERSION},
                         stage="upload")
            self.save()
        if cycle["stage"] == "upload":
            self.set_phase("uploading")
            body = {"client_id": cycle["client_id"], "base_version": cycle["base"]["version"],
                    "base_sha256": cycle["base"]["sha256"], "size": cycle["size"], "sha256": cycle["sha256"],
                    "mode": self.opts.mode, "canary": cycle["canary"], "meta": cycle["meta"]}
            if cycle.get("shard"):
                body["shard_id"] = cycle["shard"]["id"]
            elif self.opts.tag:
                body["tag"] = self.opts.tag
            body["attestation"] = terms.sign_attestation(self.token, {**body, "shard_id": body.get("shard_id"),
                                                                      "tag": body.get("tag")})
            try:
                reply = self.api.call("POST", "/api/v1/deltas", body)
            except ApiError as error:
                if error.code not in ("version_mismatch", "base_retired", "base_unknown"):
                    raise
                raise self._set_aside(cycle, "unmergeable", f"the network changed during this round ({error.code}: "
                                                            f"{error.message})") from error
            if reply["state"] == "abandoned":   # an upload left for more than a day: register it again
                cycle["client_id"] = uuid.uuid4().hex
                cycle.pop("transfer", None)
                self.save()
                return self.cycle()
            cycle["delta_id"] = reply["delta_id"]
            self.save()
            if reply["state"] == "uploading":
                transfer = cycle.setdefault("transfer", {})
                parts = upload_file(cycle["delta"], reply["upload"], lambda numbers: self.api.call(
                    "POST", f"/api/v1/uploads/{reply['upload_id']}/refresh", {"parts": numbers}), transfer, self.save)
                self.api.call("POST", f"/api/v1/deltas/{cycle['delta_id']}/complete", {"parts": parts})
            cycle["stage"] = "verdict" if cycle["canary"] else "done"
            self.state["tokens"] += cycle["meta"]["tokens"]
            self.save()
        if cycle["stage"] == "verdict":   # the canary: wait for the full validation before long rounds
            self.set_phase("waiting")
            status = self.wait_verdict(cycle["delta_id"], self.opts.canary_timeout_s)
            if status["state"] not in ("accepted", "merged"):
                self.state["outbox"].append({"delta_id": cycle["delta_id"], "path": cycle["delta"]})
                self.state["cycle"] = None
                self.save()
                raise JoinError(f"canary not accepted ({status['state']}: {status.get('reason_code')} "
                                f"{status.get('reason')}); long rounds were not started")
            cycle["stage"] = "done"
            self.save()
        self.set_phase("cleaning")
        self.state["outbox"].append({"delta_id": cycle["delta_id"], "path": cycle["delta"]})
        if cycle.get("data_dir"):   # activations (never bring-your-own data, never outside the work directory)
            shutil.rmtree(self.inside(Path(cycle["data_dir"])), ignore_errors=True)
        if cycle["canary"]:
            shutil.rmtree(self.work / ("canary-" + cycle["client_id"]), ignore_errors=True)
        else:
            self.state["rounds"] += 1
        self.state["cycle"] = None
        self.save()
        self._drain_outbox()
        self.set_phase("idle")
        return {"delta_id": cycle["delta_id"], "canary": cycle["canary"], "base_version": cycle["base"]["version"]}

    @staticmethod
    def shard_id(shard: dict[str, Any]) -> str:
        if not isinstance(shard.get("id"), str) or not SHARD_ID.match(shard["id"]):
            raise JoinError("the coordinator sent an invalid shard id; refusing to use it")
        return shard["id"]

    def inside(self, path: Path) -> Path:
        """``path`` resolved, and refused unless it lies under ``<work>/data``."""

        resolved, root = Path(path).resolve(), (self.work / "data").resolve()
        if root not in resolved.parents:
            raise JoinError(f"refusing to touch {path}: outside {root}")
        return resolved

    def verify_global(self, req: dict[str, Any], info: dict[str, Any]) -> None:
        """Check the operator's ed25519 signature on a global (key pinned at registration, or on first sight)."""

        pinned, offered = self.config.get("publish_key"), req.get("publish_key")
        if pinned is None and offered:
            pinned = self.config["publish_key"] = offered
            if self.home is not None:
                save_config(self.config, self.home)
        elif pinned and offered and offered != pinned:
            raise JoinError("the network's publishing key changed since this site registered; confirm the new key "
                            "with the operator, then update publish_key in your config")
        if not pinned:
            return
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

        try:
            Ed25519PublicKey.from_public_bytes(bytes.fromhex(pinned)).verify(
                bytes.fromhex(info.get("signature") or ""), terms.global_message(info["version"], info["sha256"]))
        except (InvalidSignature, ValueError) as error:
            raise JoinError(f"global v{info['version']} is not signed by the network's publishing key; "
                            "not training on it") from error

    def _set_aside(self, cycle: dict[str, Any], folder: str, why: str) -> JoinError:
        """Keep this cycle's delta under ``<work>/<folder>/`` for inspection and start the next cycle fresh."""

        delta = Path(cycle.get("delta") or self.work / "outbox" / f"{cycle['client_id']}.safetensors")
        kept = self.work / folder / delta.name
        if delta.is_file():
            kept.parent.mkdir(parents=True, exist_ok=True)
            delta.replace(kept)
        self.state["cycle"] = None
        self.save()
        return JoinError(f"{why}. The delta is kept at {kept}; the next `drafter-join run` starts a new round "
                         "(run `drafter-join update` first if the message asks for it).")

    def wait_verdict(self, delta_id: str, timeout_s: float) -> dict[str, Any]:
        deadline = time.time() + timeout_s
        while True:
            status = self.api.call("GET", f"/api/v1/deltas/{delta_id}")
            if status["state"] in FINAL or status["state"] == "held" or time.time() > deadline:
                return status
            self.sleep(10)

    def _drain_outbox(self) -> None:
        """Delete local deltas once the coordinator holds them and has given a verdict."""

        keep = []
        for entry in self.state.get("outbox", []):
            try:
                status = self.api.call("GET", f"/api/v1/deltas/{entry['delta_id']}")
            except ApiError:
                keep.append(entry)
                continue
            if status["state"] in FINAL:
                Path(entry["path"]).unlink(missing_ok=True)
                if status["state"] == "quarantined":
                    say(f"delta {entry['delta_id']} quarantined ({status.get('reason_code')}): {status.get('reason')}"
                        " - kept by the coordinator and re-validated if the validator is fixed")
            else:
                keep.append(entry)
        self.state["outbox"] = keep
        self.save()

    def run(self, rounds: int | None = None) -> None:
        interval = 120.0
        thread = threading.Thread(target=self._beat_loop, args=(interval,), daemon=True)
        thread.start()
        self.heartbeat()
        done = 0
        try:
            while rounds is None or done < rounds:
                try:
                    result = self.cycle()
                except JoinError as error:
                    if "no recipe shard" not in str(error) and not str(error).startswith("waiting"):
                        raise
                    say(str(error))
                    self.sleep(300)
                    continue
                done += not result["canary"]
        finally:
            self.beat_stop.set()


# --------------------------------------------------------------------------
# setup (preflight)
# --------------------------------------------------------------------------

def preflight(opts: argparse.Namespace) -> list[tuple[str, str, str]]:
    """(check, PASS|WARN|FAIL, detail or fix) for this host."""

    checks: list[tuple[str, str, str]] = []

    def add(name: str, ok: bool | None, detail: str) -> None:
        checks.append((name, "PASS" if ok else "WARN" if ok is None else "FAIL", detail))

    add("python >= 3.10", sys.version_info >= (3, 10), platform.python_version())
    try:
        import torch

        version = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        add("torch 2.11-2.12", (2, 11) <= version < (2, 13), f"{torch.__version__} (2.13 breaks compiled flex "
                                                              "attention on CPU)")
        cuda = torch.cuda.is_available()
        if cuda:
            props = torch.cuda.get_device_properties(0)
            add("GPU", True, f"{torch.cuda.device_count()} x {props.name}, {props.total_memory / 2**30:.0f} GiB")
        else:
            add("GPU", None, "no CUDA device: training will run on CPU (tests only)")
    except ImportError:
        add("torch 2.11-2.12", False, "pip install 'torch>=2.11,<2.13'")
    try:
        import speculators  # noqa: F401

        add("speculators", True, "installed")
    except ImportError:
        add("speculators", False, "bash scripts/install_speculators.sh")
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        values = {l.split(":")[0]: int(l.split()[1]) for l in meminfo.read_text().splitlines() if ":" in l}
        avail = values.get("MemAvailable", 0) / 2**20
        add("host memory", avail >= 60, f"{avail:.0f} GiB available (GB10: CPU and GPU share memory; keep >= 12 GiB "
                                        "free, the trainer stops cleanly below --min-free-gb)")
    free = shutil.disk_usage(opts.work if Path(opts.work).exists() else Path.home()).free / 2**30
    add("disk", True if free >= 500 else None if free >= 50 else False,
        f"{free:.0f} GiB free at the work directory (FP8 activations take ~24.6 kB per token; >= 500 GiB advised)")
    ib = Path("/sys/class/infiniband")
    add("RDMA link (capture pair)", True if ib.is_dir() and any(ib.iterdir()) else None,
        "present" if ib.is_dir() and any(ib.iterdir()) else "no RDMA device: capture with GLM-5.3-Flash in TP2 needs "
                                                           "the pair's 200G link (not needed for bring-your-own data)")
    if opts.peer:
        done = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", opts.peer, "nvidia-smi",
                               "--query-gpu=name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True)
        add("capture peer", done.returncode == 0, done.stdout.strip() or done.stderr.strip()[:200])
    model = Path(opts.model_dir) if opts.model_dir else None
    if model is not None:
        ok = (model / "config.json").is_file() and (model / "tokenizer.json").is_file()
        add("GLM-5.3-Flash target", ok, str(model) if ok else
            f"huggingface-cli download nvidia/GLM-5.3-Flash-NVFP4 --local-dir {model}")
    if opts.atlas_image:
        done = subprocess.run(["docker", "image", "inspect", opts.atlas_image], capture_output=True)
        add("Atlas capture image", done.returncode == 0,
            "present" if done.returncode == 0 else f"docker pull {opts.atlas_image} (or build Enntity/atlas branch "
                                                   "exp/drafter-capture-20261014, docs/capture.md)")
    return checks


def smoke(opts: argparse.Namespace) -> tuple[bool, str]:
    """Tiny train in external mode + the shared delta checks, on synthetic data (~1 minute on CPU)."""

    from . import data as sd
    from . import diloco as dl
    from . import model as sm
    from .guard import scratch_stamp
    from .validation import CheckFailed, check_delta

    started = time.time()
    with tempfile.TemporaryDirectory(prefix="drafter-smoke-") as tmp:
        root = Path(tmp)
        cfg = sm.load_config(REPO / "configs" / "tiny.json")
        sd.write_synthetic_target_dir(root / "target", cfg)
        data_args = synthetic_store(cfg, {"builder": "synthetic", "rows": [0, 16], "args": {"length": 64}},
                                    root / "features")
        init = root / "init"
        base_cmd = [sys.executable, "-m", "enntity_drafter.train", "--config", str(REPO / "configs" / "tiny.json"),
                    "--teacher", str(root / "target"), *data_args, "--device", "cpu", "--anchors", "16",
                    "--eval-every", "0", "--lr", "2e-3", "--warmup", "5"]
        _run([*base_cmd, "--output", str(init), "--lr", "0", "--steps", "1", "--stop-after", "1"], root / "smoke.log")
        from .guard import guarded_torch_load

        params = guarded_torch_load(init / "checkpoints" / "step-0000001" / "rank0.pt", "drafter")["params"]
        base = root / "base.safetensors"
        sha = dl.save_weights(base, params, "fp32", scratch_stamp(), {"version": 0})
        _run([*base_cmd, "--output", str(root / "train"), "--sync-mode", "diloco", "--diloco-transport", "external",
              "--start-weights", str(base), "--inner-steps", "20", "--steps", "20", "--delta-dtype", "bf16",
              "--delta-out", str(root / "delta.safetensors")], root / "smoke.log")
        try:
            stats = check_delta(root / "delta.safetensors", base, {"base_sha256": sha})
        except CheckFailed as failure:
            return False, f"delta check failed: {failure}"
        return True, f"20 tiny steps + delta checks in {time.time() - started:.0f} s (delta RMS {stats['rms']:.2e})"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def site_from(opts: argparse.Namespace) -> Site:
    config = load_config(Path(opts.home))
    return Site(ApiClient(config["coordinator"], config["token"]), config["token"], Path(opts.work), opts,
                config=config, home=Path(opts.home))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="drafter-join", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--home", default=str(HOME), help="config directory (token, mode 0600)")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("redeem", help="register this site with a one-use invite link")
    q.add_argument("link")
    q.add_argument("--name", default=None, help="public display name (omit to stay anonymous)")
    q.add_argument("--accept-terms", action="store_true", help="accept the printed participation terms")
    for name in ("setup", "run", "status"):
        q = sub.add_parser(name)
        q.add_argument("--work", default=str(HOME / "work"))
        q.add_argument("--config", type=Path, default=REPO / "configs" / "full.json")
        q.add_argument("--teacher", type=Path, default=None, help="your GLM-5.3-Flash directory (embedding, lm_head)")
        q.add_argument("--mode", choices=["recipe", "byo"], default="recipe")
        q.add_argument("--byo-features", type=Path, help="byo: your enntity.drafter_rows/v1 store")
        q.add_argument("--byo-capture-index", type=Path, help="byo: your capture index.json")
        q.add_argument("--byo-capture-root", type=Path)
        q.add_argument("--tag", default=None, help="byo: an opaque label for your data (shown to the operator)")
        q.add_argument("--seed", type=int, default=20261014, help="byo: trainer seed")
        q.add_argument("--gpus", type=int, default=1, help="local GPUs for training (one GB10 is enough)")
        q.add_argument("--launcher", default=None, help="custom trainer launcher prefix, e.g. a torchrun command")
        q.add_argument("--trainer-arg", action="append", default=[],
                       help="extra local trainer flag (performance, memory guards), repeatable")
        q.add_argument("--rounds", type=int, default=None, help="stop after N rounds (default: run forever)")
        q.add_argument("--canary-timeout-s", type=float, default=3600)
        q.add_argument("--no-public-check", action="store_true")
        q.add_argument("--eval-inprocess", action="store_true", help=argparse.SUPPRESS)
        q.add_argument("--teacher-url", default=None, help="recipe: your own OpenAI-compatible GLM-5.3-Flash server")
        q.add_argument("--teacher-model", default="glm-5.3-flash")
        q.add_argument("--teacher-key-env", default="OPENROUTER_API_KEY")
        q.add_argument("--gen-minutes", type=float, default=60)
        q.add_argument("--spend-cap", type=float, default=10.0)
        q.add_argument("--tokenizer-dir", type=Path, default=None)
        q.add_argument("--atlas-server", default="http://127.0.0.1:8893")
        q.add_argument("--capture-dir", action="append", default=[])
        q.add_argument("--capture-minutes", type=float, default=240)
        q.add_argument("--before-capture", default=None, help="command that starts your Atlas capture server")
        q.add_argument("--after-capture", default=None, help="command that stops it (frees memory for training)")
        if name == "setup":
            q.add_argument("--peer", default=None, help="ssh name of the second node of the capture pair")
            q.add_argument("--model-dir", default=None)
            q.add_argument("--atlas-image", default=None)
            q.add_argument("--no-canary", action="store_true")
    q = sub.add_parser("update", help="pull this checkout (fast-forward only) and reinstall")
    q.add_argument("--yes", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    opts = build_parser().parse_args(argv)
    try:
        if opts.cmd == "redeem":
            redeem(opts.link, opts.name, opts.accept_terms, Path(opts.home))
            return 0
        if opts.cmd == "update":
            if not opts.yes:
                raise JoinError("update runs `git pull --ff-only` and reinstalls; rerun with --yes")
            for cmd in (["git", "-C", str(REPO), "pull", "--ff-only"],
                        ["bash", str(REPO / "scripts" / "install_speculators.sh")],
                        [sys.executable, "-m", "pip", "install", "-e", str(REPO), "--no-deps"]):
                say("$ " + " ".join(cmd))
                if subprocess.run(cmd, env={**os.environ, "PYTHON": sys.executable}).returncode:
                    raise JoinError(f"`{' '.join(cmd)}` failed")
            return 0
        if opts.mode == "byo" and not (opts.byo_features or (opts.byo_capture_index and opts.byo_capture_root)):
            raise JoinError("--mode byo needs --byo-features or --byo-capture-index with --byo-capture-root")
        if opts.cmd == "setup":
            bad = False
            for check, status, detail in preflight(opts):
                print(f"  {status:4}  {check}: {detail}")
                bad |= status == "FAIL"
            ok, detail = smoke(opts)
            print(f"  {'PASS' if ok else 'FAIL':4}  smoke (tiny train + delta checks): {detail}")
            if bad or not ok:
                raise JoinError("setup found problems; fix the FAIL lines above and run `drafter-join setup` again")
            if not opts.no_canary:
                if opts.teacher is None:
                    raise JoinError("the canary needs --teacher (your GLM-5.3-Flash directory)")
                site = site_from(opts)
                if site.state.get("cycle") is None and not site.requirements()["canary_required"]:
                    say("a canary already passed with these versions: this site is ready for `drafter-join run`")
                else:
                    result = site.cycle()
                    say(f"canary {result['delta_id']} accepted: this site is ready for `drafter-join run`")
            return 0
        if opts.cmd == "status":
            site = site_from(opts)
            print(json.dumps({"local": {k: site.state.get(k) for k in ("rounds", "tokens", "outbox", "cycle")},
                              "pins": site.pins(), "coordinator": site.api.call("GET", "/api/v1/me")}, indent=1))
            return 0
        if opts.teacher is None:
            raise JoinError("--teacher is required (your GLM-5.3-Flash directory with the embedding and lm_head)")
        site_from(opts).run(opts.rounds)
        return 0
    except JoinError as error:
        say(f"stopped: {error}")
        return 2
    except ApiError as error:
        say(f"stopped: the coordinator answered {error.code}: {error.message}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
