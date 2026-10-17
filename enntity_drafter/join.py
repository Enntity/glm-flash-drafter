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
import math
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
from coordinator.service import REQUIRED_SETTINGS, missing_settings  # noqa: E402

CLIENT_VERSION = "0.1.0"
SHARD_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")   # coordinator-supplied ids become directory names
HOME = Path(os.environ.get("DRAFTER_HOME", Path.home() / ".enntity-drafter"))
#: Published training settings a site accepts, with their types (nothing else from the network reaches argv).
#: ``global_anchors`` (anchors per optimizer step): the trainer's probe picks the micro-batch (at most ``anchors``)
#: and derives grad_accum; without it ``anchors`` and ``grad_accum`` are used as given.
TRAINING_KEYS = {"lr": float, "warmup": int, "min_lr_ratio": float, "anchors": int, "weight_decay": float,
                 "beta1": float, "beta2": float, "clip": float, "grad_accum": int, "crop": int, "global_anchors": int}
#: CUDA sites, unless given locally: stop below 12 GiB free, but by pausing (checkpoint, exit, resume here once
#: memory is back) rather than dying; the probe keeps 4 GiB more for memory other jobs on the host take and give back.
MEMORY_GUARDS = ("--min-free-gb", "12", "--probe-headroom-gb", "4", "--low-memory", "pause")
RESUME_STEADY_S = 60    # a paused round resumes once memory stayed above its floor + headroom this long
PAUSE_MAX_H = 6.0       # ... and gives up (round kept, checkpointed) after this long without memory
FINAL = ("accepted", "merged", "quarantined")
CHECK_FRACTION = 0.05   # rows of a shard kept out of training as the public self-check set
CHECK_ROWS_MAX = 64     # ... at most this many rows of a large bring-your-own index
CANARY_ROWS = 16        # prompts of the claimed shard a recipe canary generates, captures and trains on
NOFILE_MIN = 65536      # open-file soft limit: the loader keeps up to 256 memory maps per read thread and rank


class JoinError(RuntimeError):
    """An actionable message (printed without a traceback).  ``wait``: seconds after which ``run`` simply continues
    (the state already says what happens next); None: stop, someone has to act."""

    def __init__(self, message: str, wait: float | None = None):
        super().__init__(message)
        self.wait = wait


#: Every error code the coordinator (and the transfer client) can return, and what happens to a finished delta.
#: Nothing here discards a delta: retry keeps the cycle and tries again later; park keeps the delta while the
#: required canary runs, then uploads it; new_id registers the same file again; aside keeps the file under
#: <work>/unmergeable and starts a new cycle; aside_stop does the same and stops; stop keeps everything untouched
#: and stops until someone acts (for example a revoked token).
UPLOAD_ACTIONS: dict[str, tuple[str, float | None]] = {
    "rate_limited": ("retry", 60), "low_disk": ("retry", 600), "too_many_pending": ("retry", 300),
    "too_many_uploads": ("retry", 300), "canary_pending": ("retry", 120), "conflict": ("retry", 10),
    "not_ready": ("retry", 600),
    "upload_incomplete": ("retry", 30), "unreachable": ("retry", 300), "internal": ("retry", 300),
    "upload_failed": ("retry", 60), "no_url": ("retry", 60), "download_hash": ("retry", 60),
    "download_size": ("retry", 60),
    "canary_required": ("park", 0),
    "client_id_reused": ("new_id", 0), "not_found": ("new_id", 0),
    "base_unknown": ("aside", 0), "base_retired": ("aside", 0), "shard_not_leased": ("aside", 0),
    "duplicate": ("aside", 0), "version_mismatch": ("aside", 0), "no_version": ("aside", 0),
    "too_large": ("aside_stop", None), "attestation_invalid": ("aside_stop", None),
    "bad_request": ("aside_stop", None), "bad_json": ("aside_stop", None), "length_required": ("aside_stop", None),
    "auth_required": ("stop", None), "auth_invalid": ("stop", None), "invite_invalid": ("stop", None),
    "terms_required": ("stop", None), "bad_display_name": ("stop", None), "bad_host": ("stop", None),
    "not_pending": ("stop", None), "not_validating": ("stop", None), "version_order": ("stop", None),
}


def upload_action(code: str) -> tuple[str, float | None]:
    """The action for an upload error code. A non-JSON reply from a proxy arrives as ``http_<status>``:
    server-side and throttling statuses retry (a coordinator restart behind the proxy must not set work
    aside); any other unknown code stops with the delta and cycle kept, never discarding or setting aside."""

    if code in UPLOAD_ACTIONS:
        return UPLOAD_ACTIONS[code]
    if code.startswith("http_"):
        status = code[5:]
        if status.isdigit() and (status.startswith("5") or status in ("408", "425", "429")):
            return ("retry", 60)
    return ("stop", None)


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
    if not accept and confirm is None and not (sys.stdin and sys.stdin.isatty()):
        raise JoinError("no terminal to ask on: rerun with --accept-terms after reading the terms above; nothing "
                        "was registered")
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


def _row_keys(row: dict[str, Any]) -> set[str]:
    """Names a row can be listed under: its name, id and prompt id, and name and id without a sample suffix
    (``#s1``, ``_s1``) for indexes that predate stored prompt ids."""

    from .capture_io import _norm_name

    keys = {_norm_name(str(row[k])) for k in ("name", "id", "prompt_id") if row.get(k) is not None}
    return keys | {re.sub(r"_s\d+$", "", k) for k in keys}


def byo_capture_index(index_path: Path, out: Path, exclude: list[Path] = ()) -> list[str]:
    """Bring-your-own capture index -> the site's own index under ``out`` (the capture files are read in place).

    Rows the source index does not mark ``train`` (its own held-out, pilot rows) and rows whose name, id or prompt id
    (:func:`_row_keys`) is listed in an ``exclude`` file (JSONL ``{"id": ...}``, for example an operator's private
    held-out prompt ids on the same disks) are dropped: the site never trains or self-checks on them.  The public
    check split is
    re-carved from the remaining rows by the same hash rule as recipe shards, at most ``CHECK_ROWS_MAX`` rows.
    Returns trainer data args (``--capture-index``; the caller adds ``--capture-root``)."""

    import hashlib

    from .capture_io import _norm_name

    raw = Path(index_path).read_bytes()
    index = json.loads(raw)
    excluded = {_norm_name(str(json.loads(line)["id"])) for path in exclude
                for line in Path(path).read_text().splitlines() if line.strip()}
    keep, dropped = [], {"not_train": 0, "excluded": 0}
    for row in index["rows"]:
        if row.get("split") != "train":
            dropped["not_train"] += 1
        elif excluded & _row_keys(row):
            dropped["excluded"] += 1
        else:
            keep.append({k: v for k, v in row.items() if k != "rank"})
    fraction = min(CHECK_FRACTION, CHECK_ROWS_MAX / max(len(keep), 1))
    for row in keep:
        row["split"] = "heldout" if _carved(str(row["id"]), fraction) else "train"
    tokens = (Path(index_path).parent / index["tokens_file"]).resolve()
    site = {k: v for k, v in index.items() if k not in ("rows", "world", "copy_lists", "skipped_rows")}
    site.update(rows=keep, tokens_file=str(tokens), byo={
        "source_index_sha256": hashlib.sha256(raw).hexdigest(), "dropped": dropped,
        "exclude_sha256": [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in exclude],
        "check_rows": sum(r["split"] == "heldout" for r in keep)})
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / "index.json.tmp"
    temporary.write_text(json.dumps(site))
    temporary.replace(out / "index.json")
    say(f"bring-your-own index: {len(keep)} rows ({site['byo']['check_rows']} as the public check set); dropped "
        f"{dropped['not_train']} not marked train and {dropped['excluded']} excluded")
    return ["--capture-index", str(out / "index.json")]


def byo_source_index(root: Path, captures: list[str], manifests: list[Path], out: Path, cfg: dict[str, Any]) -> Path:
    """Index captures already in this repository's format (what ``python -m enntity_drafter.capture_io index
    --data-root ROOT --fast GLOB --manifest FILE`` does by hand): ``captures`` are globs of capture directories relative
    to ``root``; a manifest is also the allowlist of rows.  Rebuilt by every ``setup`` and ``run``, so new captures
    are picked up."""

    from .capture_io import build_index

    index = build_index(Path(root), [], list(captures), [], None, out, window=int(cfg["sliding_window"]),
                        block=int(cfg["block_size"]), layers=list(cfg["target_layer_ids"]),
                        manifest=[str(m) for m in manifests] or None)
    say(f"indexed {len(index['rows'])} captured rows under {root} into {out} (skipped: {index['skipped'] or 'none'})")
    return out / "index.json"


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
    _run([py, "-m", "enntity_drafter.capture_io", "index", "--data-root", str(out), "--fast", "rows", "--meta",
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
        self.warned: set[str] = set()   # --trainer-arg overrides already reported in this process
        self.batch: dict[str, Any] | None = None   # the trainer's chosen micro-batch, reported once
        self.heartbeat_fields: list[str] = []   # optional heartbeat fields the coordinator accepts
        self.phases: list[str] = []             # heartbeat phases it accepts (older ones: no "stopped")

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
        self.heartbeat_fields = list(req.get("heartbeat_fields") or [])   # absent on older coordinators
        self.phases = list(req.get("phases") or [])
        return req

    @staticmethod
    def training(req: dict[str, Any]) -> dict[str, Any]:
        """The network's published training settings; never the trainer's command-line defaults."""

        missing = missing_settings({"training": req.get("training")}, {"training": REQUIRED_SETTINGS["training"]})
        if missing:
            raise JoinError(f"the network has not published training settings yet (missing: {', '.join(missing)}); "
                            "nothing was trained, try again later", wait=600)
        return req["training"]

    # ------------------------------------------------------------------ heartbeat
    def _beat_loop(self, interval: float) -> None:
        while not self.beat_stop.wait(interval):
            self.heartbeat()

    def heartbeat(self) -> None:
        cycle = self.state.get("cycle") or {}
        body = {"phase": self.phase, "round": self.state["rounds"], "tokens": float(self.state["tokens"]),
                "client": CLIENT_VERSION, "gpus": int(self.opts.gpus)}
        bases = [c["base"]["version"] for c in [cycle, *self.state.get("parked", [])] if c.get("base")]
        if bases:   # the oldest global still in use here: the coordinator keeps it while this site is alive
            body["version"] = int(min(bases))
        if cycle.get("shard"):
            body["shard_id"] = cycle["shard"]["id"]
        for key in ("tokens_per_s", "gpu_w", "mfu", "step"):
            if self.live.get(key) is not None:
                body[key] = self.live[key]
        failed = self.state.get("self_check") or {}
        if failed and "check_failed" in self.heartbeat_fields:   # what the operator cannot see otherwise
            body.update(check_failed=int(failed["failed"]), check_code=failed["code"],
                        check_canary=int(failed["canary"]))
            if failed.get("drop") is not None:
                body["check_drop"] = float(failed["drop"])
        try:
            self.api.call("POST", "/api/v1/heartbeat", body)
        except ApiError as error:
            say(f"heartbeat not sent ({error.code})")

    # ------------------------------------------------------------------ cycle
    def _data(self, cycle: dict[str, Any]) -> list[str]:
        if self.opts.mode == "byo":
            if self.opts.byo_features:
                return ["--features", str(self.opts.byo_features)]
            return [*byo_capture_index(self.opts.byo_capture_index, self.work / "byo-index", self.opts.byo_exclude_ids),
                    "--capture-root", str(self.opts.byo_capture_root)]
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
        delta = self.work / "outbox" / f"{cycle['client_id']}.safetensors"
        if delta.is_file():   # written atomically at the end of the round (this cycle's own client id)
            return delta
        training = self.training(req)
        canary = cycle["canary"]
        inner = int(training.get("canary_steps", 20)) if canary else int(training["inner_steps"])
        steps = inner if canary else int(training.get("steps", 10**7))
        out = self.work / ("canary-" + cycle["client_id"] if canary else "trainer")
        seed = int(cycle["shard"]["recipe"].get("train_seed", 20261014)) if cycle.get("shard") else self.opts.seed
        argv = [*self.launcher(), "-m", "enntity_drafter.train", "--config", str(self.opts.config), "--teacher",
                str(self.opts.teacher), *cycle["data"], "--output", str(out), "--sync-mode", "diloco",
                "--diloco-transport", "external", "--start-weights", cycle["base"]["path"], "--inner-steps", str(inner),
                "--steps", str(steps), "--eval-every", "0", "--checkpoint-every", str(max(1, inner // 5)),
                "--keep-checkpoints", "2", "--delta-out", str(delta),
                "--delta-dtype", str(training.get("delta_dtype", "int8")), "--seed", str(seed),
                "--delta-meta", json.dumps({"client_id": cycle["client_id"], "canary": canary}), "--log-every", "10"]
        network = {f"--{k.replace('_', '-')}": str(kind(training[k])) for k, kind in TRAINING_KEYS.items()
                   if k in training and not (k == "grad_accum" and "global_anchors" in training)}
        argv += [a for flag, value in network.items() for a in (flag, value)]
        self._warn_overrides(network)
        argv += self.local_trainer_args()
        from .train import PAUSE_EXIT

        log = self.work / "logs" / f"train-{cycle['client_id']}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        while True:
            with log.open("a") as stream:
                proc = subprocess.Popen(argv, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    while proc.poll() is None:
                        self._progress(out / "train.jsonl", inner)
                        time.sleep(2)
                except BaseException:   # stopped (Ctrl-C, SIGTERM): the trainer stops too; its checkpoint resumes
                    proc.terminate()
                    try:
                        proc.wait(60)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                    raise
            if proc.returncode != PAUSE_EXIT:
                break
            self._wait_for_memory(argv, canary)   # the trainer checkpointed and exited: resume once memory is back
        self._progress(out / "train.jsonl", inner)
        if proc.returncode or not delta.is_file():
            again = "drafter-join setup` with the same flags" if canary else "drafter-join run`"
            raise JoinError(f"training failed (exit {proc.returncode}); see {log}. Fix the cause, then rerun `{again} "
                            "to resume from the last checkpoint.")
        return delta

    def _progress(self, log: Path, inner: int) -> None:
        if not log.is_file():
            return
        lines = log.read_text().splitlines()
        batch = next((json.loads(l) for l in reversed(lines) if '"phase": "batch"' in l), None)
        if batch is not None and batch != self.batch:
            self.batch = batch
            print()
            say(f"micro-batch: {batch['anchors']} anchors x grad_accum {batch['grad_accum']} x {batch['ranks']} "
                f"rank(s) = {batch['global_anchors']} anchors per step (probed {batch['probed']})")
        train = next((json.loads(l) for l in reversed(lines) if '"phase": "train"' in l), None)
        if train:
            self.live = {"tokens_per_s": train.get("supervised_tok_per_s"), "gpu_w": train.get("gpu_w"),
                         "mfu": train.get("mfu"), "step": int(train["step"])}
            print(f"\r  step {train['step']} (round of {inner}) - {train.get('supervised_tok_per_s', 0):,.0f} tok/s "
                  f"- loss {train.get('loss', 0):.4f}   ", end="", flush=True)

    def local_trainer_args(self) -> list[str]:
        """The site's own flags, plus memory guards on CUDA unless given: GB10 CPU and GPU share memory and
        exhausting it can hang the host, so stop cleanly below 12 GiB free, and probe the worst case at every start
        and resume (memory shared with other services changes; the probe also sizes the micro-batch)."""

        local = list(self.opts.trainer_arg)
        import torch

        if torch.cuda.is_available():
            for flag, value in zip(MEMORY_GUARDS[::2], MEMORY_GUARDS[1::2]):
                if not any(a.split("=")[0] == flag for a in local):
                    local += [flag, value]
            if "--probe" not in local:
                local.append("--probe")
        return local

    def _wait_for_memory(self, argv: list[str], canary: bool) -> None:
        """The trainer paused on low memory (another job on the host took it).  Wait until MemAvailable stays above
        the trainer's floor plus its probe headroom for ``RESUME_STEADY_S``, then return to resume the round; give
        up after ``PAUSE_MAX_H`` (the round stays checkpointed for the next run)."""

        from .train import mem_available_gb

        def value(flag: str, default: float) -> float:
            for i, arg in enumerate(argv):   # the last occurrence wins, as in argparse
                if arg.startswith(flag + "="):
                    default = float(arg.split("=", 1)[1])
                elif arg == flag and i + 1 < len(argv):
                    default = float(argv[i + 1])
            return default

        need = value("--min-free-gb", 0.0) + value("--probe-headroom-gb", 0.0)
        self.phase = "paused" if "paused" in self.phases else "waiting"   # older coordinators: no "paused"
        self.heartbeat()
        say(f"paused: low memory ({mem_available_gb():.0f} GB available); the {'canary' if canary else 'round'} is "
            f"checkpointed and resumes by itself once {need:.0f} GB stay available for {RESUME_STEADY_S} s")
        started = steady = time.monotonic()
        while True:
            now = time.monotonic()
            if mem_available_gb() < need:
                steady = now
            elif now - steady >= RESUME_STEADY_S:
                break
            if now - started > PAUSE_MAX_H * 3600:
                raise JoinError(f"memory stayed below {need:.0f} GB for {PAUSE_MAX_H:.0f} h; the "
                                f"{'canary' if canary else 'round'} is kept (checkpointed). Free memory on this host "
                                "(or move the site, JOIN.md), then rerun `drafter-join run` to resume it.")
            self.sleep(10)
        self.phase = "canary" if canary else "training"
        say("memory is back: resuming from the checkpoint (the probe runs again first)")

    def _warn_overrides(self, network: dict[str, str]) -> None:
        """--trainer-arg wins over the network's settings (an escape hatch, for example a smaller --anchors on a
        host with less memory); say so once per flag."""

        local = self.opts.trainer_arg
        flags = {a.split("=")[0] for a in local}
        if "--global-anchors" in network:
            network = {**network, "--grad-accum": "derived from --global-anchors"}
        for flag in sorted(flags & set(network) - self.warned):
            note = (" (with --global-anchors this only lowers the micro-batch cap; prefer letting the probe choose)"
                    if flag == "--anchors" and "--global-anchors" in network else "")
            say(f"warning: --trainer-arg {flag} overrides the network's setting {flag} {network[flag]}{note}")
            self.warned.add(flag)

    def launcher(self) -> list[str]:
        if self.opts.launcher:
            return shlex.split(self.opts.launcher)
        if int(self.opts.gpus) > 1:
            return [str(Path(sys.executable).with_name("torchrun")), "--nnodes=1", f"--nproc_per_node={self.opts.gpus}",
                    "--master_addr=127.0.0.1", "--master_port=29517"]
        return [sys.executable]

    def _self_check(self, cycle: dict[str, Any], delta: Path, req: dict[str, Any]) -> dict[str, Any]:
        """The validator's own checks before upload.  A canary (never merged; a few steps from a fresh optimizer
        can dip) gets the sanity checks and catastrophic bound the validator gives canaries, never the merge
        thresholds."""

        from .validation import (CheckFailed, Thresholds, canary_bounds, candidate_weights, check_delta, evaluate,
                                 judge, judge_canary, metrics_of)

        base, canary = Path(cycle["base"]["path"]), cycle["canary"]
        bounds = canary_bounds(req.get("norm_bounds")) if canary else req.get("norm_bounds")
        try:
            stats = check_delta(delta, base, {"base_sha256": cycle["base"]["sha256"], **{
                k: v for k, v in self.pins().items() if k != "validator_version"}}, bounds)
        except CheckFailed as failure:
            self._check_failed(cycle, failure.code, None)
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
                decision, code, message, drops = (judge_canary if canary else judge)(base_m, cand_m, Thresholds())
                if decision == "reject":
                    self._check_failed(cycle, code, max(drops.values(), default=None))
                    raise self._set_aside(cycle, "failed-checks", f"self-check on the public check set failed "
                                                                  f"({code}): {message}; nothing was uploaded")
                stats["public_check"] = {"base": base_m, "candidate": cand_m}
        if self.state.pop("self_check", None):
            self.save()
        return stats

    def _check_failed(self, cycle: dict[str, Any], code: str, drop: float | None) -> None:
        """Count a failed self-check (nothing is uploaded, so only the heartbeat tells the operator) and report it
        at once: a reason code, the canary flag and the worst relative drop; no data."""

        from coordinator.service import REASONS

        last = self.state.get("self_check") or {"failed": 0}
        self.state["self_check"] = {"failed": last["failed"] + 1, "code": code if code in REASONS else "malformed",
                                    "canary": bool(cycle.get("canary")), "drop": drop}
        self.save()
        self.phase = "error"
        self.heartbeat()

    def cycle(self) -> dict[str, Any]:
        """One canary or round, resumable at every phase."""

        req = self.requirements()
        self._drain_outbox()
        if getattr(self.opts, "restart_cycle", False) and self.state.get("cycle"):
            say("--restart-cycle: the interrupted cycle starts over (finished deltas are kept)")
            self.state["cycle"], self.opts.restart_cycle = None, False
            self.save()
        cycle = self.state.get("cycle")
        if cycle is not None:
            self.check_paths(cycle)
        if cycle is None and not req["canary_required"] and self.state.get("parked"):
            cycle = self.state["cycle"] = self.state["parked"].pop(0)   # a delta that waited for the canary
            self.save()
        if cycle is None:
            self.training(req)   # before claiming or building anything
            if not req["canary_required"] and req.get("open_deltas", 0) >= req.get("max_open_deltas", 10**9):
                raise JoinError("waiting: this site's earlier deltas are not merged yet (the coordinator limits "
                                "unmerged deltas per site)", wait=300)
            cycle = {"client_id": uuid.uuid4().hex, "canary": bool(req["canary_required"]), "stage": "data"}
            key = f"{req['trainer']}|{req['config_sha256']}"
            if self.state.get("trainer_key") not in (None, key):   # pins changed: the old optimizer state is void
                shutil.rmtree(self.work / "trainer", ignore_errors=True)
            self.state.update(cycle=cycle, trainer_key=key)
            self.save()
        if cycle["stage"] == "data":
            self.set_phase("claiming")
            if self.opts.mode == "recipe" and not cycle.get("shard"):
                try:
                    shard = self.api.call("POST", "/api/v1/shards/claim")["shard"]
                except ApiError as error:   # the network is not set up yet: wait, nothing was built
                    if error.code != "not_ready":
                        raise
                    raise JoinError(f"waiting ({error.code}: {error.message})", UPLOAD_ACTIONS["not_ready"][1]) \
                        from error
                if shard is None:
                    self.state["cycle"] = None
                    self.save()
                    raise JoinError("no recipe shard is open right now; retry later or run with --mode byo",
                                    wait=300)
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
                if cycle["shard"].get("lease_id"):
                    body["lease_id"] = cycle["shard"]["lease_id"]
            elif self.opts.tag:
                body["tag"] = self.opts.tag
            body["attestation"] = terms.sign_attestation(self.token, {**body, "shard_id": body.get("shard_id"),
                                                                      "tag": body.get("tag")})
            try:
                reply = self.api.call("POST", "/api/v1/deltas", body)
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
                        "POST", f"/api/v1/uploads/{reply['upload_id']}/refresh", {"parts": numbers}), transfer,
                        self.save)
                    self.api.call("POST", f"/api/v1/deltas/{cycle['delta_id']}/complete", {"parts": parts})
            except ApiError as error:
                raise self.upload_error(cycle, error) from error
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
                                f"{status.get('reason')}); long rounds were not started. {self.next_step(cycle)}")
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

    def upload_error(self, cycle: dict[str, Any], error: ApiError) -> JoinError:
        """Apply :data:`UPLOAD_ACTIONS` to a failed upload of a finished delta; never discards it."""

        action, wait = upload_action(error.code)
        why = f"{error.code}: {error.message}"
        if action == "retry":
            return JoinError(f"waiting ({why}); the delta is kept and its upload resumes", wait)
        if action == "park":
            self.state.setdefault("parked", []).append(cycle)
            self.state["cycle"] = None
            self.save()
            return JoinError(f"waiting ({why}); the finished delta is kept and uploaded right after the canary", 0)
        if action == "new_id":
            cycle["client_id"] = uuid.uuid4().hex
            for key in ("transfer", "delta_id"):
                cycle.pop(key, None)
            self.save()
            return JoinError(f"waiting ({why}); registering the same delta again", 0)
        if action in ("aside", "aside_stop"):
            kept = self._set_aside(cycle, "unmergeable", why)
            kept.wait = 0 if action == "aside" else None
            return kept
        return JoinError(f"stopped ({why}); the delta and this cycle are kept, and the upload resumes on the next "
                         "`drafter-join run` once this is resolved")

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
        return JoinError(f"{why}. The delta is kept at {kept}. {self.next_step(cycle)}")

    @staticmethod
    def check_paths(cycle: dict[str, Any]) -> None:
        """An interrupted cycle resumes only where its files are (a site moved to another host resumes when the
        work and data paths are the same there); otherwise stop with what to do."""

        stage = cycle.get("stage")
        data = cycle.get("data") or []
        paths = [data[i + 1] for i, flag in enumerate(data[:-1])
                 if flag in ("--features", "--capture-index", "--capture-root")]
        if stage in ("train", "upload", "verdict") and cycle.get("base"):
            paths.append(cycle["base"]["path"])
        if stage in ("upload", "verdict") and cycle.get("delta"):
            paths.append(cycle["delta"])
        missing = [p for p in paths if not Path(p).exists()]
        if missing:
            raise JoinError(f"the interrupted {'canary' if cycle.get('canary') else 'round'} uses files that are not "
                            f"on this host: {', '.join(missing[:3])}. Make them available at the same paths (same "
                            "--work and data directories, or symlinks), or start the cycle over with "
                            "`drafter-join run --restart-cycle` (its progress is lost; finished deltas are kept).")

    @staticmethod
    def next_step(cycle: dict[str, Any]) -> str:
        """The exact command after a cycle was given up (JOIN.md: `setup` ends with the canary)."""

        if cycle.get("canary"):
            return ("Fix the cause, then rerun `drafter-join setup` with the same flags: it checks this host again and "
                    "runs a new canary (run `drafter-join update --yes` first if the message asks for it).")
        return ("The next `drafter-join run` starts a new round (run `drafter-join update --yes` first if the message "
                "asks for it).")

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
                    if error.wait is None:
                        raise
                    say(str(error))
                    self.sleep(error.wait)
                    continue
                done += not result["canary"]
        finally:
            self.beat_stop.set()
            self.stopped()

    def stopped(self) -> None:
        """A last heartbeat on any exit, so the dashboard does not show a stopped site as training."""

        self.phase = "stopped" if "stopped" in self.phases else "idle"   # older coordinators: no "stopped"
        self.live = {}
        self.heartbeat()


# --------------------------------------------------------------------------
# setup (preflight)
# --------------------------------------------------------------------------

BUILD_FIX = ("install the Python headers and a C compiler (Debian/Ubuntu: sudo apt install python3-dev "
             "build-essential); or, without root, a uv-managed CPython: pip install uv && uv python install 3.12 && "
             "\"$(uv python find 3.12)\" -m venv .venv, then redo JOIN.md step 1 in it; or the container in JOIN.md "
             "step 1")
TRITON_PROBE = """import torch, triton, triton.language as tl


@triton.jit
def add_one(x, n: tl.constexpr):
    offsets = tl.arange(0, n)
    tl.store(x + offsets, tl.load(x + offsets) + 1)


x = torch.zeros(16, device="cuda")
add_one[(1,)](x, 16)
torch.cuda.synchronize()
assert float(x.sum()) == 16.0
"""


def build_check(gpu: bool, include: Path | None = None) -> tuple[bool | None, str]:
    """Triton compiles kernels (and its launcher, against Python.h) at the first compiled GPU step, which the CPU smoke
    never reaches: check the headers, and on a GPU compile and run a trivial kernel."""

    import sysconfig

    include = Path(include or sysconfig.get_paths()["include"])
    if not (include / "Python.h").is_file():
        return (False if gpu else None), f"no Python.h in {include}, so Triton cannot compile GPU kernels; {BUILD_FIX}"
    if not gpu:
        return True, f"Python.h in {include} (no GPU: Triton not compiled)"
    with tempfile.TemporaryDirectory(prefix="drafter-triton-") as tmp:
        probe = Path(tmp) / "probe.py"   # triton.jit reads the kernel's source file
        probe.write_text(TRITON_PROBE)
        try:
            done = subprocess.run([sys.executable, str(probe)], capture_output=True, text=True, timeout=900)
        except subprocess.TimeoutExpired:
            return False, "compiling a trivial Triton kernel took over 15 minutes"
    if done.returncode:
        tail = " ".join((done.stderr or done.stdout).strip().splitlines()[-2:])[-300:]
        return False, f"compiling a trivial Triton kernel failed ({tail}); {BUILD_FIX}"
    return True, "compiled and ran a trivial Triton kernel"


def open_file_check() -> tuple[bool | None, str]:
    """The capture loader memory-maps segment files (one open file each); a low soft limit fails mid-round."""

    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)

    def limit(value: int) -> float:
        return math.inf if value == resource.RLIM_INFINITY else value

    if limit(soft) >= NOFILE_MIN:
        return True, f"soft limit {soft}"
    detail = (f"soft limit {soft} < {NOFILE_MIN}: run `ulimit -n $(ulimit -Hn)` in the shell that runs drafter-join "
              f"(hard limit {hard})")
    if limit(hard) < NOFILE_MIN:
        detail += f"; the hard limit is below {NOFILE_MIN} too: ask the host's admin to raise nofile"
    return None, detail


SWING_WARN_GB = 8.0     # MemAvailable moving more than this while sampled: other jobs come and go on this host
BIG_PROCESS_GB = 4.0    # another process this large (or any other GPU process) is reported, by count only


def memory_steadiness(seconds: float, read: Callable[[], float] | None = None,
                      sleep: Callable[[float], None] = time.sleep) -> tuple[bool | None, str]:
    """Sample MemAvailable for ``seconds``: a GB10 shared with other users' containers swings by tens of GB, which
    the trainer then has to pause through (or a host-level guard kills it)."""

    from .train import mem_available_gb

    read = read or mem_available_gb
    samples = [read()]
    if math.isinf(samples[0]):
        return None, "not measured (no MemAvailable on this OS)"
    for _ in range(int(seconds // 2)):
        sleep(2)
        samples.append(read())
    low, high = min(samples), max(samples)
    detail = f"MemAvailable {low:.0f}-{high:.0f} GB over {seconds:.0f} s"
    if high - low > SWING_WARN_GB:
        return None, (f"{detail}: other jobs on this host move memory; a dedicated host is strongly recommended "
                      "(the trainer pauses and resumes through dips, which costs time)")
    return True, detail


def other_processes(me: int | None = None) -> tuple[int, int]:
    """(other GPU compute processes, other processes above ``BIG_PROCESS_GB`` resident): counts only."""

    me = os.getpid() if me is None else me
    gpu = 0
    try:
        done = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=30)
        gpu = sum(1 for line in done.stdout.split() if line.strip().isdigit() and int(line) != me)
    except (OSError, subprocess.TimeoutExpired):
        pass
    big = 0
    for status in Path("/proc").glob("[0-9]*/status"):
        try:
            if int(status.parent.name) == me:
                continue
            rss = next((int(l.split()[1]) for l in status.read_text().splitlines() if l.startswith("VmRSS:")), 0)
        except (OSError, ValueError):
            continue
        big += rss / 2**20 >= BIG_PROCESS_GB
    return gpu, big


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
        add("Triton build (Python.h, compiler)", *build_check(cuda))
    except ImportError:
        add("torch 2.11-2.12", False, "pip install 'torch>=2.11,<2.13'")
    add("open-file limit", *open_file_check())
    add("memory steadiness", *memory_steadiness(getattr(opts, "memory_sample_s", 60)))
    gpu, big = other_processes()
    add("other workloads", True if not gpu and not big else None,
        f"{gpu} other GPU process(es), {big} other process(es) above {BIG_PROCESS_GB:.0f} GB"
        + ("" if not gpu and not big else ": this host is shared; expect their memory peaks during training (a "
                                          "dedicated host is strongly recommended)"))
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
        q.add_argument("--byo-capture-root", type=Path, help="byo: the directory capture index paths (and "
                       "--byo-captures globs) are relative to")
        q.add_argument("--byo-captures", action="append", default=[], metavar="GLOB",
                       help="byo: instead of --byo-capture-index, index captures in this repo's format under "
                            "--byo-capture-root (for example 'fast-*/p*'); repeatable")
        q.add_argument("--byo-manifest", type=Path, action="append", default=[],
                       help="byo, with --byo-captures: capture manifest (JSON/JSONL); only rows it lists are indexed")
        q.add_argument("--byo-exclude-ids", type=Path, action="append", default=[],
                       help="byo: JSONL of row ids ({\"id\": ...}) this site must never train or self-check on, for "
                            "example private held-out rows on the same disks; repeatable")
        q.add_argument("--tag", default=None, help="byo: an opaque label for your data (shown to the operator)")
        q.add_argument("--seed", type=int, default=20261014, help="byo: trainer seed")
        q.add_argument("--gpus", type=int, default=1, help="local GPUs for training (one GB10 is enough)")
        q.add_argument("--launcher", default=None, help="custom trainer launcher prefix, e.g. a torchrun command")
        q.add_argument("--trainer-arg", action="append", default=[],
                       help="extra local trainer flag (performance, memory guards), repeatable")
        q.add_argument("--rounds", type=int, default=None, help="stop after N rounds (default: run forever)")
        q.add_argument("--canary-timeout-s", type=float, default=3600)
        q.add_argument("--restart-cycle", action="store_true",
                       help="drop an interrupted cycle and start it over (finished deltas are kept)")
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
            q.add_argument("--memory-sample-s", type=float, default=60,
                           help="how long setup samples MemAvailable for swings caused by other jobs")
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
        if (opts.byo_captures or opts.byo_manifest) and (opts.byo_capture_index or not opts.byo_captures
                                                         or not opts.byo_capture_root):
            raise JoinError("--byo-captures GLOB [--byo-manifest FILE] needs --byo-capture-root and replaces "
                            "--byo-capture-index")
        if opts.byo_captures and opts.cmd in ("setup", "run"):
            from . import model as sm

            opts.byo_capture_index = byo_source_index(opts.byo_capture_root, opts.byo_captures, opts.byo_manifest,
                                                      Path(opts.work) / "byo-source-index", sm.load_config(opts.config))
        if opts.mode == "byo" and not (opts.byo_features or (opts.byo_capture_root and (opts.byo_capture_index or
                                                                                       opts.byo_captures))):
            raise JoinError("--mode byo needs --byo-features, or --byo-capture-root with --byo-capture-index or "
                            "--byo-captures")
        if opts.byo_exclude_ids and not (opts.byo_capture_index or opts.byo_captures):
            raise JoinError("--byo-exclude-ids applies to --byo-capture-index or --byo-captures")
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
            print(json.dumps({"local": {k: site.state.get(k) for k in ("rounds", "tokens", "outbox", "cycle", "parked",
                                                                         "self_check")},
                              "pins": site.pins(), "coordinator": site.api.call("GET", "/api/v1/me")}, indent=1))
            return 0
        if opts.teacher is None:
            raise JoinError("--teacher is required (your GLM-5.3-Flash directory with the embedding and lm_head)")
        import signal

        def stop(number: int, frame) -> None:   # SIGTERM / SIGHUP (a closed shell): unwind, stop the trainer, report
            raise SystemExit(128 + number)

        for name in ("SIGTERM", "SIGHUP"):
            if signal.getsignal(getattr(signal, name)) is not signal.SIG_IGN:   # nohup keeps SIGHUP ignored
                signal.signal(getattr(signal, name), stop)
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
