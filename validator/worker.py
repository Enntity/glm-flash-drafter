"""Validator and merger: the operator runs it on trusted GPU hardware next to the private held-out set.

Loop (:meth:`Validator.step`): announce the validator version (a new version makes the coordinator re-queue every
quarantined or held delta), validate pending deltas (canaries first), then merge accepted deltas once a round is
ready.  Validation of one delta: download and verify size and SHA-256, the shared checks
(:func:`enntity_drafter.validation.check_delta`, the same code sites run before uploading), optional spot check
(recipe mode), then the held-out eval of ``base - delta`` against the base, judged conservatively
(:func:`~enntity_drafter.validation.judge`).  A crash leaves the delta claimed; the coordinator hands it out again
after its claim timeout, so no verdict is lost or doubled.

Merge (:func:`merge_streamed`): token-weighted plain average of the accepted deltas (staleness down-weighted by
``stale_decay ** versions_behind``, borderline ones by their reduced factor), outer step lr 1 / momentum 0 by
default, one tensor at a time so memory stays bounded.  The fp32 master stays here; the published copy (bf16 by
default) goes to object storage through a signed upload.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch

from coordinator.client import ApiClient, ApiError, download, sha256_file, upload_file
from enntity_drafter import diloco as dl
from enntity_drafter.guard import guarded_safe_open
from enntity_drafter.validation import (
    VALIDATOR_VERSION, CheckFailed, StreamWriter, Thresholds, candidate_weights, check_delta, judge, layout,
    norm_bounds, read_header, weights_stamp,
)


def merge_weights(items: list[dict[str, Any]], current: int, thresholds: Thresholds,
                  token_cap: float | None = None) -> list[float]:
    """Merge weight per accepted delta: tokens (self-reported, so capped at ``token_cap``) x borderline factor x
    stale_decay ** versions behind."""

    cap = token_cap or float("inf")
    return [min(max(float(item["meta"].get("tokens") or 0.0), 1.0), cap)
            * float(item["verdict"].get("weight_factor") or 1.0)
            * thresholds.stale_decay ** max(0, current - item["base_version"]) for item in items]


def token_cap(history: list[float], thresholds: Thresholds) -> float | None:
    """A delta weighs at most ``token_factor`` x the median tokens of accepted deltas (once there is history)."""

    if len(history) < thresholds.history_min:
        return None
    return sorted(history)[len(history) // 2] * thresholds.token_factor


def _decode(handle, name: str, shape: tuple[int, ...], dtype: str) -> torch.Tensor:
    keys = [name + suffix for suffix in dl.INT8_KEYS] if dtype == "int8" else [name]
    return dl.decode_named({k: handle.get_tensor(k) for k in keys}, {name: shape}, dtype)[name]


def merge_streamed(master: Path, deltas: list[tuple[Path, float]], out_master: Path, out_published: Path | None,
                   publish_dtype: str, meta: dict[str, Any], lr: float = 1.0, momentum: float = 0.0,
                   nesterov: bool = True, momentum_in: Path | None = None, momentum_out: Path | None = None
                   ) -> tuple[str, str]:
    """New global = master - lr * u (``dl.outer_update``) with u from the weighted mean of the deltas.  Equal
    weights take the plain mean in arrival order (bitwise the trainer's own DiLoCo average)."""

    shapes = {name: shape for name, (_, shape) in layout(master).items()}
    weights = [w for _, w in deltas]
    equal = len(set(weights)) == 1
    stamp = weights_stamp(master)
    header = {**stamp, "weights": json.dumps({"schema": dl.WEIGHTS_SCHEMA, "dtype": "fp32", **meta})}
    entries = [(n, torch.float32, s) for n, s in shapes.items()]
    writer = StreamWriter(out_master, entries, header)
    published = None
    if out_published is not None and publish_dtype != "fp32":
        published = StreamWriter(out_published, [(n, dl.WEIGHT_DTYPES[publish_dtype], s) for n, s in shapes.items()],
                                 {**stamp, "weights": json.dumps({"schema": dl.WEIGHTS_SCHEMA, "dtype": publish_dtype,
                                                                  **meta})})
    buffer = StreamWriter(momentum_out, entries, stamp) if momentum and momentum_out else None
    handles = [guarded_safe_open(path, "drafter") for path, _ in deltas]
    dtypes = [json.loads(read_header(path)["__metadata__"]["delta"])["delta_dtype"] for path, _ in deltas]
    momentum_handle = guarded_safe_open(momentum_in, "drafter") if momentum and momentum_in else None
    try:
        with guarded_safe_open(master, "drafter") as base:
            for name, shape in shapes.items():
                param = base.get_tensor(name).float()
                mean = torch.zeros_like(param)
                for handle, dtype, weight in zip(handles, dtypes, weights):
                    mean.add_(_decode(handle, name, shape, dtype), alpha=1.0 if equal else weight)
                mean.div_(len(weights) if equal else sum(weights))
                moment = None
                if momentum:
                    moment = momentum_handle.get_tensor(name).float() if momentum_handle else torch.zeros_like(param)
                dl.outer_update(param, mean, moment, lr=lr, momentum=momentum, nesterov=nesterov)
                writer.write(name, param)
                if published is not None:
                    published.write(name, param)
                if buffer is not None:
                    buffer.write(name, moment)
    finally:
        for handle in handles:
            handle.__exit__(None, None, None)
    sha_master = writer.close()
    if buffer is not None:
        buffer.close()
    if published is None:
        if out_published is not None:
            shutil.copyfile(out_master, out_published)
        return sha_master, sha_master
    return sha_master, published.close()


def cosine(a_path: Path, b_path: Path, base_path: Path) -> float:
    """Cosine similarity of two delta files of the model in ``base_path`` (streamed)."""

    dtypes = [json.loads(read_header(p)["__metadata__"]["delta"])["delta_dtype"] for p in (a_path, b_path)]
    dot = na = nb = 0.0
    with guarded_safe_open(a_path, "drafter") as a, guarded_safe_open(b_path, "drafter") as b:
        for name, (_, shape) in layout(base_path).items():
            x, y = _decode(a, name, shape, dtypes[0]).double(), _decode(b, name, shape, dtypes[1]).double()
            dot, na, nb = dot + float((x * y).sum()), na + float(x.square().sum()), nb + float(y.square().sum())
    return dot / math.sqrt(max(na * nb, 1e-300))


class Validator:
    def __init__(self, api: ApiClient, workdir: str | Path, evaluate: Callable[[Path], dict[str, float]], *,
                 requirements: dict[str, Any] | None = None, thresholds: Thresholds | None = None,
                 version: str = VALIDATOR_VERSION, publish_dtype: str = "bf16", min_deltas: int = 2,
                 min_round_s: float = 0.0, round_timeout_s: float = 3600.0, outer_lr: float = 1.0,
                 outer_momentum: float = 0.0, outer_nesterov: bool = True, heldout_id: str = "",
                 keep_local: int = 9, replayer: Callable[[dict[str, Any], Path], Path] | None = None,
                 spot_check_rate: float = 0.0, spot_cosine: float = 0.5, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] = print):
        self.api, self.workdir, self.evaluate = api, Path(workdir), evaluate
        self.requirements = dict(requirements or {})
        self.thresholds, self.version, self.publish_dtype = thresholds or Thresholds(), version, publish_dtype
        self.min_deltas, self.min_round_s, self.round_timeout_s = min_deltas, min_round_s, round_timeout_s
        self.outer = {"lr": outer_lr, "momentum": outer_momentum, "nesterov": outer_nesterov}
        self.heldout_id, self.keep_local, self.clock, self.log = heldout_id, keep_local, clock, log
        self.replayer, self.spot_check_rate, self.spot_cosine = replayer, spot_check_rate, spot_cosine
        self.rng = random.Random(0)
        for sub in ("versions", "inbox", "tmp"):
            (self.workdir / sub).mkdir(parents=True, exist_ok=True)
        self.state_path = self.workdir / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.is_file() else {"history": []}

    # ------------------------------------------------------------------ local files
    def _save(self) -> None:
        temporary = self.state_path.with_name("state.json.tmp")
        temporary.write_text(json.dumps(self.state, indent=1))
        temporary.replace(self.state_path)

    def master(self, version: int) -> Path:
        return self.workdir / "versions" / f"v{version:06d}.master.safetensors"

    def published(self, version: int) -> Path:
        return self.workdir / "versions" / f"v{version:06d}.published.safetensors"

    def ensure_version(self, version: int, sha256: str) -> Path:
        path = self.published(version)
        if path.is_file() and self.state.get("sha", {}).get(str(version)) == sha256:
            return path
        info = self.api.call("GET", f"/api/v1/validator/versions/{version}")
        download(info["url"], path, sha256=sha256, size=info["size"])
        self.state.setdefault("sha", {})[str(version)] = sha256
        self._save()
        return path

    def metrics(self, version: int, path: Path) -> dict[str, float]:
        cache = self.workdir / "versions" / f"v{version:06d}.eval.json"
        if cache.is_file():
            saved = json.loads(cache.read_text())
            if saved.get("heldout_id") == self.heldout_id:
                return saved["metrics"]
        metrics = self.evaluate(path)
        cache.write_text(json.dumps({"heldout_id": self.heldout_id, "metrics": metrics}))
        return metrics

    # ------------------------------------------------------------------ protocol
    def hello(self) -> dict[str, Any]:
        reply = self.api.call("POST", "/api/v1/validator/hello", {"version": self.version})
        self.requirements = {**reply.get("requirements", {}), **self.requirements}
        return reply

    def bootstrap(self, weights: dict[str, torch.Tensor], stamp: dict[str, Any]) -> int:
        """Publish version 0 from fp32 tensors (e.g. a trainer checkpoint's params)."""

        if self.hello().get("current") is not None:
            raise RuntimeError("the coordinator already has a global version")
        master = self.master(0)
        dl.save_weights(master, weights, "fp32", stamp, {"version": 0})
        if self.publish_dtype == "fp32":
            shutil.copyfile(master, self.published(0))
        else:
            dl.save_weights(self.published(0), weights, self.publish_dtype, stamp, {"version": 0})
        return self._publish(0, [], self.published(0))

    def _publish(self, version: int, merged: list[dict[str, Any]], path: Path) -> int:
        sha, size = sha256_file(path), path.stat().st_size
        self.state.setdefault("sha", {})[str(version)] = sha
        metrics = self.metrics(version, path)
        upload_key = f"version-{version}"
        progress = self.state.setdefault("uploads", {}).setdefault(upload_key, {})
        if "upload_id" not in progress:
            reply = self.api.call("POST", f"/api/v1/validator/versions/{version}/upload",
                                  {"size": size, "sha256": sha, "dtype": self.publish_dtype})
            progress.update(upload_id=reply["upload_id"], plan=reply["upload"])
            self._save()
        refresh = lambda numbers: self.api.call(   # noqa: E731
            "POST", f"/api/v1/validator/uploads/{progress['upload_id']}/refresh", {"parts": numbers})
        parts = upload_file(path, progress["plan"], refresh, progress.setdefault("transfer", {}), self._save)
        self.api.call("POST", f"/api/v1/validator/versions/{version}/publish", {
            "upload_id": progress["upload_id"], "parts": parts, "merged": merged, "dtype": self.publish_dtype,
            "evals": {"metrics": metrics}, "heldout_id": self.heldout_id,
            "norm_bounds": norm_bounds(self.state["history"], self.thresholds)})
        self.state["uploads"].pop(upload_key, None)
        self.state.update(current=version, last_merge=self.clock())
        self._save()
        self._prune(version)
        self.log(f"published v{version} ({len(merged)} deltas merged) {metrics}")
        return version

    def _prune(self, current: int) -> None:
        for path in (self.workdir / "versions").glob("v*.safetensors"):
            version = int(path.name[1:7])
            if path.name.endswith(".master.safetensors") and version < current - 1:
                path.unlink()
            elif version <= current - self.keep_local:
                path.unlink()

    def step(self) -> dict[str, Any]:
        self.hello()
        verdicts = [self.process(item) for item in self.api.call("GET", "/api/v1/validator/pending")]
        merged = self.maybe_merge()
        return {"verdicts": verdicts, "merged": merged}

    def process(self, item: dict[str, Any]) -> dict[str, Any]:
        claim = self.api.call("POST", f"/api/v1/validator/deltas/{item['delta_id']}/claim")
        path = self.workdir / "inbox" / f"{item['delta_id']}.safetensors"
        candidate = self.workdir / "tmp" / f"{item['delta_id']}.candidate.safetensors"
        verdict: dict[str, Any] = {"validator_version": self.version,
                                   "staleness": self.state.get("current", item["base_version"]) - item["base_version"]}
        try:
            try:   # network errors propagate: the claim times out and the delta is retried
                download(claim["url"], path, sha256=item["sha256"], size=item["size"], retries=2)
            except ApiError as error:
                if error.code not in ("download_hash", "download_size"):
                    raise
                raise CheckFailed("hash_mismatch", "the stored file differs from the declared size/SHA-256; "
                                                   "upload it again") from error
            base = self.ensure_version(item["base_version"], item["base_sha256"])
            try:   # anything unparseable in an uploaded file is the file's fault: quarantine, never crash-loop
                stats = check_delta(path, base, {"base_sha256": item["base_sha256"],
                                                 "trainer": self.requirements.get("trainer"),
                                                 "config_sha256": self.requirements.get("config_sha256")},
                                    norm_bounds(self.state["history"], self.thresholds))
            except CheckFailed:
                raise
            except Exception as error:  # noqa: BLE001
                raise CheckFailed("malformed", f"the delta file could not be read ({type(error).__name__})") from error
            verdict["checks"] = {"rms": stats["rms"], "max_abs": stats["max_abs"]}
            if (self.replayer is not None and item["mode"] == "recipe" and claim.get("recipe")
                    and (item["canary"] or self.rng.random() < self.spot_check_rate)):
                similarity = cosine(path, self.replayer(claim, base), base)
                verdict["checks"]["spot_cosine"] = similarity
                if similarity < self.spot_cosine:
                    raise CheckFailed("spot_check", f"replaying the recipe gives a different update (cosine "
                                                    f"{similarity:.2f} < {self.spot_cosine})")
            candidate_weights(base, path, candidate)
            cand = self.evaluate(candidate)
            base_metrics = self.metrics(item["base_version"], base)
            decision, code, message, _ = judge(base_metrics, cand, self.thresholds)
            verdict.update(metrics=cand, base_metrics=base_metrics)
        except CheckFailed as failure:
            decision, code, message = "reject", failure.code, failure.message
        finally:
            candidate.unlink(missing_ok=True)
        verdict.update(decision=decision, code=code, reason=message,
                       weight_factor=self.thresholds.reduce_weight if decision == "reduce" else 1.0)
        if decision in ("accept", "reduce"):
            self.state["history"] = (self.state["history"] + [verdict["checks"]["rms"]])[-200:]
            if not item["canary"]:
                self.state["tokens"] = (self.state.get("tokens", []) + [float(item["meta"].get("tokens") or 0)])[-200:]
            self._save()
        else:
            path.unlink(missing_ok=True)   # the coordinator keeps the object (quarantine)
        self.api.call("POST", f"/api/v1/validator/deltas/{item['delta_id']}/verdict", verdict)
        self.log(f"{item['delta_id']} ({item['site']}, base v{item['base_version']}"
                 f"{', canary' if item['canary'] else ''}): {decision} {code or ''} {message}")
        return {"delta_id": item["delta_id"], "decision": decision, "code": code}

    def maybe_merge(self, force: bool = False) -> int | None:
        items = self.api.call("GET", "/api/v1/validator/mergeable")
        current = self.hello()["current"]
        if not items or current is None:
            return None
        oldest = min(item["received"] for item in items)
        ready = len(items) >= self.min_deltas or self.clock() - oldest >= self.round_timeout_s
        if not force and (not ready or self.clock() - self.state.get("last_merge", 0) < self.min_round_s):
            return None
        version = current["version"]
        weights = merge_weights(items, version, self.thresholds,
                                token_cap(self.state.setdefault("tokens", []), self.thresholds))
        files = []
        for item in items:
            path = self.workdir / "inbox" / f"{item['delta_id']}.safetensors"
            if not path.is_file():
                info = self.api.call("GET", f"/api/v1/validator/deltas/{item['delta_id']}")
                download(info["url"], path, sha256=info["sha256"], size=info["size"])
            files.append(path)
        master = self.master(version)
        if not master.is_file():   # a new validator host: continue from the published copy
            shutil.copyfile(self.ensure_version(version, current["sha256"]), master)
        meta = {"version": version + 1, "parent_sha256": current["sha256"], "merged": len(items)}
        momentum = self.workdir / "versions" / "momentum.safetensors"
        merge_streamed(master, list(zip(files, weights)), self.master(version + 1), self.published(version + 1),
                       self.publish_dtype, meta, momentum_in=momentum if momentum.is_file() else None,
                       momentum_out=self.workdir / "tmp" / "momentum.safetensors", **self.outer)
        if self.outer["momentum"]:
            (self.workdir / "tmp" / "momentum.safetensors").replace(momentum)
        merged = [{"delta_id": item["delta_id"], "weight": weight, "site": item["site"],
                   "base_version": item["base_version"]} for item, weight in zip(items, weights)]
        published = self._publish(version + 1, merged, self.published(version + 1))
        for path in files:
            path.unlink(missing_ok=True)
        return published
