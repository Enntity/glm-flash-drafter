"""Validator and merger: the operator runs it on trusted GPU hardware next to the private held-out set.

Loop (:meth:`Validator.step`): announce the validator version (a new version makes the coordinator re-queue every
quarantined or held delta), validate pending deltas (canaries first), then merge accepted deltas once a round is
ready.  Validation of one delta: download and verify size and SHA-256, the shared checks
(:func:`enntity_drafter.validation.check_delta`, the same code sites run before uploading), optional spot check
(recipe mode), then the held-out eval of ``base - delta`` against the base, judged conservatively
(:func:`~enntity_drafter.validation.judge`; a canary, never merged, gets the sanity checks and only a catastrophic
bound, :func:`~enntity_drafter.validation.judge_canary`).  A crash leaves the delta claimed; the coordinator hands it
out again after its claim timeout, so no verdict is lost or doubled.

Merge (:func:`merge_streamed`): token-weighted plain average of the accepted deltas (staleness down-weighted by
``stale_decay ** versions_behind``, borderline ones by their reduced factor), outer step lr 1 / momentum 0 by
default, one tensor at a time so memory stays bounded.  The fp32 master stays here; the published copy (bf16 by
default) goes to object storage through a signed upload.
"""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import random
import shutil
import time
from pathlib import Path
from typing import Any, Callable

import torch

from coordinator import terms
from coordinator.client import ApiClient, ApiError, download, sha256_file, upload_file
from enntity_drafter import diloco as dl
from enntity_drafter.guard import guarded_safe_open
from enntity_drafter.validation import (
    VALIDATOR_VERSION, CheckFailed, StreamWriter, Thresholds, candidate_weights, canary_bounds, check_delta, judge,
    judge_canary, layout, norm_bounds, read_header, weights_stamp,
)


def score(metrics: dict[str, float]) -> float:
    groups = [v for k, v in metrics.items() if "." not in k]
    return sum(groups) / max(len(groups), 1)


def merge_weights(items: list[dict[str, Any]], current: int, thresholds: Thresholds, *,
                  attenuate: bool = True) -> list[float]:
    """Merge weight per accepted delta.  Tokens are self-reported, so they are trusted only across sites: each site's
    total for the round is capped at ``token_factor`` x the median site total (one value per site), and with fewer
    than ``min_token_sites`` sites every site weighs the same.  A site's budget is shared by its deltas in
    proportion to their tokens, then multiplied by the borderline factor and ``stale_decay ** versions behind``."""

    def tokens(item: dict[str, Any]) -> float:
        return max(float(item["meta"].get("tokens") or 0.0), 1.0)

    def factor(item: dict[str, Any]) -> float:
        value = item["verdict"].get("weight_factor")
        return 1.0 if value is None else float(value)

    totals: dict[str, float] = {}
    for item in items:
        totals[item["site"]] = totals.get(item["site"], 0.0) + tokens(item)
    if len(totals) >= thresholds.min_token_sites:
        median = sorted(totals.values())[len(totals) // 2]
        budget = {site: min(total, thresholds.token_factor * median) for site, total in totals.items()}
    else:
        budget = dict.fromkeys(totals, 1.0)
    return [budget[item["site"]] * tokens(item) / totals[item["site"]]
            * ((factor(item)
                * thresholds.stale_decay ** max(0, current - item["base_version"])) if attenuate else 1.0)
            for item in items]


def _decode(handle, name: str, shape: tuple[int, ...], dtype: str) -> torch.Tensor:
    keys = dl.delta_keys(name, dtype, set(handle.keys()))
    return dl.decode_named({k: handle.get_tensor(k) for k in keys}, {name: shape}, dtype)[name]


def merge_streamed(master: Path, deltas: list[tuple[Path, float]], out_master: Path, out_published: Path | None,
                   publish_dtype: str, meta: dict[str, Any], lr: float = 1.0, momentum: float = 0.0,
                   nesterov: bool = True, momentum_in: Path | None = None, momentum_out: Path | None = None,
                   normalizer: float | None = None
                   ) -> tuple[str, str]:
    """New global = master - lr * u (``dl.outer_update``), with the weighted delta sum divided by normalizer.
    Omitting normalizer preserves legacy normalization by the sum of weights. Equal, unattenuated weights
    take the plain mean in arrival order (bitwise the trainer's own DiLoCo average)."""

    shapes = {name: shape for name, (_, shape) in layout(master).items()}
    weights = [w for _, w in deltas]
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("merge requires finite nonnegative weights and at least one delta")
    denominator = sum(weights) if normalizer is None else normalizer
    if not math.isfinite(denominator) or denominator <= 0:
        raise ValueError("merge normalizer must be finite and positive")
    # Preserve the unattenuated site's budget in the denominator. Otherwise
    # a singleton (or equally penalized batch) cancels its reduction entirely.
    equal = len(set(weights)) == 1 and weights[0] > 0 and denominator == sum(weights)
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
                mean.div_(len(weights) if equal else denominator)
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


SKETCH_PER_TENSOR = 256


def sketch(delta_path: Path, base_path: Path) -> torch.Tensor:
    """A fixed random sample of every tensor's delta values (the same coordinates for every delta): cosine of two
    sketches estimates the cosine of the deltas, so near-duplicates are found without re-reading old files."""

    dtype = json.loads(read_header(delta_path)["__metadata__"]["delta"])["delta_dtype"]
    parts = []
    with guarded_safe_open(delta_path, "drafter") as handle:
        for name, (_, shape) in layout(base_path).items():
            value = _decode(handle, name, shape, dtype).reshape(-1)
            generator = torch.Generator().manual_seed(int(hashlib.sha256(name.encode()).hexdigest()[:12], 16))
            parts.append(value[torch.randint(0, value.numel(), (min(value.numel(), SKETCH_PER_TENSOR),),
                                             generator=generator)])
    return torch.cat(parts).double()


def sketch_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a @ b) / max(float(a.norm() * b.norm()), 1e-300))


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
                 version: str = VALIDATOR_VERSION, publish_dtype: str = "bf16", min_deltas: int = 2, min_sites: int = 1,
                 min_round_s: float = 0.0, round_timeout_s: float = 3600.0, outer_lr: float = 1.0,
                 outer_momentum: float = 0.0, outer_nesterov: bool = True, heldout_id: str = "",
                 keep_local: int = 9, replayer: Callable[[dict[str, Any], Path], Path] | None = None,
                 spot_check_rate: float = 0.0, spot_cosine: float = 0.5, clock: Callable[[], float] = time.time,
                 log: Callable[[str], None] = print, signing_key: Any = None, eval_repeats: int = 3):
        self.api, self.workdir, self.evaluate = api, Path(workdir), evaluate
        self.requirements = dict(requirements or {})
        self.thresholds, self.version, self.publish_dtype = thresholds or Thresholds(), version, publish_dtype
        self.min_deltas, self.min_round_s, self.round_timeout_s = min_deltas, min_round_s, round_timeout_s
        if min_sites < 1:
            raise ValueError("min_sites must be positive")
        self.min_sites = min_sites
        self.outer = {"lr": outer_lr, "momentum": outer_momentum, "nesterov": outer_nesterov}
        self.heldout_id, self.keep_local, self.clock, self.log = heldout_id, keep_local, clock, log
        self.replayer, self.spot_check_rate, self.spot_cosine = replayer, spot_check_rate, spot_cosine
        self.rng = random.Random(0)
        self.signing_key = signing_key   # cryptography Ed25519PrivateKey: signs every published global
        self.eval_repeats = eval_repeats  # evaluations of version 0 that estimate the held-out noise margin
        for sub in ("versions", "inbox", "tmp", "sketches"):
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

    def metrics(self, sha256: str, path: Path) -> dict[str, float]:
        """Held-out metrics of a weights file, cached by its content hash and the held-out set."""

        cache = self.workdir / "versions" / f"{sha256[:24]}.eval.json"
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
        """Publish version 0 from fp32 tensors (e.g. a trainer checkpoint's params); it is also the fixed reference
        every later merged global is gated against."""

        if self.hello().get("current") is not None:
            raise RuntimeError("the coordinator already has a global version")
        master = self.master(0)
        dl.save_weights(master, weights, "fp32", stamp, {"version": 0})
        if self.publish_dtype == "fp32":
            shutil.copyfile(master, self.published(0))
        else:
            dl.save_weights(self.published(0), weights, self.publish_dtype, stamp, {"version": 0})
        metrics = self.metrics(sha256_file(self.published(0)), self.published(0))
        repeats = [metrics] + [self.evaluate(self.published(0)) for _ in range(self.eval_repeats - 1)]
        spread = [(max(r[g] for r in repeats) - min(r[g] for r in repeats)) / abs(metrics[g])
                  for g in metrics if "." not in g and metrics[g]]
        self.state.update(reference_metrics=metrics, noise=2.0 * max(spread, default=0.0))   # eval noise margin
        return self._publish(0, [], self.published(0), metrics)

    def _publish(self, version: int, merged: list[dict[str, Any]], path: Path, metrics: dict[str, float],
                 forced: bool = False) -> int:
        sha, size = sha256_file(path), path.stat().st_size
        self.state.setdefault("sha", {})[str(version)] = sha
        uploads = self.state.setdefault("uploads", {})
        for key in [k for k in uploads if k.startswith(f"{version}-") and k != f"{version}-{sha}"]:
            uploads.pop(key)   # progress of a different file for this version must never be reused
        progress = uploads.setdefault(f"{version}-{sha}", {})
        if "upload_id" not in progress:
            reply = self.api.call("POST", f"/api/v1/validator/versions/{version}/upload",
                                  {"size": size, "sha256": sha, "dtype": self.publish_dtype})
            progress.update(upload_id=reply["upload_id"], plan=reply["upload"])
            self._save()
        refresh = lambda numbers: self.api.call(   # noqa: E731
            "POST", f"/api/v1/validator/uploads/{progress['upload_id']}/refresh", {"parts": numbers})
        parts = upload_file(path, progress["plan"], refresh, progress.setdefault("transfer", {}), self._save)
        body = {"upload_id": progress["upload_id"], "parts": parts, "merged": merged, "dtype": self.publish_dtype,
                "evals": {"metrics": metrics}, "heldout_id": self.heldout_id,
                "norm_bounds": norm_bounds(self.state["history"], self.thresholds)}
        if self.signing_key is not None:
            body["signature"] = self.signing_key.sign(terms.global_message(version, sha)).hex()
        if forced:
            body["forced"] = True
        self.api.call("POST", f"/api/v1/validator/versions/{version}/publish", body)
        uploads.pop(f"{version}-{sha}", None)
        best = self.state.get("best")   # one coherent version, never a per-group maximum across versions
        if best is None or score(metrics) > score(best["metrics"]):
            self.state["best"] = {"version": version, "metrics": metrics}
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
        for path in (self.workdir / "versions").glob("momentum-v*.safetensors"):
            if int(path.name[10:16]) < current:
                path.unlink()

    def step(self) -> dict[str, Any]:
        self.hello()
        verdicts = []
        for item in self.api.call("GET", "/api/v1/validator/pending"):
            try:
                verdicts.append(self.process(item))
            except (urllib.error.URLError, ConnectionError, TimeoutError) as error:
                self.log(f"{item['delta_id']}: network error ({error!r}); retried after the claim times out")
            except ApiError as error:
                if error.code in ("not_pending", "unreachable") or error.status == 0:
                    self.log(f"{item['delta_id']}: skipped ({error.code})")
                    continue
                verdicts.append(self._validator_error(item, error))
            except Exception as error:  # noqa: BLE001 - one bad delta must not stop the loop
                verdicts.append(self._validator_error(item, error))
        merged = self.maybe_merge()
        return {"verdicts": verdicts, "merged": merged}

    def _validator_error(self, item: dict[str, Any], error: BaseException) -> dict[str, Any]:
        """Quarantine with ``validator_error``: re-validated automatically after the next validator version bump."""

        self.log(f"{item['delta_id']}: validator error {error!r}")
        (self.workdir / "inbox" / f"{item['delta_id']}.safetensors").unlink(missing_ok=True)
        self.api.call("POST", f"/api/v1/validator/deltas/{item['delta_id']}/verdict", {
            "decision": "reject", "code": "validator_error", "reason": type(error).__name__,
            "validator_version": self.version})
        return {"delta_id": item["delta_id"], "decision": "reject", "code": "validator_error"}

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
            bounds = norm_bounds(self.state["history"], self.thresholds)
            try:   # anything unparseable in an uploaded file is the file's fault: quarantine, never crash-loop
                stats = check_delta(path, base, {"base_sha256": item["base_sha256"],
                                                 "trainer": self.requirements.get("trainer"),
                                                 "config_sha256": self.requirements.get("config_sha256")},
                                    canary_bounds(bounds) if item["canary"] else bounds)
            except CheckFailed:
                raise
            except Exception as error:  # noqa: BLE001
                raise CheckFailed("malformed", f"the delta file could not be read ({type(error).__name__})") from error
            verdict["checks"] = {"rms": stats["rms"], "max_abs": stats["max_abs"]}
            mine = sketch(path, base)
            relaxed = None
            for other, info in self.state.setdefault("sketches", {}).items():   # accepted, awaiting the merge
                if other == item["delta_id"] or not (self.workdir / "sketches" / f"{other}.sketch.safetensors").is_file():
                    continue
                theirs = dl.load_weights(self.workdir / "sketches" / f"{other}.sketch.safetensors")[0]["sketch"].double()
                if theirs.shape != mine.shape or sketch_cosine(mine, theirs) <= self.thresholds.near_duplicate:
                    continue
                if (item["mode"] == "recipe" and item.get("shard_id") == info.get("shard_id")
                        and item.get("lease_id") and info.get("lease_id") and item["lease_id"] != info["lease_id"]):
                    relaxed = other   # the same shard re-leased after a lease expired: fine, at reduced weight
                    continue
                raise CheckFailed("near_duplicate", f"nearly identical to delta {other} of this round")
            if (self.replayer is not None and item["mode"] == "recipe" and claim.get("recipe")
                    and (item["canary"] or self.rng.random() < self.spot_check_rate)):
                similarity = cosine(path, self.replayer(claim, base), base)
                verdict["checks"]["spot_cosine"] = similarity
                if similarity < self.spot_cosine:
                    raise CheckFailed("spot_check", f"replaying the recipe gives a different update (cosine "
                                                    f"{similarity:.2f} < {self.spot_cosine})")
            candidate_weights(base, path, candidate)
            cand = self.evaluate(candidate)
            base_metrics = self.metrics(item["base_sha256"], base)
            decision, code, message, _ = (judge_canary if item["canary"] else judge)(base_metrics, cand,
                                                                                    self.thresholds)
            if relaxed is not None and decision == "accept":
                decision, code, message = "reduce", "duplicate_lease", f"same shard as {relaxed}, another lease"
            verdict.update(metrics=cand, base_metrics=base_metrics)
        except CheckFailed as failure:
            decision, code, message = "reject", failure.code, failure.message
        finally:
            candidate.unlink(missing_ok=True)
        verdict.update(decision=decision, code=code, reason=message,
                       weight_factor=self.thresholds.reduce_weight if decision == "reduce" else 1.0)
        if decision in ("accept", "reduce"):
            if item["canary"]:   # canaries are not merged: never compare later deltas or norms against them
                path.unlink(missing_ok=True)
            else:   # stored once; later deltas of the round are compared against it
                self.state["history"] = (self.state["history"] + [verdict["checks"]["rms"]])[-200:]
                dl._save(self.workdir / "sketches" / f"{item['delta_id']}.sketch.safetensors", {"sketch": mine.float()},
                         {dl.SCRATCH_STAMP_KEY: json.dumps({"lineage_root": "enntity-scratch-init",
                                                            "warm_start": "none", "kind": "sketch"})})
                self.state["sketches"][item["delta_id"]] = {"shard_id": item.get("shard_id"),
                                                            "lease_id": item.get("lease_id"), "site": item["site"]}
            self._save()
        else:
            path.unlink(missing_ok=True)   # the coordinator keeps the object (quarantine)
        self.api.call("POST", f"/api/v1/validator/deltas/{item['delta_id']}/verdict", verdict)
        self.log(f"{item['delta_id']} ({item['site']}, base v{item['base_version']}"
                 f"{', canary' if item['canary'] else ''}): {decision} {code or ''} {message}")
        return {"delta_id": item["delta_id"], "decision": decision, "code": code}

    def _late(self, delta_id: str, decision: str, code: str, reason: str) -> None:
        self.api.call("POST", f"/api/v1/validator/deltas/{delta_id}/verdict", {
            "decision": decision, "code": code, "reason": reason, "validator_version": self.version})

    def maybe_merge(self, force: bool = False) -> int | None:
        """Merge accepted deltas into the next global.  The inputs (deltas and weights) are saved before merging, so
        a crash and re-run produce the same file; the result must pass the merge gate before it is published."""

        reply = self.hello()
        current, force = reply["current"], bool(reply.get("force_merge"))
        if current is None:
            return None
        version = current["version"]
        plan = self.state.get("pending_merge")
        if plan is not None and plan["version"] != version + 1:
            plan = self.state["pending_merge"] = None   # stale (already published, or the network moved on)
        if plan is None:
            items = self.api.call("GET", "/api/v1/validator/mergeable")
            if not items or len({i["site"] for i in items}) < self.min_sites:
                return None
            oldest = min(item["received"] for item in items)
            ready = len(items) >= self.min_deltas or self.clock() - oldest >= self.round_timeout_s
            if not force and (not ready or self.clock() - self.state.get("last_merge", 0) < self.min_round_s):
                return None
            usable = []
            for item in items:
                path = self.workdir / "inbox" / f"{item['delta_id']}.safetensors"
                if not path.is_file():
                    info = self.api.call("GET", f"/api/v1/validator/deltas/{item['delta_id']}")
                    try:
                        download(info["url"], path, sha256=info["sha256"], size=info["size"], retries=2)
                    except ApiError as error:
                        if error.code not in ("download_hash", "download_size"):
                            raise
                        self._late(item["delta_id"], "reject", "hash_mismatch", "stored object changed after "
                                                                                "acceptance")
                        self._forget_sketches([item])
                        continue
                usable.append(item)
            if not usable or len({i["site"] for i in usable}) < self.min_sites:
                return None
            weights = merge_weights(usable, version, self.thresholds)
            plan = {"version": version + 1, "parent_sha256": current["sha256"],
                    "normalizer": sum(merge_weights(usable, version, self.thresholds, attenuate=False)),
                    "items": [{"delta_id": i["delta_id"], "weight": w, "site": i["site"],
                               "base_version": i["base_version"]} for i, w in zip(usable, weights)]}
            self.state["pending_merge"] = plan
            self._save()   # the inputs are fixed before any output exists
        files = [self.workdir / "inbox" / f"{item['delta_id']}.safetensors" for item in plan["items"]]
        for item, path in zip(plan["items"], files):
            if not path.is_file():   # re-run on a host that lost its inbox: the same bytes, hash-checked
                info = self.api.call("GET", f"/api/v1/validator/deltas/{item['delta_id']}")
                download(info["url"], path, sha256=info["sha256"], size=info["size"])
        master = self.master(version)
        if not master.is_file():   # a new validator host: continue from the published copy
            shutil.copyfile(self.ensure_version(version, current["sha256"]), master)
        momentum_in = self.workdir / "versions" / f"momentum-v{version:06d}.safetensors"
        momentum_out = self.workdir / "versions" / f"momentum-v{version + 1:06d}.safetensors"
        merge_streamed(master, list(zip(files, [i["weight"] for i in plan["items"]])), self.master(version + 1),
                       self.published(version + 1), self.publish_dtype,
                       {"version": version + 1, "parent_sha256": plan["parent_sha256"], "merged": len(files)},
                       normalizer=plan.get("normalizer"),
                       momentum_in=momentum_in if momentum_in.is_file() else None, momentum_out=momentum_out,
                       **self.outer)   # reads only version-keyed inputs: a re-run is idempotent
        published = self.published(version + 1)
        metrics = self.metrics(sha256_file(published), published)
        gate = self._gate(metrics)
        if gate is not None and force:   # the operator released these deltas with --publish (audited)
            self.log(f"merge gate overridden by the operator: {gate}")
        elif gate is not None:   # refuse to publish; hold the deltas for the operator, never discard them
            for item in plan["items"]:
                self._late(item["delta_id"], "hold", "merge_regression", gate)
            for path in (self.master(version + 1), published, momentum_out, *files):
                path.unlink(missing_ok=True)
            self._forget_sketches(plan["items"])
            self.state["pending_merge"] = None
            self._save()
            self.log(f"merge into v{version + 1} refused: {gate}")
            return None
        result = self._publish(version + 1, plan["items"], published, metrics, forced=gate is not None)
        self.state["pending_merge"] = None
        self._forget_sketches(plan["items"])
        self._save()
        for path in files:
            path.unlink(missing_ok=True)
        return result

    def _forget_sketches(self, items: list[dict[str, Any]]) -> None:
        for item in items:
            self.state.get("sketches", {}).pop(item["delta_id"], None)
            (self.workdir / "sketches" / f"{item['delta_id']}.sketch.safetensors").unlink(missing_ok=True)

    def _gate(self, metrics: dict[str, float]) -> str | None:
        """None if a merged global may be published: no group more than max(``gate_drop``, measured eval noise)
        below the best published version (its whole metric vector), and no clear regression against version 0."""

        margin = max(self.thresholds.gate_drop, float(self.state.get("noise", 0.0)))
        gate = Thresholds(hold_drop=margin, reject_drop=margin)
        best = self.state.get("best")
        if best:
            decision, _, message, _ = judge(best["metrics"], metrics, gate)
            if decision != "accept":
                return f"below the best published global v{best['version']} ({message})"
        reference = self.state.get("reference_metrics")
        if reference:
            decision, _, message, _ = judge(reference, metrics, self.thresholds)
            if decision == "reject":
                return f"below the reference global ({message})"
        return None
