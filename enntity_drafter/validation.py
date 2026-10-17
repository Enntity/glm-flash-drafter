"""Delta validation shared by the site client (self-check before upload) and the validator (before merge).

The same functions run on both sides, so a delta that passes locally fails on the validator only for what the
site cannot see: the private held-out set.

* :func:`check_delta`: provenance stamp (guard), header (schema, trainer version, portable config digest, base hash),
  key/shape/dtype layout against the base weights, finite values, RMS and max-abs bounds.  Streams one parameter
  at a time.
* :func:`candidate_weights`: ``base - delta`` (the site's trained weights) written tensor by tensor.
* :func:`judge`: compare held-out metrics of candidate and base; conservative thresholds: accept unless a group's
  expected accepted length (``eal``) drops; a drop beyond ``hold_drop`` is borderline (merged at reduced weight or
  held for the operator), only a drop beyond ``reject_drop`` is a rejection.
* :func:`judge_canary`: the canary's verdict.  A canary (a few steps from a fresh optimizer, never merged) proves the
  pipeline and may legitimately dip, so it gets the tensor sanity checks (:func:`canary_bounds`) and only a
  catastrophic bound: a held-out group below ``canary_floor`` x the base score.  Never the merge thresholds.
* :func:`evaluate`: the trainer's held-out proxy (``train.py --eval-only``) on a weights file.

Bump :data:`VALIDATOR_VERSION` on any change to these rules: the coordinator then re-validates every quarantined or
held delta with the new rules, and sites refresh their canary.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path
from typing import Any, Callable

import torch

from . import diloco as dl
from .guard import ForbiddenCheckpointError, check_path, guarded_safe_open

VALIDATOR_VERSION = "2"   # 2: canaries are judged by judge_canary
ST_DTYPES = {torch.float32: "F32", torch.bfloat16: "BF16", torch.int8: "I8"}
ST_NAMES = {v: k for k, v in ST_DTYPES.items()}


@dataclasses.dataclass
class Thresholds:
    hold_drop: float = 0.01        # relative eal drop on any held-out group that makes a delta borderline
    reject_drop: float = 0.03      # relative eal drop that rejects (clear regression)
    borderline: str = "reduce"     # reduce: merge at reduce_weight; hold: wait for the operator
    reduce_weight: float = 0.5
    max_abs: float = 10.0          # no single delta value may exceed this (weights are O(0.1))
    norm_factor: float = 4.0       # RMS must lie within [median / f^2, median * f] of accepted deltas ...
    history_min: int = 3           # ... once this many were accepted
    stale_decay: float = 0.5       # merge weight factor per version of staleness
    token_factor: float = 2.0      # a site's self-reported tokens count up to this x the median site of the round
    min_token_sites: int = 3       # with fewer sites in a round, every site weighs the same
    near_duplicate: float = 0.995  # cosine above which two deltas of a round count as copies
    gate_drop: float = 0.01        # a merged global may score at most this much below the best published one
    canary_floor: float = 0.5      # a canary is rejected only if a held-out group falls below this x the base score


class CheckFailed(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code, self.message = code, message


def read_header(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise CheckFailed("format", "not a safetensors file")
        (length,) = struct.unpack("<Q", raw)
        if length > 16 * 2**20:
            raise CheckFailed("format", "implausible safetensors header")
        try:
            return json.loads(stream.read(length))
        except ValueError as error:
            raise CheckFailed("format", "unreadable safetensors header") from error


def layout(path: str | Path) -> dict[str, tuple[str, tuple[int, ...]]]:
    return {k: (v["dtype"], tuple(v["shape"])) for k, v in read_header(path).items() if k != "__metadata__"}


def norm_bounds(history: list[float], thresholds: Thresholds) -> dict[str, float]:
    """Bounds published to sites and applied by the validator (``max_abs`` always; RMS once history exists)."""

    bounds = {"max_abs": thresholds.max_abs}
    if len(history) >= thresholds.history_min:
        median = sorted(history)[len(history) // 2]
        bounds.update(max_rms=median * thresholds.norm_factor, min_rms=median / thresholds.norm_factor ** 2)
    return bounds


def check_delta(delta_path: str | Path, base_path: str | Path, expect: dict[str, Any],
                bounds: dict[str, float] | None = None) -> dict[str, Any]:
    """Raise :class:`CheckFailed` (code, message) or return stats.  ``expect``: ``base_sha256`` and optionally
    ``trainer`` and ``config_sha256``."""

    try:
        check_path(delta_path, "drafter")
    except ForbiddenCheckpointError as error:
        raise CheckFailed("provenance", str(error)) from error
    header = read_header(delta_path)
    try:
        meta = json.loads(header.get("__metadata__", {}).get("delta", "{}"))
    except ValueError as error:
        raise CheckFailed("header", "delta header is not JSON") from error
    if meta.get("schema") != dl.DELTA_SCHEMA or meta.get("delta_dtype") not in dl.DELTA_DTYPES:
        raise CheckFailed("header", f"not an {dl.DELTA_SCHEMA} file")
    if (meta.get("base") or {}).get("sha256") != expect["base_sha256"]:
        raise CheckFailed("base_mismatch", "the delta was not trained from the declared base weights")
    for key in ("trainer", "config_sha256"):
        if expect.get(key) is not None and meta.get(key) != expect[key]:
            raise CheckFailed("version_mismatch", f"{key} {meta.get(key)!r} != required {expect[key]!r}")
    base = layout(base_path)
    shapes = {name: shape for name, (_, shape) in base.items()}
    got = {k: v for k, v in layout(delta_path).items()}
    want = {k: (ST_DTYPES[d], s) for k, (d, s) in dl.delta_layout(shapes, meta["delta_dtype"], got).items()}
    if got != want:
        missing = sorted(set(want) - set(got))[:3]
        wrong = sorted(k for k in set(want) & set(got) if want[k] != got[k])[:3]
        raise CheckFailed("layout", f"keys/shapes/dtypes differ from the model (missing {missing}, wrong {wrong}, "
                                    f"extra {sorted(set(got) - set(want))[:3]})")
    squares, numel, peak = 0.0, 0, 0.0
    with guarded_safe_open(delta_path, "drafter") as handle:
        for name, shape in shapes.items():
            keys = dl.delta_keys(name, meta["delta_dtype"], got)
            value = dl.decode_named({k: handle.get_tensor(k) for k in keys}, {name: shape}, meta["delta_dtype"])[name]
            if not bool(torch.isfinite(value).all()):
                raise CheckFailed("non_finite", f"{name} has NaN or Inf")
            squares += float(value.double().square().sum())
            numel += value.numel()
            peak = max(peak, float(value.abs().max()) if value.numel() else 0.0)
    rms = math.sqrt(squares / max(numel, 1))
    bounds = bounds or {}
    if rms == 0.0:
        raise CheckFailed("zero", "the delta is all zeros (no training happened)")
    if peak > bounds.get("max_abs", math.inf):
        raise CheckFailed("norm_high", f"max |delta| {peak:.3g} > {bounds['max_abs']:.3g}")
    if rms > bounds.get("max_rms", math.inf):
        raise CheckFailed("norm_high", f"delta RMS {rms:.3g} > {bounds['max_rms']:.3g} (typical accepted deltas "
                                       "are smaller)")
    if rms < bounds.get("min_rms", 0.0):
        raise CheckFailed("norm_low", f"delta RMS {rms:.3g} < {bounds['min_rms']:.3g} (too little training?)")
    return {"rms": rms, "max_abs": peak, "numel": numel, "meta": meta}


class StreamWriter:
    """Write a safetensors file one tensor at a time (header first: names, dtypes and shapes are known)."""

    def __init__(self, path: str | Path, entries: list[tuple[str, torch.dtype, tuple[int, ...]]],
                 metadata: dict[str, str]):
        self.path, self.entries, self.index = Path(path), entries, 0
        header, offset = {}, 0
        for name, dtype, shape in entries:
            size = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            header[name] = {"dtype": ST_DTYPES[dtype], "shape": list(shape), "data_offsets": [offset, offset + size]}
            offset += size
        header["__metadata__"] = metadata
        raw = json.dumps(header, separators=(",", ":")).encode()
        raw += b" " * (-len(raw) % 8)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = self.path.with_name(self.path.name + ".tmp")
        self.stream = self.temporary.open("wb")
        self.digest = hashlib.sha256()
        self._write(struct.pack("<Q", len(raw)) + raw)

    def _write(self, data: bytes) -> None:
        self.digest.update(data)
        self.stream.write(data)

    def write(self, name: str, tensor: torch.Tensor) -> None:
        want, dtype, shape = self.entries[self.index]
        if name != want:
            raise ValueError(f"expected tensor {want}, got {name}")
        value = tensor.detach().to("cpu", dtype).contiguous().reshape(shape)
        self._write(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        self.index += 1

    def close(self) -> str:
        if self.index != len(self.entries):
            raise ValueError("not every tensor was written")
        self.stream.close()
        os.replace(self.temporary, self.path)
        return self.digest.hexdigest()


def weights_stamp(path: str | Path) -> dict[str, str]:
    """Header metadata to carry over (scratch stamp) when deriving weights from ``path``."""

    meta = read_header(path).get("__metadata__", {})
    return {k: v for k, v in meta.items() if k == dl.SCRATCH_STAMP_KEY}


def candidate_weights(base_path: str | Path, delta_path: str | Path, out_path: str | Path, dtype: str = "fp32"
                      ) -> str:
    """Write ``base - delta`` (fp32 by default); returns its sha256."""

    meta = json.loads(read_header(delta_path)["__metadata__"]["delta"])
    shapes = {name: shape for name, (_, shape) in layout(base_path).items()}
    writer = StreamWriter(out_path, [(n, dl.WEIGHT_DTYPES[dtype], s) for n, s in shapes.items()],
                          {**weights_stamp(base_path), "weights": json.dumps({"schema": dl.WEIGHTS_SCHEMA,
                                                                               "dtype": dtype, "candidate": True})})
    with guarded_safe_open(base_path, "drafter") as base, guarded_safe_open(delta_path, "drafter") as delta:
        for name, shape in shapes.items():
            keys = dl.delta_keys(name, meta["delta_dtype"], set(delta.keys()))
            value = dl.decode_named({k: delta.get_tensor(k) for k in keys}, {name: shape}, meta["delta_dtype"])[name]
            writer.write(name, base.get_tensor(name).float() - value)
    return writer.close()


def metrics_of(record: dict[str, Any], key: str = "eal") -> dict[str, float]:
    """Held-out metric per group from a trainer eval record: the first group is logged unprefixed
    (``heldout``), others as ``NAME/eal``, per-category buckets as ``NAME.VALUE/eal``."""

    out = {}
    for name, value in record.items():
        if name == key:
            out["heldout"] = float(value)
        elif name.endswith("/" + key) and isinstance(value, (int, float)):
            out[name[: -len(key) - 1]] = float(value)
    return out


def _drops(base: dict[str, float], candidate: dict[str, float]
           ) -> tuple[tuple[str, str, str, dict] | None, dict[str, float], str, str]:
    """(early verdict or None, relative drop per group, worst judged group, its detail).  Judged groups are the ones
    without a dot; per-category buckets are reported, not judged (too few rows to decide on).  Fails closed: every
    base group must be scored, and every score must be a finite number."""

    judged = [g for g in base if "." not in g]
    if not judged:
        return ("hold", "no_metrics", "no held-out group to judge on", {}), {}, "", ""
    for g in judged:
        value, reference = candidate.get(g), base[g]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            return ("reject", "heldout_invalid", f"held-out group {g} has no finite score", {}), {}, "", ""
        if not isinstance(reference, (int, float)) or not math.isfinite(reference) or reference <= 0:
            return (("reject", "heldout_invalid", f"base score of held-out group {g} is not a positive number", {}),
                    {}, "", "")
    drops = {g: (base[g] - candidate[g]) / base[g] for g in base
             if isinstance(candidate.get(g), (int, float)) and math.isfinite(candidate[g])
             and isinstance(base[g], (int, float)) and math.isfinite(base[g]) and base[g] > 0}
    worst_group = max(judged, key=drops.get)
    worst = drops[worst_group]
    return None, drops, worst_group, (f"{worst_group}: eal {base[worst_group]:.4f} -> {candidate[worst_group]:.4f} "
                                      f"({-worst:+.2%})")


def judge(base: dict[str, float], candidate: dict[str, float], thresholds: Thresholds
          ) -> tuple[str, str | None, str, dict[str, float]]:
    """(decision accept|reduce|hold|reject, code, message, relative drop per group) for a delta that would be
    merged.  Decided on groups only (:func:`_drops`)."""

    early, drops, worst_group, detail = _drops(base, candidate)
    if early is not None:
        return early
    worst = drops[worst_group]
    if worst > thresholds.reject_drop:
        return "reject", "heldout_regression", f"clear held-out regression, {detail}", drops
    if worst > thresholds.hold_drop:
        decision = "reduce" if thresholds.borderline == "reduce" else "hold"
        return decision, "heldout_borderline", f"borderline held-out change, {detail}", drops
    return "accept", None, f"ok, worst group {detail}", drops


def judge_canary(base: dict[str, float], candidate: dict[str, float], thresholds: Thresholds
                 ) -> tuple[str, str | None, str, dict[str, float]]:
    """:func:`judge` for a canary: accept unless the scores are invalid or a held-out group collapses below
    ``canary_floor`` x its base score.  A canary is never merged, so a dip says nothing about the site."""

    early, drops, worst_group, detail = _drops(base, candidate)
    if early is not None:
        return early
    if 1.0 - drops[worst_group] < thresholds.canary_floor:
        return "reject", "canary_collapse", f"the canary collapsed a held-out group, {detail}", drops
    return "accept", None, f"canary ok (sanity checks only), worst group {detail}", drops


def canary_bounds(bounds: dict[str, float] | None) -> dict[str, float]:
    """The norm bounds a canary must meet: ``max_abs`` and ``max_rms``, not ``min_rms`` (a canary trains a few steps,
    so it is smaller than the round deltas the bounds are measured on; an all-zero delta still fails)."""

    return {k: v for k, v in (bounds or {}).items() if k != "min_rms"}


def evaluate(weights: str | Path, eval_args: list[str], inprocess: bool = True) -> dict[str, Any]:
    """The trainer's held-out eval of ``weights``; ``eval_args`` give config, teacher and data
    (``--features`` or ``--capture-index``/``--capture-root``, ``--eval-groups`` ...)."""

    with tempfile.TemporaryDirectory(prefix="drafter-eval-") as tmp:
        argv = [*eval_args, "--eval-only", "--start-weights", str(weights), "--output", tmp]
        if inprocess:
            from . import train

            return train.main(argv)["eval"]
        import subprocess
        import sys

        done = subprocess.run([sys.executable, "-m", "enntity_drafter.train", *argv], capture_output=True, text=True)
        if done.returncode:
            raise RuntimeError(f"held-out eval failed:\n{done.stderr[-3000:]}")
        return [json.loads(line) for line in (Path(tmp) / "train.jsonl").read_text().splitlines()
                if json.loads(line).get("phase") == "eval"][-1]


EvalFn = Callable[[Path], dict[str, float]]
