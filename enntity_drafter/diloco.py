"""DiLoCo weight synchronization (``train.py --sync-mode diloco``) and the weight/delta file formats shared with the
volunteer coordinator (:mod:`.coordinator`) and site client (:mod:`.join`).

Credit (ideas, not code; implemented here from the papers):

* DiLoCo, Douillard et al. 2023, arXiv:2311.08105: inner AdamW for H steps, then an outer step on the
  pseudo-gradient ``theta_global - theta_worker`` averaged over workers.
* OpenDiLoCo, Jaghouar et al. 2024, arXiv:2407.07852, github.com/PrimeIntellect-ai/OpenDiloco (Apache-2.0): outer
  Nesterov settings (lr 0.7, momentum 0.9, H ~500) and low-precision pseudo-gradients.
* Streaming DiLoCo, Douillard et al. 2025, arXiv:2501.18512: exchanging the model in fragments (here every
  fragment still syncs at the same step).

Every ``H`` inner optimizer steps each worker computes ``delta = theta_global - theta_local`` per fragment (~``fragment_mb``
of fp32), encodes it (fp32 / bf16 / blockwise-absmax int8), and the transport returns every worker's payload in
worker order.  Every worker decodes all of them (its own included) and sums in that order, so the new global weights
are bitwise identical everywhere whatever the transport.  Outer step (PyTorch SGD convention, :func:`outer_update`):
``m = mu*m + mean``; ``u = mean + mu*m`` (Nesterov) or ``m``; ``theta_global -= lr*u``; then every worker resets its
local weights to ``theta_global``.  Inner optimizer state is kept across rounds.

Default outer step: plain averaging (lr 1, momentum 0).  In a 2-site, 3-sync A/B on the full model (H=500), plain
averaging matched or beat synchronous training on held-out groups, while lr 0.7 / Nesterov 0.9 fell well behind;
with few outer rounds, high outer momentum costs the most (docs/volunteer.md).

Transports (``exchange(round, fragment, payload, meta) -> [payload of worker 0, 1, ...]``):

* ``ProcessGroupTransport``: each rank of the existing torch process group is a worker (``all_gather``).
* ``DirectoryTransport``: each *site* (an independent torchrun world, no shared process group) is a worker; the
  site's rank 0 writes ``round-R/fragment-F/siteK.safetensors`` (stamped, atomic rename) into a shared directory,
  reads every site's file of that fragment through the guard, and broadcasts the mean to the other ranks of its site.
* external (``transport=None`` with ``external=True``): the outer step happens elsewhere.  A round starts from
  weights handed in (:meth:`Diloco.set_global`) and ends by writing this worker's delta (:meth:`Diloco.write_delta`);
  the volunteer coordinator validates and merges the deltas of many sites.

Files (safetensors only, never pickle):

* weights: one tensor per trainable parameter name (fp32 or bf16), header ``weights`` (JSON) + scratch stamp;
* delta: ``encode_named``: fp32/bf16 tensors under the parameter name, or int8 ``<name>::q`` + fp32
  ``<name>::scale`` (blockwise absmax), with ``<name>::bf16`` instead for any tensor whose int8 error would exceed
  ``INT8_MAX_REL_ERR`` (heavy-tailed tensors); header ``delta`` (JSON) + scratch stamp.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn.functional as F

from .guard import SCRATCH_STAMP_KEY, guarded_load_file, safetensors_metadata, sha256_file

DELTA_DTYPES = ("fp32", "bf16", "int8")
WEIGHT_DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}
INT8_BLOCK = 1024
INT8_KEYS = ("::q", "::scale")
INT8_FALLBACK = "::bf16"      # an int8 delta file stores a tensor in bf16 when int8 would lose too much of it
INT8_MAX_REL_ERR = 0.01       # relative L2 error above which a tensor falls back to bf16 (heavy-tailed tensors)
DELTA_SCHEMA = "enntity.drafter.delta/v1"
WEIGHTS_SCHEMA = "enntity.drafter.weights/v1"
#: Links the per-sync record prices the exchange on (bytes/s): a slow and a fast internet uplink.
LINKS = {"18MBps": 18e6, "125MBps": 125e6}


def encode(flat: torch.Tensor, dtype: str) -> dict[str, torch.Tensor]:
    """Wire form of an fp32 pseudo-gradient fragment.  int8: absmax scale per ``INT8_BLOCK`` values (fp32),
    round to nearest (deterministic, so a resumed run re-sends the same bytes)."""

    if dtype == "fp32":
        return {"q": flat.float().contiguous()}
    if dtype == "bf16":
        return {"q": flat.to(torch.bfloat16)}
    blocks = F.pad(flat.float(), (0, (-flat.numel()) % INT8_BLOCK)).view(-1, INT8_BLOCK)
    scale = blocks.abs().amax(1) / 127.0
    q = torch.round(blocks / scale.clamp_min(torch.finfo(torch.float32).tiny)[:, None]).clamp_(-127, 127)
    return {"q": q.to(torch.int8).reshape(-1), "scale": scale}


def decode(payload: dict[str, torch.Tensor], numel: int) -> torch.Tensor:
    q = payload["q"]
    if "scale" in payload:
        return (q.view(-1, INT8_BLOCK).float() * payload["scale"][:, None]).reshape(-1)[:numel]
    return q.float()


def payload_bytes(payload: dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in payload.values())


def outer_update(param: torch.Tensor, mean: torch.Tensor, momentum_buf: torch.Tensor | None, *, lr: float,
                 momentum: float, nesterov: bool) -> torch.Tensor:
    """``param -= lr * u`` in place (PyTorch SGD convention; ``mean`` may be overwritten).  Returns ``u``."""

    if momentum:
        momentum_buf.mul_(momentum).add_(mean)
        update = mean.add_(momentum_buf, alpha=momentum) if nesterov else momentum_buf
    else:
        update = mean
    param.add_(update, alpha=-lr)
    return update


# --------------------------------------------------------------------------
# Named weight / delta files
# --------------------------------------------------------------------------

def encode_named(deltas: dict[str, torch.Tensor], dtype: str) -> dict[str, torch.Tensor]:
    """Per-parameter wire tensors (see module docstring)."""

    if dtype not in DELTA_DTYPES:
        raise ValueError(f"delta dtype {dtype!r}")
    out: dict[str, torch.Tensor] = {}
    for name, value in deltas.items():
        flat = value.detach().reshape(-1).float()
        payload = encode(flat, dtype)
        if dtype == "int8":
            norm = float(flat.norm())
            error = float((decode(payload, flat.numel()) - flat).norm()) / norm if norm else 0.0
            if error > INT8_MAX_REL_ERR:
                out[name + INT8_FALLBACK] = flat.to(torch.bfloat16)
            else:
                out[name + INT8_KEYS[0]], out[name + INT8_KEYS[1]] = payload["q"], payload["scale"]
        else:
            out[name] = payload["q"].view(value.shape)
    return out


def delta_keys(name: str, dtype: str, available) -> list[str]:
    """The file keys holding parameter ``name`` (int8 files may store some tensors in bf16)."""

    if dtype != "int8":
        return [name]
    return [name + INT8_FALLBACK] if name + INT8_FALLBACK in available else [name + k for k in INT8_KEYS]


def delta_layout(shapes: dict[str, tuple[int, ...]], dtype: str, available=()) -> dict[str, tuple[torch.dtype,
                                                                                               tuple[int, ...]]]:
    """Expected (dtype, shape) of every key of a delta file of these parameters (``available``: the file's keys,
    which say which int8 tensors fell back to bf16)."""

    if dtype == "int8":
        layout = {}
        for name, shape in shapes.items():
            if name + INT8_FALLBACK in available:
                layout[name + INT8_FALLBACK] = (torch.bfloat16, (math.prod(shape),))
                continue
            blocks = -(-math.prod(shape) // INT8_BLOCK)
            layout[name + INT8_KEYS[0]] = (torch.int8, (blocks * INT8_BLOCK,))
            layout[name + INT8_KEYS[1]] = (torch.float32, (blocks,))
        return layout
    return {name: (WEIGHT_DTYPES[dtype], tuple(shape)) for name, shape in shapes.items()}


def decode_named(tensors: dict[str, torch.Tensor], shapes: dict[str, tuple[int, ...]], dtype: str
                 ) -> dict[str, torch.Tensor]:
    """fp32 deltas from a delta file's tensors; ``ValueError`` on any missing, extra or mis-typed key."""

    layout = delta_layout(shapes, dtype, tensors)
    if set(tensors) != set(layout):
        missing, extra = sorted(set(layout) - set(tensors)), sorted(set(tensors) - set(layout))
        raise ValueError(f"delta keys differ from the model (missing {missing[:3]}, extra {extra[:3]})")
    for key, (want_dtype, want_shape) in layout.items():
        if tensors[key].dtype != want_dtype or tuple(tensors[key].shape) != want_shape:
            raise ValueError(f"delta {key}: {tensors[key].dtype} {tuple(tensors[key].shape)} "
                             f"!= {want_dtype} {want_shape}")
    out = {}
    for name, shape in shapes.items():
        if dtype == "int8" and name + INT8_FALLBACK in tensors:
            out[name] = tensors[name + INT8_FALLBACK].float().view(shape)
        elif dtype == "int8":
            payload = {"q": tensors[name + INT8_KEYS[0]], "scale": tensors[name + INT8_KEYS[1]]}
            out[name] = decode(payload, math.prod(shape)).view(shape)
        else:
            out[name] = tensors[name].float()
    return out


def _save(path: Path, tensors: dict[str, torch.Tensor], metadata: dict[str, str]) -> None:
    """Write a safetensors file whose bytes depend only on the tensors and metadata (sorted names and keys, written
    by ``validation.StreamWriter``, atomically).  safetensors' own writer orders metadata keys at random, so writing
    the same weights twice gave two SHA-256s (a rerun of ``validator init`` then started a second upload)."""

    from .validation import StreamWriter

    items = {k: tensors[k].detach().cpu().contiguous() for k in sorted(tensors)}
    writer = StreamWriter(Path(path), [(k, v.dtype, tuple(v.shape)) for k, v in items.items()],
                          dict(sorted(metadata.items())))
    for name, value in items.items():
        writer.write(name, value)
    writer.close()


def save_weights(path: str | Path, tensors: dict[str, torch.Tensor], dtype: str, stamp: dict[str, Any],
                 meta: dict[str, Any]) -> str:
    """Write a weights file (``dtype`` fp32 or bf16); returns its sha256."""

    cast = WEIGHT_DTYPES[dtype]
    _save(Path(path), {k: v.to(cast) for k, v in tensors.items()},
          {SCRATCH_STAMP_KEY: json.dumps(stamp), "weights": json.dumps({"schema": WEIGHTS_SCHEMA, "dtype": dtype,
                                                                        **meta})})
    return sha256_file(Path(path))


def load_weights(path: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Tensors (fp32) and header of a weights file, through the guard (scratch stamp required)."""

    tensors = guarded_load_file(path, "drafter")
    meta = json.loads(safetensors_metadata(Path(path)).get("weights", "{}"))
    return {k: v.float() for k, v in tensors.items()}, meta


def config_sha256(cfg: dict[str, Any]) -> str:
    """Portable architecture digest (sites pin it): the config without local fields (target path, comments)."""

    portable = {k: v for k, v in cfg.items() if k != "target_name" and not k.startswith("_")}
    return hashlib.sha256(json.dumps(portable, sort_keys=True).encode()).hexdigest()


def file_metadata(path: str | Path, key: str) -> dict[str, Any]:
    """One JSON entry of a safetensors header (``{}`` when absent)."""

    return json.loads(safetensors_metadata(Path(path)).get(key, "{}"))


# --------------------------------------------------------------------------
# Transports
# --------------------------------------------------------------------------

class ProcessGroupTransport:
    """Workers = the ranks of the default process group."""

    def __init__(self, world: int):
        self.world = world

    def exchange(self, round_: int, fragment: int, payload: dict[str, torch.Tensor],
                 meta: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        if self.world == 1:
            return [payload]
        gathered: list[dict[str, torch.Tensor]] = [{} for _ in range(self.world)]
        for key in sorted(payload):
            parts = [torch.empty_like(payload[key]) for _ in range(self.world)]
            dist.all_gather(parts, payload[key].contiguous())
            for worker, part in enumerate(parts):
                gathered[worker][key] = part
        return gathered


class DirectoryTransport:
    """Workers = sites sharing ``root`` (a shared or synced directory); no process group between sites.

    A file appears only complete (``os.replace``).  A site that sees every peer's file of round R knows every
    peer has finished reading round R-1, so it then deletes its own round R-1 file of that fragment."""

    def __init__(self, root: str | Path, site: int, sites: int, stamp: dict[str, Any],
                 timeout_s: float = 6 * 3600, poll_s: float = 0.5):
        self.root, self.site, self.sites, self.stamp = Path(root), site, sites, stamp
        self.timeout_s, self.poll_s = timeout_s, poll_s

    def path(self, round_: int, fragment: int, site: int) -> Path:
        return self.root / f"round-{round_:06d}" / f"fragment-{fragment:04d}" / f"site{site}.safetensors"

    def exchange(self, round_: int, fragment: int, payload: dict[str, torch.Tensor],
                 meta: dict[str, Any]) -> list[dict[str, torch.Tensor]]:
        device = next(iter(payload.values())).device
        own = self.path(round_, fragment, self.site)
        header = {**meta, "round": round_, "fragment": fragment, "site": self.site}
        _save(own, payload, {SCRATCH_STAMP_KEY: json.dumps(self.stamp), "diloco": json.dumps(header)})
        deadline = time.monotonic() + self.timeout_s
        gathered = []
        for site in range(self.sites):
            peer = self.path(round_, fragment, site)
            while not peer.is_file():
                if time.monotonic() > deadline:
                    raise TimeoutError(f"diloco: no {peer} after {self.timeout_s:.0f} s")
                time.sleep(self.poll_s)
            info = file_metadata(peer, "diloco")
            if (info["round"], info["fragment"], info["site"]) != (round_, fragment, site) or any(
                    info.get(k) != v for k, v in meta.items()):
                raise RuntimeError(f"diloco: {peer} does not match this site ({info} vs {meta})")
            gathered.append({k: v.to(device) for k, v in guarded_load_file(peer, "drafter").items()})
        if round_:
            stale = self.path(round_ - 1, fragment, self.site)
            stale.unlink(missing_ok=True)
            for directory in (stale.parent, stale.parent.parent):
                with contextlib.suppress(OSError):
                    directory.rmdir()
        return gathered


# --------------------------------------------------------------------------
# Outer loop state
# --------------------------------------------------------------------------

class Diloco:
    """Global weights + outer momentum over ``named`` (the tensors the inner optimizer owns, fp32).

    ``transport`` is None on the non-leader ranks of a multi-rank site (they receive the mean by broadcast);
    ``site_world`` > 1 enables that broadcast.  ``external``: no outer step here (see module docstring)."""

    def __init__(self, named: list[tuple[str, torch.Tensor]], transport, *, lr: float, momentum: float,
                 nesterov: bool, delta_dtype: str, fragment_mb: int = 512, site_world: int = 1,
                 meta: dict[str, Any] | None = None, external: bool = False):
        if delta_dtype not in DELTA_DTYPES:
            raise ValueError(f"delta dtype {delta_dtype!r}")
        self.named, self.transport, self.site_world, self.external = named, transport, site_world, external
        self.lr, self.momentum, self.nesterov, self.delta_dtype = lr, momentum, nesterov, delta_dtype
        if external:
            self.momentum = 0.0   # the coordinator owns the outer optimizer
        self.meta = dict(meta or {})
        self.round = 0
        self.base: dict[str, Any] = {}   # external: which weights this round started from (sha256, ...)
        limit = fragment_mb * 2**20 // 4
        self.fragments: list[list[int]] = [[]]
        size = 0
        for index, (_, tensor) in enumerate(named):
            if size and size + tensor.numel() > limit:
                self.fragments.append([])
                size = 0
            self.fragments[-1].append(index)
            size += tensor.numel()
        with torch.no_grad():
            self.global_flat = [torch.cat([named[i][1].detach().float().reshape(-1) for i in fragment])
                                for fragment in self.fragments]
        self.momentum_flat = [torch.zeros_like(g) for g in self.global_flat] if self.momentum else []
        self.views = self._views(self.global_flat)
        self.momentum_views = self._views(self.momentum_flat) if self.momentum else {}

    def _views(self, flats: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """Per-parameter views into per-fragment flat buffers."""

        views = {}
        for fragment, flat in zip(self.fragments, flats):
            offset = 0
            for i in fragment:
                name, tensor = self.named[i]
                views[name] = flat[offset: offset + tensor.numel()].view_as(tensor)
                offset += tensor.numel()
        return views

    def global_named(self) -> list[tuple[str, torch.Tensor]]:
        return [(name, self.views[name]) for name, _ in self.named]

    @torch.no_grad()
    def outer_step(self, step: int) -> dict[str, Any]:
        """One synchronization round; local weights end equal to the new global weights."""

        if self.external:
            raise RuntimeError("external DiLoCo: write_delta, the coordinator does the outer step")
        started = time.monotonic()
        sent = workers = 0
        exchange_s = 0.0
        sums = dict.fromkeys(("delta", "pseudo_grad", "update", "int8_err"), 0.0)
        for f, fragment in enumerate(self.fragments):
            flat = self.global_flat[f]
            local = [self.named[i][1] for i in fragment]
            delta = flat - torch.cat([t.detach().reshape(-1) for t in local])
            sums["delta"] += float(delta.square().sum())
            if self.transport is not None:
                payload = encode(delta, self.delta_dtype)
                if self.delta_dtype != "int8":   # what int8 would lose on this delta (diagnostic)
                    sums["int8_err"] += float((decode(encode(delta, "int8"), delta.numel()) - delta).square().sum())
                sent += payload_bytes(payload)
                exchanged = time.monotonic()
                gathered = self.transport.exchange(self.round, f, payload, self.meta)
                exchange_s += time.monotonic() - exchanged
                workers = len(gathered)
                mean = torch.zeros_like(flat)
                for part in gathered:
                    mean.add_(decode(part, flat.numel()))
                mean.div_(workers)
                del payload, gathered
            else:
                mean = torch.empty_like(flat)
            if self.site_world > 1:
                dist.broadcast(mean, 0)
            del delta
            sums["pseudo_grad"] += float(mean.square().sum())
            update = outer_update(flat, mean, self.momentum_flat[f] if self.momentum else None, lr=self.lr,
                                  momentum=self.momentum, nesterov=self.nesterov)
            sums["update"] += float(update.square().sum())
            del mean, update
            for i in fragment:
                name, tensor = self.named[i]
                tensor.copy_(self.views[name])
        self.round += 1
        numel = sum(g.numel() for g in self.global_flat)
        wire = sent * max(workers - 1, 0)   # bytes each worker sends and receives (full duplex, all-to-all)
        record = {"round": self.round, "step": step, "workers": workers, "delta_dtype": self.delta_dtype,
                  "fragments": len(self.fragments), "payload_bytes": sent, "wire_bytes_each_way": wire,
                  **{f"est_s_at_{k}": round(wire / v, 1) for k, v in LINKS.items()},
                  "exchange_s": round(exchange_s, 3), "outer_s": round(time.monotonic() - started, 3),
                  "local_delta_rms": (sums["delta"] / numel) ** 0.5,
                  "pseudo_grad_rms": (sums["pseudo_grad"] / numel) ** 0.5,
                  "outer_update_rms": (sums["update"] / numel) ** 0.5}
        if self.delta_dtype != "int8" and sums["delta"]:
            record["int8_rel_err"] = (sums["int8_err"] / sums["delta"]) ** 0.5
        return record

    @torch.no_grad()
    def set_global(self, tensors: dict[str, torch.Tensor], base: dict[str, Any]) -> None:
        """External round start: global and local weights become ``tensors`` (all parameters required)."""

        if set(tensors) != set(self.views):
            raise RuntimeError("start weights: parameter set differs from the model")
        for name, tensor in self.named:
            if tuple(tensors[name].shape) != tuple(tensor.shape):
                raise RuntimeError(f"start weights: shape mismatch for {name}")
            self.views[name].copy_(tensors[name])
            tensor.copy_(self.views[name])
        self.base = dict(base)

    @torch.no_grad()
    def write_delta(self, path: str | Path, stamp: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
        """External round end: write ``global - local`` (``delta_dtype``) with ``meta`` in the header."""

        deltas = {name: self.views[name] - tensor.detach().float() for name, tensor in self.named}
        squares = sum(float(d.square().sum()) for d in deltas.values())
        numel = sum(d.numel() for d in deltas.values())
        tensors = encode_named(deltas, self.delta_dtype)
        record = {"schema": DELTA_SCHEMA, "delta_dtype": self.delta_dtype, "base": self.base,
                  "delta_rms": (squares / numel) ** 0.5, "payload_bytes": payload_bytes(tensors), **meta}
        _save(Path(path), tensors, {SCRATCH_STAMP_KEY: json.dumps(stamp), "delta": json.dumps(record)})
        self.round += 1
        return record

    @contextlib.contextmanager
    def global_weights(self, sync: Callable[[], None] | None = None):
        """Local tensors temporarily hold the global weights (evaluation); restored exactly afterwards."""

        with torch.no_grad():
            saved = [t.detach().clone() for _, t in self.named]
            for name, tensor in self.named:
                tensor.copy_(self.views[name])
        if sync:
            sync()
        try:
            yield
        finally:
            with torch.no_grad():
                for (_, tensor), value in zip(self.named, saved):
                    tensor.copy_(value)
            del saved
            if sync:
                sync()

    def state_dict(self) -> dict[str, Any]:
        def cpu(views: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            return {name: views[name].detach().cpu().clone() for name, _ in self.named}

        return {"round": self.round, "global": cpu(self.views), "base": dict(self.base),
                "momentum": cpu(self.momentum_views) if self.momentum else {},
                "config": {"lr": self.lr, "momentum": self.momentum, "nesterov": self.nesterov,
                           "delta_dtype": self.delta_dtype, "fragments": len(self.fragments),
                           "external": self.external}}

    @torch.no_grad()
    def load_state_dict(self, state: dict[str, Any]) -> None:
        if set(state["global"]) != set(self.views):
            raise RuntimeError("diloco resume: parameter set differs")
        if bool(state.get("config", {}).get("external")) != self.external:
            raise RuntimeError("diloco resume: the checkpoint's transport (external or not) differs")
        for name, value in state["global"].items():
            self.views[name].copy_(value)
        if self.momentum:
            if not state["momentum"]:
                raise RuntimeError("diloco resume: checkpoint has no outer momentum")
            for name, value in state["momentum"].items():
                self.momentum_views[name].copy_(value)
        self.round = int(state["round"])
        self.base = dict(state.get("base") or {})
