"""From-scratch data-parallel trainer for an openly trained DFlash2-style GLM-5.3-Flash drafter.

Model: Speculators ``DFlash2DraftModel`` with a fresh deterministic init
(:func:`.model.scratch_init`); the chain proxy (``eal``) is evaluated on fixed
anchors (:func:`.model.install_fixed_anchor_mask`).

Parallelism: the 1.17B body is *replicated* on every rank (fp32 master,
optimizer and gradients fit one GB10).  Gradients accumulate locally in fp32
over ``--grad-accum`` micro-steps; only the last micro-step's backward
all-reduces them, in bf16 (``--grad-reduce-dtype``) buckets launched as backward
produces them (``GradSync``).  FSDP2 sharding moved ~7 GB per rank per micro-step
over a two-node 200G fabric and left the GPUs waiting on NCCL; this moves
~2.3 GB per optimizer step.  Held-out evals split the rows across ranks and sum
the metric totals.

* no drafter tensor is read from anywhere except this trainer's own stamped
  checkpoints (:mod:`.guard`); no QAT, no MXFP8 metadata file;
* AdamW with warmup + cosine;
* objective switch (upstream / selector_prefix / selector_inset) with an
  optional late switch step;
* rank-local resume (params + optimizer state + step); deterministic per-step
  row and anchor schedule so a resumed run replays the same stream;
* CPU/gloo and CUDA/NCCL; ``torchrun --nproc_per_node=2`` locally or
  ``--nnodes=2`` across nodes; single process also works;
* ``--sync-mode diloco`` (:mod:`.diloco`): workers take H independent inner
  steps and exchange only weights, over the process group, a shared directory,
  or (``--diloco-transport external``) a volunteer coordinator that validates
  and merges every site's delta (docs/volunteer.md).

See docs/training.md.  CPU tests: ``python -m pytest tests``.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gc
import hashlib
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Callable

from . import model as sm  # first: sets the torch.compile / HF offline defaults

import torch
import torch.distributed as dist
from torch import nn

from . import capture_io as cd
from . import diloco as dl
from . import fastpath as fp
from .data import RowStore, eligible_anchors
from .export import export_engine_checkpoint, verify_export
from .guard import guarded_torch_load, scratch_stamp, sha256_file
from .model import install_fixed_anchor_mask
from .objectives import OBJECTIVES, loss_kwargs, training_loss

TRAINER_VERSION = "scratch-v1"   # provenance field; unchanged so existing checkpoints stay loadable
#: ``--cpu-autocast`` (tests): bf16 autocast on CPU too, to emulate the CUDA numerics.
CPU_AUTOCAST = False


# --------------------------------------------------------------------------
# Distributed / sharding
# --------------------------------------------------------------------------

def init_distributed(device_type: str) -> tuple[int, int, torch.device]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if device_type == "cuda":
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:   # cpu, or mps (Apple GPU: evaluation only, e.g. a validator on a Mac; fp32, no autocast)
        device = torch.device(device_type)
    if world > 1 and not dist.is_initialized():
        dist.init_process_group("nccl" if device_type == "cuda" else "gloo",
                                timeout=dt.timedelta(minutes=30))
    return rank, world, device


def barrier(world: int) -> None:
    if world > 1:
        dist.barrier()


class GradSync:
    """Data-parallel gradient averaging for a replicated model.

    Fixed buckets (reverse registration order, ~``bucket_mb`` on the wire each)
    are all-reduced in ``dtype`` when ``arm()`` was called before a backward: a
    post-accumulate hook counts each bucket's ready parameters and launches the
    ready buckets strictly in bucket order (identical on every rank), so the
    communication overlaps the rest of that backward.  ``finish()`` launches any
    bucket still waiting (unused parameters), waits, and writes the rank mean
    back into the fp32 ``.grad``.  Non-armed backwards just accumulate locally.
    ``owner`` maps a hooked parameter to the tensor holding its accumulated
    gradient (the fp32 master under ``--param-dtype bf16``; itself otherwise).
    """

    def __init__(self, params: list[nn.Parameter], world: int, dtype: torch.dtype, bucket_mb: int = 128,
                 owner=None):
        self.world, self.dtype = world, dtype
        self.owner = owner or (lambda parameter: parameter)
        limit = bucket_mb * 2**20 // torch.finfo(dtype).bits * 8
        self.buckets: list[list[nn.Parameter]] = [[]]
        size = 0
        for parameter in reversed(params):
            if size >= limit:
                self.buckets.append([])
                size = 0
            self.buckets[-1].append(parameter)
            size += parameter.numel()
        self.bucket_of = {id(p): index for index, bucket in enumerate(self.buckets) for p in bucket}
        self.armed = False
        self.waiting: list[int] = []
        self.next = 0
        self.inflight: list[tuple[Any, torch.Tensor, list[nn.Parameter]]] = []
        for parameter in params:
            parameter.register_post_accumulate_grad_hook(self._ready)

    def arm(self) -> None:
        self.armed, self.next, self.inflight = True, 0, []
        self.waiting = [len(bucket) for bucket in self.buckets]

    def _ready(self, parameter: nn.Parameter) -> None:
        if not self.armed:
            return
        self.waiting[self.bucket_of[id(parameter)]] -= 1
        while self.next < len(self.buckets) and self.waiting[self.next] == 0:
            self._launch()

    def _launch(self) -> None:
        params = self.buckets[self.next]
        self.next += 1
        flat = torch.empty(sum(p.numel() for p in params), dtype=self.dtype, device=params[0].device)
        offset = 0
        for parameter in params:
            view = flat[offset: offset + parameter.numel()]
            grad = self.owner(parameter).grad
            view.copy_(grad.reshape(-1)) if grad is not None else view.zero_()
            offset += parameter.numel()
        self.inflight.append((dist.all_reduce(flat, async_op=True), flat, params))

    def finish(self) -> None:
        while self.next < len(self.buckets):
            self._launch()
        for work, flat, params in self.inflight:
            work.wait()
            offset = 0
            for parameter in params:
                target = self.owner(parameter)
                if target.grad is None:
                    target.grad = torch.zeros_like(target)
                target.grad.copy_(flat[offset: offset + parameter.numel()].view_as(target))
                target.grad.div_(self.world)
                offset += parameter.numel()
        self.armed, self.inflight = False, []


def flop_parts(cfg: dict[str, Any], queries: int, context: int, attended: int | None = None) -> dict[str, float]:
    """Matmul FLOPs of one training micro-step (forward + backward).

    Body over the ``queries`` block tokens (q/o/MLP/conv projections, selector projection),
    k/v projections over context + block keys, QK^T and PV, ``fc`` over the context rows
    (forward + weight gradient), and the vocab heads (draft forward + input gradient,
    teacher forward).  ``total`` counts dense attention (every query x every key, as
    round 1 computed it); ``useful`` counts only the ``attended`` (query, key) pairs the
    anchor-block mask allows (``fastpath.attended_keys``; = dense when unknown)."""

    h, inter, layers = cfg["hidden_size"], cfg["intermediate_size"], cfg["num_hidden_layers"]
    q_dim, kv_dim = cfg["num_attention_heads"] * cfg["head_dim"], cfg["num_key_value_heads"] * cfg["head_dim"]
    conv_out = 2 * cfg["conv_kernel_size"] * h // cfg["conv_group_size"]
    keys = context + queries
    per_layer = queries * h * (2 * q_dim + 3 * inter + 2 * conv_out) + keys * h * 2 * kv_dim + 2 * queries * keys * q_dim
    body = 2 * (layers * per_layer + queries * h * cfg["selector_rank"])
    fc = 2 * context * len(cfg["target_layer_ids"]) * cfg["target_hidden_size"] * h
    head = 2 * queries * h * cfg["vocab_size"]
    total = 3.0 * body + 2.0 * fc + 3.0 * head
    dense = 3.0 * 2 * layers * 2 * queries * keys * q_dim
    useful = 3.0 * 2 * layers * 2 * (attended if attended is not None else queries * keys) * q_dim
    return {"total": total, "useful": total - dense + useful, "gemm": total - dense,
            "attention_dense": dense, "attention_useful": useful}


def micro_flops(cfg: dict[str, Any], queries: int, context: int) -> float:
    """Round-1 MFU numerator (dense attention as computed)."""

    return flop_parts(cfg, queries, context)["total"]


def gpu_watts(device: torch.device) -> float:
    try:
        return torch.cuda.power_draw(device) / 1000.0 if device.type == "cuda" else 0.0
    except Exception:  # noqa: BLE001 - NVML missing: just not reported
        return 0.0


def fixed_anchors(row: dict[str, Any], anchors: dict[str, list[int]], count: int, seed: int) -> list[int]:
    """``count`` of the row's eligible anchors, drawn deterministically from ``seed``."""

    values = list(anchors[row["id"]])
    if len(values) < count:
        raise AssertionError((row["id"], len(values), count))
    return random.Random(seed).sample(values, count)


def scalar_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Sum of every tensor metric, as Python floats."""

    return {key: float(value.detach().float().sum().cpu()) for key, value in metrics.items()
            if isinstance(value, torch.Tensor)}


def trainable(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def rank_consensus(payload: dict[str, Any], world: int) -> None:
    """All ranks must have identical config, init and data manifest (replicas start equal)."""

    if world == 1:
        return
    gathered: list[Any] = [None] * world
    dist.all_gather_object(gathered, payload)
    if any(item != gathered[0] for item in gathered):
        raise RuntimeError(f"rank consensus mismatch: {[g.get('digest') for g in gathered]}")


# --------------------------------------------------------------------------
# Optimizer and schedule
# --------------------------------------------------------------------------

def build_optimizer(named: list[tuple[str, torch.Tensor]], args: argparse.Namespace) -> torch.optim.Optimizer:
    """``named``: the tensors the optimizer owns (fp32 masters under ``--param-dtype bf16``)."""

    decay, no_decay = [], []
    for name, parameter in named:
        # 2-D Linear weights decay; norms, conv base kernels and the
        # embedding-like selector codebooks do not.
        (decay if name.endswith(".weight") and parameter.ndim == 2 and "codebook" not in name
         else no_decay).append(parameter)
    groups = [{"params": decay, "weight_decay": args.weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    impl = {"fused": True} if args.adamw_impl == "fused" else {"foreach": True}
    return torch.optim.AdamW(groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=1e-8, **impl)


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup then cosine to ``min_lr_ratio * lr`` (step is 0-based)."""

    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    progress = min(1.0, (step - args.warmup) / max(1, args.steps - args.warmup))
    floor = args.lr * args.min_lr_ratio
    return floor + 0.5 * (args.lr - floor) * (1 + math.cos(math.pi * progress))


# --------------------------------------------------------------------------
# Deterministic data schedule
# --------------------------------------------------------------------------

def scheduled_row(rows: list[dict[str, Any]], global_index: int, seed: int) -> dict[str, Any]:
    epoch, offset = divmod(global_index, len(rows))
    order = list(range(len(rows)))
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    return rows[order[offset]]


def to_device(inputs: dict[str, torch.Tensor], device: torch.device,
              dtype: torch.dtype | None = None) -> dict[str, torch.Tensor]:
    """Move a sample to the device (FP8 features are dequantized there, ``cd.materialize_fp8``); floating
    features optionally cast to ``dtype``."""

    if any(key in inputs for key, _, _ in cd.FP8_INPUTS):
        inputs = cd.materialize_fp8({k: v.to(device, non_blocking=True) for k, v in inputs.items()})
    return {k: v.to(device, dtype if dtype is not None and v.is_floating_point() else None, non_blocking=True)
            for k, v in inputs.items()}


@contextlib.contextmanager
def fixed_anchors_installed(model: nn.Module, anchors: list[int] | None):
    """Use the sample's own anchors (capture samples) instead of ``select_anchors``."""

    if anchors is None:
        yield
        return
    old = install_fixed_anchor_mask(model, anchors)
    try:
        yield
    finally:
        model._build_attention_mask = old


def autocast(device: torch.device):
    if device.type == "cuda" or CPU_AUTOCAST:
        return torch.autocast(device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------
# Smoke / memory safety (GB10 unified memory: exhaustion hangs the host)
# --------------------------------------------------------------------------

def mem_available_gb() -> float:
    """Host MemAvailable (GB10: CPU and GPU share it).  Tests may fake it."""

    fake = os.environ.get("DRAFTER_FAKE_MEMAVAIL_GB")
    if fake:
        return float(fake)
    try:
        with open("/proc/meminfo") as stream:
            for line in stream:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024**2
    except OSError:
        pass
    return float("inf")


def peak_gb(device: torch.device) -> float:
    return torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0


def memory_consensus(device: torch.device, world: int) -> tuple[float, float]:
    """(min MemAvailable GB, max peak allocated GB) over ranks; one tiny collective."""

    values = torch.tensor([-min(mem_available_gb(), 1e9), peak_gb(device)], dtype=torch.float64, device=device)
    if world > 1:
        dist.all_reduce(values, op=dist.ReduceOp.MAX)
    return -float(values[0]), float(values[1])


def abort_run(emit, record: dict[str, Any], world: int, code: int = 3) -> None:
    """Every rank reached the same decision: stop without saving (saving copies
    shards to host memory, the resource that is short); the last periodic
    checkpoint is the resume point."""

    emit({"phase": "abort", **record})
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    sys.stdout.flush()
    raise SystemExit(code)


def worst_case_probe(model: nn.Module, args: argparse.Namespace, cfg: dict[str, Any], device: torch.device,
                     dtype: torch.dtype, objective: str, kwargs: dict[str, Any],
                     selector_prefix: dict[str, Any]) -> dict[str, float]:
    """One forward/backward at the largest sample the stream can produce
    (``crop + window + block`` context rows, ``--anchors`` anchors), identical
    on every rank (synthetic tensors, no data needed); gradients discarded."""

    window, block = cfg["sliding_window"], cfg["block_size"]
    span = args.crop or 4096
    rows = span + window + block
    generator = torch.Generator().manual_seed(args.seed)
    anchors = torch.linspace(window, window + span - 1, args.anchors).long().unique().tolist()
    inputs = {
        "hidden_states": torch.randn(1, rows, len(cfg["target_layer_ids"]) * cfg["target_hidden_size"],
                                     generator=generator),
        "input_ids": torch.randint(0, cfg["vocab_size"] - 1, (1, rows), generator=generator),
        "loss_mask": (torch.arange(rows) >= window).unsqueeze(0),
        "verifier_last_hidden_states": torch.randn(1, rows, cfg["target_hidden_size"], generator=generator),
        "document_ids": torch.zeros(1, rows, dtype=torch.long),
        "position_ids": torch.arange(rows).unsqueeze(0),
    }
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.monotonic()
    with fixed_anchors_installed(model, anchors), autocast(device):
        loss, _ = training_loss(objective, model, to_device(inputs, device, dtype), max_anchors=len(anchors),
                                on_policy=True, kwargs=kwargs, selector_prefix=selector_prefix)
    loss.backward()
    for _, parameter in trainable(model):
        parameter.grad = None
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return {"probe_rows": rows, "probe_anchors": len(anchors), "probe_s": round(time.monotonic() - started, 3)}


MIN_MICRO_ANCHORS = 8   # smallest micro-batch --global-anchors may pick


def batch_candidates(anchors: int, grad_accum: int, global_anchors: int | None, ranks: int
                     ) -> list[tuple[int, int]]:
    """(micro-batch anchors, grad_accum) pairs to try, the largest micro-batch first.  Without ``global_anchors``
    the given pair; with it, every split of ``global_anchors / ranks`` into micro x accum with a micro-batch of
    ``MIN_MICRO_ANCHORS``..``anchors`` (``anchors`` is an upper bound, never forced)."""

    if not global_anchors:
        return [(anchors, grad_accum)]
    per_rank, left = divmod(global_anchors, ranks)
    pairs = [(per_rank // accum, accum) for accum in range(1, per_rank + 1)
             if not left and per_rank % accum == 0 and MIN_MICRO_ANCHORS <= per_rank // accum <= anchors]
    if not pairs:
        raise SystemExit(f"--global-anchors {global_anchors} over {ranks} rank(s) has no micro-batch of "
                         f"{MIN_MICRO_ANCHORS}..{anchors} anchors that divides it")
    return pairs


def choose_batch(candidates: list[tuple[int, int]], probe: Callable[[int], tuple[bool, dict[str, Any]]] | None
                 ) -> tuple[tuple[int, int] | None, list[dict[str, Any]]]:
    """The first (largest) candidate whose probe fits, and every probe record; ``probe`` None: the first, untried.
    (None, records) when nothing fits."""

    if probe is None:
        return candidates[0], []
    records = []
    for anchors, accum in candidates:
        fits, record = probe(anchors)
        records.append({**record, "fits": fits})
        if fits:
            return (anchors, accum), records
    return None, records


def select_batch(args: argparse.Namespace, resumed: dict[str, Any], model: nn.Module, cfg: dict[str, Any],
                 device: torch.device, dtype: torch.dtype, kwargs: dict[str, Any], selector_prefix: dict[str, Any],
                 optimizer: torch.optim.Optimizer, world: int, emit, step: int) -> None:
    """Set ``args.anchors`` and ``args.grad_accum``: :func:`batch_candidates`, each checked by a worst-case probe
    (``--probe``, run at every start and resume: memory shared with other services changes) that must leave
    ``--min-free-gb`` + ``--probe-headroom-gb`` free.  A resumed run first tries its checkpoint's micro-batch (the
    stream then replays exactly), smaller ones only if that no longer fits."""

    candidates = batch_candidates(args.anchors, args.grad_accum, args.global_anchors, world)
    if resumed.get("anchors") and args.global_anchors:
        candidates = [c for c in candidates if c[0] <= int(resumed["anchors"])] or candidates[-1:]

    def probe(anchors: int) -> tuple[bool, dict[str, Any]]:
        args.anchors = anchors
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()   # a larger probe's cached blocks would count against this one
        try:
            record, oom = worst_case_probe(model, args, cfg, device, dtype, args.objective, kwargs, selector_prefix), 0
        except torch.OutOfMemoryError as error:
            record, oom = {"probe_error": str(error)[:300]}, 1
        optimizer.zero_grad(set_to_none=True)  # fp32 master gradients of the probe
        if oom and device.type == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
        free, peak = memory_consensus(device, world)
        flag = torch.tensor([float(oom)], device=device)
        if world > 1:   # every rank takes the same decision
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        # AdamW state (2 fp32 tensors per parameter) is allocated at the first
        # optimizer step, after the probe: count it against the margin now.
        pending = 0.0 if optimizer.state else 2 * sum(p.numel() * 4 for _, p in trainable(model)) / 1e9
        record.update(anchors=anchors, probe_peak_mem_gb=round(peak, 2), probe_min_mem_available_gb=round(free, 2),
                      probe_pending_optimizer_gb=round(pending, 2))
        return not flag.item() and free - pending >= args.min_free_gb + args.probe_headroom_gb, record

    chosen, records = choose_batch(candidates, probe if args.probe else None)
    for record in records:
        emit({"phase": "probe", "step": step, **record})
    if chosen is None:
        abort_run(emit, {"reason": "probe_low_memory", "step": step, **records[-1]}, world)
    args.anchors, args.grad_accum = chosen
    if args.global_anchors or resumed.get("anchors") not in (None, args.anchors):
        emit({"phase": "batch", "step": step, "global_anchors": args.anchors * args.grad_accum * world,
              "anchors": args.anchors, "grad_accum": args.grad_accum, "ranks": world,
              "resumed_anchors": resumed.get("anchors"), "probed": [r["anchors"] for r in records]})


def plan_steps(args: argparse.Namespace, step_times: list[float], eval_s: float, run_started: float,
               completed: int, device: torch.device, world: int) -> int:
    """Total steps that fit the time budget, from the measured smoke step time (rank 0 decides)."""

    times = sorted(step_times[2:] or step_times)
    step_s = times[len(times) // 2]
    remaining = args.time_budget_h * 3600 - (time.monotonic() - run_started)
    overhead = ((args.eval_points + 1) * eval_s + args.checkpoint_points * args.checkpoint_s_estimate
                + args.reserve_s)
    planned = completed + max(0, int((remaining - overhead) / step_s))
    value = torch.tensor([planned], dtype=torch.float64, device=device)
    if world > 1:
        dist.broadcast(value, 0)
    return int(value.item())


# --------------------------------------------------------------------------
# Chain-proxy evaluation (fixed-anchor EAL)
# --------------------------------------------------------------------------

EVAL_KEYS = ("eal", "unary_top_{k}_oracle_accepted_length", "unary_candidate_recall_at_{k}",
             "teacher_forced_selector_acc", "full_acc", "unary_loss", "selector_loss", "loss")


def rowstore_eval_samples(store: RowStore, rows: list[dict[str, Any]], args: argparse.Namespace,
                          block_size: int, rank: int = 0, world: int = 1):
    """(name, inputs, fixed anchors) per held-out row-store row (this rank's share)."""

    for index, row in enumerate(rows):
        if index % world != rank:
            continue
        inputs = store.load(row)
        eligible = eligible_anchors(inputs, block_size)
        count = min(args.eval_anchors, len(eligible))
        if count:
            yield row["id"], inputs, sorted(fixed_anchors(row, {row["id"]: eligible}, count,
                                                          args.seed + 50_000 + index))


def capture_eval_samples(data: "cd.CaptureIndex", rows: list[dict[str, Any]], args: argparse.Namespace,
                         rank: int = 0, world: int = 1):
    """This rank's share of the held-out rows (anchors depend only on the row name)."""

    for sample in cd.eval_samples(data, rows[rank::world], seed=args.seed, max_anchors=args.eval_anchors,
                                  crop=args.eval_crop, threads=args.loader_workers, fp8_device=args.fp8_device):
        yield sample["row"], sample["inputs"], sample["anchors"]


@contextlib.contextmanager
def depth_counts_recorded():
    """Record per-draft-depth counts of the selector chain while the upstream DFlash2 metrics run.

    Wraps ``dflash2.metrics.compute_accepted_length_counts`` (the realized greedy selector path behind ``eal``):
    for draft depth d (1-based) ``reach`` counts valid slots whose earlier slots were all accepted and ``accept``
    those also accepted, so accept/reach is the conditional acceptance at depth d.  Yields the list the
    per-forward [2, depths] count tensors are appended to."""

    from speculators.models.dflash2 import metrics as dflash2_metrics

    original = dflash2_metrics.compute_accepted_length_counts
    recorded: list[torch.Tensor] = []

    def wrapper(correct: torch.Tensor, valid: torch.Tensor):
        accepted = torch.logical_and(correct, valid)
        alive = accepted.to(torch.float32).cumprod(dim=-1)
        before = torch.cat([torch.ones_like(alive[:, :1]), alive[:, :-1]], dim=-1)
        reach = before * valid.to(torch.float32)
        recorded.append(torch.stack([reach.sum(0), alive.sum(0)]).detach())
        return original(correct, valid)

    dflash2_metrics.compute_accepted_length_counts = wrapper
    try:
        yield recorded
    finally:
        dflash2_metrics.compute_accepted_length_counts = original


@torch.no_grad()
def evaluate(model: nn.Module, samples, kwargs: dict[str, Any], device: torch.device,
             dtype: torch.dtype, world: int = 1, bucket_of=None) -> dict[str, Any]:
    """Held-out chain proxy: teacher-forced greedy selector prefix + bonus.

    Each rank evaluates its share of the rows (fixed anchors); the metric
    sums are added over ranks in rank order, so the result is independent of
    the split.  ``bucket_of(row name)`` (optional) also reports the same
    metrics per bucket (e.g. per category) under ``buckets``.
    """

    was_training = model.training
    model.eval()
    parts: dict[str, list[Any]] = {}   # bucket -> [rows, totals]
    for name, inputs, chosen in samples:
        with fixed_anchors_installed(model, chosen), autocast(device), depth_counts_recorded() as depth:
            _, loss, metrics = model(**to_device(inputs, device, dtype), max_anchors=len(chosen), **kwargs)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite eval loss on {name}")
        entry = parts.setdefault(str(bucket_of(name)) if bucket_of else "", [0, {}])
        entry[0] += 1
        totals = entry[1]
        for key, value in scalar_metrics(metrics).items():
            totals[key] = totals.get(key, 0.0) + value
        for counts in depth:
            reach_row, accept_row = counts.cpu().tolist()
            for d, (reach, accept) in enumerate(zip(reach_row, accept_row), 1):
                totals[f"depth_{d}_reach"] = totals.get(f"depth_{d}_reach", 0.0) + reach
                totals[f"depth_{d}_accept"] = totals.get(f"depth_{d}_accept", 0.0) + accept
    if world > 1:
        gathered: list[Any] = [None] * world
        dist.all_gather_object(gathered, parts)
        parts = {}
        for part in gathered:
            for bucket, (part_rows, part_totals) in part.items():
                entry = parts.setdefault(bucket, [0, {}])
                entry[0] += part_rows
                for key, value in part_totals.items():
                    entry[1][key] = entry[1].get(key, 0.0) + value
    rows, totals = 0, {}
    for part_rows, part_totals in parts.values():
        rows += part_rows
        for key, value in part_totals.items():
            totals[key] = totals.get(key, 0.0) + value
    result = summarize_eval(rows, totals, model.candidate_selector.top_k)
    if bucket_of:
        result["buckets"] = {bucket: summarize_eval(part_rows, part_totals, model.candidate_selector.top_k)
                             for bucket, (part_rows, part_totals) in sorted(parts.items())}
    if was_training:
        model.train()
    return result


def summarize_eval(rows: int, totals: dict[str, float], top_k: int) -> dict[str, Any]:
    """Metric means and per-depth acceptance from summed eval totals."""

    result: dict[str, Any] = {"rows": rows}
    for template in EVAL_KEYS:
        key = template.format(k=top_k)
        if totals.get(f"{key}_total"):
            result[key] = totals[f"{key}_sum"] / totals[f"{key}_total"]
    if totals.get("eal_total"):
        result["eval_blocks"] = totals["eal_total"]
    depths = sorted(int(k.split("_")[1]) for k in totals if k.startswith("depth_") and k.endswith("_reach"))
    if depths:   # conditional acceptance per draft depth, and the share of blocks that get past it
        result["accept_by_depth"] = [round(totals[f"depth_{d}_accept"] / max(totals[f"depth_{d}_reach"], 1.0), 4)
                                     for d in depths]
        blocks = max(totals.get("depth_1_reach", 0.0), 1.0)
        result["survive_by_depth"] = [round(totals[f"depth_{d}_accept"] / blocks, 4) for d in depths]
    return result


# --------------------------------------------------------------------------
# Rank-local resume checkpoints
# --------------------------------------------------------------------------

def _optimizer_to_cpu(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    # state_dict() shares the per-parameter state dicts with the live
    # optimizer; build new dicts so the live state is untouched.
    state = optimizer.state_dict()
    state["state"] = {
        index: {key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
                for key, value in values.items()}
        for index, values in state["state"].items()
    }
    return state


def save_checkpoint(output: Path, step: int, model: nn.Module, optimizer: torch.optim.Optimizer,
                    rank: int, world: int, provenance: dict[str, Any],
                    counters: dict[str, float] | None = None,
                    named: list[tuple[str, torch.Tensor]] | None = None, keep: int = 0,
                    extra: dict[str, Any] | None = None) -> Path:
    """``named``: the fp32 tensors the optimizer owns (default: the model's trainable parameters).
    ``keep``: afterwards delete all but the ``keep`` newest ``step-*`` directories on this host.
    ``extra``: more payload entries (``diloco``: global weights, outer momentum, round)."""

    directory = output / "checkpoints" / f"step-{step:07d}"
    if rank == 0:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    barrier(world)
    directory.mkdir(parents=True, exist_ok=True)  # separate per-host filesystems
    if rank != 0:
        (directory / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2, sort_keys=True))
    if next(model.parameters()).is_cuda:
        # The host copy below is ~14 GB; give the activation cache back to the
        # (unified) host memory first.
        torch.cuda.empty_cache()
    payload = {
        "step": step, "rank": rank, "world": world, "counters": dict(counters or {}),
        "params": {n: p.detach().cpu().clone() for n, p in (named or trainable(model))},
        "optimizer": _optimizer_to_cpu(optimizer), **(extra or {}),
    }
    temporary = directory / f"rank{rank}.pt.tmp"
    torch.save(payload, temporary)
    os.replace(temporary, directory / f"rank{rank}.pt")
    barrier(world)
    (output / "checkpoints" / f"latest-rank{rank}").write_text(directory.name)
    if keep:
        old = sorted(output.glob("checkpoints/step-*"))[:-keep]
        for stale in old:   # (ranks sharing a filesystem may race here: ignore what the other removed)
            shutil.rmtree(stale, ignore_errors=True)
        barrier(world)
    return directory


@torch.no_grad()
def load_checkpoint(output: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
                    rank: int, world: int, named: list[tuple[str, torch.Tensor]] | None = None,
                    extras: dict[str, Any] | None = None) -> tuple[int, dict[str, float]]:
    """``extras`` (optional) receives the checkpoint's ``diloco`` entry (None for a synchronous run)."""

    pointer = output / "checkpoints" / f"latest-rank{rank}"
    if not pointer.is_file():
        return 0, {}
    path = output / "checkpoints" / pointer.read_text().strip() / f"rank{rank}.pt"
    payload = guarded_torch_load(path, "drafter")
    if payload["world"] != world or payload["rank"] != rank:
        raise RuntimeError(f"resume world/rank mismatch: {payload['world']}/{payload['rank']}")
    named = dict(named or trainable(model))
    if set(named) != set(payload["params"]):
        raise RuntimeError("resume parameter set differs from the model")
    for name, value in payload["params"].items():
        target = named[name]
        if target.shape != value.shape:
            raise RuntimeError(f"resume shape mismatch for {name}")
        target.copy_(value.to(target.device, target.dtype))
    optimizer.load_state_dict(payload["optimizer"])
    if extras is not None:
        extras["diloco"] = payload.get("diloco")
    return int(payload["step"]), dict(payload.get("counters", {}))


@torch.no_grad()
def init_from_checkpoint(directory: Path, model: nn.Module, rank: int, config_digest: str) -> dict[str, Any]:
    """Load the fp32 body of one of this trainer's own checkpoints (``checkpoints/step-N``) into the model.

    The checkpoint must carry this trainer's scratch stamp (``guarded_torch_load`` refuses anything else,
    including warm-started lineages) and the same config digest.  The rank's own file is used when present
    (replicas are identical), else any ``rank*.pt``; of a DiLoCo checkpoint, its global weights.  Optimizer
    state and step are not taken.
    Returns the parent record stamped into this run's provenance.
    """

    directory = Path(directory)
    path = directory / f"rank{rank}.pt"
    if not path.is_file():
        candidates = sorted(directory.glob("rank*.pt"))
        if not candidates:
            raise FileNotFoundError(f"--init-from {directory}: no rank*.pt")
        path = candidates[0]
    stamp_bytes = (directory / "PROVENANCE.json").read_bytes() if (directory / "PROVENANCE.json").is_file() else b""
    payload = guarded_torch_load(path, "drafter")   # checks the PROVENANCE.json scratch stamp
    stamp = json.loads(stamp_bytes)
    if stamp.get("config_sha256") != config_digest:
        raise RuntimeError(f"--init-from {directory}: config digest {stamp.get('config_sha256')} != {config_digest}")
    params = (payload.get("diloco") or {}).get("global") or payload["params"]
    named = dict(trainable(model))
    if set(named) != set(params):
        raise RuntimeError(f"--init-from {directory}: parameter set differs from the model")
    for name, value in params.items():
        target = named[name]
        if target.shape != value.shape:
            raise RuntimeError(f"--init-from shape mismatch for {name}")
        target.copy_(value.to(target.device, target.dtype))
    return {"path": str(directory), "step": int(payload["step"]),
            "tokens_seen": float(payload.get("counters", {}).get("tokens_seen", 0.0)),
            "provenance_sha256": hashlib.sha256(stamp_bytes).hexdigest(),
            "init_digest": stamp.get("init_digest"), "init_seed": stamp.get("init_seed"),
            "parent": stamp.get("parent")}


@torch.no_grad()
def load_start_weights(path: Path, named: list[tuple[str, torch.Tensor]]) -> dict[str, Any]:
    """Copy a weights file (:func:`.diloco.load_weights`, guarded) into ``named``; returns its record."""

    tensors, meta = dl.load_weights(path)
    targets = dict(named)
    if set(targets) != set(tensors):
        raise RuntimeError(f"--start-weights {path}: parameter set differs from the model")
    for name, value in tensors.items():
        if targets[name].shape != value.shape:
            raise RuntimeError(f"--start-weights shape mismatch for {name}")
        targets[name].copy_(value.to(targets[name].device, targets[name].dtype))
    return {"path": str(path), "sha256": sha256_file(path),
            **{k: meta[k] for k in ("version", "dtype", "schema") if k in meta}}


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def gather_body(named: list[tuple[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Full BF16 body tensors on CPU (replicas are identical)."""

    return {name: parameter.detach().to("cpu", torch.bfloat16) for name, parameter in named}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def emit_local(rank: int, output: Path, record: dict[str, Any]) -> None:
    """Per-rank record for paths where ranks cannot agree (OOM)."""

    with (output / f"abort-rank{rank}.json").open("a") as stream:
        stream.write(json.dumps(record) + "\n")
    print(json.dumps({"rank": rank, **record}), file=sys.stderr, flush=True)


def sources_record(stream: "cd.CaptureTrainStream", items: int, world: int) -> dict[str, Any]:
    """Anchors drawn per source over the first ``items`` items of every rank, and the epochs of each source pool."""

    gathered: list[Any] = [stream.source_report(items)]
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, stream.source_report(items))
    total: dict[str, dict[str, float]] = {}
    for part in gathered:
        for source, entry in part.items():
            into = total.setdefault(source, {"anchors": 0.0, "pool_anchors": 0.0, "items": 0})
            for key in into:
                into[key] += entry[key]
    anchors = sum(entry["anchors"] for entry in total.values()) or 1.0
    for entry in total.values():
        entry["epochs"] = round(entry["anchors"] / max(entry["pool_anchors"], 1.0), 3)
        entry["anchor_share"] = round(entry["anchors"] / anchors, 4)
    return {"items_per_rank": items, "total": total, "per_rank": gathered}


def source_caps(stream: "cd.CaptureTrainStream", args: argparse.Namespace, switch_item: int,
                world: int) -> dict[str, Any]:
    """Apply ``--source-max-epochs`` to this rank's stream from ``switch_item`` on; the per-rank plans."""

    fill = set(args.source_fill.split(",")) if args.source_fill else None
    report = stream.cap_epochs(args.steps * args.grad_accum, args.source_max_epochs, switch_item, fill)
    gathered: list[Any] = [report]
    if world > 1:
        gathered = [None] * world
        dist.all_gather_object(gathered, report)
    return {"switch_item": switch_item, "per_rank": gathered}


def apply_plan(args: argparse.Namespace, planned: int) -> None:
    args.steps = planned
    args.eval_every = max(1, planned // args.eval_points)
    args.checkpoint_every = max(1, planned // args.checkpoint_points)


def smoke_summary(smoke: list[dict[str, float]], world: int) -> dict[str, Any]:
    steady = smoke[2:] or smoke
    seconds = sum(s["step_s"] for s in steady)
    times = sorted(s["step_s"] for s in steady)
    return {"steps": len(smoke), "world": world, "step_s_median": times[len(times) // 2],
            "tflops_per_gpu": round(sum(s["tflops_per_gpu"] * s["step_s"] for s in steady) / seconds, 2),
            "gpu_w": round(sum(s["gpu_w"] for s in steady) / len(steady), 1),
            "step_s_max": times[-1], "supervised_tok_per_s": round(sum(s["supervised"] for s in steady) / seconds, 1),
            "context_rows_per_s": round(sum(s["context_rows"] for s in steady) / seconds, 1),
            "peak_mem_gb": max(s["peak_mem_gb"] for s in smoke),
            "min_mem_available_gb": min(s["min_mem_available_gb"] for s in smoke)}


def held_out_groups(capture: "cd.CaptureIndex", args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    """Held-out eval groups (``--eval-groups``), then ``heldout_extra``.  Rows of a group keep index order and are
    subsampled as round 1/2a did (``Random(seed).sample``, sorted by name), so a group over the same rows picks the
    same rows as those runs' single held-out eval."""

    def pick(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        if limit and len(rows) > limit:   # deterministic spread over sources
            rows = sorted(random.Random(args.seed).sample(rows, limit), key=lambda r: r["name"])
        return rows

    heldout = capture.split("heldout")
    groups: dict[str, list[dict[str, Any]]] = {}
    if args.eval_groups:
        for part in args.eval_groups.split(","):
            name, _, spec = part.partition("=")
            sources, _, limit = spec.partition(":")
            wanted = set(sources.split("+"))
            groups[name.strip()] = pick([r for r in heldout if str(r.get("source")) in wanted],
                                        int(limit) if limit else args.eval_rows)
    else:
        groups["heldout"] = pick(heldout, args.eval_rows)
    extra = pick(capture.split("heldout_extra"), args.eval_rows)
    if extra:
        groups["heldout_extra"] = extra
    return groups


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True, help="GLM-5.3-Flash target dir (embed/norm/head)")
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--features", type=Path, help="enntity.drafter_rows/v1 store")
    data.add_argument("--capture-index", type=Path,
                      help="capture_io index.json (Atlas captures read in place)")
    p.add_argument("--capture-root", type=Path, help="data root the capture index paths are relative to")
    p.add_argument("--crop", type=int, default=4096, help="capture: anchors are drawn inside a random crop "
                   "of this many eligible positions (0 = whole row)")
    p.add_argument("--eval-crop", type=int, default=None, help="capture eval crop (default: --crop)")
    p.add_argument("--loader-workers", type=int, default=4,
                   help="capture: prefetch threads (or worker processes); also the held-out eval read threads")
    p.add_argument("--loader-mode", choices=["thread", "process"], default="thread",
                   help="capture: in-process prefetch threads (no shared-memory copies) or DataLoader processes")
    p.add_argument("--source-weights", default=None,
                   help="capture: source-balanced coverage sampling, e.g. template=0.4,code+agentic=0.4,structured=0.2 "
                        "(shares of anchors; see capture_io.CaptureTrainStream)")
    p.add_argument("--pack-rows", action="store_true",
                   help="capture + --source-weights: pack several short row chunks into one micro-step (up to --anchors "
                        "anchors and crop + W + B context rows, the probe's worst case); each is its own document")
    p.add_argument("--fp8-device", action="store_true",
                   help="capture: FP8 rows go to the device as e4m3 bytes + scales and are dequantized there "
                        "(bit-identical to the CPU codec; half the bytes, no codec work in the reader threads)")
    p.add_argument("--source-max-epochs", type=float, default=None,
                   help="capture + --source-weights: once the step plan is known, cap every source at this many "
                        "epochs over the run; a capped source's share goes to the uncapped ones (by target share)")
    p.add_argument("--source-fill", default=None,
                   help="with --source-max-epochs: comma list of the sources that receive the share capped sources "
                        "give up (default: every uncapped source)")
    p.add_argument("--eval-groups", default=None,
                   help="capture: held-out groups by source, NAME=SRC[+SRC...][:ROWS],... evaluated separately; the "
                        "first group is logged unprefixed (it replaces the single 'heldout' eval), the others as "
                        "NAME/<metric>.  Default: one group of all held-out rows (--eval-rows)")
    p.add_argument("--eval-breakdown", default=None,
                   help="capture: row key (e.g. category) to also report every eval group per value of "
                        "(NAME.VALUE/<metric>)")
    p.add_argument("--init-from", type=Path, default=None,
                   help="initialize the body from this trainer's own checkpoint directory (checkpoints/step-N: fp32 "
                        "masters of rank*.pt, stamped PROVENANCE.json, same config); fresh optimizer and schedule")
    p.add_argument("--start-weights", type=Path, default=None,
                   help="initialize the body from a weights file (enntity.drafter.weights/v1 safetensors, e.g. a "
                        "coordinator's global weights; scratch stamp required); with --diloco-transport external "
                        "every round starts from it")
    p.add_argument("--eval-only", action="store_true",
                   help="evaluate the initial body (--start-weights / --init-from) on the held-out set and exit")
    p.add_argument("--agreement-mask", default="off",
                   help="capture: drop supervised slots by decode-vs-prefill agreement: off | argmax | minprob:<p>")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--attn", choices=["eager", "sdpa", "simple_flex_attention", fp.PACKED_ATTN, fp.FLEX_ATTN],
                   default="eager",
                   help=f"{fp.PACKED_ATTN}: exact block-sparse attention (gathered windows, masked SDPA per anchor "
                        f"chunk); {fp.FLEX_ATTN}: upstream block mask + flex attention compiled inside the layers "
                        f"(needs --compile-layers)")
    perf = p.add_argument_group("round-2 performance path (all off = round 1)")
    perf.add_argument("--anchors-per-chunk", type=int, default=16,
                      help=f"{fp.PACKED_ATTN}: anchors (x block queries) per attention chunk")
    perf.add_argument("--param-dtype", choices=["fp32", "bf16"], default="fp32",
                      help="bf16: Linear weights stored in bf16, fp32 masters in the optimizer/checkpoints "
                           "(same math as autocast over fp32 weights, without the per-micro casts)")
    perf.add_argument("--compile-layers", action="store_true",
                      help="torch.compile(dynamic=True) each decoder layer (losses stay eager)")
    perf.add_argument("--compile-mode", default=None, help="torch.compile mode for --compile-layers")
    perf.add_argument("--adamw-impl", choices=["foreach", "fused"], default="foreach")
    perf.add_argument("--deferred-metrics", action="store_true",
                      help="no per-micro host syncs: loss/metrics/finite checks materialize once per step")
    perf.add_argument("--train-metrics", choices=["full", "light"], default="full",
                      help="light: training micro-steps skip upstream's metric-only full-vocab passes (TV acceptance, "
                           "greedy accuracy, target logsumexp); identical loss and gradients; evals unchanged")
    perf.add_argument("--pin-loader", action="store_true",
                      help="thread loader: page-lock each sample in its reader thread (round 1 copied from pageable "
                           "memory: a synchronous ~25 ms host->device copy per micro-step on the compute stream)")
    perf.add_argument("--perf-profile", type=int, default=0,
                      help="profile N steps after --perf-warmup steps (torch.profiler), write perf-profile.{json,md} "
                           "to --output and exit (no evals, checkpoints or export)")
    perf.add_argument("--perf-warmup", type=int, default=3)
    perf.add_argument("--cpu-autocast", action="store_true", help=argparse.SUPPRESS)
    lo = p.add_argument_group("DiLoCo weight sync (diloco.py, docs/volunteer.md; default: synchronous)")
    lo.add_argument("--sync-mode", choices=["sync", "diloco"], default="sync",
                    help="diloco: workers take --inner-steps independent optimizer steps (own AdamW state, no "
                         "gradient all-reduce between workers), then one outer step on the averaged deltas")
    lo.add_argument("--inner-steps", type=int, default=500, help="H: optimizer steps between outer steps")
    lo.add_argument("--outer-lr", type=float, default=1.0, help="outer learning rate (1 with momentum 0 = plain "
                    "averaging, the default; OpenDiLoCo used 0.7 with Nesterov momentum 0.9)")
    lo.add_argument("--outer-momentum", type=float, default=0.0)
    lo.add_argument("--outer-nesterov", action=argparse.BooleanOptionalAction, default=True)
    lo.add_argument("--delta-dtype", choices=dl.DELTA_DTYPES, default="int8",
                    help="wire dtype of the deltas (int8, the default: absmax scale per 1024 values, 1.18 GB for the "
                         "full model, ~0.8%% error; bf16: 2.34 GB)")
    lo.add_argument("--diloco-fragment-mb", type=int, default=512, help="fp32 MB per exchanged fragment")
    lo.add_argument("--diloco-transport", choices=["group", "dir", "external"], default="group",
                    help="group: every rank of this torchrun world is a worker; dir: every site (its own torchrun "
                         "world, synchronous inside) is a worker and exchanges through --diloco-dir; external: each "
                         "run trains one round from --start-weights, writes its delta (--delta-out) and stops (the "
                         "volunteer coordinator merges; drafter-join drives it)")
    lo.add_argument("--diloco-dir", type=Path, default=None, help="dir transport: directory shared by the sites")
    lo.add_argument("--sites", type=int, default=1,
                    help="dir transport: number of sites; external: data partitions (this site reads partition "
                         "--site)")
    lo.add_argument("--site", type=int, default=0,
                    help="dir/external: this site (its ranks read data partition site * world + rank of an index "
                         "partitioned for sites * world)")
    lo.add_argument("--diloco-timeout-h", type=float, default=6.0, help="dir transport: wait for peers this long")
    lo.add_argument("--delta-out", type=Path, default=None,
                    help="external: delta file (default <output>/deltas/step-<N>.safetensors)")
    lo.add_argument("--delta-meta", default="{}", help="external: JSON object added to the delta header")
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="micro-steps per optimizer step; gradients are all-reduced once, after the last")
    p.add_argument("--grad-reduce-dtype", choices=["bf16", "fp32"], default="bf16",
                   help="wire dtype of the gradient all-reduce (accumulation and optimizer stay fp32)")
    p.add_argument("--anchors", type=int, default=64, help="anchor blocks per row per micro-step (with "
                   "--global-anchors: the largest micro-batch to try)")
    p.add_argument("--global-anchors", type=int, default=None,
                   help="anchor blocks per optimizer step over this run's ranks: the micro-batch (--anchors) and "
                        "--grad-accum are derived so micro x accum x ranks equals it, the micro-batch as large as "
                        "--probe finds memory for (choose_batch)")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min-lr-ratio", type=float, default=0.1)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--objective", choices=OBJECTIVES, default="upstream")
    p.add_argument("--late-objective", choices=OBJECTIVES, default=None)
    p.add_argument("--late-objective-step", type=int, default=None)
    p.add_argument("--unary-loss", default="kl_div", help="Speculators loss spec (kl_div, ce, tv, nla, JSON mix)")
    p.add_argument("--per-position", choices=["fixed-exp-decay", "dpace"], default="fixed-exp-decay")
    p.add_argument("--gamma", type=float, default=4.0)
    p.add_argument("--selector-alpha", type=float, default=1.0)
    p.add_argument("--dpace-alpha", type=float, default=0.5)
    p.add_argument("--prefix-floor", type=float, default=0.1)
    p.add_argument("--eal-weight", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=1000, help="0 = no held-out evaluation")
    p.add_argument("--eval-anchors", type=int, default=128)
    p.add_argument("--eval-rows", type=int, default=0, help="0 = all held-out rows")
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--checkpoint-every", type=int, default=2000)
    p.add_argument("--stop-after", type=int, default=None, help="exit (as if killed) after this step")
    p.add_argument("--smoke-steps", type=int, default=0,
                   help="first N steps: per-step tokens/s, step time, peak memory; summary in smoke.json")
    p.add_argument("--probe", action="store_true",
                   help="before step 0: one worst-case-size forward/backward (synthetic), memory checked")
    p.add_argument("--min-free-gb", type=float, default=0.0,
                   help="abort cleanly (exit 3) when any rank's host MemAvailable falls below this")
    p.add_argument("--probe-headroom-gb", type=float, default=0.0,
                   help="--probe: the micro-batch must leave this much above --min-free-gb (memory shared with "
                        "other services moves)")
    p.add_argument("--max-mem-fraction", type=float, default=None,
                   help="CUDA caching-allocator cap (fraction of device memory): OOM raises instead of "
                        "exhausting unified memory")
    p.add_argument("--time-budget-h", type=float, default=None,
                   help="plan --steps after the smoke from the measured step time to fit this wall budget")
    p.add_argument("--eval-points", type=int, default=0,
                   help="with --time-budget-h: evaluate this many times over the run (eval_every = steps/N)")
    p.add_argument("--checkpoint-points", type=int, default=10)
    p.add_argument("--keep-checkpoints", type=int, default=0,
                   help="keep only this many newest checkpoints per rank (0 = all)")
    p.add_argument("--empty-cache-every", type=int, default=0,
                   help="torch.cuda.empty_cache() every N steps (allocator fragmentation; 0 = never)")
    p.add_argument("--gc-freeze", action="store_true",
                   help="gc.freeze() after setup: the index and plan objects are never scanned by the cyclic GC")
    p.add_argument("--checkpoint-s-estimate", type=float, default=30.0)
    p.add_argument("--reserve-s", type=float, default=300.0, help="budget kept for export and teardown")
    p.add_argument("--export", action="store_true", help="export the final body in Atlas format")
    p.add_argument("--feature-cache-bytes", type=int, default=2 * 1024**3)
    p.add_argument("--seed", type=int, default=20261014)
    p.add_argument("--peak-tflops", type=float, default=None,
                   help="measured dense BF16 matmul ceiling of one GPU (logs mfu = achieved / this)")
    args = p.parse_args(argv)
    if args.warmup >= args.steps:
        args.warmup = max(1, args.steps // 10)
    if args.late_objective and args.late_objective_step is None:
        p.error("--late-objective needs --late-objective-step")
    if args.capture_index and args.capture_root is None:
        p.error("--capture-index needs --capture-root")
    if args.agreement_mask != "off" and not (args.agreement_mask == "argmax"
                                             or args.agreement_mask.startswith("minprob:")):
        p.error("--agreement-mask must be off, argmax or minprob:<p>")
    if args.eval_crop is None:
        args.eval_crop = args.crop
    if args.time_budget_h is not None:
        if args.smoke_steps < 4 or args.smoke_steps >= args.warmup:
            p.error("--time-budget-h needs 4 <= --smoke-steps < --warmup (planning happens during warmup)")
        if args.eval_points <= 0:
            p.error("--time-budget-h needs --eval-points")
    if args.attn == fp.FLEX_ATTN and not args.compile_layers:
        p.error(f"--attn {fp.FLEX_ATTN} needs --compile-layers (uncompiled flex materialises the scores)")
    if args.perf_profile and (args.time_budget_h is not None or args.export):
        p.error("--perf-profile is a measurement mode: no --time-budget-h / --export")
    if args.start_weights and args.init_from:
        p.error("--start-weights and --init-from are exclusive")
    if args.eval_only and not args.eval_every:
        p.error("--eval-only needs evaluation (--eval-every > 0)")
    if args.sync_mode == "diloco":
        if args.inner_steps < 1:
            p.error("--inner-steps must be >= 1")
        if args.diloco_transport == "dir":
            if args.diloco_dir is None or not 0 <= args.site < args.sites:
                p.error("--diloco-transport dir needs --diloco-dir and 0 <= --site < --sites")
            if args.time_budget_h is not None:
                p.error("--diloco-transport dir: sites must agree on --steps (no --time-budget-h)")
        if args.diloco_transport == "external":
            if args.start_weights is None or args.time_budget_h is not None or not 0 <= args.site < args.sites:
                p.error("--diloco-transport external needs --start-weights, 0 <= --site < --sites and no "
                        "--time-budget-h")
            if not isinstance(json.loads(args.delta_meta), dict):
                p.error("--delta-meta must be a JSON object")
    if args.sites != 1 and not args.eval_only and not (args.sync_mode == "diloco"
                                                       and args.diloco_transport in ("dir", "external")):
        p.error("--sites/--site need --sync-mode diloco --diloco-transport dir or external")
    return args


def main(argv: list[str] | None = None) -> dict[str, Any]:
    global CPU_AUTOCAST
    args = parse_args(argv)
    CPU_AUTOCAST = args.cpu_autocast
    run_started = time.monotonic()
    rank, world, device = init_distributed(args.device)
    data_rank, data_world = args.site * world + rank, args.sites * world   # = rank, world unless sites > 1
    external = args.sync_mode == "diloco" and args.diloco_transport == "external"
    if device.type == "cuda" and args.max_mem_fraction:
        torch.cuda.set_per_process_memory_fraction(args.max_mem_fraction, device)
    cfg = sm.load_config(args.config)
    cfg["target_name"] = str(args.teacher)
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "train.jsonl"

    def emit(record: dict[str, Any]) -> None:
        if rank == 0:
            with log_path.open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)

    # Frozen target pieces (guarded) and the fresh body.
    frozen, target_receipt = sm.load_target_frozen(args.teacher)
    compute_dtype = torch.bfloat16 if device.type == "cuda" or CPU_AUTOCAST else torch.float32
    store = capture = None
    if args.capture_index:
        capture = cd.CaptureIndex(args.capture_index, args.capture_root)
        if (capture.layers != list(cfg["target_layer_ids"]) or capture.hidden != cfg["target_hidden_size"]
                or capture.window != cfg["sliding_window"] or capture.block != cfg["block_size"]):
            raise ValueError("capture index layers/hidden/window/block do not match the config")
        final_hidden, features_digest = capture.final_hidden, capture.digest
    else:
        store = RowStore(args.features, expect_layers=cfg["target_layer_ids"], hidden_size=cfg["target_hidden_size"],
                         dtype=compute_dtype, max_cache_bytes=args.feature_cache_bytes)
        final_hidden = store.final_hidden
        features_digest = hashlib.sha256((args.features / "manifest.json").read_bytes()).hexdigest()
    if args.attn == fp.PACKED_ATTN:
        fp.register_packed_attention()
    if args.attn == fp.FLEX_ATTN:
        fp.register_flex_attention()
    model = sm.build_model(cfg, attn_implementation=args.attn)
    if args.attn == fp.PACKED_ATTN:
        fp.install_packed_attention(model, args.anchors_per_chunk)
    if args.attn == fp.FLEX_ATTN:
        fp.install_flex_attention(model)
    sm.install_frozen_target(model, {k: v.to(compute_dtype) for k, v in frozen.items()}, final_hidden)
    init_report = sm.scratch_init(model, cfg, args.seed, frozen["norm.weight"].float())
    body_params, body_keys = sm.count_body_parameters(model)
    model.to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name not in sm.FROZEN_KEYS)

    config_digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
    parent = init_from_checkpoint(args.init_from, model, rank, config_digest) if args.init_from else None
    start_record = None
    if args.start_weights:
        start_record = load_start_weights(args.start_weights, trainable(model))
        parent = {"start_weights": start_record}
    if parent is not None:
        init_report = {**init_report, "init_from": parent}
    init_digest = hashlib.sha256(b"".join(
        hashlib.sha256(p.detach().cpu().float().numpy().tobytes()).digest() for _, p in trainable(model)
    )).hexdigest()
    provenance = scratch_stamp(
        trainer=TRAINER_VERSION, config_sha256=config_digest, init_seed=args.seed,
        init_digest=init_digest, target=target_receipt, speculators_revision=sm.SPECULATORS_REVISION,
        features_manifest_sha256=features_digest,
        **({"parent": parent} if parent is not None else {}),
    )
    rank_consensus({"digest": init_digest, "config": config_digest,
                    "features": provenance["features_manifest_sha256"]}, world)

    # Round-2 path (after the init digest, so provenance is identical): bf16 compute weights with
    # fp32 masters; ``owned`` is what the optimizer, clipping, checkpoints and export see.
    master = fp.MasterWeights(trainable(model)) if args.param_dtype == "bf16" else None
    owned = master.named if master is not None else trainable(model)
    if args.compile_layers:
        fp.compile_layers(model, args.compile_mode)
    grad_sync = (GradSync([p for _, p in trainable(model)], world,
                          torch.bfloat16 if args.grad_reduce_dtype == "bf16" else torch.float32,
                          owner=master.owner_of if master is not None else None)
                 if world > 1 and not (args.sync_mode == "diloco" and args.diloco_transport == "group") else None)
    optimizer = build_optimizer(owned, args)
    resumed: dict[str, Any] = {}
    start_step, counters = load_checkpoint(output, model, optimizer, rank, world, owned, resumed)
    if start_step and (resumed["diloco"] is not None) != (args.sync_mode == "diloco"):
        raise RuntimeError(f"resume: the checkpoint's sync mode is not --sync-mode {args.sync_mode}")
    diloco = None
    if args.sync_mode == "diloco":
        if args.diloco_transport == "group":
            transport = dl.ProcessGroupTransport(world)
        elif args.diloco_transport == "dir":   # one exchange per site (its rank 0); the others get the mean by broadcast
            transport = (dl.DirectoryTransport(args.diloco_dir, args.site, args.sites, provenance,
                                               args.diloco_timeout_h * 3600) if rank == 0 else None)
        else:
            transport = None
        diloco = dl.Diloco(owned, transport, lr=args.outer_lr, momentum=args.outer_momentum,
                           nesterov=args.outer_nesterov, delta_dtype=args.delta_dtype,
                           fragment_mb=args.diloco_fragment_mb,
                           site_world=world if args.diloco_transport == "dir" else 1,
                           meta={"config": config_digest, "init": init_digest, "inner_steps": args.inner_steps},
                           external=external)
        if start_step:
            diloco.load_state_dict(resumed["diloco"])
        if external:
            if start_step % args.inner_steps == 0:   # a new round starts from the coordinator's weights
                if start_step:
                    load_start_weights(args.start_weights, owned)
                diloco.set_global(dict(owned), start_record)
            elif diloco.base.get("sha256") != start_record["sha256"]:
                raise RuntimeError("resume mid-round: --start-weights is not the weights this round started from")
    if master is not None:
        master.sync_low()
    kwargs = loss_kwargs(device.type, args.unary_loss, args.per_position, args.gamma,
                         args.selector_alpha, args.dpace_alpha)
    selector_prefix = {"floor": args.prefix_floor, "eal_weight": args.eal_weight}
    if not args.eval_only:   # before the stream: its items depend on the micro-batch
        select_batch(args, counters if start_step else {}, model, cfg, device, compute_dtype, kwargs,
                     selector_prefix, optimizer, world, emit, start_step)
    items_start = int(counters.get("items_done", start_step * args.grad_accum))   # micro-steps consumed per rank
    if capture is not None:
        if not args.eval_only:
            stream = cd.CaptureTrainStream(args.capture_index, args.capture_root, rank=data_rank, world=data_world,
                                           seed=args.seed, max_anchors=args.anchors, crop=args.crop,
                                           source_weights=args.source_weights, pack=args.pack_rows,
                                           fp8_device=args.fp8_device)
        eval_groups = held_out_groups(capture, args) if args.eval_every else {}
        first_group = next(iter(eval_groups), None)
        if args.eval_every and (not first_group or not eval_groups[first_group]):
            raise SystemExit("capture index has no held-out rows with eligible anchors")
        row_info = {r["name"]: r for rows in eval_groups.values() for r in rows}
        bucket_of = ((lambda name: row_info[name].get(args.eval_breakdown)) if args.eval_breakdown else None)

        def run_eval() -> dict[str, Any]:
            record: dict[str, Any] = {}
            for name, rows in eval_groups.items():
                if not rows:
                    continue
                metrics = evaluate(model, capture_eval_samples(capture, rows, args, rank, world), kwargs, device,
                                   compute_dtype, world, bucket_of)
                buckets = metrics.pop("buckets", {})
                record.update(metrics if name == first_group else {f"{name}/{k}": v for k, v in metrics.items()})
                if len(buckets) > 1:
                    for bucket, values in buckets.items():
                        record.update({f"{name}.{bucket}/{k}": v for k, v in values.items()})
            return record
    else:
        train_rows = store.split("train")
        heldout_rows = store.split("heldout")[: args.eval_rows or None]
        if (not train_rows and not args.eval_only) or (not heldout_rows and args.eval_every):
            raise SystemExit("feature store needs train rows (training) and heldout rows (evaluation)")

        def run_eval() -> dict[str, Any]:
            return evaluate(model, rowstore_eval_samples(store, heldout_rows, args, model.block_size, rank, world),
                            kwargs, device, compute_dtype, world)
    if args.eval_only:
        record = {"phase": "eval", "step": 0, "tokens_seen": 0, **run_eval()}
        emit(record)
        barrier(world)
        if world > 1:
            dist.destroy_process_group()
        return {"eval": record, "start_weights": start_record}
    if diloco is not None and not external:   # evaluations see the global weights
        local_eval = run_eval

        def run_eval() -> dict[str, Any]:
            with diloco.global_weights(master.sync_low if master is not None else None):
                return local_eval()
    if args.perf_profile:
        # Measurement mode: no evals, checkpoints or export; log every step.
        args.steps = start_step + args.perf_warmup + args.perf_profile
        args.eval_every = args.checkpoint_every = 1 << 60
        args.log_every = 1
        eval_s = 0.0
    elif start_step == 0:
        if rank == 0:
            (output / "receipt.json").write_text(json.dumps({
                "provenance": provenance, "init": init_report, "body_parameters": body_params,
                "body_keys": body_keys, "world": world, "args": {k: str(v) for k, v in vars(args).items()},
            }, indent=2))
        eval_started = time.monotonic()
        if args.eval_every:
            emit({"phase": "eval", "step": 0, "tokens_seen": 0, **run_eval()})
        eval_s = time.monotonic() - eval_started
    else:
        eval_s = float(counters.get("eval_s", 0.0))
        emit({"phase": "resume", "step": start_step})
    if counters.get("planned_steps"):
        apply_plan(args, int(counters["planned_steps"]))
    cap_switch = counters.get("cap_switch_item")
    if capture is not None and args.source_weights and args.source_max_epochs:
        if cap_switch is None and start_step == 0 and args.time_budget_h is None:
            cap_switch = 0   # fixed step count: cap from the first item
        if cap_switch is not None:   # (a time-budgeted run caps once the smoke has planned the steps)
            emit({"phase": "source_caps", **source_caps(stream, args, int(cap_switch), world)})

    model.train()
    started = time.monotonic()
    step = start_step
    tokens_seen = float(counters.get("tokens_seen", 0.0))
    round_start_tokens = float(counters.get("round_start_tokens", tokens_seen))   # DiLoCo: tokens of this round
    finished: dict[str, Any] | None = None   # external DiLoCo: the round's delta record
    samples = None
    loader_stats: dict[str, float] = {}
    if capture is not None:
        samples = iter(cd.train_loader(stream, items_start, 1 << 40, args.loader_workers,
                                       pin_memory=device.type == "cuda", mode=args.loader_mode,
                                       pin_threads=args.pin_loader and device.type == "cuda", stats=loader_stats))
    host = fp.HostStats(device)
    if args.gc_freeze:
        gc.collect()
        gc.freeze()   # the index, plans and model objects: never traversed by the cyclic GC again
    window = {"steps": 0, "seconds": 0.0, "wait": 0.0, "read": 0.0, "sync": 0.0, "enqueue": 0.0, "tail": 0.0,
              "supervised": 0.0, "context": 0.0, "flops": 0.0, "useful": 0.0, "watts": 0.0}
    smoke: list[dict[str, float]] = []
    power = fp.PowerSampler(device)
    profiler, profiled_seconds, profiled_flops = None, [], {}
    clean: dict[str, Any] = {"seconds": [], "flops": 0.0, "useful": 0.0, "watts": []}   # unprofiled warmup steps
    step = start_step
    while step < args.steps:
        if args.perf_profile and step == start_step + args.perf_warmup:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            activities = [torch.profiler.ProfilerActivity.CPU]
            if device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            profiler = torch.profiler.profile(activities=activities)
            profiler.__enter__()
            power.take()
        step_started = time.monotonic()
        objective = (args.late_objective if args.late_objective and step >= args.late_objective_step
                     else args.objective)
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        step_metrics: dict[str, float] = {}
        supervised = context = flops = 0.0
        step_parts: dict[str, float] = {}
        deferred: dict[str, torch.Tensor] = {}   # --deferred-metrics: device-side sums
        step_rows: list[str] = []
        for micro in range(args.grad_accum):
            global_index = (items_start + (step - start_step) * args.grad_accum + micro) * data_world + data_rank
            if samples is not None:
                waited = time.monotonic()
                sample = next(samples)
                window["wait"] += time.monotonic() - waited
                window["read"] += sample["read_s"]
                anchors, row_name, on_policy = sample["anchors"], sample["row"], sample["on_policy"]
                inputs = to_device(sample["inputs"], device, compute_dtype)
                if args.agreement_mask != "off":
                    inputs["loss_mask"] = cd.agreement_keep(model, inputs, anchors, args.agreement_mask)
                max_anchors = len(anchors)
                if args.deferred_metrics and args.agreement_mask == "off":
                    supervised += float(sample["supervised"])   # same count, from the host copy
                else:
                    supervised += float(inputs["loss_mask"][0, [a + k for a in anchors
                                                                for k in range(1, model.block_size)]].sum())
                context += sample["context_rows"]
                parts = flop_parts(cfg, max_anchors * model.block_size, sample["context_rows"],
                                   fp.attended_keys(sample["inputs"]["document_ids"], anchors,
                                                    cfg["sliding_window"], model.block_size))
            else:
                row = scheduled_row(train_rows, global_index, args.seed)
                inputs = to_device(store.load(row), device)
                anchors, row_name, on_policy = None, row["id"], bool(row.get("on_policy", True))
                max_anchors = args.anchors
                supervised += float(min(args.anchors, len(eligible_anchors(inputs, model.block_size)))
                                    * (model.block_size - 1))
                context += inputs["input_ids"].shape[1]
                parts = flop_parts(cfg, max_anchors * model.block_size, inputs["input_ids"].shape[1])
            flops += parts["total"]
            for key, value in parts.items():
                step_parts[key] = step_parts.get(key, 0.0) + value
            torch.manual_seed(args.seed + 7919 * global_index)  # anchor draw (select_anchors)
            if grad_sync is not None and micro == args.grad_accum - 1:
                grad_sync.arm()
            try:
                with fixed_anchors_installed(model, anchors), autocast(device):
                    loss, metrics = training_loss(objective, model, inputs, max_anchors=max_anchors,
                                                  on_policy=on_policy, kwargs=kwargs,
                                                  selector_prefix=selector_prefix,
                                                  light_metrics=args.train_metrics == "light")
                (loss / args.grad_accum).backward()
            except torch.OutOfMemoryError as error:
                # The peer rank is inside a collective and cannot agree; exit at once
                # (torchrun / NCCL then tear the peer down).
                emit_local(rank, output, {"phase": "abort", "reason": "oom", "step": step, "row": row_name,
                                          "context_rows": int(inputs["input_ids"].shape[1]),
                                          "error": str(error)[:500]})
                os._exit(3)
            if args.deferred_metrics:
                step_rows.append(row_name)
                held = {"__loss": loss.detach().float() / args.grad_accum,
                        "__bad": (~torch.isfinite(loss.detach())).float()}
                held.update((k, v.detach().float().sum()) for k, v in metrics.items() if isinstance(v, torch.Tensor))
                for key, value in held.items():
                    deferred[key] = deferred[key] + value if key in deferred else value
                continue
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at step {step} row {row_name}")
            step_loss += float(loss.detach()) / args.grad_accum
            for key, value in scalar_metrics(metrics).items():
                step_metrics[key] = step_metrics.get(key, 0.0) + value
        if deferred:
            keys = list(deferred)
            stacked = torch.stack([deferred[k].to(device) for k in keys])
            synced = time.monotonic()
            window["enqueue"] += synced - step_started
            values = stacked.cpu().tolist()   # one host sync per step: waits for the GPU to finish the micro-steps
            window["tail"] += time.monotonic() - synced
            step_metrics = dict(zip(keys, values))
            if step_metrics.pop("__bad"):
                raise FloatingPointError(f"non-finite loss at step {step} rows {step_rows}")
            step_loss = step_metrics.pop("__loss")
        if grad_sync is not None:
            synced = time.monotonic()
            grad_sync.finish()
            window["sync"] += time.monotonic() - synced
        grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in owned], args.clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient norm at step {step}")
        optimizer.step()
        if master is not None:
            master.sync_low()
        if args.empty_cache_every and device.type == "cuda" and (step + 1) % args.empty_cache_every == 0:
            torch.cuda.empty_cache()
        if args.perf_profile and device.type == "cuda":
            torch.cuda.synchronize(device)
        completed = step + 1
        step += 1
        round_end = diloco is not None and (completed % args.inner_steps == 0 or completed == args.steps)
        if round_end and not external:
            outer = diloco.outer_step(completed)
            if master is not None:
                master.sync_low()
            emit({"phase": "outer", **outer})
        free, peak = memory_consensus(device, world)
        watts = gpu_watts(device)
        if world > 1:
            reduced = torch.tensor([step_loss, supervised, context, flops, watts], dtype=torch.float64, device=device)
            dist.all_reduce(reduced)
            step_loss, supervised, context, flops, watts = (float(reduced[0]) / world, float(reduced[1]),
                                                           float(reduced[2]), float(reduced[3]),
                                                           float(reduced[4]) / world)
        tokens_seen += supervised
        window["steps"] += 1
        step_seconds = time.monotonic() - step_started
        window["flops"] += flops
        window["useful"] += step_parts["useful"] * world   # rank-local; logged per GPU like flops
        window["watts"] += watts
        window["seconds"] += step_seconds
        if args.perf_profile and profiler is None and completed > start_step + 1:   # step 1 compiles
            clean["seconds"].append(step_seconds)
            clean["flops"] += step_parts["total"]
            clean["useful"] += step_parts["useful"]
            clean["watts"].append(power.take()["gpu_w_mean"])
        if profiler is not None:
            profiled_seconds.append(step_seconds)
            for key, value in step_parts.items():
                profiled_flops[key] = profiled_flops.get(key, 0.0) + value
            if completed == args.steps:
                profiler.__exit__(None, None, None)
                trace = output / f"perf-trace-rank{rank}.json"
                profiler.export_chrome_trace(str(trace))
                n = len(profiled_seconds)
                summary = fp.summarize_trace(trace, n, profiled_seconds,
                                             {k: v / n for k, v in profiled_flops.items()}, args.peak_tflops,
                                             power.take())
                summary.update(peak_mem_gb=round(peak, 2), min_mem_available_gb=round(free, 2))
                if clean["seconds"]:
                    seconds = sum(clean["seconds"])
                    summary["unprofiled"] = {
                        "steps": len(clean["seconds"]), "step_s": round(seconds / len(clean["seconds"]), 4),
                        "tflops": round(clean["flops"] / seconds / 1e12, 2),
                        "tflops_useful": round(clean["useful"] / seconds / 1e12, 2),
                        "mfu": round(clean["flops"] / seconds / 1e12 / args.peak_tflops, 4) if args.peak_tflops else None,
                        "mfu_useful": (round(clean["useful"] / seconds / 1e12 / args.peak_tflops, 4)
                                       if args.peak_tflops else None),
                        "gpu_w_mean": round(sum(clean["watts"]) / len(clean["watts"]), 1)}
                setup = {"anchors": args.anchors, "grad_accum": args.grad_accum, "crop": args.crop,
                         "attn": args.attn, "param_dtype": args.param_dtype, "compile_layers": args.compile_layers,
                         "adamw_impl": args.adamw_impl, "deferred_metrics": args.deferred_metrics,
                         "anchors_per_chunk": args.anchors_per_chunk, "world": world, "rank": rank}
                if rank == 0:
                    fp.write_summary(summary, output, setup)
                emit({"phase": "perf_profile", **setup, **{k: v for k, v in summary.items()
                                                           if k not in ("categories", "top_kernels")}})
                power.close()
                barrier(world)
                if world > 1:
                    dist.destroy_process_group()
                return {"perf_profile": summary}
        if completed - start_step <= args.smoke_steps and start_step == 0:
            smoke.append({"step": completed, "step_s": round(step_seconds, 4), "supervised": supervised,
                          "tflops_per_gpu": round(flops / step_seconds / world / 1e12, 2),
                          "gpu_w": round(watts, 1),
                          "context_rows": context, "peak_mem_gb": round(peak, 2),
                          "min_mem_available_gb": round(free, 2)})
            emit({"phase": "smoke_step", **smoke[-1]})
            if completed == args.smoke_steps:
                summary = smoke_summary(smoke, world)
                if args.time_budget_h is not None:
                    planned = plan_steps(args, [s["step_s"] for s in smoke], eval_s, run_started, completed,
                                         device, world)
                    apply_plan(args, planned)
                    summary.update(planned_steps=args.steps, eval_every=args.eval_every,
                                   checkpoint_every=args.checkpoint_every, eval_s=round(eval_s, 1))
                emit({"phase": "smoke", **summary})
                if capture is not None and args.source_weights and args.source_max_epochs and cap_switch is None:
                    # after every item the prefetch can have scheduled (consumed + depth), identical on resume
                    cap_switch = completed * args.grad_accum + 2 * max(1, args.loader_workers) + 1
                    emit({"phase": "source_caps", **source_caps(stream, args, cap_switch, world)})
                if capture is not None and args.source_weights:
                    emit({"phase": "sources_planned", **sources_record(stream, args.steps * args.grad_accum, world)})
                if rank == 0:
                    (output / "smoke.json").write_text(json.dumps(summary, indent=2))
        if free < args.min_free_gb:
            abort_run(emit, {"reason": "low_memory", "step": completed, "min_mem_available_gb": round(free, 2),
                             "peak_mem_gb": round(peak, 2)}, world)
        window["supervised"] += supervised
        window["context"] += context
        if completed % args.log_every == 0 or completed == args.steps:
            train_eal = (step_metrics.get("eal_sum", 0.0) / step_metrics["eal_total"]
                         if step_metrics.get("eal_total") else None)
            seconds = max(window["seconds"], 1e-9)
            record = {"phase": "train", "step": completed, "objective": objective, "loss": step_loss,
                      "train_eal_rank0": train_eal, "lr": lr, "grad_norm": grad_norm, "tokens_seen": tokens_seen,
                      "step_s": round(seconds / window["steps"], 4),
                      "supervised_tok_per_s": round(window["supervised"] / seconds, 1),
                      "context_rows_per_s": round(window["context"] / seconds, 1),
                      "tflops_per_gpu": round(window["flops"] / seconds / world / 1e12, 2),
                      "mfu": (round(window["flops"] / seconds / world / 1e12 / args.peak_tflops, 4)
                              if args.peak_tflops else None),
                      "tflops_useful_per_gpu": round(window["useful"] / seconds / world / 1e12, 2),
                      "mfu_useful": (round(window["useful"] / seconds / world / 1e12 / args.peak_tflops, 4)
                                     if args.peak_tflops else None),
                      "gpu_w": round(window["watts"] / window["steps"], 1),
                      "elapsed_s": round(time.monotonic() - started, 2)}
            # rank-local timing and host/GPU telemetry (step-time investigation): rank 0's keys as before,
            # every other rank's under rank<N>
            local = {"data_wait_s": round(window["wait"] / window["steps"], 4),
                     "grad_sync_wait_s": round(window["sync"] / window["steps"], 4), **power.take(), **host.take()}
            if samples is not None:
                local["worker_read_s"] = round(window["read"] / (window["steps"] * args.grad_accum), 4)
                if loader_stats.get("takes"):
                    local["loader_ready_mean"] = round(loader_stats["ready_sum"] / loader_stats["takes"], 2)
                    local["loader_head_wait_frac"] = round(loader_stats["head_not_ready"] / loader_stats["takes"], 3)
                    loader_stats.clear()
            if window["enqueue"]:   # --deferred-metrics: CPU time to enqueue the micro-steps vs GPU tail after it
                local["step_enqueue_s"] = round(window["enqueue"] / window["steps"], 4)
                local["step_gpu_tail_s"] = round(window["tail"] / window["steps"], 4)
            ranks: list[Any] = [local]
            if world > 1:
                ranks = [None] * world
                dist.all_gather_object(ranks, local)
            suffixed = ("data_wait_s", "grad_sync_wait_s", "worker_read_s", "step_enqueue_s", "step_gpu_tail_s")
            record.update({f"{k}_rank0" if k in suffixed else k: v for k, v in ranks[0].items()})
            record.update({f"rank{r}": part for r, part in enumerate(ranks) if r})
            record.update(peak_mem_gb=round(peak, 2), min_mem_available_gb=round(free, 2))
            emit(record)
            window = dict.fromkeys(window, 0.0)
        if args.eval_every and (completed % args.eval_every == 0 or completed == args.steps):
            emit({"phase": "eval", "step": completed, "tokens_seen": tokens_seen, **run_eval()})
        if round_end and external:
            path = args.delta_out or output / "deltas" / f"step-{completed:07d}.safetensors"
            meta = {**json.loads(args.delta_meta), "trainer": TRAINER_VERSION, "config_sha256": dl.config_sha256(cfg),
                    "step_start": ((completed - 1) // args.inner_steps) * args.inner_steps, "step_end": completed,
                    "inner_steps": args.inner_steps, "tokens": tokens_seen - round_start_tokens,
                    "seed": args.seed, "world": world}
            if rank == 0:
                finished = {"delta": str(path), **diloco.write_delta(path, provenance, meta)}
                emit({"phase": "delta", **finished})
            else:
                diloco.round += 1
            barrier(world)
        if round_end:
            round_start_tokens = tokens_seen
        if completed % args.checkpoint_every == 0 or completed == args.steps or (round_end and external):
            save_checkpoint(output, completed, model, optimizer, rank, world, provenance,
                            {"tokens_seen": tokens_seen, "eval_s": eval_s, "anchors": args.anchors,
                             "grad_accum": args.grad_accum,
                             "items_done": items_start + (completed - start_step) * args.grad_accum,
                             **({"round_start_tokens": round_start_tokens} if diloco is not None else {}),
                             **({"planned_steps": args.steps} if args.time_budget_h is not None else {}),
                             **({"cap_switch_item": cap_switch} if cap_switch is not None else {})}, owned,
                            keep=args.keep_checkpoints,
                            extra={"diloco": diloco.state_dict()} if diloco is not None else None)
        if round_end and external:
            break   # one round per run: the coordinator merges, the next run starts from its weights
        if args.stop_after is not None and completed >= args.stop_after:
            barrier(world)
            if world > 1:
                dist.destroy_process_group()
            return {"stopped_at": completed}

    del samples
    power.close()
    if external:
        barrier(world)
        if world > 1:
            dist.destroy_process_group()
        return finished or {"completed_steps": step}
    result: dict[str, Any] = {"completed_steps": args.steps}
    if capture is not None and args.source_weights:
        emit({"phase": "sources", **sources_record(stream, args.steps * args.grad_accum, world)})
    if args.export:
        body = gather_body(owned)
        if rank == 0:
            export_dir = output / "export-atlas-bf16"
            receipt = export_engine_checkpoint(body, cfg, export_dir, provenance)
            result["export"] = {**receipt, "verify": verify_export(
                export_dir, {n: tuple(t.shape) for n, t in body.items()})}
            emit({"phase": "export", **result})
    barrier(world)
    if world > 1:
        dist.destroy_process_group()
    return result


if __name__ == "__main__":
    main()
