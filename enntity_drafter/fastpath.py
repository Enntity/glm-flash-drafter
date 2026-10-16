# SPDX-License-Identifier: Apache-2.0
# Portions are derived from vllm-project/speculators (Apache-2.0); see NOTICE.
"""Round-2 performance path for the scratch drafter trainer (every piece is opt-in).

Round 1 spent ~76% of a micro-step outside GEMMs (docs/training.md): per-micro fp32->bf16
weight casts and fp32 grad casts, unfused fp32 norm/conv/rotary/residual kernels,
and a dense-masked SDPA that computes every (block query, context key) pair
although each 8-token anchor block sees at most ``W + 8`` keys.  The pieces here
remove that work without changing what is computed:

``packed_sdpa`` attention (``--attn packed_sdpa``)
    Exact block-sparse attention for the DFlash block-vs-context mask.  Anchors
    are sorted and grouped ``anchors_per_chunk`` at a time; each chunk gathers
    only the context keys its windows can reach (``[min a - W, max a)``) plus its
    own block keys, and runs one masked SDPA per chunk with the GQA query heads
    folded into the query axis (no K/V repeat).  The mask is exactly upstream's
    ``create_anchor_block_mask_mod`` (same document, ``a - W <= p < a``, own block,
    causal option); keys outside the gathered window are masked in upstream too.
    Works on CPU (tests, backward included) and CUDA.

``MasterWeights`` (``--param-dtype bf16``)
    Linear weights are stored in bf16 for compute; an fp32 master copy is what
    the optimizer, clipping and checkpoints see.  Autocast's own cast of an fp32
    weight produces the same bf16 tensor (round to nearest even), so forward and
    per-micro gradients are bit-identical to round 1; the post-accumulate hook
    adds each micro's bf16 gradient into the fp32 master gradient exactly as
    autograd's AccumulateGrad did for the fp32 leaf.  Norm gains, conv base
    kernels and the selector codebooks stay fp32 (small; they keep the round-1
    activation dtypes).  Removes ~1.2 B params x (fp32 read + bf16 write) of
    casts per micro-step and the fp32 cast of every weight gradient.

``compile_layers`` (``--compile-layers``)
    ``torch.compile(dynamic=True)`` per decoder layer (fuses RMSNorm, the grouped
    dynamic conv, rotary, residual and casts).  The upstream ``forward`` wrapper
    (``conditional_torch_compile``, whole forward incl. the fused Triton losses)
    is unwrapped so the loss kernels stay eager -- that is what broke round-1's
    whole-forward compile.  No loss replacement is needed.

``PowerSampler``
    Background NVML sampler: mean GPU watts over an interval (the round-1
    ``gpu_w`` was one sample at the step boundary, biased low).

``ProfileSummary`` (``--perf-profile N``)
    Kernel-time breakdown of N profiled steps by category with achieved TFLOPS.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

PACKED_ATTN = "packed_sdpa"
FLEX_ATTN = "flex_compiled"


# --------------------------------------------------------------------------
# Exact block-sparse attention for the anchor-block mask
# --------------------------------------------------------------------------

class PackedAttentionMask(NamedTuple):
    """Everything the packed attention needs; tensors only (compile-friendly).

    q_rows    [C * Lq]       original query row for every packed query slot
    key_index [C, Lk]        row of the [context | blocks] key sequence per packed key slot
    mask      [C, 1, Lq, Lk] bool, True = attend
    out_rows  [Q]            packed slot of every original query row
    """

    q_rows: torch.Tensor
    key_index: torch.Tensor
    mask: torch.Tensor
    out_rows: torch.Tensor


@torch.no_grad()
def build_packed_mask(document_ids: torch.Tensor, total_seq_len: int, anchor_positions: torch.Tensor,
                      block_size: int, sliding_window: int | None = None,
                      sliding_window_non_causal: bool = False, anchors_per_chunk: int = 16,
                      key_multiple: int = 16) -> PackedAttentionMask:
    """Packed form of ``speculators.models.dflash.attention.create_anchor_block_mask_mod``."""

    device = anchor_positions.device
    doc = document_ids.reshape(-1).to(device)
    anchors = anchor_positions.reshape(-1).to(device=device, dtype=torch.long)
    n, B, G = anchors.numel(), block_size, anchors_per_chunk
    non_causal = sliding_window is None or sliding_window_non_causal
    order = torch.argsort(anchors, stable=True)
    chunks = max(1, math.ceil(n / G))
    padded = chunks * G
    # Pad with the last sorted anchor: keeps the last chunk's window tight; its outputs are dropped.
    sorted_idx = torch.cat([order, order[-1:].expand(padded - n)]) if padded > n else order
    a = anchors[sorted_idx].view(chunks, G)                                # [C, G]
    if sliding_window is not None:
        lo = (a.min(1).values - sliding_window).clamp_min(0)
    else:
        lo = torch.zeros(chunks, dtype=torch.long, device=device)
    hi = a.max(1).values                                                   # context keys are p < a
    span = int((hi - lo).max().clamp_min(1))
    Lq = G * B
    lk = span + Lq
    lk = -(-lk // key_multiple) * key_multiple
    span = lk - Lq                                                         # padded context slots are masked

    # Context part.
    pos = lo[:, None] + torch.arange(span, device=device)                  # [C, span]
    pos_c = pos.clamp(max=total_seq_len - 1)
    a_q = a.repeat_interleave(B, dim=1)                                    # [C, Lq] anchor of each query slot
    q_doc = doc[a_q]
    visible = (pos[:, None, :] < a_q[:, :, None]) & (pos[:, None, :] < total_seq_len)
    if sliding_window is not None:
        visible &= pos[:, None, :] >= (a_q - sliding_window)[:, :, None]
    visible &= doc[pos_c][:, None, :] == q_doc[:, :, None]
    visible &= (q_doc != -1)[:, :, None]

    # Own-block part: keys total_seq_len + k * B + j for the chunk's anchors k (original order).
    j = torch.arange(Lq, device=device)
    block_keys = total_seq_len + sorted_idx.view(chunks, G).repeat_interleave(B, dim=1) * B + j % B
    same = (j[:, None] // B) == (j[None, :] // B)
    if not non_causal:
        same &= j[None, :] <= j[:, None]
    mask = torch.cat([visible, same.expand(chunks, Lq, Lq)], dim=2).unsqueeze(1)
    key_index = torch.cat([pos_c, block_keys], dim=1)

    q_rows = (sorted_idx.view(chunks, G).repeat_interleave(B, dim=1) * B + j % B).reshape(-1)
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(n, device=device)
    out_rows = (inverse[:, None] * B + torch.arange(B, device=device)).reshape(-1)
    return PackedAttentionMask(q_rows, key_index, mask, out_rows)


def packed_attention_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                             attention_mask: PackedAttentionMask, scaling: float | None = None,
                             dropout: float = 0.0, **_kwargs: Any) -> tuple[torch.Tensor, None]:
    """HF attention-interface function: query [1, H, Q, d], key/value [1, KVH, T + Q, d] -> [1, Q, H, d]."""

    if not isinstance(attention_mask, PackedAttentionMask):
        raise TypeError("packed_sdpa needs the PackedAttentionMask built by install_packed_attention")
    if query.shape[0] != 1:
        raise ValueError("packed_sdpa expects batch 1 (the DFlash packed sequence)")
    H, KVH, d = query.shape[1], key.shape[1], query.shape[-1]
    g = H // KVH
    q_rows, key_index, mask, out_rows = attention_mask
    C, Lk = key_index.shape
    Lq = mask.shape[2]
    # Row gathers in the [sequence, heads, d] layout the projections produced.  Context keys are shared by
    # overlapping chunks, so their gradient is a sum over chunks: gather from fp32 so that sum (index_add in
    # backward) accumulates in fp32, as the dense kernel's dK/dV accumulation did, not in bf16.
    flat = key_index.reshape(-1)
    k = key[0].transpose(0, 1).float().index_select(0, flat).to(key.dtype).view(C, Lk, KVH, d).transpose(1, 2)
    v = value[0].transpose(0, 1).float().index_select(0, flat).to(value.dtype).view(C, Lk, KVH, d).transpose(1, 2)
    # GQA: the g query heads of one KV head are stacked along the query axis (no K/V repeat).
    q = (query[0].transpose(0, 1).index_select(0, q_rows).view(C, Lq, KVH, g, d)
         .permute(0, 2, 3, 1, 4).reshape(C, KVH, g * Lq, d))
    m = mask.unsqueeze(2).expand(C, 1, g, Lq, Lk).reshape(C, 1, g * Lq, Lk)
    out = F.scaled_dot_product_attention(q, k, v, attn_mask=m, dropout_p=dropout, scale=scaling)
    out = out.view(C, KVH, g, Lq, d).permute(0, 3, 1, 2, 4).reshape(C * Lq, H, d)
    return out.index_select(0, out_rows).unsqueeze(0), None


def register_packed_attention() -> None:
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(PACKED_ATTN, packed_attention_forward)


def install_packed_attention(model: nn.Module, anchors_per_chunk: int = 16) -> None:
    """Route the model's mask construction to :func:`build_packed_mask` (instance override,
    so upstream ``_build_attention_mask`` and the fixed-anchor override both use it)."""

    register_packed_attention()

    def _create_attention_mask(document_ids, total_seq_len, anchor_positions, device, sliding_window=None,
                               sliding_window_non_causal=False):
        return build_packed_mask(document_ids.to(device), total_seq_len, anchor_positions.to(device),
                                 model.block_size, sliding_window, sliding_window_non_causal,
                                 anchors_per_chunk=anchors_per_chunk)

    model._create_attention_mask = _create_attention_mask


def flex_compiled_forward(module: nn.Module, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                          attention_mask: Any, scaling: float | None = None, **_kwargs: Any) -> tuple[torch.Tensor, None]:
    """Upstream ``flex_attention_forward`` (same block mask) for use inside compiled layers.

    ``FORCE_USE_FLEX_ATTENTION``: rows with few anchors (< 16 -> < 128 block queries) otherwise
    route inductor to the flex *decoding* template, which has no valid config for training shapes
    on GB10 (NoValidChoicesError, 2026-10-15)."""

    from torch.nn.attention.flex_attention import flex_attention

    out = flex_attention(query.contiguous(), key.contiguous(), value.contiguous(), block_mask=attention_mask,
                         enable_gqa=query.shape[1] != key.shape[1], scale=scaling,
                         kernel_options={"FORCE_USE_FLEX_ATTENTION": True})
    return out.transpose(1, 2).contiguous(), None


def register_flex_attention() -> None:
    from transformers.modeling_utils import AttentionInterface

    AttentionInterface.register(FLEX_ATTN, flex_compiled_forward)


def install_flex_attention(model: nn.Module) -> None:
    """Block masks from upstream's compiled ``create_block_mask`` (what ``simple_flex_attention`` uses)."""

    from speculators.models.dflash.core import _compiled_create_block_mask

    register_flex_attention()
    model._create_mask_fn = _compiled_create_block_mask


def attended_keys(document_ids: torch.Tensor, anchors: list[int], window: int | None, block_size: int) -> int:
    """Number of (query, key) pairs the anchor-block mask allows (non-causal sliding layers).

    Used for the "useful FLOPs" MFU; ``micro_flops`` counts dense attention as round 1 computed it."""

    doc = document_ids.reshape(-1).tolist()
    starts: dict[int, int] = {}
    for index, value in enumerate(doc):
        starts.setdefault(value, index)
    total = 0
    for a in anchors:
        lo = starts[doc[a]]
        if window is not None:
            lo = max(lo, a - window)
        total += (a - lo) + block_size
    return total * block_size


# --------------------------------------------------------------------------
# Per-layer compile
# --------------------------------------------------------------------------

def unwrap_upstream_forward() -> None:
    """Undo ``conditional_torch_compile`` on the DFlash forwards (applied only when CUDA exists)."""

    from speculators.models.dflash.core import DFlashDraftModel
    from speculators.models.dflash2.core import DFlash2DraftModel

    for cls in (DFlashDraftModel, DFlash2DraftModel):
        forward = cls.__dict__.get("forward")
        original = getattr(forward, "_torchdynamo_orig_callable", None)
        if original is not None:
            cls.forward = original


def compile_layers(model: nn.Module, mode: str | None = None) -> None:
    """Compile each decoder layer in place (parameter names unchanged)."""

    import torch._dynamo

    torch._dynamo.config.disable = False          # the launcher pins TORCH_COMPILE_DISABLE=1 for round 1
    torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
    # 512 anchors x 8 = 4096 block queries == hidden_size: duck shaping would tie the query length
    # to the (static) hidden size and specialise the first graph on it.
    torch.fx.experimental._config.use_duck_shape = False
    unwrap_upstream_forward()
    for layer in model.layers:
        layer.compile(dynamic=True, mode=mode)


# --------------------------------------------------------------------------
# bf16 compute weights with an fp32 master
# --------------------------------------------------------------------------

def is_master_weight(name: str, parameter: nn.Parameter) -> bool:
    """Linear weights only (2-D, not the embedding-like selector codebooks)."""

    return parameter.ndim == 2 and name.endswith(".weight") and "codebook" not in name


class MasterWeights:
    """fp32 masters for bf16 Linear weights; everything else is its own master.

    ``named`` is the optimizer's view: (name, fp32 tensor) in ``trainable`` order,
    so optimizer state and checkpoints have the same layout as round 1.
    """

    def __init__(self, named_params: list[tuple[str, nn.Parameter]], dtype: torch.dtype = torch.bfloat16):
        self.named: list[tuple[str, torch.Tensor]] = []
        self.low: list[nn.Parameter] = []
        self.high: list[torch.Tensor] = []
        self.owner: dict[int, torch.Tensor] = {}
        for name, parameter in named_params:
            if is_master_weight(name, parameter) and parameter.dtype != dtype:
                master = parameter.detach().clone()
                master.requires_grad_(True)
                parameter.data = parameter.data.to(dtype)
                parameter.register_post_accumulate_grad_hook(self._accumulate)
                self.low.append(parameter)
                self.high.append(master)
                self.owner[id(parameter)] = master
                self.named.append((name, master))
            else:
                self.owner[id(parameter)] = parameter
                self.named.append((name, parameter))

    def _accumulate(self, parameter: nn.Parameter) -> None:
        master = self.owner[id(parameter)]
        if master.grad is None:
            master.grad = parameter.grad.to(master.dtype)
        else:
            master.grad.add_(parameter.grad)
        parameter.grad = None

    def owner_of(self, parameter: nn.Parameter) -> torch.Tensor:
        return self.owner[id(parameter)]

    @torch.no_grad()
    def sync_low(self) -> None:
        """Copy masters into the bf16 compute weights (after each optimizer step / resume)."""

        if self.low:
            torch._foreach_copy_(self.low, self.high)


# --------------------------------------------------------------------------
# Power sampling
# --------------------------------------------------------------------------

class PowerSampler:
    """GPU telemetry over an interval, sampled every ``period`` s on a daemon thread: NVML power (mean/median),
    and -- for the step-time investigation (round 2b) -- SM clock, utilization, temperature and the clock event
    (throttle) reasons seen, each where NVML reports it on this device."""

    def __init__(self, device: torch.device, period: float = 0.05):
        self.device, self.period = device, period
        self.samples: list[tuple[float, ...]] = []
        self.reasons = 0
        self.lock = threading.Lock()
        self.enabled = device.type == "cuda"
        self.nvml = None
        if self.enabled:
            try:
                torch.cuda.power_draw(device)
            except Exception:  # noqa: BLE001 - NVML missing: not reported
                self.enabled = False
        if self.enabled:
            try:
                import pynvml

                pynvml.nvmlInit()
                self.nvml = (pynvml, pynvml.nvmlDeviceGetHandleByIndex(device.index or 0))
            except Exception:  # noqa: BLE001
                self.nvml = None
            self.stop = threading.Event()
            threading.Thread(target=self._run, daemon=True, name="power-sampler").start()

    def _probe(self) -> tuple[float, ...]:
        nan = float("nan")
        watts = torch.cuda.power_draw(self.device) / 1000.0
        clock = util = temp = nan
        if self.nvml is not None:
            nvml, handle = self.nvml
            for name, call in (("clock", lambda: nvml.nvmlDeviceGetClockInfo(handle, nvml.NVML_CLOCK_SM)),
                               ("util", lambda: nvml.nvmlDeviceGetUtilizationRates(handle).gpu),
                               ("temp", lambda: nvml.nvmlDeviceGetTemperature(handle, nvml.NVML_TEMPERATURE_GPU))):
                try:
                    value = float(call())
                except Exception:  # noqa: BLE001 - not supported on this device
                    continue
                if name == "clock":
                    clock = value
                elif name == "util":
                    util = value
                else:
                    temp = value
            try:
                getter = getattr(nvml, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
                    nvml.nvmlDeviceGetCurrentClocksThrottleReasons
                self.reasons |= int(getter(handle))
            except Exception:  # noqa: BLE001
                pass
        return watts, clock, util, temp

    def _run(self) -> None:
        while not self.stop.wait(self.period):
            try:
                sample = self._probe()
            except Exception:  # noqa: BLE001
                continue
            with self.lock:
                self.samples.append(sample)

    def take(self) -> dict[str, float]:
        """Means (and the power median) since the last take (and reset)."""

        with self.lock:
            samples, self.samples = self.samples, []
            reasons, self.reasons = self.reasons, 0
        if not samples:
            return {"gpu_w_mean": 0.0, "gpu_w_median": 0.0, "gpu_w_samples": 0}
        watts = sorted(s[0] for s in samples)
        out = {"gpu_w_mean": round(sum(watts) / len(watts), 1), "gpu_w_median": round(watts[len(watts) // 2], 1),
               "gpu_w_samples": len(samples)}
        for k, key in ((1, "gpu_sm_mhz"), (2, "gpu_util_pct"), (3, "gpu_temp_c")):
            values = [s[k] for s in samples if s[k] == s[k]]
            if values:
                out[key] = round(sum(values) / len(values), 1)
                if k == 1:
                    out["gpu_sm_mhz_min"] = min(values)
        if self.nvml is not None:
            out["gpu_clock_event_reasons"] = reasons
        return out

    def close(self) -> None:
        if self.enabled:
            self.stop.set()


_VMSTAT = ("pgmajfault", "workingset_refault_file", "compact_stall", "compact_fail", "pgscan_direct",
           "pgscan_kswapd", "allocstall_normal", "allocstall_movable")


class HostStats:
    """Host and process counters per log window (Linux /proc; nothing elsewhere), for the round-2b step-time
    investigation: host CPU busy/iowait/steal, load, this process's CPU (all threads) and the calling (main)
    thread's CPU, its involuntary context switches and current core, RSS, page cache, MemAvailable, major
    faults, page-cache refaults, compaction/reclaim stalls, and the CUDA caching allocator's state."""

    def __init__(self, device: torch.device):
        self.device = device
        self.linux = Path("/proc/stat").is_file()
        self.prev = self._read()

    @staticmethod
    def _kv(path: str) -> dict[str, str]:
        out = {}
        try:
            for line in Path(path).read_text().splitlines():
                key, _, value = line.partition(":") if ":" in line else line.partition(" ")
                out[key.strip()] = value.strip()
        except OSError:
            pass
        return out

    def _read(self) -> dict[str, float]:
        import os

        times = os.times()
        raw: dict[str, float] = {"wall": time.monotonic(), "proc_cpu": times.user + times.system,
                                 "main_cpu": time.thread_time()}
        if self.linux:
            cpu = [float(x) for x in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
            raw.update(cpu_total=sum(cpu[:8]), cpu_idle=cpu[3], cpu_iowait=cpu[4], cpu_steal=cpu[7])
            status = self._kv("/proc/thread-self/status")
            raw["main_nonvol_ctx"] = float(status.get("nonvoluntary_ctxt_switches", 0))
            vm = self._kv("/proc/vmstat")
            for key in _VMSTAT:
                raw[key] = float(vm.get(key, 0) or 0)
        if self.device.type == "cuda":
            stats = torch.cuda.memory_stats(self.device)
            for key in ("num_alloc_retries", "num_device_alloc", "num_device_free"):
                raw[key] = float(stats.get(key, 0))
        return raw

    def take(self) -> dict[str, float]:
        now = self._read()
        prev, self.prev = self.prev, now
        wall = max(now["wall"] - prev["wall"], 1e-9)
        out: dict[str, float] = {"proc_cpu_cores": round((now["proc_cpu"] - prev["proc_cpu"]) / wall, 2),
                                 "main_thread_busy": round((now["main_cpu"] - prev["main_cpu"]) / wall, 3)}
        if self.linux:
            total = max(now["cpu_total"] - prev["cpu_total"], 1e-9)
            out.update(host_cpu_busy_pct=round(100 * (1 - (now["cpu_idle"] - prev["cpu_idle"]
                                                            + now["cpu_iowait"] - prev["cpu_iowait"]) / total), 1),
                       host_iowait_pct=round(100 * (now["cpu_iowait"] - prev["cpu_iowait"]) / total, 2),
                       host_steal_pct=round(100 * (now["cpu_steal"] - prev["cpu_steal"]) / total, 2),
                       main_nonvol_ctx_per_s=round((now["main_nonvol_ctx"] - prev["main_nonvol_ctx"]) / wall, 1))
            for key in _VMSTAT:
                out[f"vm_{key}_per_s"] = round((now[key] - prev[key]) / wall, 1)
            try:
                out["loadavg1"] = float(Path("/proc/loadavg").read_text().split()[0])
                out["main_core"] = int(Path("/proc/thread-self/stat").read_text().rsplit(")", 1)[1].split()[36])
                status = self._kv("/proc/self/status")
                out["rss_gb"] = round(float(status["VmRSS"].split()[0]) / 1024**2, 2)
                mem = self._kv("/proc/meminfo")
                for key, name in (("Cached", "page_cache_gb"), ("MemAvailable", "mem_available_gb"),
                                  ("Dirty", "dirty_gb")):
                    out[name] = round(float(mem[key].split()[0]) / 1024**2, 2)
            except (OSError, KeyError, ValueError, IndexError):
                pass
        if self.device.type == "cuda":
            stats = torch.cuda.memory_stats(self.device)
            out.update(alloc_gb=round(stats.get("allocated_bytes.all.current", 0) / 1e9, 2),
                       reserved_gb=round(stats.get("reserved_bytes.all.current", 0) / 1e9, 2),
                       inactive_split_gb=round(stats.get("inactive_split_bytes.all.current", 0) / 1e9, 2),
                       segments=int(stats.get("segment.all.current", 0)))
            for key in ("num_alloc_retries", "num_device_alloc", "num_device_free"):
                out[key] = int(now[key] - prev[key])
        return out


# --------------------------------------------------------------------------
# Profiler summary
# --------------------------------------------------------------------------

#: First match wins.  Inductor kernel names: triton_poi (pointwise), triton_red /
#: triton_per (reductions), triton_tem (templates, i.e. GEMMs).
CATEGORIES: list[tuple[str, str]] = [
    ("nccl", r"nccl"),
    ("loss (fused Triton KL/TV)", r"loss_(forward|backward)_kernel"),
    ("attention", r"fmha|flash|attention|attn|flex|sdpa|cutlassF|cutlassB"),
    ("gemm", r"gemm|nvjet|cublas|xmma|cutlass|matmul|triton_tem|_tn_|_nt_|_nn_|splitk|wgrad|dgrad"),
    ("optimizer / foreach", r"adam|multi_tensor|foreach"),
    ("compiled pointwise (inductor)", r"triton_poi"),
    ("compiled reductions (inductor)", r"triton_red|triton_per"),
    ("reductions / softmax / topk / sort", r"reduce|softmax|norm|argmax|topk|sort|scan|cumsum|logsumexp|radix|max"),
    ("copies / gather / scatter / fill", r"copy|cat|index|gather|scatter|fill|memcpy|memset|Memcpy|Memset|put"),
    ("eager elementwise", r"elementwise|vectorized|unrolled|pointwise|Functor|kernel"),
]


def categorize(name: str) -> str:
    lowered = name.lower()
    for category, pattern in CATEGORIES:
        if re.search(pattern.lower(), lowered):
            return category
    return "other"


def _union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if stop <= end:
            continue
        total += stop - max(start, end)
        end = stop
    return total


def summarize_trace(trace_path: Path, steps: int, step_seconds: list[float], flops: dict[str, float],
                    peak_tflops: float | None, power: dict[str, float] | None = None) -> dict[str, Any]:
    """Category table from a chrome trace (device kernels, or CPU ops when there are none).

    ``flops`` holds per-step analytic FLOPs: ``total`` (round-1 dense formula),
    ``useful`` (attention counted on attended pairs only), ``gemm``, ``attention_dense``,
    ``attention_useful``.
    """

    events = json.loads(trace_path.read_text())
    events = events.get("traceEvents", events) if isinstance(events, dict) else events
    device = [e for e in events if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    source = "device kernels"
    if not device:  # CPU run: top-level aten ops (approximate, overlapping CPU time)
        device = [e for e in events if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
        source = "cpu ops (no device kernels; overlapping, approximate)"
    by_category: dict[str, float] = {}
    by_kernel: dict[str, list[float]] = {}
    intervals = []
    for event in device:
        duration = float(event.get("dur", 0.0)) / 1e3  # us -> ms
        name = event.get("name", "?")
        category = categorize(name)
        by_category[category] = by_category.get(category, 0.0) + duration
        entry = by_kernel.setdefault(name, [0.0, 0])
        entry[0] += duration
        entry[1] += 1
        start = float(event.get("ts", 0.0)) / 1e3
        intervals.append((start, start + duration))
    step_ms = 1e3 * sum(step_seconds) / max(1, len(step_seconds))
    busy_ms = _union_length(intervals) / steps
    rows = []
    for category, total in sorted(by_category.items(), key=lambda kv: -kv[1]):
        ms = total / steps
        row = {"category": category, "ms_per_step": round(ms, 2), "pct_of_step": round(100 * ms / step_ms, 1)}
        if category == "gemm" and flops.get("gemm"):
            row["tflops"] = round(flops["gemm"] / (ms / 1e3) / 1e12, 1)
        if category == "attention" and flops.get("attention_useful"):
            row["tflops_useful"] = round(flops["attention_useful"] / (ms / 1e3) / 1e12, 1)
            row["tflops_dense"] = round(flops["attention_dense"] / (ms / 1e3) / 1e12, 1)
        rows.append(row)
    idle = max(0.0, step_ms - busy_ms)
    rows.append({"category": "device idle (launch gaps, CPU, syncs)", "ms_per_step": round(idle, 2),
                 "pct_of_step": round(100 * idle / step_ms, 1)})
    achieved = flops["total"] / (step_ms / 1e3) / 1e12
    useful = flops.get("useful", flops["total"]) / (step_ms / 1e3) / 1e12
    summary = {
        "source": source, "steps": steps, "step_ms": round(step_ms, 1), "busy_ms": round(busy_ms, 1),
        "tflops": round(achieved, 2), "tflops_useful": round(useful, 2),
        "mfu": round(achieved / peak_tflops, 4) if peak_tflops else None,
        "mfu_useful": round(useful / peak_tflops, 4) if peak_tflops else None,
        "flops_per_step": flops, "categories": rows,
        "top_kernels": [{"name": name[:160], "ms_per_step": round(total / steps, 2), "calls_per_step": count / steps,
                         "category": categorize(name)}
                        for name, (total, count) in sorted(by_kernel.items(), key=lambda kv: -kv[1][0])[:30]],
    }
    if power:
        summary.update(power)
    return summary


def write_summary(summary: dict[str, Any], output: Path, extra: dict[str, Any] | None = None) -> Path:
    payload = {**(extra or {}), **summary}
    (output / "perf-profile.json").write_text(json.dumps(payload, indent=2))
    lines = [f"# perf profile ({summary['steps']} steps, {summary['source']})", "",
             f"step {summary['step_ms']} ms; device busy {summary['busy_ms']} ms; "
             f"{summary['tflops']} TFLOPS (round-1 dense formula), {summary['tflops_useful']} useful; "
             f"MFU {summary['mfu']} / useful {summary['mfu_useful']}"
             + (f"; GPU {summary.get('gpu_w_mean')} W mean" if summary.get("gpu_w_mean") else ""),
             (f"unprofiled: {summary['unprofiled']}" if summary.get("unprofiled") else ""), "",
             "| category | ms/step | % of step | TFLOPS |", "|---|---|---|---|"]
    for row in summary["categories"]:
        tflops = row.get("tflops", "")
        if "tflops_useful" in row:
            tflops = f"{row['tflops_useful']} useful / {row['tflops_dense']} dense"
        lines.append(f"| {row['category']} | {row['ms_per_step']} | {row['pct_of_step']} | {tflops} |")
    lines += ["", "| top kernels | ms/step | calls/step | category |", "|---|---|---|---|"]
    for row in summary["top_kernels"]:
        lines.append(f"| `{row['name'][:90]}` | {row['ms_per_step']} | {row['calls_per_step']:.1f} | {row['category']} |")
    path = output / "perf-profile.md"
    path.write_text("\n".join(lines) + "\n")
    return path


__all__ = [
    "FLEX_ATTN", "MasterWeights", "PACKED_ATTN", "install_flex_attention", "register_flex_attention", "PackedAttentionMask", "PowerSampler", "attended_keys", "build_packed_mask",
    "categorize", "compile_layers", "install_packed_attention", "is_master_weight", "packed_attention_forward",
    "register_packed_attention", "summarize_trace", "unwrap_upstream_forward", "write_summary",
]
