"""Feature rows for scratch training: on-disk format, Atlas packer, synthetic fixture.

Row store layout (``schema = enntity.drafter_rows/v1``)::

    <root>/manifest.json
    <root>/rows/<id>.safetensors

``manifest.json`` records the target identity, capture-layer convention and
one entry per row: ``id``, ``split`` (``train`` | ``heldout``), ``file``,
``sha256``, ``tokens`` and ``on_policy`` (the row's continuation was emitted
by the target, so source tokens equal target greedy/served tokens).

Each row file holds one target sequence (prompt + target-generated
continuation, contiguous from absolute position 0):

    hidden_states  [n, L, H]  bfloat16, or float8_e4m3fn + hidden_scale [n, L] fp32
    final_hidden   [n, H]     bfloat16 (pre- or post-final-norm per manifest)
    token_ids      [n]        int64 (int32 accepted)
    loss_mask      [n]        uint8, 1 where the position may anchor/be supervised
                              (normally the generated region only)

``RowStore.load`` returns the Speculators DFlash2 forward contract:
``hidden_states [1,n,L*H]``, ``input_ids``, ``loss_mask``,
``verifier_last_hidden_states [1,n,H]``, ``document_ids``, ``position_ids``.
"""

from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch

from .guard import check_path, guarded_load_file

SCHEMA = "enntity.drafter_rows/v1"
FP8_MAX = 448.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_fp8_rows(hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-(token, layer) amax scaling to E4M3 (halves cache bytes vs BF16)."""

    values = hidden.float()
    scale = (values.abs().amax(-1) / FP8_MAX).clamp_min(1e-12)
    return (values / scale[..., None]).to(torch.float8_e4m3fn), scale


class RowStore:
    """Bounded LRU cache of verified feature rows."""

    def __init__(self, root: str | Path, *, expect_layers: list[int], hidden_size: int,
                 dtype: torch.dtype = torch.bfloat16, max_cache_bytes: int = 512 * 1024**2):
        self.root = Path(root)
        check_path(self.root / "manifest.json", "features")
        self.manifest = json.loads((self.root / "manifest.json").read_text())
        if self.manifest.get("schema") != SCHEMA:
            raise ValueError(f"{root}: schema {self.manifest.get('schema')!r} != {SCHEMA}")
        if list(self.manifest["capture_layers"]) != list(expect_layers):
            raise ValueError(f"capture layers {self.manifest['capture_layers']} != config {expect_layers}")
        if self.manifest["hidden_size"] != hidden_size:
            raise ValueError("hidden size mismatch between features and config")
        self.final_hidden = self.manifest["final_hidden"]
        self.rows = list(self.manifest["rows"])
        if len({r["id"] for r in self.rows}) != len(self.rows):
            raise ValueError("duplicate row ids")
        self.dtype = dtype
        self.max_bytes = max_cache_bytes
        self.cache: OrderedDict[str, tuple[dict[str, torch.Tensor], int]] = OrderedDict()
        self.cached_bytes = 0
        self.verified: set[str] = set()

    def split(self, name: str) -> list[dict[str, Any]]:
        return [row for row in self.rows if row["split"] == name]

    def load(self, row: dict[str, Any]) -> dict[str, torch.Tensor]:
        key = row["id"]
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key][0]
        path = self.root / row["file"]
        if key not in self.verified:
            if _sha256(path) != row["sha256"]:
                raise ValueError(f"feature hash mismatch for {key}")
            self.verified.add(key)
        raw = guarded_load_file(path, "features")
        hidden = raw["hidden_states"]
        if hidden.dtype == torch.float8_e4m3fn:
            hidden = hidden.float() * raw["hidden_scale"].float()[..., None]
        final = raw["final_hidden"]
        if final.dtype == torch.float8_e4m3fn:
            final = final.float() * raw["final_scale"].float()[..., None]
        ids = raw["token_ids"].long()
        n = ids.numel()
        if hidden.shape[0] != n or final.shape != (n, hidden.shape[-1]):
            raise ValueError(f"row {key}: inconsistent shapes {tuple(hidden.shape)} {tuple(final.shape)}")
        if not torch.isfinite(hidden).all() or not torch.isfinite(final).all():
            raise ValueError(f"row {key}: non-finite features")
        value = {
            "hidden_states": hidden.to(self.dtype).flatten(1).unsqueeze(0),
            "input_ids": ids.unsqueeze(0),
            "loss_mask": raw["loss_mask"].bool().unsqueeze(0),
            "verifier_last_hidden_states": final.to(self.dtype).unsqueeze(0),
            "document_ids": torch.zeros((1, n), dtype=torch.long),
            "position_ids": torch.arange(n).unsqueeze(0),
        }
        nbytes = sum(t.numel() * t.element_size() for t in value.values())
        while self.cache and self.cached_bytes + nbytes > self.max_bytes:
            _, (_, removed) = self.cache.popitem(last=False)
            self.cached_bytes -= removed
        if nbytes <= self.max_bytes:
            self.cache[key] = (value, nbytes)
            self.cached_bytes += nbytes
        return value


def eligible_anchors(row_inputs: dict[str, torch.Tensor], block_size: int) -> list[int]:
    """Positions Speculators' ``select_anchors`` could draw (loss mask, not last block)."""

    mask = row_inputs["loss_mask"][0].clone()
    mask[-block_size:] = False
    return mask.nonzero().flatten().tolist()


def write_row_store(root: str | Path, rows: list[dict[str, Any]], *, capture_layers: list[int],
                    hidden_size: int, final_hidden: str, target: dict[str, Any],
                    fp8: bool = False) -> Path:
    """Write rows (dicts with tensors + id/split/on_policy) and the manifest."""

    from safetensors.torch import save_file

    root = Path(root)
    (root / "rows").mkdir(parents=True, exist_ok=True)
    records = []
    for row in rows:
        tensors = {
            "token_ids": row["token_ids"].long().contiguous(),
            "loss_mask": row["loss_mask"].to(torch.uint8).contiguous(),
        }
        if fp8:
            tensors["hidden_states"], tensors["hidden_scale"] = quantize_fp8_rows(row["hidden_states"])
            tensors["final_hidden"], tensors["final_scale"] = quantize_fp8_rows(row["final_hidden"])
        else:
            tensors["hidden_states"] = row["hidden_states"].to(torch.bfloat16).contiguous()
            tensors["final_hidden"] = row["final_hidden"].to(torch.bfloat16).contiguous()
        relative = f"rows/{row['id']}.safetensors"
        save_file(tensors, str(root / relative), metadata={"schema": SCHEMA, "row": row["id"]})
        records.append({
            "id": row["id"], "split": row["split"], "file": relative,
            "sha256": _sha256(root / relative), "tokens": int(row["token_ids"].numel()),
            "on_policy": bool(row.get("on_policy", True)),
        })
    manifest = {
        "schema": SCHEMA,
        "target": target,
        "capture_layers": capture_layers,
        "layer_indexing": "Atlas dflash_capture_rows (engine-native drafter feed); "
                          "= vLLM aux_hidden_state_layers minus 1",
        "final_hidden": final_hidden,
        "hidden_size": hidden_size,
        "hidden_dtype": "fp8_e4m3+row_scale" if fp8 else "bf16",
        "rows": records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return root


def pack_atlas_capture(capture_dir: str | Path, *, row_id: str, split: str, loss_start: int,
                       on_policy: bool = True) -> dict[str, Any]:
    """Assemble one request's ``atlas.hidden_capture.chunk/v1`` chunks into a row.

    Expects ``<capture_dir>/chunks/p<pos0>-n<rows>.{bin,json}`` for a single
    request (BF16, slot-major ``[slots, rows, hidden]``: capture layers then the
    contracted pre-final-norm hidden), contiguous from position 0.
    ``loss_start`` is the first generated-token position.
    """

    chunks = sorted(Path(capture_dir, "chunks").glob("p*-n*.json"))
    if not chunks:
        raise FileNotFoundError(f"no capture chunks under {capture_dir}")
    metas = [json.loads(p.read_text()) for p in chunks]
    metas, chunks = zip(*sorted(zip(metas, chunks), key=lambda item: item[0]["pos0"]))
    layers = metas[0]["layers"]
    hidden = metas[0]["hidden_size"]
    position = 0
    parts, ids = [], []
    for meta, sidecar in zip(metas, chunks):
        if meta.get("schema") != "atlas.hidden_capture.chunk/v1" or meta["dtype"] != "bf16":
            raise ValueError(f"unsupported chunk {sidecar}")
        if meta["layers"] != layers or meta["hidden_size"] != hidden:
            raise ValueError(f"layer/hidden mismatch in {sidecar}")
        if meta["pos0"] != position:
            raise ValueError(f"capture gap at {position}: next chunk starts at {meta['pos0']}")
        binary = sidecar.with_suffix(".bin")
        check_path(binary, "features")
        raw = torch.frombuffer(bytearray(binary.read_bytes()), dtype=torch.bfloat16)
        slots = len(layers) + 1
        parts.append(raw.view(slots, meta["rows"], hidden))
        ids.extend(meta["token_ids"])
        position += meta["rows"]
    stacked = torch.cat(parts, dim=1)  # [slots, n, H]
    token_ids = torch.tensor(ids, dtype=torch.long)
    loss_mask = torch.arange(position) >= loss_start
    return {
        "id": row_id, "split": split, "on_policy": on_policy, "token_ids": token_ids,
        "loss_mask": loss_mask, "hidden_states": stacked[:-1].permute(1, 0, 2).contiguous(),
        "final_hidden": stacked[-1].contiguous(), "capture_layers": layers,
    }


# --------------------------------------------------------------------------
# Synthetic deterministic fixture for CPU tests
# --------------------------------------------------------------------------

def synthetic_target(cfg: dict[str, Any], seed: int = 7) -> dict[str, torch.Tensor]:
    """Frozen fake target: random embeddings, orthonormal lm_head rows, unit norm."""

    vocab, hidden = cfg["vocab_size"], cfg["hidden_size"]
    if vocab > hidden:
        raise ValueError("synthetic target needs vocab <= hidden for orthonormal head rows")
    generator = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(hidden, vocab, generator=generator))
    return {
        "embed_tokens.weight": torch.randn(vocab, hidden, generator=generator).bfloat16(),
        "lm_head.weight": q.T.contiguous().bfloat16(),  # [V, H], orthonormal rows
        "norm.weight": torch.ones(hidden).bfloat16(),
    }


def write_synthetic_target_dir(root: str | Path, cfg: dict[str, Any], seed: int = 7) -> Path:
    """A miniature on-disk "target" (MIT card, sharded index) for loader tests."""

    from safetensors.torch import save_file

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tensors = synthetic_target(cfg, seed)
    keys = {"embed_tokens.weight": "model.language_model.embed_tokens.weight",
            "lm_head.weight": "lm_head.weight", "norm.weight": "model.language_model.norm.weight"}
    save_file({keys[k]: v for k, v in tensors.items()}, str(root / "model-00001-of-00001.safetensors"))
    (root / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {v: "model-00001-of-00001.safetensors" for v in keys.values()}}))
    (root / "config.json").write_text(json.dumps({
        "architectures": ["Glm5NextForConditionalGeneration"],
        "text_config": {"hidden_size": cfg["hidden_size"], "vocab_size": cfg["vocab_size"]}}))
    (root / "README.md").write_text("---\nlicense: mit\n---\nSynthetic CPU fixture target.\n")
    return root


def synthetic_rows(cfg: dict[str, Any], *, train_rows: int, heldout_rows: int, length: int,
                   seed: int = 11, noise: float = 0.1) -> list[dict[str, Any]]:
    """Deterministic arithmetic-progression sequences the target "generates".

    Row r: ``tok_t = (start_r + t * stride_r) mod (V-1)`` (the mask token V-1
    never occurs).  Captured layer ``l`` hidden at t is ``A_l[tok_t] +
    B_l[stride] + noise``; the pre-norm final hidden at t is a scaled lm_head row of
    ``tok_{t+1}``, so the frozen target head predicts the true next token.  A
    drafter must read the stride from context and the anchor token from its
    embedding to propose ``anchor + k*stride``: learnable but not trivial.
    Held-out rows use start/stride combinations not seen in training.
    """

    vocab, hidden = cfg["vocab_size"], cfg["hidden_size"]
    layers = len(cfg["target_layer_ids"])
    target = synthetic_target(cfg)
    head = target["lm_head.weight"].float()
    generator = torch.Generator().manual_seed(seed)
    strides = [1, 2, 3, 5, 7]
    token_table = torch.randn(layers, vocab, hidden, generator=generator)
    stride_table = torch.randn(layers, max(strides) + 1, hidden, generator=generator)
    modulus = vocab - 1
    rows = []
    for index in range(train_rows + heldout_rows):
        heldout = index >= train_rows
        stride = strides[index % len(strides)]
        start = (index * 17 + (31 if heldout else 0)) % modulus
        tokens = (start + torch.arange(length + 1) * stride) % modulus
        layer_hidden = token_table[:, tokens[:-1]] + stride_table[:, stride][:, None, :]
        layer_hidden = layer_hidden + noise * torch.randn(layer_hidden.shape, generator=generator)
        final = 3.0 * head[tokens[1:]] + 0.01 * torch.randn(length, hidden, generator=generator)
        loss_mask = torch.arange(length) >= 4
        rows.append({
            "id": f"{'heldout' if heldout else 'train'}-{index:03d}",
            "split": "heldout" if heldout else "train",
            "on_policy": True,
            "token_ids": tokens[:-1].clone(),
            "loss_mask": loss_mask,
            "hidden_states": layer_hidden.permute(1, 0, 2).contiguous(),
            "final_hidden": final,
        })
    return rows


__all__ = [
    "RowStore", "SCHEMA", "eligible_anchors", "pack_atlas_capture", "quantize_fp8_rows",
    "synthetic_rows", "synthetic_target", "write_row_store", "write_synthetic_target_dir",
]
