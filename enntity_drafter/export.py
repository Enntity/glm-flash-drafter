"""Export a trained scratch drafter in the Atlas DFlash drafter format.

Atlas (``crates/spark-model/src/weight_loader/dflash_loader.rs``) reads a
safetensors store with bare keys (no ``model.`` prefix): BF16 tensors are used
as-is (Atlas builds its own NVFP4 copies at load); ``embed_tokens``/``lm_head``
are absent and filled from the target.  Key names and shapes are an
architecture fact of the Apache-2.0 DFlash2 code, so the export is BF16 with
the 81 body keys plus a ``config.json`` generated from *our* config (no file
is copied from any other drafter checkpoint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .guard import SCRATCH_STAMP_KEY, guarded_load_file, guarded_safe_open

def engine_config(cfg: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    layers = cfg["num_hidden_layers"]
    config = {
        "architectures": ["DFlash2DraftModel"],
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": None,
        "dflash_config": {
            "block_size": cfg["block_size"],
            "conv_group_size": cfg["conv_group_size"],
            "conv_kernel_size": cfg["conv_kernel_size"],
            "mask_token_id": cfg["mask_token_id"],
            "selector_rank": cfg["selector_rank"],
            "selector_top_k": cfg["selector_top_k"],
            "target_layer_ids": list(cfg["target_layer_ids"]),
        },
        "dtype": "bfloat16",
        "eos_token_id": cfg.get("eos_token_id"),
        "head_dim": cfg["head_dim"],
        "hidden_act": "silu",
        "hidden_size": cfg["hidden_size"],
        "initializer_range": cfg.get("initializer_range", 0.02),
        "intermediate_size": cfg["intermediate_size"],
        "is_causal": False,
        "layer_types": ["sliding_attention"] * layers,
        "max_position_embeddings": cfg.get("max_position_embeddings", 1048576),
        "max_window_layers": layers,
        "model_type": "qwen3",
        "num_attention_heads": cfg["num_attention_heads"],
        "num_hidden_layers": layers,
        "num_key_value_heads": cfg["num_key_value_heads"],
        "num_target_layers": cfg.get("num_target_layers"),
        "pad_token_id": cfg.get("pad_token_id"),
        "rms_norm_eps": cfg["rms_norm_eps"],
        "rope_parameters": {"rope_theta": float(cfg["rope_theta"]), "rope_type": "default"},
        "sliding_window": cfg["sliding_window"],
        "tie_word_embeddings": False,
        "use_cache": False,
        "use_sliding_window": True,
        "vocab_size": cfg["vocab_size"],
        SCRATCH_STAMP_KEY: provenance,
    }
    return config


def export_engine_checkpoint(body: dict[str, torch.Tensor], cfg: dict[str, Any], output: str | Path,
                             provenance: dict[str, Any]) -> dict[str, Any]:
    """Write ``model.safetensors`` + ``config.json`` + ``PROVENANCE.json``."""

    from safetensors.torch import save_file

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    tensors = {name: t.detach().to("cpu", torch.bfloat16).contiguous() for name, t in sorted(body.items())}
    if not all(torch.isfinite(t).all() for t in tensors.values()):
        raise FloatingPointError("non-finite tensor in export")
    metadata = {"format": "pt", SCRATCH_STAMP_KEY: json.dumps(provenance, sort_keys=True)}
    save_file(tensors, str(output / "model.safetensors"), metadata=metadata)
    (output / "config.json").write_text(json.dumps(engine_config(cfg, provenance), indent=2))
    digest = hashlib.sha256((output / "model.safetensors").read_bytes()).hexdigest()
    receipt = {**provenance, "model_sha256": digest, "body_keys": len(body), "format": "atlas-dflash2-bf16"}
    (output / "PROVENANCE.json").write_text(json.dumps(receipt, indent=2, sort_keys=True))
    return receipt


def verify_export(output: str | Path, expected: dict[str, tuple[int, ...]]) -> dict[str, Any]:
    """Reload through the guard and check the engine contract."""

    output = Path(output)
    tensors = guarded_load_file(output / "model.safetensors", "drafter")
    base = tensors
    if set(base) != set(expected):
        raise AssertionError(f"export keys differ: {sorted(set(base) ^ set(expected))[:8]}")
    for name, shape in expected.items():
        if tuple(base[name].shape) != tuple(shape):
            raise AssertionError((name, tuple(base[name].shape), shape))
    if any(k.startswith("model.") or k.startswith(("embed_tokens", "lm_head", "verifier_")) for k in base):
        raise AssertionError("engine export must use bare keys and omit target tensors")
    config = json.loads((output / "config.json").read_text())
    for field in ("hidden_size", "num_hidden_layers", "intermediate_size", "num_attention_heads",
                  "num_key_value_heads", "head_dim", "vocab_size", "dflash_config"):
        if field not in config:
            raise AssertionError(f"config.json missing {field}")
    with guarded_safe_open(output / "model.safetensors", "drafter") as handle:
        stamp = json.loads(handle.metadata()[SCRATCH_STAMP_KEY])
    return {"keys": len(base), "dtypes": sorted({str(t.dtype) for t in tensors.values()}), "stamp": stamp}


__all__ = ["engine_config", "export_engine_checkpoint", "verify_export"]
