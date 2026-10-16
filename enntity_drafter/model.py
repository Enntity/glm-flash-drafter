"""Build the DFlash2-style drafter from scratch on top of the pinned Speculators code.

Architecture: ``speculators.DFlash2DraftModel`` (vllm-project/speculators
``824125a``, Apache-2.0; its DFlash2 files carry the Z Lab MIT notice).  No
drafter tensor is ever loaded: every body parameter is freshly initialized by
:func:`scratch_init`.  The only external tensors are the frozen GLM-5.3-Flash
target embedding, final norm and lm_head (MIT), read through :mod:`.guard`.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

# Whole-forward torch.compile of the upstream model is not used (see docs/training.md);
# --compile-layers compiles the decoder layers explicitly.
os.environ.setdefault("TORCH_COMPILE_DISABLE", "1")
# Speculators resolves the verifier norm convention via AutoConfig; never reach
# the network for that (GLM uses the plain x*w final norm).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from .guard import check_target_dir, guarded_safe_open  # noqa: E402

#: Speculators revision this code is tested against (pyproject.toml pins it).
SPECULATORS_REVISION = "824125a36600b5af8fdd1b18286da51ed60f60a8"

#: Frozen target copies inside the model; never trained, never exported.  The
#: engine fills embed/lm_head from the target at load.
FROZEN_KEYS = ("embed_tokens.weight", "lm_head.weight", "verifier_lm_head.weight", "verifier_norm.weight")
FULL_BODY_PARAMETERS = 1_171_080_448
FULL_BODY_KEYS = 81


def load_config(path: str | Path) -> dict[str, Any]:
    """Our architecture config (configs/*.json); never the incoai config.json."""

    cfg = json.loads(Path(path).read_text())
    required = {
        "hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "vocab_size", "rms_norm_eps", "rope_theta",
        "sliding_window", "block_size", "conv_kernel_size", "conv_group_size",
        "selector_rank", "selector_top_k", "mask_token_id", "target_layer_ids",
        "target_hidden_size",
    }
    missing = required - set(cfg)
    if missing:
        raise KeyError(f"config {path} missing {sorted(missing)}")
    if not 0 <= cfg["mask_token_id"] < cfg["vocab_size"]:
        raise ValueError("mask_token_id must be inside the shared target vocabulary")
    return cfg


def speculators_config(cfg: dict[str, Any], attn_implementation: str = "eager"):
    """Translate our flat config into the Speculators DFlash2 config object."""

    from speculators import DFlash2SpeculatorConfig, SpeculatorsConfig, VerifierConfig
    from speculators.proposals.greedy import GreedyTokenProposalConfig
    from transformers import Qwen3Config

    layers = cfg["num_hidden_layers"]
    transformer = Qwen3Config(
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=layers,
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_parameters={"rope_theta": float(cfg["rope_theta"]), "rope_type": "default"},
        sliding_window=cfg["sliding_window"],
        use_sliding_window=True,
        max_window_layers=layers,
        layer_types=["sliding_attention"] * layers,
        max_position_embeddings=cfg.get("max_position_embeddings", 1048576),
        tie_word_embeddings=False,
        attention_bias=False,
        initializer_range=cfg.get("initializer_range", 0.02),
    )
    transformer._attn_implementation = attn_implementation
    return DFlash2SpeculatorConfig(
        transformer_layer_config=transformer,
        draft_vocab_size=cfg["vocab_size"],
        target_hidden_size=cfg["target_hidden_size"],
        block_size=cfg["block_size"],
        aux_hidden_state_layer_ids=list(cfg["target_layer_ids"]),
        mask_token_id=cfg["mask_token_id"],
        sliding_window_non_causal=True,
        sample_from_anchor=False,
        conv_kernel_size=cfg["conv_kernel_size"],
        conv_group_size=cfg["conv_group_size"],
        selector_rank=cfg["selector_rank"],
        selector_top_k=cfg["selector_top_k"],
        speculators_config=SpeculatorsConfig(
            algorithm="dflash2",
            proposal_methods=[GreedyTokenProposalConfig(speculative_tokens=cfg["block_size"] - 1)],
            default_proposal_method="greedy",
            verifier=VerifierConfig(
                name_or_path=cfg.get("target_name", "zai-org/GLM-5.3-Flash"),
                architectures=cfg.get("target_architectures", ["Glm5NextForConditionalGeneration"]),
            ),
        ),
    )


def body_parameter_names(model: nn.Module) -> list[str]:
    return [name for name, _ in model.named_parameters() if name not in FROZEN_KEYS]


def count_body_parameters(model: nn.Module) -> tuple[int, int]:
    names = set(body_parameter_names(model))
    total = sum(p.numel() for n, p in model.named_parameters() if n in names)
    return total, len(names)


def build_model(cfg: dict[str, Any], *, device: str | torch.device = "cpu",
                attn_implementation: str = "eager") -> nn.Module:
    """Construct the drafter; parameters are random until :func:`scratch_init`."""

    from speculators import DFlash2DraftModel

    with torch.device(device):
        model = DFlash2DraftModel(speculators_config(cfg, attn_implementation))
    return model


@torch.no_grad()
def scratch_init(model: nn.Module, cfg: dict[str, Any], seed: int,
                 target_final_norm: torch.Tensor | None = None) -> dict[str, Any]:
    """Deterministic fresh initialization of every body parameter.

    * Linear weights: N(0, initializer_range); residual-output projections
      (``o_proj``, ``down_proj``) scaled by 1/sqrt(2*num_layers) (GPT-2 style)
      so the 5-layer residual stream starts near the noise embedding.
    * RMSNorm gains: 1.  The output ``norm`` optionally starts as a copy of the
      target's final-norm gain (allowed frozen piece), so the drafter's last
      hidden enters the shared target lm_head on the scale it expects.
    * Grouped convolutions: identity (upstream ``reset_parameters``).
    * Candidate selector: codebooks and hidden projection N(0, std), but the
      successor codebook starts at zero so the initial selector is exactly
      the unary top-k order (a LoRA-style zero branch); it still receives a
      gradient on the first step.
    """

    generator = torch.Generator(device="cpu").manual_seed(seed)
    std = float(cfg.get("initializer_range", 0.02))
    residual_std = std / math.sqrt(2 * cfg["num_hidden_layers"])
    report: dict[str, Any] = {"seed": seed, "std": std, "residual_std": residual_std, "tensors": {}}

    def normal_(param: torch.Tensor, scale: float) -> None:
        values = torch.randn(param.shape, generator=generator, dtype=torch.float32) * scale
        param.copy_(values.to(device=param.device, dtype=param.dtype))

    for name, param in model.named_parameters():
        if name in FROZEN_KEYS:
            continue
        if name.endswith("conv.base_kernel"):
            param.zero_()
            param[:, 0].fill_(1.0)
            rule = "identity-conv"
        elif name.endswith("kernel_projection.weight"):
            param.zero_()
            rule = "zero (identity-conv delta)"
        elif name.endswith("norm.weight"):
            if name == "norm.weight" and target_final_norm is not None and cfg.get("init_norm_from_target", True):
                param.copy_(target_final_norm.to(param.dtype))
                rule = "copy of target final norm"
            else:
                param.fill_(1.0)
                rule = "ones"
        elif name == "candidate_selector.successor_codebook":
            param.zero_()
            rule = "zero"
        elif name.endswith(("o_proj.weight", "down_proj.weight")):
            normal_(param, residual_std)
            rule = "normal residual-scaled"
        elif param.ndim == 2:
            normal_(param, std)
            rule = "normal"
        else:
            raise AssertionError(f"no scratch-init rule for {name} {tuple(param.shape)}")
        report["tensors"][name] = rule
    return report


def target_tensor_keys(root: Path) -> dict[str, str]:
    """Locate embedding / final norm / lm_head keys in a sharded target index.

    Excludes decoder layers, the appended MTP layer (``.45.``) and vision towers.
    """

    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    found: dict[str, str] = {}
    for suffix in ("embed_tokens.weight", "lm_head.weight", "norm.weight"):
        matches = [k for k in index if k.endswith(suffix) and ".layers." not in k
                   and "vision" not in k and ".45." not in k and "mtp" not in k]
        if suffix == "norm.weight":
            matches = [k for k in matches if k in (
                "model.language_model.norm.weight", "model.norm.weight",
                "language_model.model.norm.weight", "language_model.norm.weight")]
        if len(matches) != 1:
            raise AssertionError((suffix, matches))
        found[suffix] = matches[0]
    return {suffix: f"{index[key]}::{key}" for suffix, key in found.items()}


def load_target_frozen(root: str | Path) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Read the target embedding, final norm and lm_head through the guard."""

    root = Path(root)
    receipt = check_target_dir(root)
    tensors: dict[str, torch.Tensor] = {}
    for suffix, location in target_tensor_keys(root).items():
        filename, key = location.split("::", 1)
        with guarded_safe_open(root / filename, "target") as handle:
            tensors[suffix] = handle.get_tensor(key)
        receipt.setdefault("keys", {})[suffix] = key
    return tensors, receipt


def install_fixed_anchor_mask(model: nn.Module, anchor_positions) -> Any:
    """Override only ``_build_attention_mask`` on this instance so the given anchors are used.

    Upstream ``select_anchors`` draws anchors at random; capture samples and evals bring
    their own.  Masks are still built by upstream ``_create_attention_mask``, so attention
    semantics and dtypes are unchanged.  Returns the original method (restore it after).
    """

    original = model._build_attention_mask

    def _build_attention_mask(loss_mask, max_anchors, document_ids, device):  # noqa: ANN001
        total_seq_len = loss_mask.shape[1]
        anchors = torch.as_tensor(anchor_positions, dtype=torch.long, device=device)
        assert anchors.numel() == max_anchors, (anchors.numel(), max_anchors)
        anchor_valid = torch.ones(max_anchors, dtype=torch.bool, device=device)
        full_attn_mask = None
        if model.uses_full_attn:
            full_attn_mask = model._create_attention_mask(
                document_ids=document_ids, total_seq_len=total_seq_len, anchor_positions=anchors,
                device=device, sliding_window=None)
        sliding_window_attn_mask = None
        if model.uses_sliding_window_attn:
            sliding_window_attn_mask = model._create_attention_mask(
                document_ids=document_ids, total_seq_len=total_seq_len, anchor_positions=anchors,
                device=device, sliding_window=model.sliding_window,
                sliding_window_non_causal=model.sliding_window_non_causal)
        return full_attn_mask, sliding_window_attn_mask, anchors, anchor_valid

    model._build_attention_mask = _build_attention_mask
    return original


@torch.no_grad()
def install_frozen_target(model: nn.Module, tensors: dict[str, torch.Tensor],
                          final_hidden: str) -> None:
    """Install frozen target tensors (shared, never trained, never exported).

    ``final_hidden`` is the capture convention for the verifier slot:
    ``pre-final-norm`` applies the target norm; ``post-final-norm`` uses Identity.
    """

    embed, head, norm = tensors["embed_tokens.weight"], tensors["lm_head.weight"], tensors["norm.weight"]
    if embed.shape != model.embed_tokens.weight.shape or head.shape != model.lm_head.weight.shape:
        raise AssertionError((tuple(embed.shape), tuple(head.shape)))
    model.embed_tokens.weight = nn.Parameter(embed, requires_grad=False)
    model.lm_head.weight = nn.Parameter(head, requires_grad=False)
    model.verifier_lm_head.weight = model.lm_head.weight
    if final_hidden == "pre-final-norm":
        model.verifier_norm.weight = nn.Parameter(norm.clone(), requires_grad=False)
    elif final_hidden == "post-final-norm":
        model.verifier_norm = nn.Identity()
    else:
        raise ValueError(final_hidden)
    for name in FROZEN_KEYS:
        module_name, _, _ = name.rpartition(".")
        module = model.get_submodule(module_name)
        for parameter in module.parameters():
            parameter.requires_grad_(False)


__all__ = [
    "FULL_BODY_KEYS", "FULL_BODY_PARAMETERS", "FROZEN_KEYS", "body_parameter_names", "build_model",
    "count_body_parameters", "install_fixed_anchor_mask", "install_frozen_target", "load_config",
    "load_target_frozen", "scratch_init", "speculators_config",
]
