"""GLM-5.3-Flash chat rendering and token counting.

Uses the target's own ``chat_template.jinja`` and ``tokenizer.json`` (from your
copy of the GLM-5.3-Flash checkpoint; nothing is vendored here) through
transformers' chat-template renderer (trim_blocks/lstrip_blocks, as HF).
Point ``GLM_TOKENIZER_DIR`` (or :func:`set_tokenizer_dir`) at that directory.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

GEN_PREFIX = "<|assistant|><think>"
_TOK_DIR: Path | None = None


def set_tokenizer_dir(path: str | Path) -> None:
    global _TOK_DIR
    _TOK_DIR = Path(path)
    tokenizer.cache_clear()


@lru_cache(maxsize=1)
def tokenizer():
    from transformers import PreTrainedTokenizerFast

    root = _TOK_DIR or (Path(os.environ["GLM_TOKENIZER_DIR"]) if os.environ.get("GLM_TOKENIZER_DIR") else None)
    if root is None:
        raise RuntimeError("set GLM_TOKENIZER_DIR to a directory with the target's tokenizer.json and "
                           "chat_template.jinja")
    tok = PreTrainedTokenizerFast(tokenizer_file=str(root / "tokenizer.json"))
    tok.chat_template = (root / "chat_template.jinja").read_text()
    return tok


def _template_messages(messages):
    """Tool-call arguments as dicts (the template iterates them)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                fn = dict(tc.get("function", tc))
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"raw": args}
                if not isinstance(args, dict):
                    args = {"value": args}
                fn["arguments"] = args
                tcs.append({"id": tc.get("id"), "type": "function", "function": fn})
            m["tool_calls"] = tcs
        if m.get("content") is None:
            m["content"] = ""
        out.append(m)
    return out


def render(messages, tools=None, reasoning_effort=None, add_generation_prompt=True) -> str:
    kw = {}
    if reasoning_effort:
        kw["reasoning_effort"] = reasoning_effort
    return tokenizer().apply_chat_template(_template_messages(messages), tools=tools or None, tokenize=False,
                                           add_generation_prompt=add_generation_prompt, **kw)


def encode(text: str) -> list[int]:
    return tokenizer()(text, add_special_tokens=False)["input_ids"]


def count(text: str) -> int:
    return len(encode(text))


def completion_text(messages, tools, reasoning_effort, assistant: dict) -> str:
    """Text the model generated after `<|assistant|><think>` for this turn."""
    head = render(messages, tools, reasoning_effort, True)
    full = render(messages + [assistant], tools, reasoning_effort, False)
    assert head.endswith(GEN_PREFIX), head[-40:]
    base = head[: -len(GEN_PREFIX)]
    assert full.startswith(base + GEN_PREFIX), "assistant render does not extend the generation prompt"
    return full[len(base) + len(GEN_PREFIX):]
