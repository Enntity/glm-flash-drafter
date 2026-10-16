#!/usr/bin/env python3
"""Turn prompt rows + teacher generations into capture items (token ids only).

Each item is one full sequence for the hidden-state capture (scripts/capture.py):

* prompt ids: rendered by Atlas's own ``/tokenize`` at capture time
  (``atlas_tokenize``: messages, tools, reasoning effort), so the drafter sees
  exactly the serving template's tokens;
* completion ids: the generated turn (reasoning, ``</think>``, answer, tool
  calls) rendered with the target's chat template and tokenized with the
  target tokenizer (``--tokenizer-dir``: a directory holding the target's
  ``tokenizer.json`` and ``chat_template.jinja``);
* stop id: ``<|observation|>`` after tool calls, ``<|user|>`` after a natural
  stop, none when the generation was truncated (finish=length).

Generations marked ``junk`` or ``error`` by scripts/api_generate.py are dropped.
Rows with a ``check`` field (structured coverage) are also validated
(``enntity_drafter.prompts.validate``); a ``validity.jsonl`` records every
verdict.  Items are interleaved by category; a prompt's samples stay adjacent
so a time-boxed capture still covers every category and samples share a prefix.

    python scripts/prepare_items.py --prompts data/prompts/templates/train.jsonl \\
        --generations data/gen/templates/generations.jsonl --tokenizer-dir /models/GLM-5.3-Flash \\
        --out data/items/templates.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from enntity_drafter import glm_render  # noqa: E402

STOP_TOOL, STOP_TEXT = 154829, 154827   # <|observation|>, <|user|>
MAX_SEQ = 49152


def assistant_msg(resp: dict) -> dict:
    message = {"role": "assistant", "content": resp.get("content") or "",
               "reasoning_content": resp.get("reasoning") or ""}
    if resp.get("tool_calls"):
        message["tool_calls"] = resp["tool_calls"]
    return message


def item(row: dict, gen: dict, extra: dict) -> dict:
    resp = {"reasoning": gen.get("reasoning"), "content": gen.get("content"), "tool_calls": gen.get("tool_calls")}
    effort = row.get("reasoning_effort")
    pids = glm_render.encode(glm_render.render(row["messages"], row.get("tools"), effort, True))
    cids = glm_render.encode(glm_render.completion_text(row["messages"], row.get("tools"), effort, assistant_msg(resp)))
    finish = gen.get("finish")
    stop = STOP_TOOL if finish in ("tool_calls", "tool-calls", "toolUse") or resp["tool_calls"] else (
        STOP_TEXT if finish == "stop" else None)
    comp = cids + ([stop] if stop is not None else [])
    truncated = len(pids) + len(comp) > MAX_SEQ
    if truncated:
        comp = comp[: MAX_SEQ - len(pids)]
    return {"id": gen["id"], "name": gen["id"].replace("#", "_"), "prompt_id": row["id"], "source": row["source"],
            "category": row["category"], "kind": "api", "group": row["id"], "prompt_ids": pids,
            "completion_ids": comp, "stop_id": stop, "finish": finish, "truncated_to_max_seq": truncated,
            "prompt_tokens": len(pids), "completion_tokens": len(comp), "on_policy": False,
            "provider": gen.get("provider"), "temperature": gen.get("temperature"),
            "atlas_tokenize": {"messages": row["messages"], "tools": row.get("tools"), "reasoning_effort": effort},
            **extra}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompts", type=Path, action="append", required=True, help="prompt rows (repeatable)")
    ap.add_argument("--generations", type=Path, action="append", required=True, help="generations.jsonl (repeatable)")
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="items.jsonl to write")
    ap.add_argument("--seed", type=int, default=20261015)
    a = ap.parse_args(argv)
    glm_render.set_tokenizer_dir(a.tokenizer_dir)
    rows = {r["id"]: r for path in a.prompts for r in map(json.loads, path.open())}
    gens: dict[str, dict] = {}
    for path in a.generations:
        for line in path.open():
            g = json.loads(line)
            gens[g["id"]] = g   # last attempt wins
    validity, items = [], []
    for g in sorted(gens.values(), key=lambda g: g.get("seq", 0)):
        row = rows.get(g["prompt_id"])
        if row is None:
            continue
        verdict = {"id": g["id"], "prompt_id": row["id"], "category": row["category"]}
        if g.get("error") or g.get("junk"):
            verdict.update(validity="junk:" + (g.get("junk") or "error"), keep=False)
        elif row.get("check"):
            from enntity_drafter.prompts.validate import validate

            cls, detail, keep = validate(row, g)
            verdict.update(validity=cls, detail=detail, keep=keep)
        else:
            verdict.update(validity="unchecked", keep=True)
        if verdict["keep"]:
            try:
                items.append(item(row, g, {"validity": verdict["validity"]}))
            except Exception as error:  # noqa: BLE001 - a row that does not render is recorded, not fatal
                verdict.update(keep=False, validity="render_error", detail=[repr(error)[:160]])
        validity.append(verdict)
    # Interleave categories; a prompt's samples stay adjacent (prefix sharing).
    rng = random.Random(a.seed)
    groups: dict[str, dict[str, list]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for it in items:
        groups[it["category"]][it["group"]].append(it)
    queues = {c: list(v.values()) for c, v in groups.items()}
    for queue in queues.values():
        rng.shuffle(queue)
    order: list[dict] = []
    while any(queues.values()):
        for category in sorted(queues):
            if queues[category]:
                order += sorted(queues[category].pop(), key=lambda x: x["id"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w") as stream:
        for i, it in enumerate(order):
            it["order"] = i
            stream.write(json.dumps(it) + "\n")
    with a.out.with_name(a.out.stem + "-validity.jsonl").open("w") as stream:
        for verdict in validity:
            stream.write(json.dumps(verdict) + "\n")
    print(json.dumps({"items": len(order), "dropped": sum(not v["keep"] for v in validity),
                      "validity": dict(collections.Counter(v["validity"] for v in validity).most_common()),
                      "tokens": sum(i["prompt_tokens"] + i["completion_tokens"] for i in order),
                      "truncated_to_max_seq": sum(i["truncated_to_max_seq"] for i in order)}, indent=1))


if __name__ == "__main__":
    main()
