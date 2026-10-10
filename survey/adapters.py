"""Convert each dataset's native shape into `messages` + `tools` for ftgate data.

Every adapter is a guess about the author's intent, written down. Where a
dataset's own format is lossy or odd, the adapter keeps it that way and
the linter reports it — that is the point of the survey, not a nuisance to
paper over. Each adapter returns a row dict or None (row unparseable).
"""
from __future__ import annotations

import json
import re

TOOL_CALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
FUNC_CALLS = re.compile(r"<function_calls>\s*(.*?)\s*</function_calls>", re.S)
INVOKE = re.compile(r'<invoke name="([^"]+)">(.*?)</invoke>', re.S)
PARAM = re.compile(r'<parameter name="([^"]+)">(.*?)</parameter>', re.S)


def _wrap_tools(tools) -> list[dict]:
    """Bare function objects → OpenAI `{"type":"function","function":{...}}`."""
    out = []
    for t in tools or []:
        if isinstance(t, dict) and "function" in t and "type" in t:
            out.append(t)
        elif isinstance(t, dict):
            out.append({"type": "function", "function": t})
    return out


def _xlam_schema(params: dict) -> dict:
    """xLAM's flat `{name: {type, description, default}}` → JSON schema.

    xLAM marks optional parameters with a `default` key; everything else is
    required. That convention is implicit, so it is made explicit here.
    """
    props, required = {}, []
    for name, spec in (params or {}).items():
        if not isinstance(spec, dict):
            continue
        props[name] = {k: v for k, v in spec.items() if k in ("type", "description", "enum")}
        if "default" not in spec:
            required.append(name)
    return {"type": "object", "properties": props, "required": required}


def _parse_json(s, default=None):
    try:
        return json.loads(s) if isinstance(s, str) else s
    except json.JSONDecodeError:
        return default


# --- NousResearch/hermes-function-calling-v1 ---------------------------

def hermes(row: dict) -> dict | None:
    tools = _wrap_tools(_parse_json(row.get("tools"), []))
    msgs = []
    for m in row["conversations"]:
        role = {"system": "system", "human": "user", "gpt": "assistant", "tool": "tool"}.get(m["from"], m["from"])
        text = m["value"]
        if role == "assistant" and "<tool_call>" in text:
            calls = []
            for block in TOOL_CALL.findall(text):
                d = _parse_json(block)
                calls.append({"type": "function", "function": {"name": d.get("name") if d else None,
                              "arguments": d.get("arguments") if d else block}})   # unparseable → string → linter flags it
            msgs.append({"role": "assistant", "content": TOOL_CALL.sub("", text).strip(), "tool_calls": calls})
        else:
            msgs.append({"role": role, "content": text})
    # hermes writes the signatures into its own system prompt; the tools
    # field is for schema validation only.
    return {"messages": msgs, "tools": tools, "tools_in_prompt": True}


# --- glaiveai/glaive-function-calling-v2 (original text format) --------

def glaive(row: dict) -> dict | None:
    system = row["system"]
    tools = []
    head = system.split("Use them if required -", 1)[-1] if "Use them if required" in system else ""
    # one or more JSON objects back to back
    depth, start = 0, None
    for i, ch in enumerate(head):
        if ch == "{":
            if depth == 0: start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                t = _parse_json(head[start:i + 1]); start = None
                if t: tools.append(t)
    tools = _wrap_tools(tools)
    sys_text = system.split("SYSTEM:", 1)[-1].strip()
    msgs = [{"role": "system", "content": sys_text}] if sys_text else []

    parts = re.split(r"\n*(USER:|ASSISTANT:|FUNCTION RESPONSE:)\s*", row["chat"])
    for tag, body in zip(parts[1::2], parts[2::2]):
        body = body.replace("<|endoftext|>", "").strip()
        if tag == "USER:":
            msgs.append({"role": "user", "content": body})
        elif tag == "FUNCTION RESPONSE:":
            msgs.append({"role": "tool", "content": body})
        else:
            if "<functioncall>" in body:
                raw = body.split("<functioncall>", 1)[1].strip()
                d = _parse_json(raw)
                args = d.get("arguments") if d else None
                # glaive's famous quirk: arguments is a *string* in single quotes
                if d is None:
                    m = re.match(r"\{\s*\"name\"\s*:\s*\"([^\"]+)\"\s*,\s*\"arguments\"\s*:\s*'(.*)'\s*\}\s*$", raw, re.S)
                    if m:
                        d = {"name": m.group(1)}; args = m.group(2)   # kept as string → linter checks it parses
                msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                    "name": d.get("name") if d else None, "arguments": args if args is not None else raw}}]})
            else:
                msgs.append({"role": "assistant", "content": body})
    # glaive writes "ASSISTANT: Sure, let me check." and then "ASSISTANT:
    # <functioncall> …" as two turns. In the OpenAI shape that is one
    # assistant message with content and tool_calls, so merge them; a
    # naive conversion that keeps them separate is what role_repeat would
    # catch, and that is the converter's problem, not the data's.
    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == "assistant" and m["role"] == "assistant" \
                and merged[-1].get("content") and not merged[-1].get("tool_calls") and m.get("tool_calls"):
            merged[-1] = {"role": "assistant", "content": merged[-1]["content"], "tool_calls": m["tool_calls"]}
        else:
            merged.append(m)
    return {"messages": merged, "tools": tools, "tools_in_prompt": True}


# --- lockon/xlam-function-calling-60k and argilla/apigen-function-calling

def xlam(row: dict) -> dict | None:
    tools_raw = _parse_json(row.get("tools"), [])
    tools = []
    for t in tools_raw:
        if not isinstance(t, dict):
            continue
        if "function" in t:
            # apigen's distilabel subset already ships OpenAI-shaped tools
            # with JSON-schema parameters; keep them as they are.
            tools.append(t if t.get("type") else {"type": "function", **t})
        else:
            tools.append({"type": "function", "function": {"name": t.get("name"), "description": t.get("description", ""),
                          "parameters": _xlam_schema(t.get("parameters"))}})
    answers = _parse_json(row.get("answers"), None)
    if answers is None:
        calls = [{"type": "function", "function": {"name": None, "arguments": row.get("answers")}}]
    else:
        calls = [{"type": "function", "function": {"name": a.get("name"), "arguments": a.get("arguments", {})}} for a in answers if isinstance(a, dict)]
    return {"messages": [{"role": "user", "content": row["query"]},
                         {"role": "assistant", "content": "", "tool_calls": calls}], "tools": tools}


# --- hiyouga/glaive-function-calling-v2-sharegpt (LLaMA-Factory roles) --

def hiyouga(row: dict) -> dict | None:
    tools = _wrap_tools(_parse_json(row.get("tools"), []))
    msgs = []
    for m in row["conversations"]:
        f, v = m["from"], m["value"]
        if f == "human": msgs.append({"role": "user", "content": v})
        elif f == "gpt": msgs.append({"role": "assistant", "content": v})
        elif f == "observation": msgs.append({"role": "tool", "content": v})
        elif f == "function_call":
            d = _parse_json(v)
            msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {
                "name": d.get("name") if d else None, "arguments": d.get("arguments") if d else v}}]})
        elif f == "system": msgs.append({"role": "system", "content": v})
    return {"messages": msgs, "tools": tools}


# --- allenai/Dolci-Instruct-SFT-Tool-Use ------------------------------

def dolci(row: dict) -> dict | None:
    msgs, tools = [], []
    for m in row["messages"]:
        role, text = m["role"], m.get("content") or ""
        if role == "system":
            fm = re.search(r"<functions>(.*?)</functions>", text, re.S)
            if fm:
                # one JSON object per line, or a JSON list
                blob = fm.group(1).strip()
                lst = _parse_json(blob)
                if isinstance(lst, list): tools = _wrap_tools(lst)
                else:
                    for line in blob.splitlines():
                        t = _parse_json(line.strip())
                        if isinstance(t, dict): tools.append(t)
                    tools = _wrap_tools(tools)
            msgs.append({"role": "system", "content": text})
        elif role == "environment":
            msgs.append({"role": "tool", "content": text})
        elif role == "assistant" and "<function_calls>" in text:
            calls = []
            for block in FUNC_CALLS.findall(text):
                for name, body in INVOKE.findall(block):
                    args = {k: v for k, v in PARAM.findall(body)}
                    calls.append({"type": "function", "function": {"name": name, "arguments": args}})
                if not INVOKE.search(block):
                    d = _parse_json(block)
                    if isinstance(d, dict): calls.append({"type": "function", "function": {"name": d.get("name"), "arguments": d.get("arguments", {})}})
                    elif isinstance(d, list):
                        calls += [{"type": "function", "function": {"name": x.get("name"), "arguments": x.get("arguments", {})}} for x in d if isinstance(x, dict)]
                    else: calls.append({"type": "function", "function": {"name": None, "arguments": block}})
            msgs.append({"role": "assistant", "content": FUNC_CALLS.sub("", text).strip(), "tool_calls": calls})
        else:
            msgs.append({"role": role, "content": text})
    return {"messages": msgs, "tools": tools, "tools_in_prompt": True}


ADAPTERS = {
    "NousResearch__hermes-function-calling-v1": hermes,
    "glaiveai__glaive-function-calling-v2": glaive,
    "lockon__xlam-function-calling-60k": xlam,
    "argilla__apigen-function-calling": xlam,
    "hiyouga__glaive-function-calling-v2-sharegpt": hiyouga,
    "allenai__Dolci-Instruct-SFT-Tool-Use": dolci,
}
