"""llama-server: both halves come straight from its endpoints, no emulation."""
from __future__ import annotations

import requests

from ftgate.runtimes import RuntimeRendering


def render(base_url: str, messages: list[dict], tools: list[dict]) -> RuntimeRendering:
    body = {"messages": messages}
    if tools:
        body["tools"] = tools
    r = requests.post(f"{base_url}/apply-template", json=body, timeout=60); r.raise_for_status()
    text = r.json()["prompt"]
    r = requests.post(f"{base_url}/tokenize", json={"content": text, "add_special": True, "parse_special": True}, timeout=60); r.raise_for_status()
    ids = r.json()["tokens"]
    r = requests.post(f"{base_url}/tokenize", json={"content": "", "add_special": True, "parse_special": True}, timeout=60); r.raise_for_status()
    prefix = r.json()["tokens"]
    notes = []
    try:
        props = requests.get(f"{base_url}/props", timeout=10).json()
        notes.append(f"model: {props.get('model_path', '?').rsplit('/', 1)[-1]}")
    except Exception:  # noqa: BLE001
        pass
    return RuntimeRendering("llama-server", text, ids, prefix, notes)
