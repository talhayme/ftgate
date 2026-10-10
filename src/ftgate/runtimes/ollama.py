"""Ollama: no render or tokenize endpoint exists (ollama/ollama#12030), so the
TEMPLATE is fetched from /api/show and rendered with Ollama's own `template`
package through a small Go helper. That makes it Ollama's rendering, not a
guess — and the token count is cross-checked against what Ollama reports
for a real request.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import requests

from ftgate.runtimes import RuntimeRendering


def renderer_path() -> str | None:
    env = os.environ.get("FTGATE_OLLAMA_RENDER")
    candidates = [env] if env else []
    candidates += [shutil.which("ollama-render"), str(Path(__file__).resolve().parents[3] / "bin" / "ollama-render")]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def render(base_url: str, model: str, messages: list[dict], tools: list[dict], verify: bool = True) -> RuntimeRendering:
    r = requests.post(f"{base_url}/api/show", json={"name": model}, timeout=30); r.raise_for_status()
    show = r.json()
    template = show["template"]
    notes_pre = []
    # The server prepends the Modelfile's SYSTEM when the request carries no
    # system message. The model then sees a system prompt the training rows
    # may never have had — so the renderer must do the same, and the
    # comparison names it.
    if show.get("system") and not any(m.get("role") == "system" for m in messages):
        messages = [{"role": "system", "content": show["system"]}] + list(messages)
        notes_pre.append(f"Modelfile SYSTEM injected by the runtime: {show['system'][:60]!r}")

    helper = renderer_path()
    if helper is None:
        raise RuntimeError("ollama-render helper not found; build it from bin/ (see bin/README.md) "
                           "or set FTGATE_OLLAMA_RENDER")
    proc = subprocess.run([helper], input=json.dumps({"template": template, "tools": tools, "messages": messages}),
                          capture_output=True, text=True, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"ollama-render failed: {proc.stderr.strip()[:200]}")
    text = proc.stdout
    notes = notes_pre + [f"rendered with Ollama's template package (helper: {Path(helper).name})"]

    if verify:
        # One real request: Ollama reports how many prompt tokens it evaluated.
        body = {"model": model, "messages": messages, "stream": False, "options": {"num_predict": 1, "temperature": 0}}
        if tools:
            body["tools"] = tools
        try:
            resp = requests.post(f"{base_url}/api/chat", json=body, timeout=180).json()
            n = resp.get("prompt_eval_count")
            if n is not None:
                notes.append(f"ollama evaluated {n} prompt tokens for this request")
                return RuntimeRendering("ollama", text, None, [], notes + [f"__prompt_eval_count={n}"])
        except Exception as exc:  # noqa: BLE001
            notes.append(f"could not cross-check with a live request: {exc}")
    return RuntimeRendering("ollama", text, None, [], notes)
