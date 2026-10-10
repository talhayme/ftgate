"""Compare a runtime's bytes with the reference and say what kind of difference it is.

Findings are ordered by how much they matter to a model that was trained
on the reference:

  schema_not_json     a tool definition reached the model as something other
                      than JSON — the schema itself is gone
  injected_system     the runtime added a system prompt the training bytes
                      did not have (Ollama's Modelfile SYSTEM)
  special_prefix      the runtime prepends a special token the training
                      bytes did not have (the BOS case)
  structural          text differs beyond whitespace/JSON spacing
  spacing             same content, different whitespace or JSON separators —
                      different tokens, same meaning
  tokenizer           identical text, different ids — the runtime's
                      tokenizer disagrees with the training tokenizer
  identical           byte-for-byte and id-for-id
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

from ftgate.reference import Rendering
from ftgate.runtimes import RuntimeRendering

_TOOLS_BLOCK = re.compile(r"<tools>(.*?)</tools>", re.S)


@dataclass
class Finding:
    kind: str
    detail: str
    diff: str = ""


@dataclass
class Comparison:
    conversation: str
    runtime: str
    reference_tokens: int
    runtime_tokens: int | None
    findings: list[Finding] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def verdict(self) -> str:
        kinds = {f.kind for f in self.findings}
        if kinds & {"schema_not_json", "structural", "special_prefix", "injected_system"}:
            return "mismatch"
        if kinds & {"spacing", "tokenizer"}:
            return "drift"
        return "identical"


def _canon(text: str) -> str:
    """Collapse whitespace and JSON separators so spacing-only differences vanish."""
    t = re.sub(r"[ \t]+", " ", text)
    t = re.sub(r"\s*([,:{}\[\]])\s*", r"\1", t)
    return t.strip()


def _tool_schemas_valid(text: str) -> list[str]:
    """Every line inside <tools> must parse as JSON with a function name."""
    bad = []
    for block in _TOOLS_BLOCK.findall(text):
        for line in block.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                fn = obj.get("function", obj)
                if not isinstance(fn, dict) or "name" not in fn:
                    bad.append(line[:120])
            except json.JSONDecodeError:
                bad.append(line[:120])
    return bad


def compare(name: str, ref: Rendering, rt: RuntimeRendering, ref_ids_of_runtime_text: list[int]) -> Comparison:
    c = Comparison(name, rt.runtime, len(ref.ids), len(rt.ids) if rt.ids is not None else None, notes=list(rt.notes))

    bad = _tool_schemas_valid(rt.text)
    if bad and not _tool_schemas_valid(ref.text):
        c.findings.append(Finding("schema_not_json",
            f"{len(bad)} tool definition(s) are not JSON in the runtime prompt", "\n".join(bad)))

    if any(n.startswith("Modelfile SYSTEM injected") for n in rt.notes):
        c.findings.append(Finding("injected_system", "runtime prepends a system prompt the training conversation does not have"))

    if rt.special_prefix and not ref.ids[:len(rt.special_prefix)] == rt.special_prefix:
        c.findings.append(Finding("special_prefix",
            f"runtime prepends special token(s) {rt.special_prefix} that the training bytes do not start with"))

    if rt.text != ref.text:
        diff = "\n".join(difflib.unified_diff(ref.text.splitlines(), rt.text.splitlines(), "training", rt.runtime, lineterm="", n=1))
        if _canon(rt.text) == _canon(ref.text):
            c.findings.append(Finding("spacing", "same content, different whitespace or JSON spacing — different tokens", diff))
        elif not bad and not any(f.kind == "injected_system" for f in c.findings):
            c.findings.append(Finding("structural", "prompt text differs from the training bytes", diff))
        else:
            c.findings[0].diff = diff
    elif rt.ids is not None and rt.ids != ref.ids:
        c.findings.append(Finding("tokenizer", f"identical text, different ids ({len(rt.ids)} vs {len(ref.ids)})"))

    if rt.ids is None:
        c.runtime_tokens = len(ref_ids_of_runtime_text)
        c.notes.append("runtime cannot tokenize; ids computed with the training tokenizer on the runtime's text")
        for n in rt.notes:
            if n.startswith("__prompt_eval_count="):
                reported = int(n.split("=", 1)[1])
                c.notes = [x for x in c.notes if not x.startswith("__")]
                c.notes.append("cross-check: matches the runtime's reported prompt token count" if reported == c.runtime_tokens
                               else f"cross-check FAILED: runtime reported {reported} prompt tokens, render has {c.runtime_tokens}")
    c.notes = [x for x in c.notes if not x.startswith("__")]
    return c
