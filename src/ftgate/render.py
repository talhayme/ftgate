"""Text and JSON reports."""
from __future__ import annotations

import json

from ftgate.compare import Comparison

MARK = {"identical": "ok ", "drift": " ~ ", "mismatch": "!! "}


def to_text(results: list[Comparison], model: str, show_diff: bool) -> str:
    out = [f"ftgate template — reference: {model}", ""]
    width = max(len(r.conversation) for r in results) if results else 10
    for r in results:
        rt_tok = "?" if r.runtime_tokens is None else str(r.runtime_tokens)
        out.append(f"{MARK[r.verdict]}{r.conversation:{width}}  {r.runtime:13} {rt_tok:>5} tokens vs {r.reference_tokens:<5} {r.verdict}")
        for f in r.findings:
            out.append(f"      [{f.kind}] {f.detail}")
            if show_diff and f.diff:
                out.extend("        " + line for line in f.diff.splitlines()[:40])
        for n in r.notes:
            if not n.startswith("model:") and not n.startswith("rendered with"):
                out.append(f"      · {n}")
    total = len(results)
    mism = sum(r.verdict == "mismatch" for r in results); drift = sum(r.verdict == "drift" for r in results)
    out += ["", f"  {total} comparison(s): {total - mism - drift} identical, {drift} drift, {mism} mismatch"]
    if mism:
        out.append("  A mismatch means the runtime sends bytes the model never saw in training.")
    return "\n".join(out) + "\n"


def to_json(results: list[Comparison], model: str) -> str:
    return json.dumps({"schema": "ftgate/template/v0", "reference_model": model, "results": [
        {"conversation": r.conversation, "runtime": r.runtime, "verdict": r.verdict,
         "reference_tokens": r.reference_tokens, "runtime_tokens": r.runtime_tokens,
         "findings": [{"kind": f.kind, "detail": f.detail, "diff": f.diff} for f in r.findings], "notes": r.notes}
        for r in results]}, indent=2, ensure_ascii=False) + "\n"
