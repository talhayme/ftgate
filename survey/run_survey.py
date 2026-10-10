"""Lint six public tool-calling datasets with ftgate data and tabulate.

Each raw sample is converted with its adapter, written as messages+tools
JSONL, and linted with the Qwen2.5 tokenizer and template at two common
cutoffs. Every number is a rate over the sampled rows, with the sample
size shown, and the adapter's own parse failures are a column — the
conversion is part of what is being measured.
"""
import json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from adapters import ADAPTERS  # noqa: E402
from ftgate import data  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
CUTS = (2048, 4096)
CODES = ["tool_args_not_json", "undeclared_tool", "tool_args_missing_required", "tool_args_unknown_key",
         "tool_call_id_mismatch", "tool_without_call", "empty_assistant", "no_assistant_target",
         "role_repeat", "duplicate_row", "template_error", "zero_target_tokens", "no_eos"]

results = {}
(HERE / "adapted").mkdir(exist_ok=True)
for name, fn in ADAPTERS.items():
    raw = HERE / "raw" / f"{name}.jsonl"
    if not raw.exists():
        print(f"  (missing sample) {name}"); continue
    rows = [json.loads(l) for l in raw.open(encoding="utf-8")]
    # Safety net against overlapping sample windows: drop byte-identical raw rows.
    seen, uniq = set(), []
    for l in raw.open(encoding="utf-8"):
        if l not in seen: seen.add(l); uniq.append(json.loads(l))
    raw_dups, rows = len(rows) - len(uniq), uniq
    adapted, fails = [], 0
    for r in rows:
        try:
            a = fn(r)
        except Exception:  # noqa: BLE001
            a = None
        if a and a["messages"]: adapted.append(a)
        else: fails += 1
    out = HERE / "adapted" / f"{name}.jsonl"
    out.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in adapted) + "\n", encoding="utf-8")

    per_cut = {}
    for cut in CUTS:
        rep = data.lint(out, model=MODEL, max_seq_len=cut)
        per_cut[cut] = rep
    rep = per_cut[CUTS[0]]
    n = rep.rows
    rows_with_code = {c: len({i.row for i in rep.issues if i.code == c}) for c in CODES}
    call_rows = sum(1 for a in adapted if any(m.get("tool_calls") for m in a["messages"]))
    results[name] = {
        "sampled": len(rows), "raw_dups_removed": raw_dups, "adapter_failed": fails, "linted": n, "rows_with_tool_calls": call_rows,
        "tokens": rep.stats.get("tokens"), "over": {str(c): per_cut[c].stats.get("over_max_seq_len", 0) for c in CUTS},
        "truncated_before_target": {str(c): len({i.row for i in per_cut[c].issues if i.code == "truncated_before_target"}) for c in CUTS},
        "rows_with_code": rows_with_code, "errors": rep.count("error"), "warnings": rep.count("warning"),
        "examples": {c: next((i.detail for i in rep.issues if i.code == c), None) for c in CODES if rows_with_code[c]},
    }
    print(f"  {name[:44]:44} linted {n:4}  errors {rep.count('error'):5}  warnings {rep.count('warning'):5}")

(HERE / "results.json").write_text(json.dumps({"model": MODEL, "results": results}, indent=1, ensure_ascii=False))

# ---- markdown table
def pct(k, n): return f"{100*k/n:.1f}%" if n else "–"
lines = [f"| dataset | sampled | adapter fail | rows w/ calls | tokens p50 / p95 / max | >2048 | cut before answer @2048 | >4096 | args not JSON | undeclared tool | missing required | unknown key | tool msg w/o call | empty assistant | no final assistant turn | dup rows |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, r in results.items():
    t = r["tokens"] or {}; n = r["linted"]; c = r["rows_with_code"]
    lines.append(f"| {name.replace('__', '/')} | {r['sampled']} | {r['adapter_failed']} | {pct(r['rows_with_tool_calls'], n)} | "
                 f"{t.get('p50','–')} / {t.get('p95','–')} / {t.get('max','–')} | {pct(r['over']['2048'], n)} | {pct(r['truncated_before_target']['2048'], n)} | {pct(r['over']['4096'], n)} | "
                 f"{pct(c['tool_args_not_json'], n)} | {pct(c['undeclared_tool'], n)} | {pct(c['tool_args_missing_required'], n)} | {pct(c['tool_args_unknown_key'], n)} | "
                 f"{pct(c['tool_without_call'], n)} | {pct(c['empty_assistant'], n)} | {pct(c['no_assistant_target'], n)} | {pct(c['duplicate_row'], n)} |")
(HERE / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n" + "\n".join(lines))
