"""hermes-function-calling-v1, every tool-calling config, every row.

The survey sampled one config (`func_calling`) and its headline — "38% cut
before the answer at 2,048" — was then quoted for the whole dataset. The
configs are small enough to read in full, so this does: all rows of the
three tool-calling configs, same adapter, same linter, same model.

    python survey/hermes_full.py [model]
"""
import json, sys, time, urllib.parse, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "src"))
from adapters import hermes  # noqa: E402
from ftgate import data  # noqa: E402

DS = "NousResearch/hermes-function-calling-v1"
CONFIGS = ["func_calling", "func_calling_singleturn", "glaive_func_calling"]
MODEL = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"


def fetch(cfg: str) -> list[dict]:
    cache = HERE / "raw" / f"hermes_full__{cfg}.jsonl"
    if cache.exists():
        return [json.loads(l) for l in cache.open(encoding="utf-8")]
    rows, off = [], 0
    while True:
        url = (f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(DS)}"
               f"&config={cfg}&split=train&offset={off}&length=100")
        d = None
        for attempt in range(6):
            try:
                d = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ftgate-survey"}), timeout=60))
                break
            except Exception:  # noqa: BLE001 — 429s and timeouts: wait and retry
                time.sleep(4 + 4 * attempt)
        if not d or not d.get("rows"):
            break
        rows += [r["row"] for r in d["rows"]]
        off += 100
        if off >= d["num_rows_total"]:
            break
        time.sleep(0.7)
    cache.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
    return rows


out = {}
for cfg in CONFIGS:
    rows = fetch(cfg)
    adapted, fails = [], 0
    for r in rows:
        try:
            a = hermes(r)
        except Exception:  # noqa: BLE001
            a = None
        if a and a["messages"]:
            adapted.append(a)
        else:
            fails += 1
    path = HERE / "adapted" / f"hermes_full__{cfg}.jsonl"
    path.parent.mkdir(exist_ok=True)
    path.write_text("\n".join(json.dumps(a, ensure_ascii=False) for a in adapted) + "\n", encoding="utf-8")
    res = {"rows": len(rows), "unique_ids": len({r.get("id") for r in rows}), "adapter_failed": fails}
    for cut in (2048, 4096):
        rep = data.lint(path, model=MODEL, max_seq_len=cut)
        res["linted"] = rep.rows
        res["tokens"] = rep.stats.get("tokens")
        res[f"over_{cut}"] = rep.stats.get("over_max_seq_len", 0)
        res[f"cut_before_answer_{cut}"] = len({i.row for i in rep.issues if i.code == "truncated_before_target"})
        if cut == 2048:
            res["no_final_assistant"] = len({i.row for i in rep.issues if i.code == "no_assistant_target"})
    out[cfg] = res
    n = res["linted"] or 1
    print(f"{cfg:26} rows {res['rows']:5} linted {res['linted']:5} adapter-fail {fails:3} | tokens {res['tokens']} | "
          f">2048 {100*res['over_2048']/n:.1f}% | cut before answer @2048 {100*res['cut_before_answer_2048']/n:.1f}% "
          f"@4096 {100*res['cut_before_answer_4096']/n:.1f}% | no final assistant {100*res['no_final_assistant']/n:.1f}%", flush=True)
(HERE / "hermes_full.json").write_text(json.dumps({"model": MODEL, "results": out}, indent=1), encoding="utf-8")
tot = sum(r["linted"] for r in out.values()); cut = sum(r["cut_before_answer_2048"] for r in out.values())
print(f"all three configs: {cut} of {tot} rows cut before the answer at 2,048 = {100*cut/tot:.1f}%")
