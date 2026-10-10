"""The same hard set through llama-server's own tool-call parser (/v1/chat/completions).

The raw judge is strict JSON — what a hermes-style parser (vLLM) does.
llama-server's parser is lenient about the doubled braces, so this is the
best case for the embedded template: the number a user of llama-server
would see.
"""
import json, sys, time, requests
from pathlib import Path
root = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(root / "scripts"))
from day4_hard_cases import TOOLS, CASES  # noqa: E402
from day4_eval import norm  # noqa: E402

ARMS = {"embedded-{{": "http://localhost:8090", "hf-{": "http://localhost:8091"}
LABEL = sys.argv[1] if len(sys.argv) > 1 else "model"

def calls_via_parser(base, user):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user}]
    r = requests.post(f"{base}/v1/chat/completions", json={"model": "m", "messages": msgs, "tools": TOOLS, "temperature": 0, "max_tokens": 400}, timeout=300).json()
    m = r["choices"][0]["message"]; out = []
    for tc in m.get("tool_calls") or []:
        try: args = json.loads(tc["function"].get("arguments") or "{}")
        except json.JSONDecodeError: args = None
        out.append({"name": tc["function"].get("name"), "arguments": args})
    return out

def exact(calls, want):
    if any(c["arguments"] is None for c in calls): return 0
    a = sorted(json.dumps(norm(c["arguments"]), sort_keys=True) + "|" + str(c["name"]) for c in calls)
    b = sorted(json.dumps(norm(c["arguments"]), sort_keys=True) + "|" + str(c["name"]) for c in want)
    return int(a == b)

res = {a: [] for a in ARMS}; t0 = time.time()
for i, case in enumerate(CASES):
    for a, base in ARMS.items():
        calls = calls_via_parser(base, case["user"])
        res[a].append({"any": int(bool(calls)), "names": int(sorted(c["name"] for c in calls) == sorted(c["name"] for c in case["calls"])),
                       "exact": exact(calls, case["calls"]), "kind": case["kind"]})
n = len(CASES)
print(f"\n{LABEL}: hard set through llama-server's lenient parser\n")
print(f"{'arm':13} {'any call':>9} {'right tools':>12} {'exact args':>11}")
for a, rs in res.items():
    print(f"{a:13} {sum(r['any'] for r in rs):>6}/{n:<3} {sum(r['names'] for r in rs):>8}/{n:<3} {sum(r['exact'] for r in rs):>7}/{n:<3}")
(root / f"out/day4-lenient-{LABEL}.json").write_text(json.dumps(res, indent=1))
