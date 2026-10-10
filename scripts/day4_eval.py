"""Does the doubled-brace instruction change what the model emits?

Two llama-servers on the same GGUF blob, temperature 0. One renders the
template embedded in the file ({{"name": …}}), the other the current HF
template ({"name": …}). Raw mode — /apply-template then /completion — so
the model's output is read unparsed and every <tool_call> block is judged
here: valid JSON or not, right tools or not, exact arguments or not,
doubled braces in the output or not.
"""
import json, re, sys, time, requests
from pathlib import Path
root = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(root / "scripts"))
from day4_hard_cases import TOOLS, CASES  # noqa: E402

ARMS = {"embedded-{{": "http://localhost:8090", "hf-{": "http://localhost:8091"}
LABEL = sys.argv[1] if len(sys.argv) > 1 else "model"
BLOCK = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)

def run(base, user):
    msgs = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": user}]
    prompt = requests.post(f"{base}/apply-template", json={"messages": msgs, "tools": TOOLS}, timeout=60).json()["prompt"]
    r = requests.post(f"{base}/completion", json={"prompt": prompt, "temperature": 0, "n_predict": 400, "stop": ["<|im_end|>"]}, timeout=300).json()
    return r.get("content", "")

def norm(v):
    if isinstance(v, dict): return {k: norm(x) for k, x in sorted(v.items())}
    if isinstance(v, list): return [norm(x) for x in v]
    if isinstance(v, bool): return v
    if isinstance(v, (int, float)): return float(v)
    return str(v).strip()

def judge(text, case):
    blocks = BLOCK.findall(text)
    calls, malformed = [], 0
    for b in blocks:
        try:
            d = json.loads(b); calls.append({"name": d.get("name"), "arguments": d.get("arguments") if isinstance(d.get("arguments"), dict) else {}})
        except json.JSONDecodeError:
            malformed += 1
    want = case["calls"]
    names_ok = sorted(c["name"] for c in calls) == sorted(c["name"] for c in want)
    exact = names_ok and sorted(json.dumps(norm(c["arguments"]), sort_keys=True) for c in calls) == sorted(json.dumps(norm(c["arguments"]), sort_keys=True) for c in want)
    return {"any": int(bool(blocks)), "malformed": malformed, "names": int(names_ok), "exact": int(bool(exact)),
            "doubled": int("{{" in text), "n_calls": len(blocks), "raw": text[:400]}

def main():
    results = {a: [] for a in ARMS}; t0 = time.time()
    for i, case in enumerate(CASES):
        for a, base in ARMS.items():
            results[a].append({"case": i, "kind": case["kind"], **judge(run(base, case["user"]), case)})
        if i % 10 == 9: print(f"  {i+1}/{len(CASES)} ({time.time()-t0:.0f}s)", file=sys.stderr)

    n = len(CASES)
    print(f"\n{LABEL}: {n} hard cases, same blob, temperature 0\n")
    print(f"{'arm':13} {'any call':>9} {'malformed':>10} {'right tools':>12} {'exact args':>11} {'{{ in output':>13}")
    for a, rs in results.items():
        print(f"{a:13} {sum(r['any'] for r in rs):>6}/{n:<3} {sum(r['malformed'] for r in rs):>7}    {sum(r['names'] for r in rs):>8}/{n:<3} {sum(r['exact'] for r in rs):>7}/{n:<3} {sum(r['doubled'] for r in rs):>9}")
    print("\nexact args by kind:")
    kinds = list(dict.fromkeys(c["kind"] for c in CASES))
    print(f"  {'kind':15}" + "".join(f"{a:>14}" for a in ARMS))
    for k in kinds:
        idx = [i for i, c in enumerate(CASES) if c["kind"] == k]
        print(f"  {k:15}" + "".join(f"{sum(results[a][i]['exact'] for i in idx):>10}/{len(idx):<3}" for a in ARMS))
    (root / "out").mkdir(exist_ok=True)
    (root / f"out/day4-{LABEL}.json").write_text(json.dumps({"label": LABEL, "results": results}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
