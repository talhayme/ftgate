"""Same weights, two prompts: does Ollama's garbled tool schema cost accuracy?

Three arms, all on the GGUF blob Ollama itself uses, temperature 0:

  ollama         Ollama /api/chat with tools — the garbled schema, as shipped
  llama-correct  llama-server /v1/chat/completions — the model's own template
  llama-garbled  llama-server /completion fed the prompt Ollama would build,
                 rendered with Ollama's template package — same engine as
                 llama-correct, same prompt as ollama

If llama-garbled tracks ollama and both trail llama-correct, the template
is the cause and the engine is not.
"""
import json, re, subprocess, sys, time, requests
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "scripts"))
from day2_cases import TOOLS, CASES  # noqa: E402

OLLAMA, LLAMA = "http://localhost:11434", "http://localhost:8090"
MODEL = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:0.5b-instruct"
SYSTEM = "You are a helpful assistant."
RENDER = root / "bin/ollama-render"
TEMPLATE = requests.post(f"{OLLAMA}/api/show", json={"name": MODEL}, timeout=30).json()["template"]

TAG = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

def parse_text(text):
    """Pull a tool call out of raw model text the way a parser would."""
    m = TAG.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(1)); return {"name": d.get("name"), "arguments": d.get("arguments") or {}}
    except json.JSONDecodeError:
        return {"name": None, "arguments": {}, "malformed": True}

def arm_ollama(user):
    r = requests.post(f"{OLLAMA}/api/chat", json={"model": MODEL, "tools": TOOLS, "stream": False,
        "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user}],
        "options": {"temperature": 0, "num_predict": 200}}, timeout=180).json()
    msg = r.get("message", {})
    calls = msg.get("tool_calls") or []
    if calls:
        f = calls[0]["function"]; return {"name": f.get("name"), "arguments": f.get("arguments") or {}}
    return parse_text(msg.get("content", ""))

def arm_llama_correct(user):
    r = requests.post(f"{LLAMA}/v1/chat/completions", json={"model": "m", "tools": TOOLS, "temperature": 0, "max_tokens": 200,
        "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user}]}, timeout=180).json()
    msg = r["choices"][0]["message"]
    calls = msg.get("tool_calls") or []
    if calls:
        f = calls[0]["function"]
        try: args = json.loads(f.get("arguments") or "{}")
        except json.JSONDecodeError: return {"name": f.get("name"), "arguments": {}, "malformed": True}
        return {"name": f.get("name"), "arguments": args}
    return parse_text(msg.get("content", ""))

def arm_llama_garbled(user):
    prompt = subprocess.run([str(RENDER)], input=json.dumps({"template": TEMPLATE, "tools": TOOLS,
        "messages": [{"role":"system","content":SYSTEM},{"role":"user","content":user}]}),
        capture_output=True, text=True, check=True).stdout
    r = requests.post(f"{LLAMA}/completion", json={"prompt": prompt, "temperature": 0, "n_predict": 200,
        "stop": ["<|im_end|>"]}, timeout=180).json()
    return parse_text(r.get("content", ""))

def score(got, case):
    if not got: return {"called": 0, "tool": 0, "args": 0}
    tool = int(got.get("name") == case["tool"])
    args = got.get("arguments") or {}
    def norm(v): return str(v).strip().lower()
    exact = int(tool and all(k in args and norm(args[k]) == norm(v) for k, v in case["args"].items()))
    return {"called": 1, "tool": tool, "args": exact}

arms = {"ollama": arm_ollama, "llama-correct": arm_llama_correct, "llama-garbled": arm_llama_garbled}
results = {a: [] for a in arms}
t0 = time.time()
for i, case in enumerate(CASES):
    for a, fn in arms.items():
        try: got = fn(case["user"])
        except Exception as exc: got = {"error": str(exc)[:80]}
        results[a].append({"case": i, "got": got, **score(got, case)})
    if i % 10 == 9: print(f"  {i+1}/{len(CASES)} done ({time.time()-t0:.0f}s)", file=sys.stderr)

n = len(CASES)
rows = []
print(f"\nmodel: {MODEL}  cases: {n}  (same GGUF blob in every arm, temperature 0)\n")
print(f"{'arm':15} {'any call':>9} {'right tool':>11} {'exact args':>11}")
for a in arms:
    c = sum(r["called"] for r in results[a]); t = sum(r["tool"] for r in results[a]); x = sum(r["args"] for r in results[a])
    print(f"{a:15} {c:>5}/{n:<3} {t:>7}/{n:<3} {x:>7}/{n:<3}")
    rows.append((a, c, t, x))
(root / "out").mkdir(exist_ok=True)
(root / "out/day2.json").write_text(json.dumps({"model": MODEL, "n": n, "results": results, "summary": rows}, indent=1))
