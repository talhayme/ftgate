"""The Ollama side: the prompt Ollama actually assembles from its own TEMPLATE.

Ollama has no apply-template or tokenize endpoint (ollama/ollama#12030 is
open), so the only authoritative source is the server itself: with
OLLAMA_DEBUG=1 it logs the final prompt for each request. We send the
fixture conversation and read it back from the log.
"""
import json, re, sys, time, requests
from pathlib import Path

model = sys.argv[1] if len(sys.argv) > 1 else "qwen2.5:0.5b-instruct"
log = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/w/ollama.log")
root = Path(__file__).resolve().parent.parent
fx = json.loads((root / "fixtures/toolcall.json").read_text())

mark = log.stat().st_size if log.exists() else 0
r = requests.post("http://localhost:11434/api/chat", json={
    "model": model, "messages": fx["messages"], "tools": fx["tools"],
    "stream": False, "options": {"num_predict": 1},
}, timeout=120)
r.raise_for_status()
time.sleep(1)

tail = log.read_bytes()[mark:].decode("utf-8", "replace")
# Ollama logs the assembled prompt as a quoted Go string: prompt="..."
m = re.search(r'prompt="((?:[^"\\]|\\.)*)"', tail)
if not m:
    print("prompt not found in log tail; last lines:"); print(tail[-1500:]); sys.exit(1)
text = json.loads('"' + m.group(1) + '"')   # Go %q escaping is JSON-compatible for our purposes
out = {"model": model, "side": "ollama", "text": text}
(root / "out").mkdir(exist_ok=True)
(root / "out/ollama.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"ollama: prompt of {len(text)} chars captured")
print("----- assembled prompt -----"); print(text)
