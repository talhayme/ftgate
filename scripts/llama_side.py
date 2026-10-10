"""The llama.cpp side, via llama-server's own endpoints — no emulation."""
import difflib, hashlib, json, sys, requests
from pathlib import Path
from transformers import AutoTokenizer

base = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8089"
root = Path(__file__).resolve().parent.parent
fx = json.loads((root / "fixtures/toolcall.json").read_text())
hf = json.loads((root / "out/hf.json").read_text())
tok = AutoTokenizer.from_pretrained(hf["model"])

def post(path, body):
    r = requests.post(f"{base}{path}", json=body, timeout=60); r.raise_for_status(); return r.json()

props = requests.get(f"{base}/props", timeout=10).json()
gguf_template_sha = hashlib.sha256((props.get("chat_template") or "").encode()).hexdigest()[:12]

infer_text = post("/apply-template", {"messages": fx["messages"][:-1], "tools": fx["tools"]})["prompt"]
infer_ids = post("/tokenize", {"content": infer_text, "add_special": True, "parse_special": True})["tokens"]
special_prefix = post("/tokenize", {"content": "", "add_special": True, "parse_special": True})["tokens"]

out = {"side": "llama-server", "template_sha": gguf_template_sha, "bos_token": props.get("bos_token"),
       "special_prefix": special_prefix, "infer": {"text": infer_text, "ids": infer_ids}}
(root / "out/llama.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

def report(name, mine_text, mine_ids):
    ref_text, ref_ids = hf["infer"]["text"], hf["infer"]["ids"]
    print(f"\n[{name}] {len(mine_ids)} tokens vs training-side {len(ref_ids)}")
    if mine_text == ref_text: print("  template text: IDENTICAL")
    else:
        print("  template text: DIFFERS")
        for line in difflib.unified_diff(ref_text.splitlines(), mine_text.splitlines(), "hf", name, lineterm="", n=1):
            print("   ", line)
    hf_ids_same_text = tok(mine_text, add_special_tokens=False)["input_ids"]
    print("  tokenizer on identical text:", "IDENTICAL" if mine_ids == hf_ids_same_text else f"DIFFERS ({len(mine_ids)} vs {len(hf_ids_same_text)})")
    print("  final ids vs training-side:", "IDENTICAL" if mine_ids == ref_ids else "DIFFER")

print(f"GGUF embedded template sha {gguf_template_sha} vs HF {hf['template_sha']}: "
      f"{'SAME' if gguf_template_sha == hf['template_sha'] else 'DIFFERENT'}")
print(f"add_special prepends: {special_prefix} (bos_token reported: {props.get('bos_token')!r})")
report("llama-server", infer_text, infer_ids)
