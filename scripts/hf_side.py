"""The training side: the bytes a trainer actually feeds the model.

Two renderings of the fixture, because they answer different questions:
  train  — the whole conversation, as SFT formats it (what the weights saw)
  infer  — the conversation up to the final assistant turn, with the
           generation prompt appended (what a runtime sends at inference)

Runtimes are compared on `infer`. `train` is kept so a later check can ask
whether the GGUF's embedded template still renders it the same way.
"""
import hashlib, json, sys
from pathlib import Path
from transformers import AutoTokenizer

model = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-0.5B-Instruct"
root = Path(__file__).resolve().parent.parent
fx = json.loads((root / "fixtures/toolcall.json").read_text())
tok = AutoTokenizer.from_pretrained(model)

def render(messages, gen):
    text = tok.apply_chat_template(messages, tools=fx["tools"], tokenize=False, add_generation_prompt=gen)
    return {"text": text, "ids": tok(text, add_special_tokens=False)["input_ids"]}

out = {
    "model": model, "side": "hf",
    "bos": tok.bos_token, "eos": tok.eos_token,
    "template_sha": hashlib.sha256((tok.chat_template or "").encode()).hexdigest()[:12],
    "train": render(fx["messages"], gen=False),
    "infer": render(fx["messages"][:-1], gen=True),
}
(root / "out").mkdir(exist_ok=True)
(root / "out/hf.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))
print(f"hf: train {len(out['train']['ids'])} tokens, infer {len(out['infer']['ids'])} tokens, "
      f"bos={tok.bos_token!r} eos={tok.eos_token!r}, template sha {out['template_sha']}")
