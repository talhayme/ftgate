# ftgate (working name)

Regression checks for a small model you fine-tuned on tool calling: do the
bytes the runtime sends match the bytes the model was trained on, and does
it still call your tools correctly after export and quantisation?

**Status: days 1–2 of a reproduction, not a tool yet.** The premise was
checked before any tool code was written — the same way
[pr-witness](https://github.com/talhayme/pr-witness) was built.

## What has been measured so far

- **llama-server reproduces the HF training bytes exactly** for a
  tool-calling prompt (Qwen2.5-0.5B-Instruct, 250/250 tokens, text and ids
  identical) — even though the GGUF's embedded template hashes differently
  from HF's. Compare renderings, never hashes. → [`notes/day1-findings.md`](notes/day1-findings.md)
- **Ollama 0.40.2 hands the model a Go struct dump instead of the tool
  schema**: `{get_weather … {object <nil> <nil> [city] {…}}}`. Root cause
  in `template/template.go` (`templateToolFunction` has no `String()`);
  known upstream as ollama/ollama#14601 with open PRs #18318 and #18391.
- **It costs accuracy.** Same GGUF blob in every arm, temperature 0, 40
  cases: Qwen2.5-0.5B gets exact arguments 39/40 with its own template and
  31/40 with the prompt Ollama builds — every failure an invented
  parameter `amount from_currency`, read off Go's `[amount from_currency
  to_currency]` rendering of `required`. Qwen2.5-1.5B: 40/40 either way on
  this easy set. → [`notes/day2-impact.md`](notes/day2-impact.md)

## Reproduce

```
uv venv --python 3.12 ~/.venvs/ftgate && uv pip install --python ~/.venvs/ftgate/bin/python torch transformers gguf requests
# llama-server built from llama.cpp with --jinja support; Ollama ≥ 0.40 running on :11434
python scripts/hf_side.py                       # training-side bytes
python scripts/llama_side.py http://localhost:8089
python scripts/ollama_side.py                   # needs bin/ollama-render (see bin/README.md)
python scripts/day2_eval.py qwen2.5:0.5b-instruct   # three-arm accuracy comparison
```

## Plan

`ftgate template` — the three-sided byte comparison as a command.
`ftgate data` — a linter for SFT/DPO/tool-call datasets using the target
model's tokenizer and template. Tool-call evaluation builds on promptfoo's
`tool-call-f1` / `trajectory:tool-args-match` rather than duplicating them.

Apache-2.0.
