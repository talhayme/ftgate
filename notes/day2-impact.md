# Day 2 — does the garbled schema cost accuracy?

Date: 2026-10-10. Forty tool-calling requests over five tools
(`scripts/day2_cases.py`), temperature 0, **the same GGUF blob Ollama
itself uses** loaded into every arm, so quantisation is held constant:

| arm | engine | prompt |
|---|---|---|
| `ollama` | Ollama 0.40.2 | its Go template — the garbled schema, as shipped |
| `llama-correct` | llama-server, Ollama's blob | the model's own Jinja template |
| `llama-garbled` | llama-server, Ollama's blob | the prompt Ollama would build, rendered with Ollama's `template` package |

If `llama-garbled` tracks `ollama` and both trail `llama-correct`, the
template is the cause and the engine is not.

## Qwen2.5-0.5B-Instruct

| arm | any call | right tool | exact args |
|---|---|---|---|
| `ollama` | 40/40 | 40/40 | **31/40** |
| `llama-correct` | 40/40 | 40/40 | **39/40** |
| `llama-garbled` | 40/40 | 40/40 | **31/40** |

Tool *selection* is unaffected. Argument accuracy drops 20 points, and the
garbled arm reproduces Ollama's number exactly — same failures, same cases.

Every one of the eight failures is `convert_currency`, the one tool with
three required parameters, and every one has the same shape:

```
expected: {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}
got:      {"amount from_currency": "USD", "from_currency": "USD", "to_currency": "EUR"}
```

The model invents a parameter named `amount from_currency`. That string
comes straight from the prompt. The garbled schema renders `required` as
Go prints a slice — `[amount from_currency to_currency]`, no quotes, no
commas — and the model reads the first two words as one key. With the
real schema (`"required": ["amount", "from_currency", "to_currency"]`) it
gets all eight right.

The single `llama-correct` miss (`city: "Berlin, Germany"`) is a model
quirk, not a template effect; the garbled arm makes the same choice.

## Qwen2.5-1.5B-Instruct

| arm | any call | right tool | exact args |
|---|---|---|---|
| `ollama` | 40/40 | 40/40 | 40/40 |
| `llama-correct` | 40/40 | 40/40 | 40/40 |
| `llama-garbled` | 40/40 | 40/40 | 40/40 |

The 1.5B model absorbs the garbled schema on this set. These cases are
easy on purpose (one obvious tool per request); a harder set might show a
gap at 1.5B too, but that is not measured and should not be claimed.

## What this says

- The template mismatch is not cosmetic: on the smallest model it turns
  a 39/40 tool-caller into a 31/40 one, with a failure mode (a merged
  parameter name) that is invisible unless you diff the prompts.
- Small models pay first. That is the project's premise, and it is the
  population that fine-tunes on tool calling and exports to Ollama.
- The three-arm design is the right shape for `ftgate tools`: same
  weights, prompt as the only variable, Ollama's own renderer for the
  Ollama arm.

## Reproduce

```
python scripts/day2_eval.py qwen2.5:0.5b-instruct    # needs Ollama on :11434 and
python scripts/day2_eval.py qwen2.5:1.5b-instruct    # llama-server on :8090 with the same blob
```
Raw results: `out/day2-0.5b.json`, `out/day2-1.5b.json`.
