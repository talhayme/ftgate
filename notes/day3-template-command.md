# Day 3 — `ftgate template`, and two more things it found on its first run

The three-sided comparison is now a command:

```
ftgate template --model Qwen/Qwen2.5-0.5B-Instruct \
                --llama http://localhost:8090 --ollama qwen2.5:0.5b-instruct \
                [--dataset train.jsonl --sample 20] [--diff] [--format json] [--fail-on mismatch]
```

Reference = HF `apply_chat_template` + tokenizer (what SFT trains on).
llama-server = its own `/apply-template` + `/tokenize`. Ollama = TEMPLATE
from `/api/show`, rendered with Ollama's `template` package via the Go
helper, token count cross-checked against a live request's
`prompt_eval_count`. Findings are classified: `schema_not_json`,
`injected_system`, `special_prefix`, `structural`, `spacing`, `tokenizer`.

Ran on the four built-in conversations. Result on Qwen2.5-0.5B:

```
ok plain-chat              llama-server     34 vs 34   identical
!! plain-chat              ollama           34 vs 34   mismatch   [injected_system]
ok system-prompt           llama-server     30 vs 30   identical
ok system-prompt           ollama           30 vs 30   identical
!! tool-call-and-response  llama-server    252 vs 250  mismatch   [structural]
!! tool-call-and-response  ollama          222 vs 250  mismatch   [schema_not_json]
!! two-tools-three-params  llama-server    286 vs 284  mismatch   [structural]
!! two-tools-three-params  ollama          231 vs 284  mismatch   [schema_not_json, injected_system]
```

Every Ollama cross-check matches the live `prompt_eval_count` (34, 30,
222, 231), so the Ollama column is Ollama's bytes, not an approximation.

## Finding 3: Ollama's GGUF blobs embed a template with doubled braces

The two `[structural]` rows on **llama-server** were a surprise — day 1
had llama-server byte-identical. The difference: day 1 used my own f16
conversion of the HF checkpoint; today llama-server loads **Ollama's
library blob**. The diff is one line of the tool-call instruction:

```
-{"name": <function-name>, "arguments": <args-json-object>}
+{{"name": <function-name>, "arguments": <args-json-object>}}
```

Read straight out of the files with `gguf.GGUFReader`:

| file | chat_template sha | instruction line |
|---|---|---|
| HF Qwen2.5-0.5B-Instruct `tokenizer_config.json` | `cd8e9439f0` | `{"name": …}` |
| HF Qwen2.5-1.5B-Instruct | `cd8e9439f0` | `{"name": …}` |
| my `convert_hf_to_gguf.py` output | `cd8e9439f0` | `{"name": …}` |
| **Ollama blob `qwen2.5:0.5b-instruct`** | **`d5495a1e5d`** | **`{{"name": …}}`** |
| **Ollama blob `qwen2.5:1.5b-instruct`** | **`d5495a1e5d`** | **`{{"name": …}}`** |

Ollama itself never renders that Jinja — it uses its Go TEMPLATE — so
Ollama users do not see this. Anyone who pulls Ollama's blob and serves it
with llama.cpp, vLLM or anything that honours the embedded template gets a
model instructed to emit `{{"name": …}}`. On the 40-case set it changed no
answers at 0.5B (the model ignores the instruction's braces), but it is a
template that differs from training, embedded in the most widely
distributed GGUFs for this family. Where the doubling comes from is not
established; the likely place is Ollama's own HF→GGUF conversion.

## Finding 4: Ollama injects the Modelfile SYSTEM

When the request has no system message, Ollama's server prepends the
Modelfile's `SYSTEM` — for `qwen2.5` that is *"You are Qwen, created by
Alibaba Cloud. You are a helpful assistant."* A fine-tune whose training
rows had no system prompt, or a different one, is served with this one in
front of every conversation. Found because the first run's cross-check
failed (13 rendered vs 34 evaluated); the adapter now does what the server
does, and the comparison names it `injected_system`.

Not a bug — it is documented Modelfile behaviour — but it is precisely
the kind of difference the fine-tuner does not know about, and it costs
tokens of context on every request.

## Where the tool stands

- Four classes of training/runtime mismatch found on one model family in
  three days, none of them visible without diffing the bytes: a Go struct
  dump for the schema (Ollama), a doubled-brace instruction (Ollama's
  GGUFs under other runtimes), an injected system prompt (Ollama),
  compact-vs-spaced JSON (Ollama, `drift`).
- 9 unit tests on the comparison, no network. Live adapters are exercised
  by hand against both servers; a recorded-fixture test for them is next.
- `--dataset` takes the user's own JSONL, which is the real use: the rows
  the model was trained on, checked against the runtime it will run in.
