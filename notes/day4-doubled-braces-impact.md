# Day 4 — does the doubled-brace instruction matter? Yes, for the small model

Day 3 said the embedded `{{"name": …}}` instruction "changed no answers at
0.5B". That was wrong, and the reason it looked right is itself the
finding: llama-server's tool-call parser forgives the model's output.

## Setup

Two llama-servers on the **same GGUF blob** (Qwen's official file, as
shipped by Ollama), temperature 0. One renders the template embedded in
the file (`{{"name"`), the other the current HF template (`{"name"`),
via `--chat-template-file`. Nothing else differs.

35 harder cases (`scripts/day4_hard_cases.py`): nested objects and arrays,
an object-valued argument, a no-argument tool, strings containing braces
and JSON, parallel calls, mixed parallel calls. Two judges:

- **strict** — raw `/completion` output, every `<tool_call>` block through
  `json.loads`. This is what a hermes-style parser does (vLLM's
  `hermes_tool_parser.py` line 92).
- **lenient** — llama-server's own `/v1/chat/completions` parser.

## Qwen2.5-0.5B

| judge | arm | any call | malformed blocks | right tools | exact args | `{{` in output |
|---|---|---|---|---|---|---|
| strict | embedded `{{` | 33/35 | **39** | **0/35** | **0/35** | **35/35** |
| strict | HF `{` | 35/35 | 1 | 32/35 | 22/35 | 1 |
| lenient | embedded `{{` | 35/35 | — | 31/35 | 26/35 | — |
| lenient | HF `{` | 35/35 | — | 31/35 | 26/35 | — |

The model copies the instruction. Re-checked on day 2's 40 easy cases:
`{{` in **40/40** raw outputs, zero strictly valid blocks. Day 2's 39/40
came through the lenient parser.

## Qwen2.5-1.5B

| judge | arm | right tools | exact args | `{{` in output |
|---|---|---|---|---|
| strict | embedded `{{` | 34/35 | 30/35 | 1 |
| strict | HF `{` | 34/35 | 30/35 | 1 |
| lenient | both | 35/35 | 30/35 | — |

The 1.5B model ignores the doubled braces entirely. (The single `{{` is a
`braces`-kind case whose note text contains `{{`.)

## Reading

- The shipped Qwen2.5 GGUF template makes the **0.5B** model emit invalid
  JSON on every tool call. Whether a user notices depends on the server:
  llama-server recovers all of it; a strict parser recovers none.
- Larger models absorb the bad instruction, so the people affected are
  exactly the small-model fine-tuners this project is for.
- "No visible impact" was a parser artefact. The strict judge is the
  honest one for a format instruction, because the model's job is to
  emit the format, not to be rescued from it.
- Neither size handles the `braces` cases (text containing `{…}`) — a
  model limitation, same in both arms, not a template effect.

Draft for Qwen updated: `docs/upstream/02-qwen-gguf-template.md`.
