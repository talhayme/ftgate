# To Qwen — HF discussion on `Qwen/Qwen2.5-0.5B-Instruct-GGUF` — draft, not posted

**Why there and not Ollama:** Ollama's library blobs for `qwen2.5` are
Qwen's official GGUFs byte-for-byte (same `tokenizer.chat_template` sha
`d5495a1e5d`). The template inside them is the one Qwen fixed on HF on
2024-09-19; the GGUFs were never regenerated. Ollama is a downstream
consumer here, and so is every other runtime that honours the embedded
template.

**Why a discussion, not a metadata PR:** HF's metadata editor could patch
`tokenizer.chat_template` in each file, but there are eight
`Qwen2.5-*-Instruct-GGUF` repos with several quantisations each, and the
fix belongs to the owners. A discussion with the evidence lets them decide
whether to re-embed the template or regenerate.

**Also checked:** the two existing "Update metadata (tokenizer.chat_template)"
PRs are unrelated — #2 on 1.5B was opened by mistake and closed by its
author; #7 on 7B replaces the template with a tool-less one.

**Posted from:** the user's Hugging Face account.

---

**Title:** Embedded chat_template in the Qwen2.5-*-Instruct-GGUF files predates the 2024-09-19 fix (doubled braces in the tool-call instruction)

The `tokenizer.chat_template` embedded in these GGUFs differs from the
template in `Qwen/Qwen2.5-0.5B-Instruct` (and the other sizes) in one line
of the tool-calling instruction:

```
HF tokenizer_config.json (current):   {"name": <function-name>, "arguments": <args-json-object>}
GGUF tokenizer.chat_template:         {{"name": <function-name>, "arguments": <args-json-object>}}
```

The HF repo had the doubled braces at release (2024-09-16) and fixed them
in commit `a8b602d9` on 2024-09-19 ("Upload ./tokenizer_config.json").
The GGUF repos were populated on 2024-09-18 and not regenerated, so they
still carry the pre-fix template. Read directly from the files:

| file | `tokenizer.chat_template` sha256[:10] | instruction line |
|---|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct` `tokenizer_config.json` (main) | `cd8e9439f0` | `{"name": …}` |
| `Qwen/Qwen2.5-1.5B-Instruct` (main) | `cd8e9439f0` | `{"name": …}` |
| fresh `convert_hf_to_gguf.py` of 0.5B main | `cd8e9439f0` | `{"name": …}` |
| `Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q2_k.gguf` | `d5495a1e5d` | `{{"name": …}}` |
| `Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q2_k.gguf` | `d5495a1e5d` | `{{"name": …}}` |

Any runtime that renders the embedded template — llama-server with
`--jinja`, vLLM, LM Studio — instructs the model to emit
`{{"name": …}}`. Rendered through llama-server `/apply-template`, the
prompt for a one-tool conversation differs from the HF rendering by
exactly that line (252 vs 250 tokens). Ollama ships these same files as
its `qwen2.5` library models (its blobs have the same sha), although Ollama
itself renders its own Go template and does not hit this.

Whether it matters depends on the model size and on the runtime's parser.
Measured on the same blob, temperature 0, with llama-server rendering
either the embedded template or the current HF one:

- **`qwen2.5:0.5b-instruct` copies the instruction.** Under the embedded
  template it emits `{{"name": …}}` in 40 of 40 single-tool requests and
  35 of 35 harder ones (nested arguments, parallel calls): zero strictly
  valid JSON `<tool_call>` blocks. Under the HF template the same blob
  emits valid JSON and gets 22/35 of the hard cases exactly right.
- **`qwen2.5:1.5b-instruct` ignores it** — single braces either way,
  30/35 exact in both arms.
- llama-server's tool-call parser tolerates the doubled braces and
  recovers the 0.5B calls (26/35 exact in both arms through
  `/v1/chat/completions`). A parser that `json.loads` the block — vLLM's
  `hermes_tool_parser.py` does, line 92 — gets nothing from the 0.5B
  output.

So for the smallest model the shipped template turns every tool call
into invalid JSON, and it works or fails depending on which server is in
front of it.

A one-line fix on your side: re-embedding the current `chat_template`
with `gguf-py`'s `gguf-set-metadata` (or regenerating from main) aligns
the files with what the model was trained on.

Happy to open metadata PRs per repo if that is the preferred route.

Reproduction: https://github.com/talhayme/ftgate — `notes/day3-template-command.md`
(reading the template with `gguf.GGUFReader`, rendering with llama-server).
