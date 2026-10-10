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

On a 40-request tool-calling set at 0.5B the doubled braces did not change
any answer — the model ignores the braces in the instruction — so the
practical impact looks small. It is still a template that differs from
training, embedded in the most widely distributed GGUFs for this family,
and it is a one-line fix: re-embedding the current `chat_template` with
`gguf-py`'s `gguf-set-metadata` (or regenerating from main) would align
them.

Happy to open metadata PRs per repo if that is the preferred route.

Reproduction: https://github.com/talhayme/ftgate — `notes/day3-template-command.md`
(reading the template with `gguf.GGUFReader`, rendering with llama-server).
