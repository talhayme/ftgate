# Day 1 — do runtimes send the bytes the model was trained on?

Date: 2026-10-10. Model: Qwen/Qwen2.5-0.5B-Instruct (HF snapshot 7ae5576),
converted to GGUF f16 with llama.cpp's converter; Ollama library model
`qwen2.5:0.5b-instruct` (a8b0c5157701). Fixture: `fixtures/toolcall.json`,
a five-turn tool-calling conversation with one tool, `get_weather`.

Compared on the **inference prompt**: the conversation up to the final
assistant turn, generation prompt appended. (Comparing the full
conversation is misleading — llama-server treats a trailing assistant
message as a prefill and omits its `<|im_end|>`.)

## Result

| side | how obtained | tokens | vs training bytes |
|---|---|---|---|
| HF transformers (training) | `apply_chat_template(tools=…)` | 250 | — |
| llama-server 0.6.0-dev | `/apply-template` + `/tokenize` | 250 | **identical**, text and ids |
| Ollama 0.40.2 | its own Go template, rendered with Ollama's `template` package | 222 | **differs**; count matches Ollama's reported `prompt_eval_count` exactly |

## Ollama: the tool definition reaches the model as a Go struct dump

Training bytes (HF, also what llama-server sends):

```
{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.", "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
```

What Ollama sends:

```
{"type": "function", "function": {get_weather Get the current weather for a city. {object <nil> <nil> [city] {"city":{"type":"string","description":"City name"},"unit":{"type":"string","enum":["celsius","fahrenheit"]}}}}}
```

Not JSON. No `"name":`, no `"description":`, no `"parameters":`, `<nil>`
literals, Go slice syntax for `required`. The model was trained to read a
JSON schema in `<tools>` and is handed Go's `%v` of a struct.

Root cause, in Ollama's source (v0.40.2), `template/template.go`:

- `Execute` converts the request's tools into template-friendly types
  (`convertToolsForTemplate`, line 459). `templateTool`, `templateTools` and
  `templateProperties` each define a `String()` that marshals JSON.
- **`templateToolFunction` does not** (line ~417). The Qwen2.5 TEMPLATE
  reaches into it with `{{ .Function }}`, so Go's `fmt` prints the struct
  positionally: `{Name Description {Type <nil> <nil> [Required] {…}}}`. The
  inner `{…}` is JSON only because `templateProperties` has a `String()`.
  That is exactly the shape in the captured prompt.
- Any Ollama template that renders `{{ .Function }}` rather than
  `{{ . }}` or `{{ .Function.Name }}` hits this. The library `qwen2.5`
  template does.
- `server/prompt.go:152` calls the same `Execute`, so the rendering above is
  what the server sends; the token count (222 = `prompt_eval_count`) is the
  independent confirmation.

Second, smaller mismatch: tool-call *arguments* are marshalled compactly
(`{"city":"Tbilisi","unit":"celsius"}`) where training had spaces
(`{"city": "Tbilisi", "unit": "celsius"}`). Different tokens, same meaning;
a model fine-tuned on spaced JSON sees unspaced JSON at inference.

Already known upstream, all open as of today:
- ollama/ollama#18318 — api: use value receivers for tool String methods
- ollama/ollama#18391 — template: render tool function and parameters as JSON
- ollama/ollama#14601 — Qwen3 tool calling via /api/chat: malformed tool definitions

So this is not a corner case I constructed. It is the live behaviour of
the most popular local runtime, for one of the most popular small-model
families, on exactly the surface (tool calling) the project is about.

## What this settles for the tool

1. **The premise holds, measurably.** "The runtime sends different bytes
   than training" is true for Ollama today, and the difference is not
   whitespace — it is the schema itself.
2. **Compare renderings, never template hashes.** The GGUF's embedded
   template hashes differently from HF's (`55b2f4a26ac9` vs `cd8e9439f057`)
   and renders *identically*. A hash check would have raised a false alarm
   on the one runtime that is correct.
3. **llama-server needs no emulation**: `/apply-template` and `/tokenize`
   give both halves directly.
4. **Ollama needs emulation, and the honest emulation is Ollama's own
   code.** There is no tokenize or render endpoint (#12030 open), and
   `OLLAMA_DEBUG=2` does not log the prompt in 0.40. Rendering the TEMPLATE
   from `/api/show` with Ollama's `template` package reproduces the server
   byte-for-byte — verified by the token count. This is a small Go binary
   the tool can ship, pinned to the Ollama version.
5. **BOS is a non-finding here** — llama-server reports `bos_token:
   <|endoftext|>` but `add_special` prepends nothing for Qwen. Needs
   re-checking on Llama/Gemma, where it is the FAQ's second named cause.

## Environment notes (cost a few hours, worth writing down)

- `/usr/local/bin/python3.11`, brew, Node and the brew-installed Ollama on
  this Mac are all **x86_64 under Rosetta**. torch for Intel Mac stops at
  2.2.2; brew Ollama ships no runner at all. `uv` (arm64) fixed Python,
  cmake and Go; Ollama came from the GitHub release tarball.
- A venv inside `~/Desktop` cannot be deleted cleanly — iCloud sync holds
  files. Project venv lives in `~/.venvs/ftgate`.
- llama.cpp releases carry no macOS binaries; built from source with
  `-DCMAKE_OSX_ARCHITECTURES=arm64 -DLLAMA_SERVER_WEBUI=OFF` (the web UI
  tries to download a tarball at configure time).

## Next

- Day 2: does the garbled schema actually hurt? Run the same tool-calling
  cases through llama-server and Ollama and compare tool-selection and
  argument accuracy. That number is the demo.
- Then `ftgate template`: the three-sided comparison as a command, with
  the Go renderer for Ollama and the HF tokenizer as the reference.
