# Comment on ollama/ollama#14601 — posted 2026-10-10

**Why a comment, not a new issue:** #14601 (open since 2026-03-03) describes
exactly this bug for Qwen3, and two open PRs propose fixes: #18318 (value
receivers for the tool `String()` methods) and #18391 (render tool function
and parameters as JSON). A new issue would be a duplicate. What is missing
in the thread is a measured accuracy impact and the Qwen2.5 confirmation.

**Posted from:** talhayme.

---

Confirming this on **Qwen2.5** as well (library `qwen2.5` template, Ollama
0.40.2), and adding a measurement of what it costs, since the thread has
the mechanism but not the impact.

**Mechanism, pinned to the current source.** `template/template.go`
converts tools via `convertToolsForTemplate`; `templateTool`,
`templateTools` and `templateProperties` all have a JSON `String()`, but
`templateToolFunction` does not. The library template renders
`{{ .Function }}`, so `fmt` prints the struct positionally. For a
three-parameter tool the model receives:

```
{"type": "function", "function": {convert_currency Convert an amount from one currency to another. {object <nil> <nil> [amount from_currency to_currency] {"amount":{"type":"number"},"from_currency":{...},"to_currency":{...}}}}}
```

Note `required` rendered as a Go slice: `[amount from_currency to_currency]`
— no quotes, no commas.

**Impact, same weights, prompt as the only variable.** 40 single-tool
requests over 5 tools, temperature 0. Every arm loads the *same GGUF blob*
Ollama uses, so quantisation is held constant:

| arm | prompt | right tool | exact arguments |
|---|---|---|---|
| Ollama `/api/chat` with `tools` | as shipped | 40/40 | **31/40** |
| llama-server, model's own Jinja template | correct schema | 40/40 | **39/40** |
| llama-server fed the prompt Ollama builds (rendered with Ollama's own `template` package) | as shipped | 40/40 | **31/40** |

`qwen2.5:0.5b-instruct`. The third arm reproduces Ollama's score exactly, so
the engine is not the variable; the prompt is. Tool *selection* is
unaffected; argument accuracy drops 20 points.

All eight failures are the three-parameter tool, and all eight have the
same shape:

```
expected: {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}
got:      {"amount from_currency": "USD", "from_currency": "USD", "to_currency": "EUR"}
```

The model invents a parameter called `amount from_currency` — the first two
words of the Go-printed `required` list. With the real schema it gets all
eight right.

`qwen2.5:1.5b-instruct` scores 40/40 in every arm on this set, so the
larger model absorbs it here; the cost lands on the small models people
fine-tune and export.

Either #18318 or #18391 would fix this; #18391 is the more direct one for
`{{ .Function }}` templates. Happy to re-run the 40 cases against a build
of either PR if that helps it along.

Reproduction (cases, the three arms, and the Go renderer that uses Ollama's
`template` package so the third arm is Ollama's rendering rather than a
guess): https://github.com/talhayme/ftgate
