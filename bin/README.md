# ollama-render

Renders an Ollama model's TEMPLATE exactly as Ollama does — with Ollama's
own `template` package and `api` types — so the prompt we compare against
is Ollama's rendering, not a guess. Ollama has no endpoint for this
(ollama/ollama#12030 is open) and `OLLAMA_DEBUG` does not log the prompt.

```
go build -o ollama-render .          # Go ≥ 1.24; pins github.com/ollama/ollama v0.40.2
echo '{"template": "...", "tools": [...], "messages": [...]}' | ./ollama-render
```

Pin the module version to the Ollama version you are checking against.
