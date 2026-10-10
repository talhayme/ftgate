// Render an Ollama TEMPLATE exactly as Ollama does, using Ollama's own
// template package and api types. Reads {"template": "...", "tools": [...],
// "messages": [...]} on stdin, writes the prompt to stdout. The template
// itself decides whether to append the generation prompt (Ollama's do when
// the last message is not from the assistant).
package main

import (
	"encoding/json"
	"io"
	"os"

	"github.com/ollama/ollama/api"
	"github.com/ollama/ollama/template"
)

type request struct {
	Template string        `json:"template"`
	Tools    []api.Tool    `json:"tools"`
	Messages []api.Message `json:"messages"`
}

func main() {
	raw, _ := io.ReadAll(os.Stdin)
	var req request
	if err := json.Unmarshal(raw, &req); err != nil { panic(err) }
	t, err := template.Parse(req.Template)
	if err != nil { panic(err) }
	if err := t.Execute(os.Stdout, template.Values{Messages: req.Messages, Tools: req.Tools}); err != nil { panic(err) }
}
