"""Conversations to compare: built-in fixtures, or rows from the user's own dataset.

The built-in set covers the shapes a chat template has to get right —
plain chat, a system prompt, a tool call with a response, several tools.
The user's dataset is the one that matters: those are the rows the model
was actually trained on, so a mismatch there is a mismatch that counts.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "City name"},
    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}
CONVERT = {"type": "function", "function": {"name": "convert_currency", "description": "Convert an amount between currencies.",
    "parameters": {"type": "object", "properties": {"amount": {"type": "number"}, "from_currency": {"type": "string"},
    "to_currency": {"type": "string"}}, "required": ["amount", "from_currency", "to_currency"]}}}


@dataclass
class Conversation:
    name: str
    messages: list[dict]
    tools: list[dict] = field(default_factory=list)

    @property
    def inference_messages(self) -> list[dict]:
        """Everything up to the final assistant turn — what a runtime is sent."""
        msgs = self.messages
        if msgs and msgs[-1].get("role") == "assistant":
            msgs = msgs[:-1]
        return msgs


BUILTIN = [
    Conversation("plain-chat", [
        {"role": "user", "content": "Name three primary colours."},
        {"role": "assistant", "content": "Red, blue and yellow."},
    ]),
    Conversation("system-prompt", [
        {"role": "system", "content": "You are a terse assistant. Answer in one sentence."},
        {"role": "user", "content": "Why is the sky blue?"},
        {"role": "assistant", "content": "Air scatters short wavelengths more than long ones."},
    ]),
    Conversation("tool-call-and-response", [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Tbilisi?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": {"city": "Tbilisi", "unit": "celsius"}}}]},
        {"role": "tool", "content": "{\"temp_c\": 21, \"sky\": \"clear\"}"},
        {"role": "assistant", "content": "It's 21°C and clear in Tbilisi."},
    ], tools=[WEATHER]),
    Conversation("two-tools-three-params", [
        {"role": "user", "content": "Convert 100 USD to EUR."},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "convert_currency", "arguments": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"}}}]},
    ], tools=[WEATHER, CONVERT]),
]


def load_dataset(path: Path, limit: int | None = None) -> list[Conversation]:
    """Rows from a JSONL file in the common `messages` (+ optional `tools`) shape.

    ShareGPT-style `conversations` with `from`/`value` is mapped onto it; any
    other shape is reported rather than guessed at.
    """
    out: list[Conversation] = []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "messages" in row:
                msgs = row["messages"]
            elif "conversations" in row:
                role = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}
                msgs = [{"role": role.get(m.get("from"), m.get("from")), "content": m.get("value", "")} for m in row["conversations"]]
            else:
                raise ValueError(f"{path}:{n+1}: no `messages` or `conversations` key; keys are {sorted(row)}")
            out.append(Conversation(f"{path.name}:{n+1}", msgs, row.get("tools") or []))
            if limit and len(out) >= limit:
                break
    return out
