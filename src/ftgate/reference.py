"""The reference: what the trainer fed the model.

Rendered with the model's own chat template through HF transformers — the
same call Unsloth, TRL and Axolotl make — and tokenized with its tokenizer.
Every runtime is judged against this.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from transformers import AutoTokenizer


@dataclass
class Rendering:
    text: str
    ids: list[int]
    bos: str | None = None
    eos: str | None = None


@lru_cache(maxsize=4)
def tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)


def render(model: str, messages: list[dict], tools: list[dict], add_generation_prompt: bool) -> Rendering:
    tok = tokenizer(model)
    kwargs = {"tools": tools} if tools else {}
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt, **kwargs)
    ids = tok(text, add_special_tokens=False)["input_ids"]
    return Rendering(text, ids, tok.bos_token, tok.eos_token)


def tokenize(model: str, text: str) -> list[int]:
    return tokenizer(model)(text, add_special_tokens=False)["input_ids"]
