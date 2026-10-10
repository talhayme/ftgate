"""Runtime adapters. Each answers one question: for these messages and tools,
what text does the runtime build and which token ids does it send?

`ids` may be None when the runtime cannot tokenize for us; the comparison
then falls back to the reference tokenizer on the runtime's text, and says
so in the report.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RuntimeRendering:
    runtime: str
    text: str
    ids: list[int] | None = None
    special_prefix: list[int] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
