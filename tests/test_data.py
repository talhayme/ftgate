"""The dataset linter: one test per issue code. Structural tests need no model;
the tokenizer tests use Qwen2.5-0.5B-Instruct from the local HF cache."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ftgate import data

W = {"type": "function", "function": {"name": "get_weather", "description": "w",
     "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def write(tmp_path: Path, rows, name="d.jsonl") -> Path:
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return p


def codes(rep, level=None):
    return {i.code for i in rep.issues if level is None or i.level == level}


def chat(*turns):
    return {"messages": [{"role": r, "content": c} for r, c in turns]}


# --- formats ------------------------------------------------------------

def test_formats_are_detected():
    assert data.detect_format({"messages": []}) == "messages"
    assert data.detect_format({"conversations": []}) == "sharegpt"
    assert data.detect_format({"instruction": "x", "output": "y"}) == "alpaca"
    assert data.detect_format({"chosen": "a", "rejected": "b"}) == "preference"
    assert data.detect_format({"foo": 1}) == "unknown"


def test_sharegpt_and_alpaca_map_onto_messages():
    sg = data.to_messages({"conversations": [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "yo"}]}, "sharegpt")
    assert [m["role"] for m in sg] == ["user", "assistant"]
    al = data.to_messages({"instruction": "Translate", "input": "cat", "output": "кот"}, "alpaca")
    assert al[0]["content"] == "Translate\n\ncat" and al[1]["content"] == "кот"


# --- structure ----------------------------------------------------------

def test_a_clean_row_has_no_issues(tmp_path):
    rep = data.lint(write(tmp_path, [chat(("user", "hi"), ("assistant", "hello"))]))
    assert rep.issues == []


def test_empty_assistant_is_an_error(tmp_path):
    rep = data.lint(write(tmp_path, [chat(("user", "hi"), ("assistant", "  "))]))
    assert "empty_assistant" in codes(rep, "error")


def test_row_ending_on_user_has_no_target(tmp_path):
    rep = data.lint(write(tmp_path, [chat(("user", "hi"), ("assistant", "yo"), ("user", "and?"))]))
    assert "no_assistant_target" in codes(rep, "error")


def test_consecutive_same_roles_warn(tmp_path):
    rep = data.lint(write(tmp_path, [chat(("user", "a"), ("user", "b"), ("assistant", "c"))]))
    assert "role_repeat" in codes(rep, "warning")


def test_tool_call_arguments_are_checked_against_the_declared_schema(tmp_path):
    row = {"messages": [{"role": "user", "content": "w?"},
                        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function",
                         "function": {"name": "get_weather", "arguments": {"town": "Oslo"}}}]},
                        {"role": "tool", "tool_call_id": "c9", "content": "{}"},
                        {"role": "assistant", "content": "ok"}], "tools": [W]}
    rep = data.lint(write(tmp_path, [row]))
    assert {"tool_args_missing_required", "tool_call_id_mismatch"} <= codes(rep, "error")
    assert "tool_args_unknown_key" in codes(rep, "warning")


def test_undeclared_tool_and_non_json_arguments(tmp_path):
    row = {"messages": [{"role": "user", "content": "t?"},
                        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_time", "arguments": "{"}}]},
                        {"role": "assistant", "content": "ok"}], "tools": [W]}
    rep = data.lint(write(tmp_path, [row]))
    assert {"undeclared_tool", "tool_args_not_json"} <= codes(rep, "error")


def test_tool_message_without_a_call(tmp_path):
    rep = data.lint(write(tmp_path, [chat(("user", "hi"), ("tool", "x"), ("assistant", "ok"))]))
    assert "tool_without_call" in codes(rep, "error")


def test_exact_duplicates_and_eval_leakage(tmp_path):
    base = chat(("system", "S"), ("user", "Tell me about the weather in Oslo today please"), ("assistant", "It is cold and clear in Oslo today."))
    ev = write(tmp_path, [base], "eval.jsonl")
    rep = data.lint(write(tmp_path, [base, base]), eval_path=ev)
    assert "duplicate_row" in codes(rep, "warning")
    assert "eval_leak" in codes(rep, "error")


def test_preference_pairs(tmp_path):
    rows = [{"prompt": "q", "chosen": "The red one.", "rejected": "The red one."},
            {"prompt": "q", "chosen": "A much longer answer " * 10, "rejected": "short"}]
    rep = data.lint(write(tmp_path, rows))
    assert "chosen_equals_rejected" in codes(rep, "error")
    assert "length_bias" in codes(rep, "warning")


def test_unknown_rows_are_errors_not_crashes(tmp_path):
    rep = data.lint(write(tmp_path, [{"foo": "bar"}, chat(("user", "a"), ("assistant", "b"))]))
    assert "unknown_format" in codes(rep, "error") and rep.rows == 2


# --- with the model -----------------------------------------------------

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tok_available():
    try:
        from ftgate.reference import tokenizer
        tokenizer(MODEL)
    except Exception:  # noqa: BLE001
        pytest.skip("Qwen tokenizer not available offline")


def test_truncation_before_the_target_is_an_error(tmp_path, tok_available):
    rep = data.lint(write(tmp_path, [chat(("user", "word " * 700), ("assistant", "short"))]), model=MODEL, max_seq_len=512)
    assert "truncated_before_target" in codes(rep, "error")


def test_truncation_inside_the_target_is_a_warning(tmp_path, tok_available):
    rep = data.lint(write(tmp_path, [chat(("user", "q"), ("assistant", "long answer " * 300))]), model=MODEL, max_seq_len=512)
    assert "truncated" in codes(rep, "warning")
    assert rep.stats["over_max_seq_len"] == 1


def test_token_stats_are_reported(tmp_path, tok_available):
    rep = data.lint(write(tmp_path, [chat(("user", "hi"), ("assistant", "hello"))]), model=MODEL)
    assert set(rep.stats["tokens"]) == {"min", "p50", "p95", "max"}
    assert "no_eos" not in codes(rep)  # Qwen's template ends the assistant turn with <|im_end|> == eos


def test_the_same_messages_with_different_tools_are_not_exact_duplicates(tmp_path):
    """glaive repeats a conversation under reworded tool descriptions in ~3%
    of rows. The template renders the tools, so those rows train on different
    text; the first survey called them duplicates and blamed a conversion."""
    def row(description):
        r = chat(("user", "convert 5 USD"), ("assistant", "ok"))
        r["tools"] = [{"type": "function", "function": {"name": "convert", "description": description,
                                                        "parameters": {"type": "object", "properties": {}}}}]
        return r
    rep = data.lint(write(tmp_path, [row("The amount to be converted"), row("The amount to convert"),
                                     row("The amount to convert")]))
    by_code = {i.code: i.row for i in rep.issues}
    assert by_code.get("repeated_conversation") == 1      # same messages, different tools
    assert by_code.get("duplicate_row") == 2              # same messages and same tools
