"""ftgate tools: config generation, result parsing, the JS judge. No promptfoo run."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ftgate import tools_eval
from ftgate.tools_eval import Case, provider_entry, write_config, parse_results, cases_from_dataset

TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]


def test_provider_specs_expand_to_promptfoo_ids():
    o = provider_entry("ollama:qwen2.5:0.5b-instruct", TOOLS)
    assert o["id"] == "ollama:chat:qwen2.5:0.5b-instruct" and o["config"]["tools"] == TOOLS and o["config"]["temperature"] == 0
    l = provider_entry("openai:m@http://localhost:8090/v1", TOOLS)
    assert l["id"] == "openai:chat:m" and l["config"]["apiBaseUrl"] == "http://localhost:8090/v1"
    assert provider_entry("anthropic:messages:claude-x", TOOLS)["id"] == "anthropic:messages:claude-x"


def test_config_has_a_string_chat_prompt_and_one_test_per_case(tmp_path):
    cfg = json.loads(write_config(tmp_path, [Case("hi", [{"name": "get_weather", "arguments": {"city": "Oslo"}}])],
                                  TOOLS, ["ollama:x"]).read_text())
    assert isinstance(cfg["prompts"][0], str) and json.loads(cfg["prompts"][0])[1]["role"] == "user"
    assert len(cfg["tests"]) == 1 and cfg["tests"][0]["vars"]["__expected_calls"][0]["name"] == "get_weather"
    assert (tmp_path / "assert_tools.js").exists()


def test_results_are_tallied_per_arm(tmp_path):
    def row(label, ns=None, error=None, desc="c"):
        r = {"provider": {"id": "x", "label": label}, "testCase": {"description": desc}}
        if error: r["error"] = error
        else:
            r["namedScores"] = ns; r["gradingResult"] = {"componentResults": [{"reason": "arguments differ"}]}
            if not ns.get("args"): r["error"] = "arguments differ"   # promptfoo copies a failed assertion's reason here
        return r
    data = {"results": {"results": [
        row("A", {"called": 1, "names": 1, "args": 1}), row("A", {"called": 1, "names": 1, "args": 0}),
        row("A", {"called": 1, "names": 0, "args": 0, "parser_lost": 1}), row("B", error="timeout"),
    ]}}
    p = tmp_path / "r.json"; p.write_text(json.dumps(data))
    arms = {a.provider: a for a in parse_results(p)}
    assert (arms["A"].n, arms["A"].args, arms["A"].names, arms["A"].parser_lost) == (3, 1, 2, 1)
    assert arms["B"].errors == 1 and len(arms["A"].failures) == 2


def test_cases_are_held_out_from_a_dataset(tmp_path):
    rows = [
        {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Weather in Oslo?"},
                      {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\"}"}}]},
                      {"role": "tool", "content": "{}"}, {"role": "assistant", "content": "cold"}], "tools": TOOLS},
        {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},   # no call: skipped
    ]
    p = tmp_path / "d.jsonl"; p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    cases, tools = cases_from_dataset(p, 10)
    assert len(cases) == 1 and cases[0].user == "Weather in Oslo?" and cases[0].system == "S"
    assert cases[0].calls == [{"name": "get_weather", "arguments": {"city": "Oslo"}}] and tools == TOOLS


node = shutil.which("node")


@pytest.mark.skipif(node is None, reason="node not installed")
@pytest.mark.parametrize("output,expect", [
    # OpenAI-shaped structured call, exact
    (json.dumps([{"type": "function", "function": {"name": "get_weather", "arguments": "{\"city\": \"oslo\"}"}}]), (1, 1, 1, 0, 0)),
    # right tool, wrong argument
    (json.dumps([{"function": {"name": "get_weather", "arguments": "{\"city\": \"Bergen\"}"}}]), (1, 1, 0, 0, 0)),
    # right tool, expected argument present, an extra optional one: still ok
    (json.dumps([{"function": {"name": "get_weather", "arguments": "{\"city\": \"Oslo\", \"unit\": \"celsius\"}"}}]), (1, 1, 1, 0, 0)),
    # wrong tool
    (json.dumps([{"function": {"name": "set_timer", "arguments": "{}"}}]), (1, 0, 0, 0, 0)),
    # raw text with a <tool_call> the provider did not surface → parser lost, but still judged
    ('<tool_call>\n{"name": "get_weather", "arguments": {"city": "Oslo"}}\n</tool_call>', (1, 1, 1, 1, 0)),
    # the doubled-brace output from day 4: malformed
    ('<tool_call>\n{{"name": "get_weather", "arguments": {"city": "Oslo"}}}\n</tool_call>', (1, 0, 0, 1, 1)),
    # no call at all
    ("I cannot help with that.", (0, 0, 0, 0, 0)),
])
def test_the_js_judge(tmp_path, output, expect):
    """(called, names, args, parser_lost, malformed) for each shape of output."""
    js = tools_eval.ASSET.read_text()
    runner = tmp_path / "run.js"
    runner.write_text(js.replace("module.exports =", "const judge =") + """
const r = judge(process.argv[2], {vars: {__expected_calls: [{name: "get_weather", arguments: {city: "Oslo"}}]}});
console.log(JSON.stringify(r.namedScores));
""")
    out = subprocess.run([node, str(runner), output], capture_output=True, text=True, check=True).stdout
    ns = json.loads(out)
    assert (ns["called"], ns["names"], ns["args"], ns["parser_lost"], ns["malformed"]) == expect
