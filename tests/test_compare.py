"""The comparison, one test per kind of finding. No network, no model."""

from __future__ import annotations

from ftgate.compare import compare
from ftgate.reference import Rendering
from ftgate.runtimes import RuntimeRendering

REF_TEXT = (
    "<|im_start|>system\nYou are a helpful assistant.\n\n# Tools\n\n<tools>\n"
    '{"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}\n'
    "</tools><|im_end|>\n<|im_start|>user\nWeather in Oslo?<|im_end|>\n<|im_start|>assistant\n"
)
REF_IDS = list(range(1, 51))


def ref(text=REF_TEXT, ids=None):
    return Rendering(text, ids or REF_IDS)


def rt(text=REF_TEXT, ids=None, prefix=None, notes=None, name="rt"):
    return RuntimeRendering(name, text, ids, prefix or [], notes or [])


def kinds(c):
    return [f.kind for f in c.findings]


def test_identical_bytes_and_ids():
    c = compare("x", ref(), rt(ids=REF_IDS), REF_IDS)
    assert c.findings == [] and c.verdict == "identical"


def test_json_spacing_only_is_drift_not_mismatch():
    """Same schema, Go-style compact JSON: different tokens, same meaning."""
    text = REF_TEXT.replace('"type": "function", "function": {"name": "get_weather"', '"type":"function","function":{"name":"get_weather"')
    c = compare("x", ref(), rt(text, ids=REF_IDS[:-2]), REF_IDS[:-2])
    assert kinds(c) == ["spacing"] and c.verdict == "drift"


def test_a_go_struct_dump_in_tools_is_schema_not_json():
    text = REF_TEXT.replace(
        '{"type": "function", "function": {"name": "get_weather", "description": "Weather.", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}',
        '{"type": "function", "function": {get_weather Weather. {object <nil> <nil> [city] {"city":{"type":"string"}}}}}')
    c = compare("x", ref(), rt(text), REF_IDS[:-10])
    assert kinds(c) == ["schema_not_json"] and c.verdict == "mismatch"
    assert c.findings[0].diff  # the diff is attached to the schema finding, not reported twice


def test_doubled_braces_in_the_instruction_are_structural():
    text = REF_TEXT.replace("</tools>", '</tools>\n<tool_call>\n{{"name": <function-name>}}\n</tool_call>')
    c = compare("x", ref(), rt(text), REF_IDS)
    assert "structural" in kinds(c) and c.verdict == "mismatch"


def test_runtime_prepending_bos_is_special_prefix():
    c = compare("x", ref(), rt(ids=[999] + REF_IDS, prefix=[999]), REF_IDS)
    assert "special_prefix" in kinds(c) and c.verdict == "mismatch"


def test_identical_text_different_ids_is_a_tokenizer_finding():
    c = compare("x", ref(), rt(ids=REF_IDS[:-1] + [7]), REF_IDS)
    assert kinds(c) == ["tokenizer"] and c.verdict == "drift"


def test_injected_system_is_named_and_not_double_counted_as_structural():
    text = REF_TEXT.replace("You are a helpful assistant.", "You are Qwen, created by Alibaba Cloud.")
    c = compare("x", ref(), rt(text, notes=["Modelfile SYSTEM injected by the runtime: 'You are Qwen'"]), REF_IDS)
    assert kinds(c) == ["injected_system"] and c.verdict == "mismatch"


def test_cross_check_against_reported_count_is_reported_both_ways():
    ok = compare("x", ref(), rt(notes=["__prompt_eval_count=50"]), REF_IDS)
    assert any("matches the runtime's reported" in n for n in ok.notes)
    bad = compare("x", ref(), rt(notes=["__prompt_eval_count=61"]), REF_IDS)
    assert any("cross-check FAILED" in n for n in bad.notes)
    assert not any(n.startswith("__") for n in bad.notes)


def test_runtime_without_tokenizer_uses_reference_ids_and_says_so():
    c = compare("x", ref(), rt(), REF_IDS[:-3])
    assert c.runtime_tokens == 47
    assert any("cannot tokenize" in n for n in c.notes)
