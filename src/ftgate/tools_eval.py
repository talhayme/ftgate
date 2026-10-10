"""ftgate tools — does the model still call *your* tools correctly, on the backend it will run on?

promptfoo does the evaluation. ftgate writes the config: your tools, your
cases (hand-written, or held out from your training set), one provider
per arm — base, fine-tuned, quantised, the same weights behind Ollama and
llama-server — and a deterministic judge that reports, per case, whether
the right tool was chosen, whether the arguments were exact, and whether
a call the model did make was lost by the provider's parser. That last
column is the one no generic eval gives you, and it is where a fine-tune
that "works in the notebook" dies in production.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ASSET = Path(__file__).with_name("assets") / "assert_tools.js"


@dataclass
class Case:
    user: str
    calls: list[dict]                       # [{"name": ..., "arguments": {...}}]
    system: str | None = None
    name: str | None = None


def load_cases(path: Path) -> list[Case]:
    """cases.jsonl / cases.json: `{"user": ..., "calls": [...], "system"?: ...}` per row.

    YAML is accepted when PyYAML is installed; JSON needs nothing.
    """
    text = path.read_text(encoding="utf-8")
    rows: list[dict]
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional dependency
        rows = yaml.safe_load(text)
    elif path.suffix == ".jsonl":
        rows = [json.loads(l) for l in text.splitlines() if l.strip()]
    else:
        rows = json.loads(text)
    if isinstance(rows, dict):
        rows = rows.get("cases", [])
    return [Case(r["user"], r["calls"], r.get("system"), r.get("name")) for r in rows]


def cases_from_dataset(path: Path, limit: int, fmt_hint: str | None = None) -> tuple[list[Case], list[dict]]:
    """Hold-out cases from the user's own training rows.

    Takes rows whose first assistant turn is a tool call: the user text (and
    system prompt) become the case, the call becomes the expectation, and
    the row's `tools` are collected. These are the behaviours the model was
    trained on, so a miss is a regression, not a generalisation failure.
    """
    from ftgate.data import detect_format, to_messages
    cases, tools_seen, tools = [], set(), []
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh):
            if not line.strip():
                continue
            row = json.loads(line)
            msgs = to_messages(row, detect_format(row))
            first_call = next((i for i, m in enumerate(msgs) if m.get("role") == "assistant" and m.get("tool_calls")), None)
            if first_call is None:
                continue
            system = next((m["content"] for m in msgs[:first_call] if m.get("role") == "system"), None)
            users = [m["content"] for m in msgs[:first_call] if m.get("role") == "user"]
            if not users:
                continue
            calls = []
            for tc in msgs[first_call]["tool_calls"]:
                fn = tc.get("function", tc)
                args = fn.get("arguments")
                if isinstance(args, str):
                    try: args = json.loads(args)
                    except json.JSONDecodeError: args = None
                if fn.get("name") and isinstance(args, dict):
                    calls.append({"name": fn["name"], "arguments": args})
            if not calls:
                continue
            for t in row.get("tools") or []:
                k = json.dumps(t, sort_keys=True)
                if k not in tools_seen:
                    tools_seen.add(k); tools.append(t)
            cases.append(Case(users[-1], calls, system, f"{path.name}:{n+1}"))
            if len(cases) >= limit:
                break
    return cases, tools


def provider_entry(spec: str, tools: list[dict]) -> dict:
    """Turn a short provider spec into a promptfoo provider entry.

      ollama:qwen2.5:0.5b-instruct         → ollama:chat:qwen2.5:0.5b-instruct
      openai:MODEL@http://host:8090/v1     → openai:chat:MODEL with apiBaseUrl
      anything else is passed through as a promptfoo provider id
    """
    config: dict = {"temperature": 0, "tools": tools}
    if spec.startswith("ollama:") and not spec.startswith("ollama:chat:"):
        pid = "ollama:chat:" + spec[len("ollama:"):]
    elif spec.startswith("openai:") and "@" in spec:
        model, base = spec[len("openai:"):].split("@", 1)
        pid = f"openai:chat:{model}"
        config["apiBaseUrl"] = base
        config.setdefault("apiKey", os.environ.get("OPENAI_API_KEY", "ftgate"))
    else:
        pid = spec
    return {"id": pid, "label": spec, "config": config}


def write_config(workdir: Path, cases: list[Case], tools: list[dict], providers: list[str]) -> Path:
    shutil.copy(ASSET, workdir / "assert_tools.js")
    tests = []
    for i, c in enumerate(cases):
        tests.append({
            "description": c.name or f"case {i}",
            "vars": {"user": c.user, "system": c.system or "You are a helpful assistant.", "__expected_calls": c.calls},
            "assert": [{"type": "javascript", "value": "file://assert_tools.js"}],
        })
    cfg = {
        "description": "ftgate tools",
        # A chat prompt is a JSON *string* of the message list; promptfoo
        # renders the {{vars}} and parses it into messages for chat providers.
        "prompts": [json.dumps([{"role": "system", "content": "{{system}}"}, {"role": "user", "content": "{{user}}"}])],
        "providers": [provider_entry(p, tools) for p in providers],
        "tests": tests,
    }
    path = workdir / "promptfooconfig.json"
    path.write_text(json.dumps(cfg, indent=1, ensure_ascii=False), encoding="utf-8")
    return path


def promptfoo_bin() -> list[str]:
    env = os.environ.get("FTGATE_PROMPTFOO")
    if env:
        return [env]
    local = Path(__file__).resolve().parents[2] / ".pfrunner" / "node_modules" / ".bin" / "promptfoo"
    if local.exists():
        return [str(local)]
    if shutil.which("promptfoo"):
        return ["promptfoo"]
    return ["npx", "--yes", "promptfoo"]


@dataclass
class ArmResult:
    provider: str
    n: int = 0
    called: int = 0
    names: int = 0
    args: int = 0
    parser_lost: int = 0
    malformed: int = 0
    extra_args: int = 0
    errors: int = 0
    failures: list[dict] = field(default_factory=list)


def parse_results(path: Path) -> list[ArmResult]:
    data = json.loads(path.read_text(encoding="utf-8"))
    res = data.get("results", data)
    arms: dict[str, ArmResult] = {}
    for r in res.get("results", []):
        label = (r.get("provider") or {}).get("label") or (r.get("provider") or {}).get("id") or "?"
        arm = arms.setdefault(label, ArmResult(label))
        arm.n += 1
        # promptfoo writes a failed assertion's reason into `error` too; a
        # real provider error has no grading result at all.
        if r.get("error") and not r.get("gradingResult") and not r.get("namedScores"):
            arm.errors += 1; continue
        ns = r.get("namedScores") or {}
        arm.called += int(ns.get("called", 0)); arm.names += int(ns.get("names", 0)); arm.args += int(ns.get("args", 0))
        arm.parser_lost += int(ns.get("parser_lost", 0)); arm.malformed += int(ns.get("malformed", 0))
        arm.extra_args += int(ns.get("extra_args", 0))
        if not ns.get("args"):
            comps = (r.get("gradingResult") or {}).get("componentResults") or []
            reason = comps[0].get("reason") if comps else (r.get("gradingResult") or {}).get("reason", "")
            arm.failures.append({"case": (r.get("testCase") or {}).get("description"), "reason": reason})
    return list(arms.values())


def run(cases: list[Case], tools: list[dict], providers: list[str], keep: Path | None = None,
        concurrency: int = 1) -> list[ArmResult]:
    with tempfile.TemporaryDirectory(prefix="ftgate-tools-") as tmp:
        work = Path(keep) if keep else Path(tmp)
        work.mkdir(parents=True, exist_ok=True)
        cfg = write_config(work, cases, tools, providers)
        out = work / "results.json"
        # Local backends (Ollama, llama-server) serve one request at a time;
        # promptfoo's default of four in flight turns into timeouts. One is
        # the honest default; raise it for hosted APIs.
        proc = subprocess.run([*promptfoo_bin(), "eval", "-c", str(cfg), "-o", str(out), "--no-cache", "--no-progress-bar",
                               "--max-concurrency", str(concurrency)],
                              cwd=work, capture_output=True, text=True, timeout=3600,
                              env={**os.environ, "PROMPTFOO_DISABLE_TELEMETRY": "1", "PROMPTFOO_DISABLE_UPDATE": "1"})
        if not out.exists():
            raise RuntimeError("promptfoo produced no results:\n" + (proc.stderr or proc.stdout)[-1500:])
        return parse_results(out)


def to_text(arms: list[ArmResult], show_failures: int = 8) -> str:
    out = ["ftgate tools — same cases, every arm, temperature 0", ""]
    out.append(f"{'arm':44} {'called':>8} {'right tool':>11} {'args ok':>9} {'extra args':>11} {'parser lost':>12} {'malformed':>10} {'errors':>7}")
    for a in arms:
        out.append(f"{a.provider:44} {a.called:>4}/{a.n:<3} {a.names:>7}/{a.n:<3} {a.args:>5}/{a.n:<3} {a.extra_args:>7}     {a.parser_lost:>8}     {a.malformed:>6}     {a.errors:>4}")
    out.append("")
    out.append("  right tool = the expected set of tool names; args ok = every expected argument present and equal")
    out.append("  (extra optional arguments allowed, counted under 'extra args'); parser lost = the model emitted a")
    out.append("  <tool_call> the provider did not surface; malformed = a call whose JSON does not parse.")
    for a in arms:
        if a.failures:
            out.append(f"\n  {a.provider}: first failures")
            for f in a.failures[:show_failures]:
                out.append(f"    - {f['case']}: {f['reason']}")
    return "\n".join(out) + "\n"
