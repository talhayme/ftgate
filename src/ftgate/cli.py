"""ftgate command line."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ftgate import __version__, reference, render
from ftgate.compare import compare
from ftgate.conversations import BUILTIN, load_dataset
from ftgate.runtimes import llama as llama_rt
from ftgate.runtimes import ollama as ollama_rt
from ftgate import data as data_mod
import json as _json
from ftgate import tools_eval


def cmd_tools(args) -> int:
    tools = _json.loads(Path(args.tools).read_text(encoding="utf-8")) if args.tools else []
    if args.from_dataset:
        cases, ds_tools = tools_eval.cases_from_dataset(Path(args.from_dataset), args.holdout)
        tools = tools or ds_tools
    else:
        cases = tools_eval.load_cases(Path(args.cases))
    if not cases:
        print("no cases", file=sys.stderr); return 2
    if not tools:
        print("no tools: pass --tools tools.json (or a dataset whose rows carry `tools`)", file=sys.stderr); return 2
    arms = tools_eval.run(cases, tools, args.provider, keep=Path(args.keep) if args.keep else None, concurrency=args.concurrency)
    if args.format == "json":
        print(_json.dumps([a.__dict__ for a in arms], indent=2, ensure_ascii=False))
    else:
        print(tools_eval.to_text(arms), end="")
    if args.min_exact is not None:
        worst = min((a.args / a.n) for a in arms if a.n) if arms else 0
        return 1 if worst < args.min_exact else 0
    return 0


def cmd_data(args) -> int:
    rep = data_mod.lint(Path(args.path), model=args.model, max_seq_len=args.max_seq_len,
                        eval_path=Path(args.eval) if args.eval else None, limit=args.limit)
    print(_json.dumps(rep.as_dict(), indent=2, ensure_ascii=False) if args.format == "json" else data_mod.to_text(rep), end="")
    if args.fail_on == "error":
        return 1 if rep.count("error") else 0
    if args.fail_on == "warning":
        return 1 if rep.count("error") or rep.count("warning") else 0
    return 0


def cmd_template(args) -> int:
    convs = load_dataset(Path(args.dataset), args.sample) if args.dataset else BUILTIN
    if not args.llama and not args.ollama:
        print("nothing to compare against: pass --llama URL and/or --ollama MODEL", file=sys.stderr); return 2

    results = []
    for conv in convs:
        msgs = conv.inference_messages
        ref = reference.render(args.model, msgs, conv.tools, add_generation_prompt=True)
        if args.llama:
            rt = llama_rt.render(args.llama.rstrip("/"), msgs, conv.tools)
            results.append(compare(conv.name, ref, rt, reference.tokenize(args.model, rt.text)))
        if args.ollama:
            rt = ollama_rt.render(args.ollama_url.rstrip("/"), args.ollama, msgs, conv.tools, verify=not args.no_verify)
            results.append(compare(conv.name, ref, rt, reference.tokenize(args.model, rt.text)))

    print(render.to_json(results, args.model) if args.format == "json" else render.to_text(results, args.model, args.diff), end="")
    if args.fail_on == "mismatch":
        return 1 if any(r.verdict == "mismatch" for r in results) else 0
    if args.fail_on == "drift":
        return 1 if any(r.verdict != "identical" for r in results) else 0
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="ftgate", description="Do runtimes send the bytes the model was trained on?")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="command", required=True)

    t = sub.add_parser("template", help="compare runtime prompts and tokens against the training-side rendering")
    t.add_argument("--model", required=True, help="HF model id or path whose tokenizer and chat template were used for training")
    t.add_argument("--llama", metavar="URL", help="llama-server base URL, e.g. http://localhost:8080")
    t.add_argument("--ollama", metavar="MODEL", help="Ollama model name, e.g. qwen2.5:0.5b-instruct")
    t.add_argument("--ollama-url", default="http://localhost:11434")
    t.add_argument("--no-verify", action="store_true", help="skip the live cross-check of Ollama's prompt token count")
    t.add_argument("--dataset", metavar="JSONL", help="your own conversations (messages[, tools] per line) instead of the built-in set")
    t.add_argument("--sample", type=int, default=20, help="rows to take from --dataset (default 20)")
    t.add_argument("--diff", action="store_true", help="print unified diffs for findings")
    t.add_argument("--format", choices=["text", "json"], default="text")
    t.add_argument("--fail-on", choices=["never", "drift", "mismatch"], default="never")
    t.set_defaults(func=cmd_template)

    d = sub.add_parser("data", help="lint a fine-tuning dataset with the target model's tokenizer and template")
    d.add_argument("path", help="JSONL: messages / ShareGPT / Alpaca / preference rows")
    d.add_argument("--model", help="HF model id; enables tokenizer and template checks")
    d.add_argument("--max-seq-len", type=int, help="the trainer's cutoff; rows beyond it are reported")
    d.add_argument("--eval", metavar="JSONL", help="eval set to check for leakage (near-duplicates)")
    d.add_argument("--limit", type=int)
    d.add_argument("--format", choices=["text", "json"], default="text")
    d.add_argument("--fail-on", choices=["never", "warning", "error"], default="never")
    d.set_defaults(func=cmd_data)

    tl = sub.add_parser("tools", help="evaluate tool calling on your tools and cases, base vs tuned vs quantised, via promptfoo")
    tl.add_argument("--tools", metavar="JSON", help="OpenAI-style tool list")
    tl.add_argument("--cases", metavar="FILE", help="cases: {user, calls[, system]} per row (json/jsonl/yaml)")
    tl.add_argument("--from-dataset", metavar="JSONL", help="hold out cases from your training rows instead")
    tl.add_argument("--holdout", type=int, default=50)
    tl.add_argument("--provider", action="append", required=True,
                    help="repeatable: ollama:MODEL | openai:MODEL@http://host:port/v1 | any promptfoo provider id")
    tl.add_argument("--keep", metavar="DIR", help="keep the generated promptfoo config and results here")
    tl.add_argument("--concurrency", type=int, default=1, help="requests in flight (default 1: local servers answer one at a time)")
    tl.add_argument("--min-exact", type=float, help="exit 1 if any arm's exact-args rate is below this")
    tl.add_argument("--format", choices=["text", "json"], default="text")
    tl.set_defaults(func=cmd_tools)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
