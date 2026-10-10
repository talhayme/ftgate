"""ftgate data — lint a fine-tuning dataset with the target model's own tokenizer and template.

A dataset row can be wrong in ways no schema validator sees: an assistant
turn that renders to zero trainable tokens, a conversation the trainer
silently truncates before the answer, a tool call whose arguments do not
match the schema the model is being taught, a DPO pair where chosen and
rejected are the same text. These checks run the row through the same
template the trainer will, so what they report is what the model sees.

Three levels: `error` (the row trains the wrong thing or nothing),
`warning` (likely a mistake), `info` (worth knowing). `--fail-on` turns
levels into an exit code.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROLES = {"system", "user", "assistant", "tool"}
_SHARE = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool", "function": "tool"}


@dataclass
class Issue:
    level: str          # error | warning | info
    code: str
    row: int | None
    detail: str

    def as_dict(self) -> dict:
        return {"level": self.level, "code": self.code, "row": self.row, "detail": self.detail}


@dataclass
class Report:
    path: str
    format: str
    rows: int
    issues: list[Issue] = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def count(self, level: str) -> int:
        return sum(i.level == level for i in self.issues)

    def as_dict(self) -> dict:
        return {"schema": "ftgate/data/v0", "path": self.path, "format": self.format, "rows": self.rows,
                "errors": self.count("error"), "warnings": self.count("warning"), "info": self.count("info"),
                "stats": self.stats, "issues": [i.as_dict() for i in self.issues]}


# ---------------------------------------------------------------- formats

def detect_format(row: dict) -> str:
    if "chosen" in row and "rejected" in row:
        return "preference"
    if "messages" in row:
        return "messages"
    if "conversations" in row:
        return "sharegpt"
    if "instruction" in row and ("output" in row or "response" in row):
        return "alpaca"
    return "unknown"


def to_messages(row: dict, fmt: str) -> list[dict]:
    if fmt == "messages":
        return list(row["messages"])
    if fmt == "sharegpt":
        return [{"role": _SHARE.get(m.get("from"), m.get("from")), "content": m.get("value", "")} for m in row["conversations"]]
    if fmt == "alpaca":
        user = row["instruction"] + (("\n\n" + row["input"]) if row.get("input") else "")
        return [{"role": "user", "content": user}, {"role": "assistant", "content": row.get("output", row.get("response", ""))}]
    if fmt == "preference":
        prompt = row.get("prompt")
        chosen, rejected = row["chosen"], row["rejected"]
        if isinstance(chosen, list):      # conversational preference: chosen/rejected are message lists
            return list(chosen)
        return [{"role": "user", "content": prompt or ""}, {"role": "assistant", "content": chosen}]
    return []


# ------------------------------------------------------- structural checks

def check_structure(n: int, msgs: list[dict], tools: list[dict], issues: list[Issue]) -> None:
    if not msgs:
        issues.append(Issue("error", "empty_row", n, "no messages")); return

    roles = [m.get("role") for m in msgs]
    for i, r in enumerate(roles):
        if r not in ROLES:
            issues.append(Issue("error", "bad_role", n, f"message {i}: role {r!r}"))

    if roles[-1] != "assistant":
        issues.append(Issue("error", "no_assistant_target", n,
                            f"last message is {roles[-1]!r}; SFT has nothing to train on"))
    if not any(r == "assistant" for r in roles):
        issues.append(Issue("error", "no_assistant_turn", n, "no assistant message at all"))

    for i, m in enumerate(msgs):
        if m.get("role") == "assistant" and not (m.get("content") or "").strip() and not m.get("tool_calls"):
            issues.append(Issue("error", "empty_assistant", n, f"message {i}: assistant content is empty"))
        if m.get("role") == "user" and not (m.get("content") or "").strip():
            issues.append(Issue("warning", "empty_user", n, f"message {i}: user content is empty"))

    # alternation: user/assistant should alternate once the system prompt is past
    # A tool turn sits between an assistant call and the assistant's answer,
    # so it must stay in the sequence or every tool exchange reads as a repeat.
    core = [r for r in roles if r in ("user", "assistant", "tool")]
    for a, b in zip(core, core[1:]):
        if a == b and a != "tool":
            issues.append(Issue("warning", "role_repeat", n, f"two consecutive {a!r} turns")); break
    if len(roles) > 1 and roles.count("system") > 1:
        issues.append(Issue("warning", "multiple_system", n, "more than one system message"))
    if "system" in roles[1:]:
        issues.append(Issue("warning", "system_not_first", n, "system message is not the first message"))

    # tool-call integrity
    declared = {t.get("function", {}).get("name") for t in tools or []}
    pending_ids: set[str] = set()
    for i, m in enumerate(msgs):
        for tc in m.get("tool_calls") or []:
            fn = tc.get("function", {})
            name = fn.get("name")
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    issues.append(Issue("error", "tool_args_not_json", n, f"message {i}: arguments of {name!r} are not JSON")); args = None
            if name is None:
                # Already reported as unparseable arguments; a second finding
                # for the missing name would count the same defect twice.
                continue
            if tools and name not in declared:
                issues.append(Issue("error", "undeclared_tool", n, f"message {i}: calls {name!r}, not in tools"))
            elif tools and isinstance(args, dict):
                schema = next((t["function"].get("parameters") or {} for t in tools if t.get("function", {}).get("name") == name), {})
                props = schema.get("properties") or {}
                for req in schema.get("required") or []:
                    if req not in args:
                        issues.append(Issue("error", "tool_args_missing_required", n, f"message {i}: {name!r} missing required {req!r}"))
                for k in args:
                    if props and k not in props:
                        issues.append(Issue("warning", "tool_args_unknown_key", n, f"message {i}: {name!r} has unknown argument {k!r}"))
            if tc.get("id"):
                pending_ids.add(tc["id"])
        if m.get("role") == "tool":
            prev_calls = any(x.get("tool_calls") for x in msgs[:i] if x.get("role") == "assistant")
            if not prev_calls:
                issues.append(Issue("error", "tool_without_call", n, f"message {i}: tool message with no preceding tool_calls"))
            if m.get("tool_call_id") and pending_ids and m["tool_call_id"] not in pending_ids:
                issues.append(Issue("error", "tool_call_id_mismatch", n, f"message {i}: tool_call_id {m['tool_call_id']!r} matches no call"))


def check_preference(n: int, row: dict, issues: list[Issue], ratios: list[float]) -> None:
    c, r = row["chosen"], row["rejected"]
    ct = json.dumps(c, sort_keys=True) if isinstance(c, list) else str(c)
    rt = json.dumps(r, sort_keys=True) if isinstance(r, list) else str(r)
    if ct.strip() == rt.strip():
        issues.append(Issue("error", "chosen_equals_rejected", n, "chosen and rejected are identical; the pair teaches nothing"))
    elif ct.strip().lower() == rt.strip().lower():
        issues.append(Issue("warning", "chosen_rejected_case_only", n, "chosen and rejected differ only in case"))
    if len(rt):
        ratios.append(len(ct) / len(rt))


# ------------------------------------------------------- tokenizer checks

def check_with_model(n: int, msgs: list[dict], tools: list[dict], tok, max_seq_len: int | None,
                     issues: list[Issue], lengths: list[int], tools_in_prompt: bool = False) -> None:
    # Some datasets write the tool signatures into the system message
    # themselves. Passing them to the template as well would render them
    # twice and double the measured length, so such rows say so.
    kwargs = {"tools": tools} if tools and not tools_in_prompt else {}
    try:
        full = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False, **kwargs)
    except Exception as exc:  # noqa: BLE001 — the template raising is the finding
        issues.append(Issue("error", "template_error", n, f"chat template failed on this row: {str(exc)[:120]}")); return

    ids = tok(full, add_special_tokens=False)["input_ids"]
    lengths.append(len(ids))

    # Where does the last assistant turn start? Everything from there is the
    # trainable target under train_on_responses_only-style masking.
    prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **kwargs) if msgs[-1].get("role") == "assistant" else full
    n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])
    target = len(ids) - n_prefix
    if msgs[-1].get("role") == "assistant" and target <= 0:
        issues.append(Issue("error", "zero_target_tokens", n, "the final assistant turn renders to zero tokens after the prompt"))

    if tok.eos_token and not full.rstrip().endswith(tok.eos_token):
        # Many templates end the assistant turn with their own marker (<|im_end|>); eos is that marker for Qwen.
        issues.append(Issue("warning", "no_eos", n, f"rendered row does not end with eos {tok.eos_token!r}; the model may never learn to stop"))

    if max_seq_len and len(ids) > max_seq_len:
        if n_prefix >= max_seq_len:
            issues.append(Issue("error", "truncated_before_target", n,
                                f"{len(ids)} tokens > max_seq_len {max_seq_len}; the cut falls before the assistant turn — this row trains nothing"))
        else:
            issues.append(Issue("warning", "truncated", n,
                                f"{len(ids)} tokens > max_seq_len {max_seq_len}; {len(ids) - max_seq_len} target tokens are silently dropped"))


# ------------------------------------------------------------ duplicates

def _shingles(text: str, k: int = 5) -> set[int]:
    words = re.findall(r"\w+", text.lower())
    return {hash(" ".join(words[i:i + k])) for i in range(max(1, len(words) - k + 1))}


def check_duplicates(rows_text: list[str], issues: list[Issue], eval_text: list[str] | None, threshold: float = 0.8) -> None:
    seen: dict[str, int] = {}
    for n, t in enumerate(rows_text):
        h = hashlib.sha256(t.encode()).hexdigest()
        if h in seen:
            issues.append(Issue("warning", "duplicate_row", n, f"exact duplicate of row {seen[h]}"))
        else:
            seen[h] = n
    if eval_text:
        ev = [_shingles(t) for t in eval_text]
        for n, t in enumerate(rows_text):
            s = _shingles(t)
            if not s:
                continue
            for j, e in enumerate(ev):
                if not e:
                    continue
                jac = len(s & e) / len(s | e)
                if jac >= threshold:
                    issues.append(Issue("error", "eval_leak", n, f"near-duplicate of eval row {j} (jaccard {jac:.2f}); the eval set is contaminated")); break


# ------------------------------------------------------------------ main

def lint(path: Path, model: str | None = None, max_seq_len: int | None = None,
         eval_path: Path | None = None, limit: int | None = None) -> Report:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
            if limit and len(rows) >= limit:
                break
    fmt = Counter(detect_format(r) for r in rows).most_common(1)[0][0] if rows else "unknown"
    rep = Report(str(path), fmt, len(rows))
    issues = rep.issues

    tok = None
    if model:
        from ftgate.reference import tokenizer
        tok = tokenizer(model)

    lengths: list[int] = []; ratios: list[float] = []; texts: list[str] = []
    roles = Counter()
    for n, row in enumerate(rows):
        rf = detect_format(row)
        if rf == "unknown":
            issues.append(Issue("error", "unknown_format", n, f"keys {sorted(row)[:6]} match no known format")); texts.append(""); continue
        if rf != fmt:
            issues.append(Issue("warning", "mixed_format", n, f"row is {rf}, file is mostly {fmt}"))
        msgs = to_messages(row, rf)
        tools = row.get("tools") or []
        roles.update(m.get("role") for m in msgs)
        check_structure(n, msgs, tools, issues)
        if rf == "preference":
            check_preference(n, row, issues, ratios)
        if tok is not None:
            check_with_model(n, msgs, tools, tok, max_seq_len, issues, lengths, bool(row.get("tools_in_prompt")))
        texts.append(json.dumps(msgs, sort_keys=True, ensure_ascii=False))

    eval_text = None
    if eval_path:
        with eval_path.open(encoding="utf-8") as fh:
            eval_text = [json.dumps(to_messages(r, detect_format(r)), sort_keys=True, ensure_ascii=False)
                         for r in (json.loads(l) for l in fh if l.strip())]
    check_duplicates(texts, issues, eval_text)

    rep.stats = {"roles": dict(roles)}
    if lengths:
        q = statistics.quantiles(lengths, n=100) if len(lengths) > 1 else [lengths[0]] * 99
        rep.stats["tokens"] = {"min": min(lengths), "p50": int(q[49]), "p95": min(int(q[94]), max(lengths)), "max": max(lengths)}
        if max_seq_len:
            rep.stats["over_max_seq_len"] = sum(l > max_seq_len for l in lengths)
    if ratios:
        mean = statistics.fmean(ratios)
        rep.stats["chosen_to_rejected_length"] = round(mean, 2)
        if mean > 1.5 or mean < 0.67:
            issues.append(Issue("warning", "length_bias", None,
                                f"chosen is on average {mean:.2f}× the length of rejected; the model may learn length, not quality"))
    issues.sort(key=lambda i: ({"error": 0, "warning": 1, "info": 2}[i.level], i.row if i.row is not None else -1))
    return rep


def to_text(rep: Report, max_rows: int = 40) -> str:
    out = [f"ftgate data — {rep.path}", f"  format {rep.format}, {rep.rows} rows, roles {rep.stats.get('roles', {})}"]
    if "tokens" in rep.stats:
        t = rep.stats["tokens"]; out.append(f"  tokens per row: min {t['min']}  p50 {t['p50']}  p95 {t['p95']}  max {t['max']}"
                                           + (f"  ({rep.stats['over_max_seq_len']} over max_seq_len)" if "over_max_seq_len" in rep.stats else ""))
    if "chosen_to_rejected_length" in rep.stats:
        out.append(f"  chosen/rejected length ratio: {rep.stats['chosen_to_rejected_length']}")
    out.append("")
    by_code = Counter((i.level, i.code) for i in rep.issues)
    for (level, code), k in sorted(by_code.items(), key=lambda x: ({"error": 0, "warning": 1, "info": 2}[x[0][0]], -x[1])):
        out.append(f"  {level:7} {code:28} ×{k}")
    out.append("")
    for i in rep.issues[:max_rows]:
        where = f"row {i.row}" if i.row is not None else "file"
        out.append(f"  [{i.level}] {where}: {i.detail}")
    if len(rep.issues) > max_rows:
        out.append(f"  … {len(rep.issues) - max_rows} more")
    out.append("")
    out.append(f"  {rep.count('error')} error(s), {rep.count('warning')} warning(s), {rep.count('info')} info")
    return "\n".join(out) + "\n"
