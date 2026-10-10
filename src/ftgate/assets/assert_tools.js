// ftgate: judge one tool-calling response against the expected calls.
//
// Written once by ftgate, run by promptfoo for every case. It reads the
// model's output in whatever shape the provider produced — OpenAI
// tool_calls JSON, a list of function_call items, or raw text with
// <tool_call> blocks — and reports named scores rather than one pass/fail,
// so the table can separate "wrong tool", "wrong arguments" and "the
// call was there but the provider's parser lost it".
module.exports = (output, context) => {
  const expected = context.vars.__expected_calls;   // [{name, arguments}]
  const norm = (v) => {
    if (Array.isArray(v)) return v.map(norm);
    if (v && typeof v === "object") { const o = {}; for (const k of Object.keys(v).sort()) o[k] = norm(v[k]); return o; }
    if (typeof v === "number") return Number(v);
    if (typeof v === "boolean") return v;
    return String(v).trim().toLowerCase();
  };
  // Expected arguments must be present and equal; extra (optional) arguments
  // are allowed and reported separately, so a model that adds unit="celsius"
  // to a weather call is not marked wrong for being thorough.
  const subsetKey = (c, exp) => {
    const got = c.arguments || {}, want = exp.arguments || {};
    const picked = {}; for (const k of Object.keys(want)) picked[k] = got[k];
    return JSON.stringify([c.name, norm(picked)]);
  };
  const wantKey = (e) => JSON.stringify([e.name, norm(e.arguments || {})]);

  let structured = [];
  let text = typeof output === "string" ? output : JSON.stringify(output);
  const takeCall = (c) => {
    const fn = c.function || c;
    let args = fn.arguments ?? fn.input ?? {};
    if (typeof args === "string") { try { args = JSON.parse(args); } catch (e) { return { name: fn.name, arguments: null, malformed: true }; } }
    return { name: fn.name, arguments: args };
  };
  try {
    const o = typeof output === "string" ? JSON.parse(output) : output;
    const list = Array.isArray(o) ? o : (o && o.tool_calls) ? o.tool_calls : (o && o.name) ? [o] : [];
    structured = list.filter(c => c && (c.function || c.name)).map(takeCall);
  } catch (e) { /* not JSON: raw text */ }

  let fromText = [];
  const re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g; let m;
  while ((m = re.exec(text)) !== null) {
    try { const d = JSON.parse(m[1]); fromText.push({ name: d.name, arguments: d.arguments || {} }); }
    catch (e) { fromText.push({ name: null, arguments: null, malformed: true }); }
  }

  const calls = structured.length ? structured : fromText;
  const parserLost = structured.length === 0 && fromText.length > 0;   // the model called; the provider did not surface it
  const malformed = calls.some(c => c.malformed);
  const names = (a) => a.map(c => c.name).sort().join("|");
  const namesOk = names(calls) === names(expected);
  // Pair each expected call with a produced call of the same name (greedy).
  let argsOk = namesOk && !malformed, extra = false;
  if (argsOk) {
    const pool = calls.slice();
    for (const e of expected) {
      const i = pool.findIndex(c => c.name === e.name && subsetKey(c, e) === wantKey(e));
      if (i < 0) { argsOk = false; break; }
      const got = pool[i].arguments || {};
      if (Object.keys(got).some(k => !(k in (e.arguments || {})))) extra = true;
      pool.splice(i, 1);
    }
  }

  return {
    pass: argsOk,
    score: argsOk ? 1 : namesOk ? 0.5 : 0,
    reason: argsOk ? (extra ? "ok, extra optional arguments" : "exact")
      : !calls.length ? "no call"
      : malformed ? "malformed arguments: " + text.slice(0, 160)
      : !namesOk ? `tools ${names(calls) || "-"} vs expected ${names(expected)}`
      : "arguments differ: got " + JSON.stringify(calls.map(c => c.arguments)) + " expected " + JSON.stringify(expected.map(e => e.arguments)),
    namedScores: { called: calls.length ? 1 : 0, names: namesOk ? 1 : 0, args: argsOk ? 1 : 0, extra_args: extra ? 1 : 0, parser_lost: parserLost ? 1 : 0, malformed: malformed ? 1 : 0 },
  };
};
