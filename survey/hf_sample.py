"""Sample rows from a Hugging Face dataset through the datasets-server API.

No download of the full dataset: 100 rows per request, chunks spread
evenly across the split (seeded), so the sample is not just the head.
"""
import json, random, sys, time, urllib.parse, urllib.request
from pathlib import Path

def get(url, tries=8):
    """Polite: the datasets-server rate-limits anonymous callers (429)."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "ftgate-survey"}), timeout=90) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            # 429 is the rate limit; 5xx is the server having a moment. Both pass.
            if e.code in (429, 500, 502, 503, 504) and i < tries - 1:
                time.sleep((15 if e.code == 429 else 5) * (i + 1)); continue
            raise
        except Exception:  # noqa: BLE001
            if i == tries - 1: raise
            time.sleep(3 * (i + 1))

def sample(ds, cfg, split, n, out, seed=7):
    size = get(f"https://datasets-server.huggingface.co/size?dataset={urllib.parse.quote(ds)}")
    total = next(c["num_rows"] for c in size["size"]["splits"] if c["config"] == cfg and c["split"] == split)
    chunks = max(1, n // 100)
    rng = random.Random(seed)
    # 100-aligned windows so chunks never overlap; on a small split that
    # means fewer, distinct windows rather than repeated rows.
    aligned = list(range(0, max(100, total - total % 100), 100))
    offsets = sorted(rng.sample(aligned, min(chunks, len(aligned))))
    # A split this small is cheaper to read whole than to argue about: the
    # first published hermes figure came from a sample of a 1,893-row split.
    if total <= 2 * n:
        offsets = list(range(0, total, 100))
    rows = []
    for off in offsets:
        time.sleep(1.5)
        d = get(f"https://datasets-server.huggingface.co/rows?dataset={urllib.parse.quote(ds)}&config={urllib.parse.quote(cfg)}&split={split}&offset={off}&length=100")
        rows += [x["row"] for x in d["rows"]]
    with open(out, "w", encoding="utf-8") as fh:
        for r in rows: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return total, len(rows)

if __name__ == "__main__":
    targets = [
        ("NousResearch/hermes-function-calling-v1", "func_calling", "train"),
        ("glaiveai/glaive-function-calling-v2", "default", "train"),
        ("lockon/xlam-function-calling-60k", "dataset", "train"),
        ("argilla/apigen-function-calling", "default", "train"),
        ("hiyouga/glaive-function-calling-v2-sharegpt", "default", "train"),
        ("allenai/Dolci-Instruct-SFT-Tool-Use", "default", "train"),
    ]
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    raw = Path(__file__).parent / "raw"
    for ds, cfg, split in targets:
        name = ds.replace("/", "__")
        if (raw / f"{name}.jsonl").exists():
            print(f"  {ds:48} already sampled"); continue
        total, got = sample(ds, cfg, split, n, raw / f"{name}.jsonl")
        print(f"  {ds:48} {cfg:22} total {total:>7}  sampled {got}")
