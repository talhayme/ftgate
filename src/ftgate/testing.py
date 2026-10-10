"""Use ftgate from a test suite.

    from ftgate.testing import assert_dataset_clean

    def test_training_data():
        assert_dataset_clean("data/train.jsonl", model="Qwen/Qwen2.5-0.5B-Instruct",
                             max_seq_len=2048, allow={"role_repeat"})

A failing assertion prints the same report the CLI does, so the test log
says which rows and why.
"""
from __future__ import annotations

from pathlib import Path

from ftgate import data as _data


def lint_dataset(path, model: str | None = None, max_seq_len: int | None = None, eval_path=None, limit: int | None = None):
    return _data.lint(Path(path), model=model, max_seq_len=max_seq_len,
                      eval_path=Path(eval_path) if eval_path else None, limit=limit)


def assert_dataset_clean(path, model: str | None = None, max_seq_len: int | None = None, eval_path=None,
                         level: str = "error", allow: set[str] | None = None, limit: int | None = None) -> None:
    """Fail if the dataset has issues at `level` ("error" or "warning") or worse, except codes in `allow`."""
    rep = lint_dataset(path, model, max_seq_len, eval_path, limit)
    levels = ("error",) if level == "error" else ("error", "warning")
    bad = [i for i in rep.issues if i.level in levels and i.code not in (allow or set())]
    if bad:
        raise AssertionError(f"{len(bad)} {level}(s) in {path}\n" + _data.to_text(rep, max_rows=25))
