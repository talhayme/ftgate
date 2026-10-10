"""The pytest helper, the plugin switch, the pre-commit hook and the action manifest."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from ftgate.testing import assert_dataset_clean

ROOT = Path(__file__).resolve().parent.parent


def write(tmp_path, rows, name="d.jsonl"):
    p = tmp_path / name; p.write_text("\n".join(json.dumps(r) for r in rows) + "\n"); return p


def test_assert_dataset_clean_passes_and_fails_with_the_report(tmp_path):
    good = write(tmp_path, [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}])
    assert_dataset_clean(good)
    bad = write(tmp_path, [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": ""}]}], "bad.jsonl")
    with pytest.raises(AssertionError, match="empty_assistant"):
        assert_dataset_clean(bad)
    assert_dataset_clean(bad, allow={"empty_assistant"})


def test_the_plugin_adds_a_test_per_dataset(tmp_path):
    bad = write(tmp_path, [{"messages": [{"role": "user", "content": "hi"}]}], "bad.jsonl")
    (tmp_path / "test_nothing.py").write_text("def test_x():\n    pass\n")
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q", "--ftgate-dataset", str(bad), str(tmp_path)],
                          capture_output=True, text=True, cwd=tmp_path)
    assert "ftgate[" in proc.stdout and "no_assistant_target" in proc.stdout
    assert "1 failed, 1 passed" in proc.stdout


def test_cli_accepts_several_files_and_exits_nonzero(tmp_path):
    a = write(tmp_path, [{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]}], "a.jsonl")
    b = write(tmp_path, [{"messages": [{"role": "user", "content": "hi"}]}], "b.jsonl")
    proc = subprocess.run([sys.executable, "-m", "ftgate.cli", "data", str(a), str(b), "--fail-on", "error"], capture_output=True, text=True)
    assert proc.returncode == 1 and "a.jsonl" in proc.stdout and "b.jsonl" in proc.stdout


def test_pre_commit_hook_and_action_manifest_parse():
    yaml = pytest.importorskip("yaml")
    hooks = yaml.safe_load((ROOT / ".pre-commit-hooks.yaml").read_text())
    assert hooks[0]["id"] == "ftgate-data" and hooks[0]["entry"] == "ftgate data"
    action = yaml.safe_load((ROOT / "action.yml").read_text())
    assert action["runs"]["using"] == "composite"
    assert all("shell" in s for s in action["runs"]["steps"] if "run" in s)
    assert "dataset" in action["inputs"]
