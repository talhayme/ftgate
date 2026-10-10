"""pytest plugin: a fixture and a command-line switch.

    pytest --ftgate-dataset data/train.jsonl --ftgate-model Qwen/Qwen2.5-0.5B-Instruct

adds one test per dataset given on the command line; the `ftgate_lint`
fixture lets a test lint a path itself.
"""
from __future__ import annotations

import pytest

from ftgate import testing


def pytest_addoption(parser):
    g = parser.getgroup("ftgate")
    g.addoption("--ftgate-dataset", action="append", default=[], help="JSONL dataset(s) to lint as tests")
    g.addoption("--ftgate-model", default=None, help="HF model id for tokenizer/template checks")
    g.addoption("--ftgate-max-seq-len", type=int, default=None)
    g.addoption("--ftgate-fail-on", choices=["error", "warning"], default="error")


@pytest.fixture
def ftgate_lint():
    return testing.lint_dataset


def pytest_collection_modifyitems(session, config, items):
    datasets = config.getoption("--ftgate-dataset")
    if not datasets:
        return
    model, msl, level = config.getoption("--ftgate-model"), config.getoption("--ftgate-max-seq-len"), config.getoption("--ftgate-fail-on")

    class DatasetItem(pytest.Item):
        def __init__(self, name, parent, path):
            super().__init__(name, parent); self.dataset = path
        def runtest(self):
            testing.assert_dataset_clean(self.dataset, model=model, max_seq_len=msl, level=level)
        def reportinfo(self):
            return self.path, 0, f"ftgate data {self.dataset}"

    for ds in datasets:
        items.append(DatasetItem.from_parent(session, name=f"ftgate[{ds}]", path=ds))
