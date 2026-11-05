"""Contestants may use normal Python class/import features in submissions."""
import importlib.util
from pathlib import Path
import pickle
import sys

import pytest


spec = importlib.util.spec_from_file_location(
    "module_loading_support", Path(__file__).parents[1] / "common/support.py")
support = importlib.util.module_from_spec(spec)
spec.loader.exec_module(support)


def test_dataclass_postponed_annotations_and_pickle(tmp_path):
    source = tmp_path / "submission.py.golden"
    source.write_text(
        "from __future__ import annotations\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Session:\n"
        "    batch: int = 2\n")
    module = support.load_module(source)
    session = module.Session(4)
    assert pickle.loads(pickle.dumps(session)) == session


def test_independent_submissions_keep_their_module_identity(tmp_path):
    sources = [tmp_path / name for name in ("first.py", "second.py")]
    for source in sources:
        source.write_text("class Session:\n    pass\n")
    first, second = [support.load_module(source) for source in sources]
    assert first.__name__ != second.__name__
    assert type(pickle.loads(pickle.dumps(first.Session()))) is first.Session
    assert type(pickle.loads(pickle.dumps(second.Session()))) is second.Session


def test_failed_reload_preserves_previous_module(tmp_path):
    source = tmp_path / "submission.py"
    source.write_text("class Session:\n    pass\n")
    module = support.load_module(source)
    source.write_text("raise RuntimeError('submission import failed')\n")
    with pytest.raises(RuntimeError, match="submission import failed"):
        support.load_module(source)
    assert sys.modules[module.__name__] is module


def test_failed_first_import_is_not_left_registered(tmp_path):
    source = tmp_path / "submission.py"
    source.write_text("raise RuntimeError('submission import failed')\n")
    before = set(sys.modules)
    with pytest.raises(RuntimeError, match="submission import failed"):
        support.load_module(source)
    assert not (set(sys.modules) - before)
