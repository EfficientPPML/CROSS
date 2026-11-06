"""Exact frozen collector __main__ AST; no child, payload, or model work."""
import ast
import contextlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import pytest


def main_exit(passed):
    path = Path(__file__).with_name('preflight_original.py')
    tree = ast.parse(path.read_bytes())
    main = next(n for n in tree.body if isinstance(n, ast.If)
                and ast.unparse(n.test) == "__name__ == '__main__'")
    class Parser:
        def add_mutually_exclusive_group(self, **kwargs): return self
        def add_argument(self, *args, **kwargs): pass
        def parse_args(self):
            return SimpleNamespace(test_worker=None, output=Path('/unused-synthetic-output'))
    report = dict(passed=passed, source_count=300,
                  tests=dict(tests=45, failures=0, errors=0, skipped=0),
                  source_archive_sha256='synthetic')
    ns = dict(argparse=SimpleNamespace(ArgumentParser=Parser), Path=Path, json=json,
              finish=lambda *args: report, collect=lambda *args: None,
              c=SimpleNamespace(write=lambda *args: None))
    out = io.StringIO()
    status = 0
    try:
        with contextlib.redirect_stdout(out):
            exec(compile(ast.Module(body=main.body, type_ignores=[]), str(path), 'exec'), ns)
    except SystemExit as error:
        status = error.code
    assert json.loads(out.getvalue())['passed'] is passed
    return status


def test_failed_returned_report_must_exit_nonzero():
    assert main_exit(False) == 1


def test_successful_returned_report_exits_zero():
    assert main_exit(True) == 0
