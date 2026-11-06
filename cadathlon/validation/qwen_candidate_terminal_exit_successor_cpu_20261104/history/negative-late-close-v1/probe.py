"""Synthetic exact-function terminal review; never imports runtime/model code."""
import ast
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent

def exact_functions(filename, names, namespace):
    body = ast.parse((HERE / 'source' / filename).read_bytes()).body
    selected = [node for node in body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), filename, 'exec'), namespace)

def namespace():
    ns = dict(time=time)
    exec(compile((HERE / 'source' / 'cleanup_origin.py').read_bytes(), 'cleanup_origin.py', 'exec'), ns)
    @contextmanager
    def signals(state=None, shield=False):
        yield
        if state is not None:
            state['restored'] = True
    ns['signals'] = signals
    ns['source_end'] = lambda root, expected: expected
    ns['input_gate'] = SimpleNamespace(typed_equal=lambda x, y: json.dumps(x, sort_keys=True) == json.dumps(y, sort_keys=True))
    def require_int(value, expected):
        if type(value) is not int or value != expected:
            raise ValueError('typed integer')
    binding = SimpleNamespace(manifest_sha256='1'*64, ordinal=1, initial_report_sha256='2'*64, selection_proof_sha256='3'*64)
    ns['checker_admission'] = SimpleNamespace(require_candidate=lambda _: binding, require_int=require_int)
    exact_functions('checker_process.py', {'supervise', 'accumulate', 'accept_candidate_worker', 'failed_revision_coverage'}, ns)
    exact_functions('test_checker_process.py', {'terminal'}, ns)
    return ns, binding

def fixture(ns, binding, passing):
    args = SimpleNamespace(manifest_sha256='4'*64, grant_sha256='5'*64)
    expected = {'synthetic.py': '6'*64}
    report = ns['terminal'](binding, args, expected)
    report.update(passed=passing, candidate_validation_admissible=passing,
                  failed_attempt_terminal_candidate=not passing,
                  revision_eligible_ledger=not passing, semantic_passes=3200,
                  input_ranges={'native': {'contexts': 3200}, 'exact': {'contexts': 3200}})
    report['by_kind'] = {kind: dict(expected=count, records=count, top1_denominator=count,
        complete=count, candidate_invalid=0, comparison_invalid=0, unmeasured=0)
        for kind, count in [('prefill', 3072), ('teacher', 64), ('free', 64)]}
    # A finite numeric failure can retain every measured context.
    assert passing or ns['failed_revision_coverage'](report)
    return report, args, expected

def child(mode, directory):
    ns, binding = namespace()
    passing = mode == 'pass_late_close'
    fault = mode != 'failed_clean'
    report, args, expected = fixture(ns, binding, passing)
    class Output:
        def __init__(self, publication=False):
            self.path = str(directory)
            self.owner = (0, 1)
            self.publication = publication
        @classmethod
        def reopen(cls, path, owner):
            return cls(True)
        def json(self, name, value):
            (directory / name).write_text(json.dumps(value, sort_keys=True))
        def close(self):
            if self.publication and fault:
                raise OSError('injected final publication close failure')
    ns['Output'] = Output
    # The production entrypoint raises SystemExit(dispatch(...)); an unhandled
    # OSError also becomes OS exit1. No exception conversion is injected here.
    raise SystemExit(ns['supervise'](Output(), report, lambda: None, [], directory, expected))

def main():
    ns, binding = namespace()
    results = []
    for mode in ('failed_clean', 'failed_late_close', 'pass_late_close'):
        directory = HERE / mode
        directory.mkdir()
        proc = subprocess.Popen([sys.executable, '-I', '-B', __file__, '--child', mode, str(directory)],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        stdout, stderr = proc.communicate(timeout=20)
        (directory / 'stdout.log').write_bytes(stdout)
        (directory / 'stderr.log').write_bytes(stderr)
        report = json.loads((directory / 'report.json').read_bytes())
        _, args, expected = fixture(ns, binding, mode == 'pass_late_close')
        try:
            ns['accept_candidate_worker'](report, proc.returncode, expected, args, {})
            accepted, error = True, None
        except BaseException as exc:
            accepted, error = False, type(exc).__name__
        try:
            os.killpg(proc.pid, 0)
            absent = False
        except ProcessLookupError:
            absent = True
        results.append(dict(mode=mode, pid=proc.pid, pgid=proc.pid, returncode=proc.returncode,
            waited=True, group_absent=absent, producer_accepts=accepted, rejection_type=error,
            report_passed=report['passed'], report_failed_candidate=report['failed_attempt_terminal_candidate'],
            report_cleanup=report['cleanup_completed'], report_error_type=report.get('error_type'),
            final_close_exception_in_stderr=b'injected final publication close failure' in stderr))
    (HERE / 'result.json').write_text(json.dumps(results, sort_keys=True, indent=2)+'\n')
    assert results[0]['producer_accepts'] and results[0]['returncode'] == 1
    assert not results[2]['producer_accepts'] and results[2]['returncode'] == 1
    assert all(row['waited'] and row['group_absent'] for row in results)
    # This requirement intentionally fails against the frozen predecessor.
    assert not results[1]['producer_accepts'], 'Late infrastructure close failure was admitted as scientific exit1'

if __name__ == '__main__':
    if len(sys.argv) > 1:
        child(sys.argv[2], Path(sys.argv[3]))
    else:
        main()
