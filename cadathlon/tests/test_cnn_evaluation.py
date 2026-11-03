"""Evaluation failures and timeout cleanup must never look like a pass."""
import importlib.util
import contextlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


SRC = Path(__file__).resolve().parents[1] / 'non_agent_pset/p3_cnn/src'
spec = importlib.util.spec_from_file_location('cnn_evaluate', SRC / 'evaluate.py')
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


def test_constructor_failure_still_collects_successful_worker():
    resource = object()

    def build(case):
        if case == 'bad':
            raise ValueError('construction failed')
        return resource

    prepared, failures = evaluate.prepare_all(('bad', 'good'), build)
    assert prepared == {'good': resource}
    assert len(failures) == 1
    assert 'construction failed' in failures[0]


def test_timeout_terminates_worker_and_writes_failed_report(monkeypatch, tmp_path):
    popen = subprocess.Popen
    processes = []

    def sleeping_worker(command, **kwargs):
        process = popen([sys.executable, '-c', 'import time; time.sleep(60)'], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(evaluate.subprocess, 'Popen', sleeping_worker)
    assert evaluate.evaluate(SimpleNamespace(output=tmp_path, budget_s=0.1)) == 1
    assert processes[0].poll() is not None
    report = json.loads((tmp_path / 'report.json').read_text())
    assert not report['passed']
    assert 'wall-time budget' in report['error']


def test_missing_worker_report_cannot_reuse_stale_pass(monkeypatch, tmp_path):
    popen = subprocess.Popen
    (tmp_path / 'worker.json').write_text('{"passed": true}')

    def empty_worker(command, **kwargs):
        return popen([sys.executable, '-c', 'pass'], **kwargs)

    monkeypatch.setattr(evaluate.subprocess, 'Popen', empty_worker)
    assert evaluate.evaluate(SimpleNamespace(output=tmp_path, budget_s=10)) == 1
    assert not json.loads((tmp_path / 'report.json').read_text())['passed']


def test_release_failure_does_not_skip_other_mapping():
    calls = []

    def release_bad():
        calls.append('bad')
        raise RuntimeError('release failed')

    def release_good():
        calls.append('good')

    import pytest
    with pytest.raises(RuntimeError, match='cleanup failed'):
        evaluate.release_all({'bad': (SimpleNamespace(release=release_bad), 0),
                              'good': (SimpleNamespace(release=release_good), 0)})
    assert calls == ['bad', 'good']


def test_one_shared_prepared_resource_is_released_once():
    calls = []
    resource = SimpleNamespace(release=lambda: calls.append('released'))
    evaluate.release_all({'first': (resource, 0), 'alias': (resource, 0)})
    assert calls == ['released']


def test_cleanup_failure_preserves_original_evaluation_error():
    original = ValueError('numeric gate failed')
    def fail():
        raise RuntimeError('cleanup broken')
    evaluate.release_all({'first': (SimpleNamespace(release=fail), 0)}, primary_error=original)
    assert str(original) == 'numeric gate failed'
    assert original.__notes__ == ['cleanup failed: first: RuntimeError: cleanup broken']


def test_constructor_interruption_releases_already_prepared_resources():
    calls = []
    resource = SimpleNamespace(release=lambda: calls.append('released'))
    def build(case):
        if case == 'interrupt':
            raise KeyboardInterrupt('stop')
        return resource, 0
    with pytest.raises(KeyboardInterrupt, match='stop'):
        evaluate.prepare_all(('first', 'interrupt'), build)
    assert calls == ['released']


def test_malformed_worker_report_overwrites_prior_pass(monkeypatch, tmp_path):
    popen = subprocess.Popen
    (tmp_path / 'report.json').write_text('{"passed": true}')

    def malformed_worker(command, **kwargs):
        script = 'from pathlib import Path; import sys; Path(sys.argv[1]).write_text("{broken")'
        return popen([sys.executable, '-c', script, command[-1]], **kwargs)

    monkeypatch.setattr(evaluate.subprocess, 'Popen', malformed_worker)
    assert evaluate.evaluate(SimpleNamespace(output=tmp_path, budget_s=10)) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert not report['passed']
    assert 'JSONDecodeError' in report['error']


def test_post_success_diagnostic_failure_is_not_a_pass(monkeypatch, tmp_path):
    root = SRC.parents[3]

    def fail_stats():
        raise RuntimeError('memory stats unavailable')

    device = SimpleNamespace(memory_stats=fail_stats)
    monkeypatch.setitem(sys.modules, 'runner', SimpleNamespace(run=lambda *a, **k: {'passed': True}))
    monkeypatch.setitem(sys.modules, 'support', SimpleNamespace(
        cross_root=lambda: root, read_json=lambda path: {'batch': 1, 'model': 'compact_gray_cnn'},
        load_module=lambda path: sys.modules['baseline_student']))
    monkeypatch.setitem(sys.modules, 'reference', SimpleNamespace(
        make_case=lambda case: (object(), (1,), None, None)))
    monkeypatch.setitem(sys.modules, 'baseline_student', SimpleNamespace(
        pack_model=lambda *a: object(),
        build_packed_mapping=lambda *a, **k: SimpleNamespace(release=lambda: None)))
    monkeypatch.setitem(sys.modules, 'jax', SimpleNamespace(
        config=SimpleNamespace(update=lambda *a: None), devices=lambda platform: [device]))
    monkeypatch.setitem(sys.modules, 'jaxite_word.mapping', SimpleNamespace(
        defer_cache_clearing=contextlib.nullcontext))
    path = tmp_path / 'worker.json'
    assert evaluate.worker(path, baseline=True) == 1
    report = json.loads(path.read_text())
    assert not report['passed']
    assert report['error'] == 'RuntimeError: memory stats unavailable'


def _process_is_running(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split()[2] != 'Z'
    except FileNotFoundError:
        return False


def _await_file(path, timeout=5):
    import time
    deadline = time.monotonic() + timeout
    while not path.exists() and time.monotonic() < deadline:
        time.sleep(.01)
    assert path.exists(), f'worker did not create {path}'


def _await_stopped(pid, timeout=5):
    import time
    deadline = time.monotonic() + timeout
    while _process_is_running(pid) and time.monotonic() < deadline:
        time.sleep(.01)
    assert not _process_is_running(pid), f'owned worker {pid} survived cleanup'


def test_successful_leader_with_lingering_helper_is_cleaned_and_failed(monkeypatch, tmp_path):
    import os
    import signal
    popen = subprocess.Popen
    ready = tmp_path / 'helper.pid'
    helper = ("import os,signal,time; from pathlib import Path; "
              "signal.signal(signal.SIGTERM,signal.SIG_IGN); "
              f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
    def worker_with_helper(command, **kwargs):
        report = command[command.index('--worker-report') + 1]
        leader = ("import subprocess,sys,time; from pathlib import Path; "
                  f"child=subprocess.Popen([sys.executable,'-c',{helper!r}]); "
                  f"ready=Path({str(ready)!r}); deadline=time.monotonic()+5\n"
                  "while not ready.exists() and time.monotonic()<deadline: time.sleep(.01)\n"
                  "assert ready.exists()\n"
                  f"Path({report!r}).write_text('{{\"passed\": true}}')\n")
        return popen([sys.executable, '-c', leader], **kwargs)
    monkeypatch.setattr(evaluate.subprocess, 'Popen', worker_with_helper)
    try:
        assert evaluate.evaluate(SimpleNamespace(output=tmp_path, budget_s=10)) == 1
        _await_file(ready)
        _await_stopped(int(ready.read_text()))
        report = json.loads((tmp_path / 'report.json').read_text())
        assert report['worker_returncode'] == 0
        assert report['passed'] is False and 'descendant' in report['error']
    finally:
        if ready.exists() and _process_is_running(int(ready.read_text())):
            os.kill(int(ready.read_text()), signal.SIGKILL)


def test_outer_sigterm_cleans_worker_and_preserves_failed_report(tmp_path):
    import os
    import signal
    ready = tmp_path / 'child.pid'
    child = ("import os,time; from pathlib import Path; "
             f"Path({str(ready)!r}).write_text(str(os.getpid())); time.sleep(60)")
    wrapper = tmp_path / 'wrapper.py'
    wrapper.write_text(
        'import importlib.util,sys\nfrom pathlib import Path\nfrom types import SimpleNamespace\n'
        f"spec=importlib.util.spec_from_file_location('cnn_evaluate',{str(SRC / 'evaluate.py')!r})\n"
        'module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n'
        'original_popen=module.subprocess.Popen\n'
        f"def worker(command,**kwargs): return original_popen([sys.executable,'-c',{child!r}],**kwargs)\n"
        'module.subprocess.Popen=worker\n'
        'with module.termination_signals():\n'
        f'    module.evaluate(SimpleNamespace(output=Path({str(tmp_path)!r}),budget_s=60))\n')
    process = subprocess.Popen([sys.executable, str(wrapper)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, start_new_session=True,
                               env=dict(os.environ, JAX_PLATFORMS='cpu'))
    try:
        _await_file(ready)
        child_pid = int(ready.read_text())
        process.terminate()
        process.communicate(timeout=8)
        assert process.returncode != 0
        _await_stopped(child_pid)
        report = json.loads((tmp_path / 'report.json').read_text())
        assert report['passed'] is False
        assert 'SystemExit' in report['error']
        assert report['elapsed_s'] > 0 and report['within_budget'] is True
    finally:
        if ready.exists() and _process_is_running(int(ready.read_text())):
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)


def test_launch_failure_replaces_stale_pass_with_failed_report(monkeypatch, tmp_path):
    (tmp_path / 'report.json').write_text('{"passed": true}')
    (tmp_path / 'worker.json').write_text('{"passed": true}')
    def fail_launch(*args, **kwargs):
        raise OSError('worker exec failed')
    monkeypatch.setattr(evaluate.subprocess, 'Popen', fail_launch)
    assert evaluate.evaluate(SimpleNamespace(output=tmp_path, budget_s=10)) == 1
    report = json.loads((tmp_path / 'report.json').read_text())
    assert report['passed'] is False
    assert report['error'] == 'OSError: worker exec failed'
    assert not (tmp_path / 'worker.json').exists()


@pytest.mark.parametrize('raise_inside', [False, True])
def test_termination_signal_handlers_restore_on_all_exits(raise_inside):
    import signal
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGHUP)}
    try:
        with evaluate.termination_signals():
            for sig, handler in previous.items():
                assert callable(signal.getsignal(sig))
                assert signal.getsignal(sig) is not handler
            if raise_inside:
                raise RuntimeError('test cleanup')
    except RuntimeError:
        assert raise_inside
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


@pytest.mark.parametrize('interruption', [KeyboardInterrupt('cleanup interrupted'), SystemExit(143)])
def test_interrupted_release_finishes_other_mappings_before_reraising(interruption):
    calls = []
    def interrupted():
        calls.append('interrupted')
        raise interruption
    def remaining():
        calls.append('remaining')
    with pytest.raises(type(interruption)) as caught:
        evaluate.release_all({'first': (SimpleNamespace(release=interrupted), 0),
                              'next': (SimpleNamespace(release=remaining), 0)})
    assert caught.value is interruption
    assert calls == ['interrupted', 'remaining']
    assert len(interruption.__notes__) == 1
    assert 'cleanup failed: first:' in interruption.__notes__[0]


@pytest.mark.parametrize('interruption', [KeyboardInterrupt('cleanup interrupted'), SystemExit(143)])
def test_release_interruption_keeps_request_error_and_finishes_cleanup(interruption):
    calls = []
    primary = ValueError('encrypted numeric gate failed')
    def interrupted():
        calls.append('interrupted')
        raise interruption
    def remaining():
        calls.append('remaining')
    evaluate.release_all({'first': (SimpleNamespace(release=interrupted), 0),
                          'next': (SimpleNamespace(release=remaining), 0)}, primary_error=primary)
    assert str(primary) == 'encrypted numeric gate failed'
    assert calls == ['interrupted', 'remaining']
    assert len(primary.__notes__) == 1
    assert type(interruption).__name__ in primary.__notes__[0]
