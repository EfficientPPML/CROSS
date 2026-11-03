"""Complete evaluations account for failures and enforce their process budget."""
from pathlib import Path
import json
import os
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
import evaluate
import install_suite
import score


def test_command_failure_is_not_a_successful_evaluation(tmp_path):
    with (tmp_path / 'log').open('w') as log:
        with pytest.raises(RuntimeError, match='exited 7'):
            evaluate.run_command([sys.executable, '-c', 'raise SystemExit(7)'],
                                 cwd=tmp_path, env=os.environ, log=log,
                                 deadline=time.monotonic() + 5)


def test_command_timeout_is_bounded(tmp_path):
    started = time.monotonic()
    with (tmp_path / 'log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            evaluate.run_command([sys.executable, '-c', 'import time; time.sleep(30)'],
                                 cwd=tmp_path, env=os.environ, log=log,
                                 deadline=started + .2)
    assert time.monotonic() - started < 5


def test_expired_budget_starts_no_process(tmp_path):
    with (tmp_path / 'log').open('w') as log:
        with pytest.raises(TimeoutError, match='budget'):
            evaluate.run_command(['not-an-executable'], cwd=tmp_path, env=os.environ,
                                 log=log, deadline=time.monotonic() - 1)


def test_timeout_kills_child_that_ignores_sigterm(tmp_path):
    child_source = (
        "import os, signal, time; from pathlib import Path; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "Path('child.pid').write_text(str(os.getpid())); time.sleep(30)")
    parent_source = (
        "import subprocess, sys, time; "
        f"subprocess.Popen([sys.executable, '-c', {child_source!r}]); "
        "time.sleep(30)")
    with (tmp_path / 'log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            evaluate.run_command([sys.executable, '-c', parent_source], cwd=tmp_path,
                                 env=os.environ, log=log, deadline=time.monotonic() + 1)
    pid = int((tmp_path / 'child.pid').read_text())
    # A zombie has exited and closed its device handles, but container PID1
    # may not have reaped it yet.
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        path = Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2] == 'Z':
            break
        time.sleep(.01)
    else:
        os.kill(pid, 9)
        pytest.fail('timed-out evaluation left a live grandchild')


def assert_process_exited(pid):
    deadline = time.monotonic()+2
    while time.monotonic() < deadline:
        path = Path(f'/proc/{pid}/stat')
        if not path.exists() or path.read_text().split()[2] == 'Z':
            return
        time.sleep(.01)
    os.kill(pid, signal.SIGKILL)
    pytest.fail('evaluation left a live child')


def test_successful_leader_does_not_leave_a_running_child(tmp_path):
    child = ("import os, signal, time; from pathlib import Path; "
             "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
             "Path('child.pid').write_text(str(os.getpid())); time.sleep(30)")
    parent = ("import subprocess, sys, time; from pathlib import Path; "
              f"subprocess.Popen([sys.executable,'-c',{child!r}]); "
              "\nwhile not Path('child.pid').exists(): time.sleep(.01)")
    with (tmp_path/'log').open('w') as log:
        evaluate.run_command([sys.executable, '-c', parent], cwd=tmp_path,
            env=os.environ, log=log, deadline=time.monotonic()+5)
    assert_process_exited(int((tmp_path/'child.pid').read_text()))


@pytest.mark.parametrize('cancel_signal', [signal.SIGTERM, signal.SIGHUP])
def test_outer_cancellation_runs_cleanup_and_reaps_active_group(tmp_path, cancel_signal):
    child = ("import os, time; from pathlib import Path; "
             "Path('child.pid').write_text(str(os.getpid())); time.sleep(30)")
    source = f'''import sys, os, time
from pathlib import Path
sys.path.insert(0, {str(SUITE)!r})
import evaluate
try:
    with evaluate.cancellation_handlers():
        with open('log', 'w') as log:
            evaluate.run_command([sys.executable, '-c', {child!r}], cwd=Path.cwd(), env=os.environ, log=log, deadline=time.monotonic()+60)
finally:
    Path('cleanup').write_text('completed')
'''
    with (tmp_path/'outer.log').open('w') as log:
        process = subprocess.Popen([sys.executable, '-c', source], cwd=tmp_path, stdout=log, stderr=log)
        deadline = time.monotonic()+5
        while not (tmp_path/'child.pid').exists():
            if time.monotonic() > deadline:
                process.kill()
                pytest.fail('child did not start')
            time.sleep(.01)
        process.send_signal(cancel_signal)
        assert process.wait(timeout=5) != 0
    assert (tmp_path/'cleanup').read_text() == 'completed'
    assert_process_exited(int((tmp_path/'child.pid').read_text()))


def test_new_failed_attempt_invalidates_default_and_custom_reports(monkeypatch, tmp_path):
    suite = tmp_path/'checkout/cadathlon/non_agent_pset/p1/src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    monkeypatch.setattr(evaluate, 'model_identity', lambda *args: {})
    monkeypatch.setattr(evaluate, 'cleanup', lambda: None)
    monkeypatch.setattr(install_suite, 'harness_is_pristine', lambda root: (False, 'changed harness'))
    default = suite/'run/evaluation-student-candidate/report.json'
    custom = tmp_path/'custom.json'
    default.parent.mkdir(parents=True)
    for path in (default, custom):
        path.write_text('{"passed": true}')
    args = SimpleNamespace(profile='student', baseline=False, budget_s=300., suite_root=tmp_path, report=custom)
    assert evaluate.run(args) == 1
    for path in (default, custom):
        report = json.loads(path.read_text())
        assert report['passed'] is False
        assert 'changed harness' in report['error']


@pytest.mark.parametrize('failure', ['metadata', 'suite_resolution'])
def test_early_environment_failure_invalidates_previous_success(monkeypatch, tmp_path, failure):
    suite = tmp_path/'checkout/cadathlon/non_agent_pset/p1/src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    monkeypatch.setattr(evaluate, 'cleanup', lambda: None)
    if failure == 'metadata':
        def version(name):
            raise evaluate.importlib.metadata.PackageNotFoundError(name)
        monkeypatch.setattr(evaluate.importlib.metadata, 'version', version)
    else:
        def missing():
            raise SystemExit('suite not installed')
        monkeypatch.setattr(install_suite, 'suite_root', missing)
    path = suite/'run/evaluation-student-candidate/report.json'
    path.parent.mkdir(parents=True)
    path.write_text('{"passed": true}')
    args = SimpleNamespace(profile='student', baseline=False, budget_s=300.,
                           suite_root=tmp_path if failure == 'metadata' else None, report=None)
    assert evaluate.run(args) == 1
    report = json.loads(path.read_text())
    assert report['passed'] is False
    assert ('PackageNotFoundError' if failure == 'metadata' else 'suite not installed') in report['error']


def test_student_budget_cannot_be_relaxed_at_cli():
    result = subprocess.run([sys.executable, str(SUITE/'evaluate.py'), '--budget-s', '301'],
        capture_output=True, text=True, env=dict(os.environ, JAX_PLATFORMS='no_backend_for_cli_check'))
    assert result.returncode == 2
    assert 'at most 300' in result.stderr


def test_evaluations_cannot_share_daemon_ownership(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, 'SUITE', tmp_path)
    with evaluate.evaluation_lock():
        with pytest.raises(RuntimeError, match='already owns'):
            with evaluate.evaluation_lock():
                pytest.fail('second evaluator acquired an owned checkout')
    with evaluate.evaluation_lock():
        pass


@pytest.mark.parametrize('final_cleanup', ['success', 'failure', 'budget'])
def test_calibration_published_only_after_cleanup_and_budget(monkeypatch, tmp_path, final_cleanup):
    clock = SimpleNamespace(now=0., cleanups=0)
    suite = tmp_path / 'checkout/cadathlon/non_agent_pset/p1/src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    monkeypatch.setattr(score, 'SUITE_DIR', suite)
    monkeypatch.setattr(evaluate, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    config = dict(score.CONFIG, cases=[{'id': 'C3', 'instance': 1, 'batch': 100, 'devices': 1}])
    config['evaluation_profiles'] = {'student': {'case_ids': ['C3']}}
    monkeypatch.setattr(score, 'CONFIG', config)
    monkeypatch.setattr(install_suite, 'harness_is_pristine', lambda root: (True, ''))
    monkeypatch.setattr(install_suite, 'install', lambda root: None)
    monkeypatch.setattr(score, 'read_case', lambda *args: ([], {}, tmp_path))
    monkeypatch.setattr(score, 'summarize', lambda *args: {'baseline_profile': 'student'})
    monkeypatch.setattr(score, 'gate', lambda *args: (1, []))
    monkeypatch.setattr(evaluate, 'model_identity', lambda *args: {})
    monkeypatch.setattr(evaluate, 'archive_hashes', lambda *args: {})
    monkeypatch.setattr(score, 'validate_student_evaluation', lambda report, **kwargs: report)
    monkeypatch.setattr(score, 'calibration_payload', lambda *args: {'new_calibration': True})
    monkeypatch.setenv('CROSS_ROOT', str(tmp_path / 'external-cross'))

    def command(args, **kwargs):
        assert kwargs['env']['CROSS_ROOT'] == str(tmp_path / 'external-cross')

    def cleanup():
        clock.cleanups += 1
        if clock.cleanups == 3:
            if final_cleanup == 'failure':
                raise RuntimeError('daemon would not exit')
            if final_cleanup == 'budget':
                clock.now = 101

    monkeypatch.setattr(evaluate, 'run_command', command)
    monkeypatch.setattr(evaluate, 'cleanup', cleanup)
    manifest = suite / 'baseline_manifest_student.json'
    manifest.write_text('previous verified calibration')
    args = SimpleNamespace(profile='student', baseline=True, budget_s=100.,
                           suite_root=tmp_path, report=tmp_path / 'report.json')
    code = evaluate.run(args)
    assert (code == 0) is (final_cleanup == 'success')
    if final_cleanup == 'success':
        assert json.loads(manifest.read_text()) == {'new_calibration': True}
    else:
        assert manifest.read_text() == 'previous verified calibration'


@pytest.mark.parametrize('previous', [None, 'verified calibration'])
def test_slow_calibration_publication_rolls_back(monkeypatch, tmp_path, previous):
    staged, target = tmp_path / 'staged.json', tmp_path / 'baseline.json'
    staged.write_text('new calibration')
    if previous is not None:
        target.write_text(previous)
    ticks = iter((0., 2.))
    monkeypatch.setattr(evaluate, 'time', SimpleNamespace(monotonic=lambda: next(ticks)))
    with pytest.raises(TimeoutError, match='during calibration'):
        evaluate.publish_calibration(staged, target, deadline=1.)
    assert (target.read_text() if target.exists() else None) == previous


@pytest.mark.parametrize('previous', [None, 'previous verified calibration'])
@pytest.mark.parametrize('failure', ['snapshot_io', 'report_io', 'deadline'])
def test_post_publish_failure_restores_manifest_and_invalidates_proof(monkeypatch, tmp_path, previous, failure):
    clock = SimpleNamespace(now=1.)
    monkeypatch.setattr(evaluate, 'time', SimpleNamespace(monotonic=lambda: clock.now))
    staged, target = tmp_path/'new.json', tmp_path/'manifest.json'
    staged.write_text('new calibration')
    if previous is not None:
        target.write_text(previous)
    snapshot, report = tmp_path/'immutable-proof.json', tmp_path/'report.json'
    original = Path.write_text

    def write(path, text, *args, **kwargs):
        committed = target.exists() and target.read_text() == 'new calibration'
        if committed and ((failure == 'snapshot_io' and path == snapshot)
                          or (failure == 'report_io' and path == report)):
            raise OSError('injected post-publish write failure')
        value = original(path, text, *args, **kwargs)
        if committed and failure == 'deadline':
            clock.now = 301.
        return value

    monkeypatch.setattr(Path, 'write_text', write)
    result = {'passed': True, 'budget_s': 300.}
    evaluate.finalize_reports(result, [report], started=0., deadline=300.,
                              staged=staged, destination=target, snapshot=snapshot)
    assert result['passed'] is False
    assert (target.read_text() if target.exists() else None) == previous
    assert not snapshot.exists()
    assert json.loads(report.read_text())['passed'] is False


def test_daemon_cleanup_attempts_every_owned_pidfile(monkeypatch, tmp_path):
    monkeypatch.setattr(evaluate, 'SUITE', tmp_path)
    for name, pid in (('first', 111), ('second', 222)):
        path = tmp_path/'run'/name/'service.pid'
        path.parent.mkdir(parents=True)
        path.write_text(str(pid))
    stopped = []

    def stop(pid, **kwargs):
        stopped.append(pid)
        if pid == 111:
            raise TimeoutError('first daemon stuck')

    monkeypatch.setattr(evaluate, 'stop_daemon', stop)
    with pytest.raises(RuntimeError, match='first daemon stuck'):
        evaluate.cleanup()
    assert sorted(stopped) == [111, 222]
    assert not (tmp_path/'run/second/service.pid').exists()


def test_source_identity_includes_schedule_and_gates(monkeypatch, tmp_path):
    suite = tmp_path/'src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(evaluate, 'SUITE', suite)
    config = suite/'suite_config.json'
    config.write_text('{"gates": 1}')
    before = evaluate.source_hashes(tmp_path)
    config.write_text('{"gates": 2}')
    after = evaluate.source_hashes(tmp_path)
    assert before['problem/suite/suite_config.json'] != after['problem/suite/suite_config.json']


def test_repeated_cancellation_is_ignored_during_final_cleanup(tmp_path):
    source = f'''import sys, os, signal
sys.path.insert(0, {str(SUITE)!r})
import evaluate
with evaluate.cancellation_handlers(), evaluate.cancellation_shield():
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        os.kill(os.getpid(), sig)
print('cleanup completed')
'''
    result = subprocess.run([sys.executable, '-c', source], cwd=tmp_path,
        capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    assert 'cleanup completed' in result.stdout


def test_client_shims_override_inherited_tpu_backend(tmp_path):
    install_suite.install(tmp_path)
    shim = tmp_path / 'submissions/mnist/client_key_generation.py'
    assert 'os.environ["JAX_PLATFORMS"] = "cpu"' in shim.read_text()
    server = tmp_path / 'submissions/mnist/server_preprocess_model.py'
    assert 'os.environ["JAX_PLATFORMS"] = "cpu"' not in server.read_text()


def test_legacy_archives_cannot_receive_student_execution_weights(monkeypatch, capsys):
    def read_case(root, instance, runs, case_id, label):
        archive = SUITE / 'results' / (case_id + '_baseline')
        meta = json.loads((archive / 'case.json').read_text())
        records = [json.loads((archive / f'results-{i}.json').read_text()) for i in range(1, runs + 1)]
        return records, meta, archive
    monkeypatch.setattr(score, 'read_case', read_case)
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'student'])
    score.main()
    result = json.loads(capsys.readouterr().out)
    assert set(result['cases']) == {'C3', 'C4'}
    assert result['execution_points'] == 0
    assert result['speed_points'] == 0
