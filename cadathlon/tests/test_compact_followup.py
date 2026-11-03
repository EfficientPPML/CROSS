"""Validation wrappers preserve source bytes and allow evaluator cleanup."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time

SCRIPT = Path(__file__).resolve().parents[1] / 'validation/check_p3_compact_followup.py'
spec = importlib.util.spec_from_file_location('compact_followup_test', SCRIPT)
followup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(followup)


def test_prepare_only_archives_exact_sources_without_inference(tmp_path):
    output = tmp_path / 'prepared'
    result = subprocess.run([sys.executable, str(SCRIPT), '--prepare-only', '--output', str(output)],
                            text=True, capture_output=True, env=dict(os.environ, JAX_PLATFORMS='cpu'))
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)['encrypted_evaluation_passed'] is False
    report = json.loads((output / 'process.json').read_text())
    assert report['prepared_only'] and report['passed'] is False
    import hashlib
    with tarfile.open(output / 'sources.tar.gz') as archive:
        assert set(archive.getnames()) == set(report['source_sha256_start'])
        for name, digest in report['source_sha256_start'].items():
            assert hashlib.sha256(archive.extractfile(name).read()).hexdigest() == digest


def test_child_failure_is_retained(tmp_path):
    report = followup.wait_for_evaluator([sys.executable, '-c', 'raise SystemExit(7)'],
                                         env=dict(os.environ, JAX_PLATFORMS='cpu'), output=tmp_path)
    assert report['returncode'] == 7 and not report['forced_kill']
    assert json.loads((tmp_path / 'evaluator-process.json').read_text())['returncode'] == 7


def test_outer_sigterm_allows_nested_evaluator_cleanup(tmp_path):
    ready, cleaned = tmp_path / 'ready', tmp_path / 'cleaned'
    child = (
        'import os,signal,time\nfrom pathlib import Path\n'
        'def cleanup(signum,frame):\n'
        '    time.sleep(2.2)\n'
        f'    Path({str(cleaned)!r}).write_text("cleaned")\n'
        '    raise SystemExit(0)\n'
        'signal.signal(signal.SIGTERM,cleanup)\n'
        f'Path({str(ready)!r}).write_text(str(os.getpid()))\n'
        'time.sleep(60)\n')
    wrapper = tmp_path / 'wrapper.py'
    wrapper.write_text(
        'import importlib.util,os,sys\nfrom pathlib import Path\n'
        f'spec=importlib.util.spec_from_file_location("followup",{str(SCRIPT)!r})\n'
        'module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)\n'
        f'module.wait_for_evaluator([sys.executable,"-c",{child!r}],env=dict(os.environ),output=Path({str(tmp_path)!r}))\n')
    process = subprocess.Popen([sys.executable, str(wrapper)], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=dict(os.environ, JAX_PLATFORMS='cpu'))
    try:
        deadline = time.monotonic() + 8
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(.01)
        assert ready.exists()
        process.terminate()
        _, stderr = process.communicate(timeout=10)
        assert process.returncode == 0, stderr
        assert cleaned.read_text() == 'cleaned'
        report = json.loads((tmp_path / 'evaluator-process.json').read_text())
        assert report['returncode'] == 0
        assert report['cancelled_signals'] == [signal.SIGTERM]
        assert report['evaluator_process_elapsed_s'] >= 2.2
        assert not report['forced_kill']
    finally:
        if process.poll() is None:
            process.kill()
        process.communicate(timeout=5)
        if ready.exists():
            try:
                os.killpg(int(ready.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
