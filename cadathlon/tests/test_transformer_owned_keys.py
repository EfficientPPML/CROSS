"""Owned directory lifetime across real child success, termination and failures."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'validation/check_transformer_suite.py'
spec = importlib.util.spec_from_file_location('owned_keys_suite', SCRIPT)
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)


def parent(monkeypatch, tmp_path):
    shared = tmp_path / 'shared'
    shared.mkdir()
    (shared / 'unrelated.key').write_bytes(b'leave alone')
    monkeypatch.setenv('CROSS_PUBLIC_KEY_DIRECTORY', str(shared))
    return shared


def assert_parent(shared):
    assert (shared / 'unrelated.key').read_bytes() == b'leave alone'
    assert os.environ['CROSS_PUBLIC_KEY_DIRECTORY'] == str(shared)


KEY_CODE = ("import json,os; from pathlib import Path; "
    "p=Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY']); "
    "(p/'nested').mkdir(); (p/'nested'/'public-key.bin').write_bytes(b'public'); ")


def test_normal_exit_removes_only_owned_child_keys(monkeypatch, tmp_path):
    shared = parent(monkeypatch, tmp_path)
    code = KEY_CODE + "print(json.dumps({'directory':str(p)}))"
    record = suite.run_case([sys.executable, '-c', code], tmp_path, timeout=None)
    assert not record.get('error') and record['returncode'] == 0
    assert record['process_reaped'] and record['public_key_store_removed']
    owned = Path(record['result']['directory'])
    assert owned.parent == shared and owned != shared and not owned.exists()
    assert_parent(shared)


def test_forced_kill_removes_orphaned_key_files(monkeypatch, tmp_path):
    shared = parent(monkeypatch, tmp_path)
    code = KEY_CODE + 'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'
    record = suite.run_case([sys.executable, '-c', code], tmp_path, timeout=.5, cleanup_grace_s=.1)
    assert record['returncode'] == -9
    assert record['process_reaped'] and record['public_key_store_removed']
    assert not Path(record['owned_public_key_directory']).exists()
    assert 'deadline' in record['error']
    assert_parent(shared)


def test_launch_failure_still_records_and_removes_owned_directory(monkeypatch, tmp_path):
    shared = parent(monkeypatch, tmp_path)
    with pytest.raises(FileNotFoundError):
        suite.run_case(['/definitely-missing-owned-key-test'], tmp_path, timeout=None)
    record = json.loads((tmp_path / 'process.json').read_text())
    assert record['process_reaped'] and record['public_key_store_removed']
    assert not Path(record['owned_public_key_directory']).exists()
    assert 'FileNotFoundError' in record['error']
    assert_parent(shared)


@pytest.mark.parametrize('primary_failure', [False, True])
@pytest.mark.parametrize('cancel_cleanup', [False, True])
def test_cleanup_failure_prevents_pass_and_preserves_primary(monkeypatch, tmp_path, primary_failure, cancel_cleanup):
    shared = parent(monkeypatch, tmp_path)
    original = shutil.rmtree
    def fail_cleanup(path):
        raise (KeyboardInterrupt if cancel_cleanup else RuntimeError)('simulated cleanup failure')
    monkeypatch.setattr(suite.shutil, 'rmtree', fail_cleanup)
    if primary_failure:
        with pytest.raises(FileNotFoundError):
            suite.run_case(['/definitely-missing-owned-key-test'], tmp_path, timeout=None)
        record = json.loads((tmp_path / 'process.json').read_text())
        assert 'FileNotFoundError' in record['error']
    elif cancel_cleanup:
        with pytest.raises(KeyboardInterrupt, match='simulated cleanup failure'):
            suite.run_case([sys.executable, '-c', 'print("{}")'], tmp_path, timeout=None)
        record = json.loads((tmp_path / 'process.json').read_text())
    else:
        record = suite.run_case([sys.executable, '-c', 'print("{}")'], tmp_path, timeout=None)
        assert record['error'] == 'owned public key directory cleanup failed'
    assert not record['passed'] and not record['public_key_store_removed']
    assert ('KeyboardInterrupt' if cancel_cleanup else 'RuntimeError') in record['key_cleanup_error']
    original(record['owned_public_key_directory'])
    assert_parent(shared)


def test_process_stop_failure_does_not_replace_primary_cancellation(monkeypatch, tmp_path):
    shared = parent(monkeypatch, tmp_path)
    class FakeProcess:
        pid = 99999999
        returncode = None
        def wait(self, **kwargs):
            raise KeyboardInterrupt('primary cancellation')
        def poll(self):
            return None
    monkeypatch.setattr(suite.subprocess, 'Popen', lambda *a, **kw: FakeProcess())
    def failed_stop(*a, **kw):
        raise RuntimeError('secondary stop failure')
    monkeypatch.setattr(suite, 'stop_process_group', failed_stop)
    with pytest.raises(KeyboardInterrupt, match='primary cancellation'):
        suite.run_case(['fake-never-launched'], tmp_path, timeout=None)
    record = json.loads((tmp_path / 'process.json').read_text())
    assert not record['process_reaped'] and not record['public_key_store_removed']
    assert 'secondary stop failure' in record['process_cleanup_error']
    shutil.rmtree(record['owned_public_key_directory'])
    assert_parent(shared)


def test_live_group_retains_key_files(monkeypatch, tmp_path):
    shared = parent(monkeypatch, tmp_path)
    monkeypatch.setattr(suite, 'process_group_alive', lambda pid: True)
    monkeypatch.setattr(suite, 'stop_process_group', lambda *a, **kw: None)
    record = suite.run_case([sys.executable, '-c', KEY_CODE + 'print("{}")'], tmp_path, timeout=None)
    assert not record['process_reaped'] and not record['public_key_store_removed']
    owned = Path(record['owned_public_key_directory'])
    assert (owned / 'nested/public-key.bin').read_bytes() == b'public'
    assert 'still present' in record['key_cleanup_error']
    shutil.rmtree(owned)
    assert_parent(shared)


def test_replaced_directory_cannot_redirect_cleanup(monkeypatch, tmp_path):
    shared = parent(monkeypatch, tmp_path)
    code = ("import json,os; from pathlib import Path; "
            "p=Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY']); "
            "p.rmdir(); p.symlink_to(p.parent, target_is_directory=True); print('{}')")
    record = suite.run_case([sys.executable, '-c', code], tmp_path, timeout=None)
    assert record['process_reaped'] and not record['public_key_store_removed']
    assert 'symlink' in record['key_cleanup_error']
    Path(record['owned_public_key_directory']).unlink()
    assert_parent(shared)


@pytest.mark.parametrize('mode', ['primary', 'cleanup', 'none'])
def test_final_artifact_failure_preserves_existing_cancellation(monkeypatch, tmp_path, mode):
    shared = parent(monkeypatch, tmp_path)
    original_write, original_remove = suite.write_json, shutil.rmtree
    writes = 0
    def write(path, record):
        nonlocal writes
        writes += 1
        if writes == 3:
            raise OSError('final evidence unavailable')
        original_write(path, record)
    monkeypatch.setattr(suite, 'write_json', write)
    if mode == 'primary':
        class FakeProcess:
            pid = 99999999
            returncode = 0
            def wait(self, **kwargs):
                raise KeyboardInterrupt('primary cancellation')
            def poll(self):
                return 0
        monkeypatch.setattr(suite.subprocess, 'Popen', lambda *a, **kw: FakeProcess())
        monkeypatch.setattr(suite, 'process_group_alive', lambda pid: False)
    elif mode == 'cleanup':
        def cancel_cleanup(path):
            raise KeyboardInterrupt('cleanup cancellation')
        monkeypatch.setattr(suite.shutil, 'rmtree', cancel_cleanup)
    expected = OSError if mode == 'none' else KeyboardInterrupt
    with pytest.raises(expected) as error:
        suite.run_case([sys.executable, '-c', 'print("{}")'], tmp_path, timeout=None)
    assert writes == 3
    if mode != 'none':
        assert any('final evidence unavailable' in note for note in error.value.__notes__)
    record = json.loads((tmp_path / 'process.json').read_text())
    owned = Path(record['owned_public_key_directory'])
    if owned.exists():
        original_remove(owned)
    assert_parent(shared)


def test_replaced_real_directory_is_retained_by_inode(monkeypatch,tmp_path):
    shared=parent(monkeypatch,tmp_path)
    foreign=shared/'unrelated-directory';foreign.mkdir();(foreign/'sentinel').write_text('keep')
    expected_inode=foreign.stat().st_ino
    code=("import os; from pathlib import Path; p=Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY']); "
          "p.rmdir(); (p.parent/'unrelated-directory').rename(p); print('{}')")
    record=suite.run_case([sys.executable,'-c',code],tmp_path,timeout=None)
    replacement=Path(record['owned_public_key_directory'])
    assert record['process_reaped'] and not record['public_key_store_removed']
    assert record['owned_directory_handle_closed']
    assert record['owned_public_key_directory_identity']['inode']!=expected_inode
    assert replacement.stat().st_ino==expected_inode and (replacement/'sentinel').read_text()=='keep'
    assert 'identity was replaced' in record['key_cleanup_error']
    shutil.rmtree(replacement);assert_parent(shared)


def test_moved_original_directory_is_not_reported_removed(monkeypatch,tmp_path):
    shared=parent(monkeypatch,tmp_path)
    code=KEY_CODE+"p.rename(p.with_name(p.name+'.moved')); print('{}')"
    record=suite.run_case([sys.executable,'-c',code],tmp_path,timeout=None)
    original=Path(record['owned_public_key_directory']);moved=original.with_name(original.name+'.moved')
    assert record['process_reaped'] and not record['public_key_store_removed']
    assert record['owned_directory_handle_closed'] and not original.exists()
    assert moved.stat().st_ino==record['owned_public_key_directory_identity']['inode']
    assert (moved/'nested/public-key.bin').read_bytes()==b'public'
    assert 'was moved' in record['key_cleanup_error']
    shutil.rmtree(moved);assert_parent(shared)


def test_child_removing_original_empty_directory_is_complete(monkeypatch,tmp_path):
    shared=parent(monkeypatch,tmp_path)
    code="import os; from pathlib import Path; Path(os.environ['CROSS_PUBLIC_KEY_DIRECTORY']).rmdir(); print('{}')"
    record=suite.run_case([sys.executable,'-c',code],tmp_path,timeout=None)
    assert not record.get('error') and record['process_reaped'] and record['public_key_store_removed']
    assert record['owned_directory_handle_closed'];assert_parent(shared)


@pytest.mark.parametrize('error_type',[OSError,KeyboardInterrupt])
def test_directory_identity_acquisition_failure_closes_descriptor(monkeypatch,tmp_path,error_type):
    directory=tmp_path/'owned';directory.mkdir()
    original_open,original_fstat=os.open,os.fstat
    captured=[]
    def opened(*args,**kwargs):
        descriptor=original_open(*args,**kwargs);captured.append(descriptor);return descriptor
    def failed(*args,**kwargs):raise error_type('identity interrupted')
    with monkeypatch.context() as patch:
        patch.setattr(suite.os,'open',opened);patch.setattr(suite.os,'fstat',failed)
        with pytest.raises(error_type,match='identity interrupted'):suite.open_owned_directory(directory)
    assert len(captured)==1
    with pytest.raises(OSError):original_fstat(captured[0])
    assert directory.is_dir()
