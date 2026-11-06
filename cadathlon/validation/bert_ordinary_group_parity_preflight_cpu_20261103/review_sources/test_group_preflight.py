"""Preparation collector failure evidence; all subprocesses are inert stubs."""
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import contextmanager
import json
import pytest
import group_preflight as c

@pytest.fixture
def case(tmp_path,monkeypatch):
    raw={'source.py':b'original'};events=[];failure={}
    monkeypatch.setattr(c,'HERE',tmp_path)
    monkeypatch.setattr(c.protocol,'environment',lambda:{})
    monkeypatch.setattr(c.protocol,'sources',lambda:dict(raw))
    monkeypatch.setattr(c.protocol,'check_sources',lambda **kw:{'ready':False})
    def run(command,directory,*,timeout,cleanup_grace_s):
        assert timeout is None and cleanup_grace_s==30
        assert c.process.child_environment_overrides()['JAX_PLATFORMS']=='cpu'
        events.append(command[-2])
        row=dict(returncode=0,process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True,
            result=dict(passed=True,tests_passed=214))
        if failure.get('cancel'):
            c.write(directory/'process.json',row);raise failure['cancel']
        if failure.get('unknown'):raise failure['unknown']
        if failure.get('source'):raw['source.py']=b'drift'
        if failure.get('exit'):row['returncode']=1
        return row
    monkeypatch.setattr(c.process,'run_case',run)
    return NS(output=tmp_path/'proof',raw=raw,events=events,failure=failure)

def read(case):return json.loads((case.output/'report.json').read_bytes())

def test_public_only_three_children_and_exact_archive(case):
    assert c.main(case.output)==0
    r=read(case);assert r['passed'] and r['source_archive_verified'] and r['terminal_scope_completed']
    assert case.events==['tests','installed','public'] and not r['actual_execution_attempted']

def test_late_source_drift_fails_report_and_exit(case):
    case.failure['source']=True
    assert c.main(case.output)==1
    assert not read(case)['passed'] and not read(case)['source_files_unchanged']

def test_child_nonzero_fails_with_preserved_evidence(case):
    case.failure['exit']=True
    with pytest.raises(ValueError,match='child failed'):c.main(case.output)
    assert len(case.events)==1 and not read(case)['passed']

def test_primary_cancel_preserved_and_reaped_evidence_read(case):
    error=KeyboardInterrupt('cancel');case.failure['cancel']=error
    with pytest.raises(KeyboardInterrupt) as got:c.main(case.output)
    assert got.value is error and read(case)['all_processes_reaped'] and not read(case)['passed']

def test_missing_process_evidence_cannot_claim_cleanup(case):
    error=KeyboardInterrupt('unknown');case.failure['unknown']=error
    with pytest.raises(KeyboardInterrupt) as got:c.main(case.output)
    assert got.value is error and not read(case)['all_processes_reaped'] and not read(case)['cleanup_completed']

def test_initial_archive_failure_has_failed_report(case,monkeypatch):
    monkeypatch.setattr(c.collector,'archive',lambda *args:(_ for _ in ()).throw(OSError('archive')))
    with pytest.raises(OSError,match='archive'):c.main(case.output)
    assert not case.events and not read(case)['passed']

def test_terminal_shield_failure_republishes_failure(case,monkeypatch):
    error=KeyboardInterrupt('exit')
    @contextmanager
    def shield():
        yield
        raise error
    monkeypatch.setattr(c,'cleanup_signals',shield)
    with pytest.raises(KeyboardInterrupt) as got:c.main(case.output)
    assert got.value is error and not read(case)['passed'] and not read(case)['terminal_scope_completed']
