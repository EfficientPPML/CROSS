"""Mocked collector child/terminal/provenance tests; no retained payload reads."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace as NS
from dataclasses import replace
import json,hashlib
import pytest
import ranked_collect as c,ranked_execution as ex,ranked_case_gate as cases
import test_ranked_stage_gate as fixtures
HERE=Path(__file__).resolve().parent


@pytest.fixture
def harness(tmp_path,monkeypatch):
    monkeypatch.setenv('JAX_PLATFORMS','cpu');events=[];fail={}
    case_set=cases.authenticate(HERE/'public_evidence/fixture',cases.FIXTURE_REPORT_SHA,HERE/'public_evidence/reference',cases.REFERENCE_REPORT_SHA)
    fixture=tmp_path/'original_fixture';reference=tmp_path/'original_reference'
    fixture.mkdir();reference.mkdir();(fixture/'sentinel').write_text('keep');(reference/'sentinel').write_text('keep')
    def workspace(p):return cases.Workspace(p,p.stat().st_dev,p.stat().st_ino)
    old=case_set.case(1,42);case=replace(old,fixture_workspace=workspace(fixture),reference_workspace=workspace(reference))
    case_set=replace(case_set,cases=(case,),fixture_workspace=case.fixture_workspace,reference_workspace=case.reference_workspace)
    monkeypatch.setattr(c.worker.cases,'authenticate',lambda *a:case_set)
    root=tmp_path/'attempt';sha=ex.initialize_attempt(root,case)
    directories={name:tmp_path/name for name in ('fixture_evidence','reference_evidence','key_evidence','model_evidence')}
    for p in directories.values():p.mkdir()
    request=dict(stage=3,size=1,seed=42,staged_root=str(root),attempt_sha256=sha,
        fixture_report_sha256='1'*64,reference_report_sha256='2'*64,key_report_sha256='3'*64,model_report_sha256='4'*64,
        predecessors=[],**{n:str(p) for n,p in directories.items()})
    paths=ex.paths(root,case);bindings=dict(case_binding=c.worker.stage_gate.case_binding(case),context_id='5'*64,model_id='6'*64,
        key_report_sha256='3'*64,model_report_sha256='4'*64,previous_stage_report_sha256=None)
    raw={'ranked_worker.py':b'# mock worker\n','marker.py':b'# original\n'}
    monkeypatch.setattr(c,'sources',lambda:dict(raw));monkeypatch.setattr(c.shutil,'disk_usage',lambda p:NS(free=2<<40))
    def auth(req,expected):
        events.append('gates')
        if fail.get('gate'):raise fail['gate']
        return case_set,case,paths,{}, {},None,bindings
    monkeypatch.setattr(c.worker,'authenticate',auth)
    def launch(command,directory,*,timeout,cleanup_grace_s):
        events.append('launch');assert timeout is None and cleanup_grace_s>=30
        assert Path(command[1]).parent.name=='execution_source'
        (root/'retained-output').write_text('retain on any later failure')
        if fail.get('launch'):raise fail['launch']
        expected=c.hashes(raw);proof=dict(passed=True,stage=3,actual_stage_started=True,actual_stage_completed=True,
            readback_complete=True,source_files_unchanged=True,cleanup_completed=True,terminal_scope_completed=True,termination_scope_completed=True,
            source_hashes_start=expected,source_hashes_end=expected,bindings=bindings,readback=fixtures.summary(3,case),
            phases=dict(encryption=True,decryption=False,server_compute=False,model_forward=False,key_generation=False),all_ordered_input_bindings_validated=True)
        proof.update(fail.get('proof',{}));c.write(directory/'worker_report.json',proof)
        record=dict(returncode=0,process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True,
            result=dict(passed=True,worker_report_sha256=hashlib.sha256((directory/'worker_report.json').read_bytes()).hexdigest()))
        record.update(fail.get('process',{}));c.write(directory/'process.json',record)
        if fail.get('source'):raw['marker.py']=b'# changed\n'
        return record
    monkeypatch.setattr(c.process,'run_case',launch)
    req=tmp_path/'request.json';req.write_text(json.dumps(request)+'\n')
    args=dict(output=tmp_path/'evidence',request_path=req,request_sha256=hashlib.sha256(req.read_bytes()).hexdigest(),execute=True)
    return NS(args=args,events=events,fail=fail,root=root,protected=(fixture,reference,*directories.values()),request=request,raw=raw,bindings=bindings)


def report(h):return json.loads((h.args['output']/'report.json').read_text())


def test_complete_source_bound_worker_retains_fresh_attempt(harness):
    r=c.run(**harness.args)
    assert r['passed'] and r['operation_complete'] and r['terminal_scope_completed'] and r['source_archive_verified']
    assert harness.events==['gates','launch','gates'] and (harness.root/'retained-output').exists()
    assert not r['official_harness_timing']


def test_preparation_never_calls_handler(harness):
    harness.args['execute']=False;r=c.run(**harness.args)
    assert r['passed'] and not r['actual_execution_attempted'] and not r['operation_complete']
    assert 'launch' not in harness.events


@pytest.mark.parametrize('index',range(6))
@pytest.mark.parametrize('alias',[False,True])
def test_output_containment_precedes_mkdir_and_gates(harness,tmp_path,index,alias):
    p=harness.protected[index]
    if alias:
        a=tmp_path/'alias';a.symlink_to(p,target_is_directory=True);p=a
    harness.args['output']=p/'forbidden'
    with pytest.raises(ValueError):c.run(**harness.args)
    assert not harness.args['output'].exists() and not harness.events


@pytest.mark.parametrize('flag',['process_reaped','public_key_store_removed','owned_directory_handle_closed'])
def test_cleanup_unknown_fails_closed_and_retains(harness,flag):
    harness.fail['process']={flag:False}
    with pytest.raises(ValueError):c.run(**harness.args)
    r=report(harness);assert not r['passed'] and not r['all_processes_reaped'] and (harness.root/'retained-output').exists()


@pytest.mark.parametrize('fault',['terminal_scope_completed','readback_complete','source_hashes_end','bindings'])
def test_worker_partial_source_terminal_proof_rejected(harness,fault):
    harness.fail['proof']={fault:{} if fault in ('source_hashes_end','bindings') else False}
    with pytest.raises(ValueError):c.run(**harness.args)
    assert not report(harness)['passed']


def test_source_mutation_prevents_pass(harness):
    harness.fail['source']=True;r=c.run(**harness.args)
    assert not r['passed'] and not r['source_files_unchanged']


def test_initial_report_failure_recovers_failed_terminal(harness,monkeypatch):
    old=c.write;count=[0]
    def write(path,value):
        count[0]+=1
        if count[0]==1:raise OSError('initial')
        old(path,value)
    monkeypatch.setattr(c,'write',write)
    with pytest.raises(OSError):c.run(**harness.args)
    assert not report(harness)['passed'] and not harness.events


def test_primary_cancel_and_unknown_child_preserved(harness):
    error=KeyboardInterrupt('cancel');harness.fail['launch']=error
    with pytest.raises(KeyboardInterrupt) as seen:c.run(**harness.args)
    assert seen.value is error and not report(harness)['all_processes_reaped'] and not report(harness)['passed']
    assert (harness.root/'retained-output').exists()


def test_cleanup_scope_exit_cannot_leave_success(harness,monkeypatch):
    @contextmanager
    def shield():
        yield
        raise OSError('restoration')
    monkeypatch.setattr(c,'cleanup_signals',shield)
    with pytest.raises(OSError):c.run(**harness.args)
    r=report(harness);assert not r['passed'] and not r['terminal_scope_completed'] and not r['cleanup_completed']


def test_public_error_does_not_include_private_exception_body(harness):
    harness.fail['gate']=ValueError('synthetic-private-token-123')
    with pytest.raises(ValueError):c.run(**harness.args)
    assert 'synthetic-private-token-123' not in (harness.args['output']/'report.json').read_text()
