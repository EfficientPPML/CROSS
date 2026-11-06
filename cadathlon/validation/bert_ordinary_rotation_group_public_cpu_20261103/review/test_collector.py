"""Backend-free source/terminal/lifecycle regressions; subprocesses are mocked."""
import json,os,signal
from pathlib import Path
import pytest
import collect_public as collect

@pytest.fixture
def harness(tmp_path,monkeypatch):
    root=tmp_path/'lab';root.mkdir()
    for name in collect.SOURCE_NAMES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b'fixed source')
    monkeypatch.setattr(collect,'HERE',root)
    events=[];settings={};processes=[]
    class Child:
        def __init__(self,args,**kwargs):
            self.pid=123450+len(processes);self.args=args;self.code=None;self.waits=0;processes.append(self)
        def poll(self):return self.code
        def wait(self):
            self.waits+=1;events.append(('wait',self.pid))
            if self.waits==1 and settings.get('failure') is not None:raise settings['failure']
            if self.waits>1:self.code=-9;return self.code
            output=settings['output']
            if '-m' in self.args:
                (output/'tests.xml').write_text('<testsuites><testsuite tests="7" failures="0" errors="0" skipped="0"/></testsuites>')
            else:
                path=output/'public-parity';path.mkdir()
                value=dict(passed=True,fixtures=[{} for _ in range(12)],total_comparisons=372,total_negative_guards=180,no_context_keygen_modules_loaded=True)
                (path/'report.json').write_text(json.dumps(value))
                if settings.get('drift'):(root/'check_public_group.py').write_bytes(b'changed')
            self.code=0;return 0
    def kill(pid,sig):
        events.append(('kill',pid,sig))
        if settings.get('repeat_signals'):
            for signum in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):os.kill(os.getpid(),signum)
        if settings.get('kill_failure'):raise settings['kill_failure']
    monkeypatch.setattr(collect.subprocess,'Popen',Child);monkeypatch.setattr(collect.os,'killpg',kill)
    return root,settings,events,processes

def execute(harness,name='out'):
    root,settings,_,_=harness;settings['output']=root/name
    return collect.run(settings['output'])

def test_nested_output_checkout_is_excluded(harness):
    root,_,_,_=harness;before=collect.hashes(collect.sources())
    path=root/'source-bound-v1/snapshot/checkout/jaxite_word/herot.py';path.parent.mkdir(parents=True);path.write_bytes(b'output copy')
    assert collect.hashes(collect.sources())==before

def test_required_source_missing_fails(harness):
    root,_,_,_=harness;(root/'check_public_group.py').unlink()
    with pytest.raises(FileNotFoundError):collect.sources()

def test_success_records_exact_inventory_and_terminal(harness,capsys):
    result=execute(harness)
    assert result['passed'] is True and result['terminal_scope_completed'] is True
    assert result['source_hashes_before']==result['source_hashes_after']==result['snapshot_hashes']
    assert all(p['reaped'] and p['returncode']==0 for p in result['processes'])
    assert json.loads(capsys.readouterr().out)['passed'] is True

def test_late_source_drift_has_false_report_and_stdout(harness,capsys):
    harness[1]['drift']=True;result=execute(harness)
    assert result['passed'] is False and result['source_files_unchanged'] is False
    assert json.loads(capsys.readouterr().out)['passed'] is False
    persisted=json.loads((harness[1]['output']/'report.json').read_text());assert persisted['passed'] is False

@pytest.mark.parametrize('passed,exit_code',[(True,0),(False,1)])
def test_main_returns_report_exit_outside_termination(monkeypatch,tmp_path,passed,exit_code):
    monkeypatch.setattr(collect,'run',lambda output:dict(passed=passed,terminal_scope_completed=True))
    assert collect.main(['--output',str(tmp_path/'unused')])==exit_code

def test_interruption_stops_and_reaps_owned_child(harness):
    error=KeyboardInterrupt('original body');harness[1]['failure']=error
    with pytest.raises(KeyboardInterrupt) as caught:execute(harness)
    assert caught.value is error
    assert [e[0] for e in harness[2]]==['wait','kill','wait']
    report=json.loads((harness[1]['output']/'report.json').read_text())
    assert report['passed'] is False and report['process_reaped'] is True
    assert report['terminal_scope_completed'] is False

def test_primary_survives_cleanup_and_publication_errors(harness,monkeypatch):
    error=KeyboardInterrupt('body');harness[1].update(failure=error,kill_failure=OSError('kill failed'))
    original=collect.write
    def write(path,value):
        if 'source_hashes_after' in value:raise OSError('final publication')
        return original(path,value)
    monkeypatch.setattr(collect,'write',write)
    with pytest.raises(KeyboardInterrupt) as caught:execute(harness)
    assert caught.value is error and harness[3][0].poll()==-9
    assert any('publication' in note for note in error.__notes__)

def test_repeat_all_three_signals_deferred_through_cleanup(harness):
    error=SystemExit(143);harness[1].update(failure=error,repeat_signals=True)
    before={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    with pytest.raises(SystemExit) as caught:execute(harness)
    assert caught.value is error and harness[3][0].poll()==-9
    assert before=={s:signal.getsignal(s) for s in before}
    report=json.loads((harness[1]['output']/'report.json').read_text())
    assert report['process_reaped'] is True and report['passed'] is False
