"""Synthetic collector lifecycle/CLI faults; no public tensor bodies read."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json,os,sys
import pytest
import collect_join as c
from component_processes import OwnedChild

@pytest.fixture
def fake_collection(tmp_path,monkeypatch):
    root=tmp_path/'lab';root.mkdir();(root/'source.py').write_text('public synthetic source\n')
    (root/'join_inventory.json').write_text(json.dumps({'expected_tests':1}))
    monkeypatch.setattr(c,'P',root);monkeypatch.setattr(c,'source_paths',lambda:[root/'source.py'])
    monkeypatch.setattr(c,'verify_inputs',lambda:None);monkeypatch.setattr(c,'public_arrays',lambda:{'synthetic':'a'*64})
    state=SimpleNamespace(mode='healthy',owners=[],primary=None,close_error=None)
    class Child:
        def __init__(self,command,*,cwd,stdout,env):
            if state.mode=='launch':raise RuntimeError('injected launch failure')
            self.pid=765000+len(state.owners);state.owners.append(self)
            self.record=dict(pid=self.pid,process_group=self.pid,direct_child_reaped=False,direct_returncode=None,
                group_empty_verified=False,cleanup_completed=False,cleanup_errors=[],signals_sent=[])
            self.command=command;self.env=env
        def wait(self):
            if state.mode in ('wait','cancel'):
                state.primary=KeyboardInterrupt('chosen cancellation') if state.mode=='cancel' else ValueError('chosen child error')
                raise state.primary
            self.record.update(direct_child_reaped=True,direct_returncode=0,group_empty_verified=state.mode=='healthy',cleanup_completed=state.mode=='healthy')
            if state.mode=='missing':self.record.pop('group_empty_verified')
            out=root/'source-bound-owner-v1'
            if '-m' in self.command:
                (out/'tests.xml').write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
                names=('layer0.S1','layer0.S8','layer0.S16','layer0.S19','layer0.S32','layer5.S1','layer9.S1')
                Path(self.env['QWEN_JOIN_METRICS_PATH']).write_text(json.dumps({n:{} for n in names}))
            else:(root/'join_actual_inventory.json').write_text('{}')
            return 0
        def close(self,primary=None):
            self.record.update(direct_child_reaped=True,direct_returncode=-15,group_empty_verified=True,cleanup_completed=True)
            if state.mode=='missing':self.record.pop('group_empty_verified')
            if state.close_error is not None:
                self.record['cleanup_completed']=False
                if primary is not None:primary.add_note('injected cleanup: '+repr(state.close_error))
                else:raise state.close_error
    monkeypatch.setattr(c,'OwnedChild',Child)
    state.root=root;state.report=lambda:json.loads((root/'source-bound-owner-v1/report.json').read_bytes())
    return state

def test_mock_success_requires_both_children_and_outer_completion(fake_collection):
    assert c.main()==0;r=fake_collection.report()
    assert r['passed']is True and r['termination_scope_completed']is True and r['terminal_scope_completed']is True
    assert len(r['children'])==2 and all(row['direct_child_reaped'] and row['group_empty_verified'] for row in r['children'])

@pytest.mark.parametrize('mode',['surviving','missing'])
def test_exited_leader_or_unknown_group_cannot_pass(fake_collection,mode):
    fake_collection.mode=mode
    with pytest.raises(RuntimeError,match='group absence'):c.main()
    r=fake_collection.report();assert r['passed']is False and r['termination_scope_completed']is False
    assert len(fake_collection.owners)==1

@pytest.mark.parametrize('mode',['wait','cancel','launch'])
def test_child_failure_is_persisted_and_primary_preserved(fake_collection,mode):
    fake_collection.mode=mode
    expected=KeyboardInterrupt if mode=='cancel' else RuntimeError if mode=='launch' else ValueError
    with pytest.raises(expected) as found:c.main()
    if mode!='launch':assert found.value is fake_collection.primary
    r=fake_collection.report();assert r['passed']is False and r['termination_scope_completed']is False

def test_body_cancellation_survives_secondary_cleanup_fault(fake_collection):
    fake_collection.mode='cancel';fake_collection.close_error=OSError('secondary')
    with pytest.raises(KeyboardInterrupt) as found:c.main()
    assert found.value is fake_collection.primary and found.value.__notes__
    assert fake_collection.report()['passed']is False

@pytest.mark.parametrize('kind',[OSError,KeyboardInterrupt,SystemExit])
def test_cleanup_scope_exit_failure_republishes_false(fake_collection,monkeypatch,kind):
    @contextmanager
    def fail():
        yield
        raise kind('cleanup exit')
    monkeypatch.setattr(c,'cleanup_signals',fail)
    with pytest.raises(kind):c.main()
    r=fake_collection.report();assert r['passed']is False and r['termination_scope_completed']is False
    assert r['terminal_scope_completed']is False

@pytest.mark.parametrize('kind',[OSError,KeyboardInterrupt,SystemExit])
def test_outer_termination_restore_failure_republishes_false(fake_collection,monkeypatch,kind):
    @contextmanager
    def fail():
        yield
        raise kind('termination restore')
    monkeypatch.setattr(c,'termination_signals',fail)
    with pytest.raises(kind):c.main()
    r=fake_collection.report();assert r['passed']is False and r['termination_scope_completed']is False

def test_actual_cli_cannot_exit_zero_after_outer_restore_failure(tmp_path):
    script=tmp_path/'cli_probe.py';output=tmp_path/'output';output.mkdir()
    script.write_text('''from pathlib import Path
from contextlib import contextmanager
import collect_join as c
c.P=Path(__file__).resolve().parent/'output'
def body():
 p=c.P/'source-bound-owner-v1';p.mkdir()
 r={'passed':True,'cleanup_completed':True,'terminal_scope_completed':True,'tests':{},'source_hashes_before':{},'source_archive_sha256':'synthetic'}
 c.write(p/'report.json',r)
 return r
@contextmanager
def fail():
 yield
 raise OSError('injected outer restore')
c.collect=body;c.termination_signals=fail
raise SystemExit(c.main())
''')
    env=dict(os.environ,PYTHONPATH=str(Path(c.__file__).parent))
    with (tmp_path/'child.log').open('w') as log:
        owner=OwnedChild([sys.executable,str(script)],cwd=tmp_path,stdout=log,env=env)
        assert owner.wait()!=0
    r=json.loads((output/'source-bound-owner-v1/report.json').read_bytes())
    assert r['passed']is False and r['termination_scope_completed']is False
    assert owner.record['direct_child_reaped']is True and owner.record['group_empty_verified']is True
    assert '"passed": true' not in (tmp_path/'child.log').read_text()
