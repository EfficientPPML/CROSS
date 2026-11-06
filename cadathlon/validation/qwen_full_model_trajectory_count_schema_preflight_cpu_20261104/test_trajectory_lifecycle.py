"""Mock-only worker/collector publication and dedicated-process boundaries."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import json,sys
import numpy as np
import pytest
import collect_trajectory as c
import trajectory_worker as w

@pytest.fixture
def fake_collection(tmp_path,monkeypatch):
    root=tmp_path/'lab';root.mkdir();(root/'source.py').write_text('public synthetic source\n')
    (root/'trajectory_inventory.json').write_text(json.dumps({'sources':['source.py','trajectory_inventory.json'],'tests':['fake.py'],'expected_tests':1}))
    (root/'trajectory_requirement.json').write_text('{}')
    monkeypatch.setattr(c,'P',root);monkeypatch.setattr(c,'requirement',lambda:{});monkeypatch.setattr(c,'runtime_gate',lambda r:None)
    state=SimpleNamespace(mode='healthy',owners=[],primary=None,close_error=None,out=root/'out')
    class Child:
        def __init__(self,command,*,cwd,stdout,env):
            if state.mode=='launch':raise RuntimeError('injected launch failure')
            self.pid=765000+len(state.owners);state.owners.append(self)
            self.record=dict(pid=self.pid,process_group=self.pid,direct_child_reaped=False,direct_returncode=None,group_empty_verified=False,cleanup_completed=False)
        def wait(self):
            if state.mode in ('wait','cancel'):
                state.primary=KeyboardInterrupt('chosen cancellation') if state.mode=='cancel'else ValueError('chosen child error');raise state.primary
            self.record.update(direct_child_reaped=True,direct_returncode=0,group_empty_verified=state.mode=='healthy',cleanup_completed=state.mode=='healthy')
            if state.mode=='missing':self.record.pop('group_empty_verified')
            (state.out/'tests.xml').write_text('<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0"/></testsuites>')
            (state.out/'audit.json').write_text(json.dumps(dict(passed=True,denied_attempts=[],backend_modules=[])))
            return 0
        def close(self,primary=None):
            self.record.update(direct_child_reaped=True,direct_returncode=-15,group_empty_verified=True,cleanup_completed=True)
            if state.close_error is not None:
                self.record['cleanup_completed']=False
                if primary is not None:primary.add_note('secondary cleanup '+repr(state.close_error))
                else:raise state.close_error
    monkeypatch.setattr(c,'OwnedChild',Child);monkeypatch.setattr(sys,'argv',['collect','--output',str(state.out)])
    state.report=lambda:json.loads((state.out/'report.json').read_bytes());return state

def test_collector_complete_and_group_gates(fake_collection):
    assert c.main()==0;r=fake_collection.report();assert r['passed'] and r['termination_scope_completed'] and r['terminal_scope_completed']
    assert len(r['children'])==1 and r['children'][0]['group_empty_verified']

@pytest.mark.parametrize('mode',['surviving','missing','wait','cancel','launch'])
def test_collector_child_failures_preserve_false_and_primary(fake_collection,mode):
    fake_collection.mode=mode
    expected=KeyboardInterrupt if mode=='cancel'else ValueError if mode=='wait'else RuntimeError
    with pytest.raises(expected)as found:c.main()
    if mode in('wait','cancel'):assert found.value is fake_collection.primary
    assert not fake_collection.report()['passed']

def test_cancel_survives_secondary_cleanup(fake_collection):
    fake_collection.mode='cancel';fake_collection.close_error=OSError('secondary')
    with pytest.raises(KeyboardInterrupt)as found:c.main()
    assert found.value is fake_collection.primary and found.value.__notes__
    assert not fake_collection.report()['passed']

@pytest.mark.parametrize('scope',['cleanup_signals','termination_signals'])
@pytest.mark.parametrize('kind',[OSError,KeyboardInterrupt,SystemExit])
def test_terminal_failures_republish_false(fake_collection,monkeypatch,scope,kind):
    @contextmanager
    def broken():yield;raise kind('restoration')
    monkeypatch.setattr(c,scope,broken)
    with pytest.raises(kind):c.main()
    assert not fake_collection.report()['passed'] and not fake_collection.report()['termination_scope_completed']

@pytest.mark.parametrize('module',[c,w])
def test_existing_output_evidence_never_overwritten(tmp_path,monkeypatch,module):
    out=tmp_path/'prior';out.mkdir();raw=b'{"passed":true,"prior_exact_receipt":"preserve"}\n';(out/'report.json').write_bytes(raw)
    argv=['program','--output',str(out)]
    if module is w:argv+=['--execute','--fixture','0']
    monkeypatch.setattr(sys,'argv',argv)
    with pytest.raises(FileExistsError):module.main()
    assert (out/'report.json').read_bytes()==raw and list(out.iterdir())==[out/'report.json']

@pytest.fixture
def fake_worker(tmp_path,monkeypatch):
    state=SimpleNamespace(out=tmp_path/'worker',calls=[],failure=ValueError('deliberate first independent domain'),close_error=None)
    state.out.mkdir()
    req={'fixtures':[{'positions':[0],'name':'fixedS1'},{'positions':list(range(8)),'name':'fixedS8'}]}
    monkeypatch.setattr(w,'source_hashes',lambda:{'synthetic':'hash'})
    monkeypatch.setattr(w,'requirement',lambda:req)
    for name in ('runtime_gate','calibration_gate','asset_gate'):
        def gate(r,name=name):state.calls.append(name);return {'passed':True}
        monkeypatch.setattr(w,name,gate)
    monkeypatch.setattr(w,'selected_ids',lambda r:(((0,),tuple(range(8))),{'fixed':True}))
    class Parameters:
        def __init__(self,r):state.calls.append('parameters');self.records={}
        def matrix(self,name):state.calls.append('matrix');return np.zeros((8,1024))
        def gamma(self,name):raise AssertionError('reference never executes')
        def readback(self):state.calls.append('readback')
        def close(self):
            state.calls.append('close')
            if state.close_error:raise state.close_error
    monkeypatch.setattr(w,'PublicParameters',Parameters)
    def failure(*a,**k):state.calls.append('reference');raise state.failure
    monkeypatch.setattr(w,'forward',failure)
    state.report=lambda:json.loads((state.out/'report.json').read_bytes());return state

def test_worker_first_failure_preserved_no_owner_or_alternate_fixture(fake_worker):
    with pytest.raises(ValueError)as found:w.execute(0,fake_worker.out)
    assert found.value is fake_worker.failure
    assert fake_worker.calls==['runtime_gate','calibration_gate','asset_gate','parameters','matrix','reference','readback','close','asset_gate']
    r=fake_worker.report();assert not r['passed'] and not r['complete'] and r['failure_detail']['message']=='deliberate first independent domain'

@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_worker_primary_cancellation_survives_cleanup(fake_worker,kind):
    fake_worker.failure=kind('chosen');fake_worker.close_error=OSError('close')
    with pytest.raises(kind)as found:w.execute(0,fake_worker.out)
    assert found.value is fake_worker.failure and found.value.__notes__
    assert not fake_worker.report()['cleanup_completed']

def test_worker_initial_source_error_publishes_failure(fake_worker,monkeypatch):
    monkeypatch.setattr(w,'source_hashes',lambda:(_ for _ in ()).throw(OSError('initial sources')))
    with pytest.raises(OSError,match='initial sources'):w.execute(0,fake_worker.out)
    r=fake_worker.report();assert not r['passed'] and r['failure_detail']['message']=='initial sources'
    assert fake_worker.calls==[]

def test_collector_runs_verified_execution_tree_without_bytecode(fake_collection):
    assert c.main()==0;r=fake_collection.report()
    assert r['execution_source_hashes_before']==r['execution_source_hashes_after']==r['source_hashes_before']
    assert (fake_collection.out/'execution/source.py').read_text()=='public synthetic source\n'
    assert not list((fake_collection.out/'execution').rglob('*.pyc'))

def test_execution_copy_drift_fails_before_child(fake_collection,monkeypatch):
    original=c.execution_tree
    def corrupt(out,before):
        root=original(out,before);(root/'source.py').write_text('changed');return root
    monkeypatch.setattr(c,'execution_tree',corrupt)
    # A copy changed after verification cannot produce a passing terminal report.
    with pytest.raises(ValueError,match='before child'):c.main()
    assert not fake_collection.report()['execution_files_unchanged'] and not fake_collection.owners
