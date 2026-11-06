"""One fresh owned child per guard-state scenario; no parent candidate import."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parent
CASES=(
    'clean_graph','byte_replace','tuple_replace','extra_field','dict_replace','mutable_graph',
    'helper_code','helper_export','keyword_default','descriptor','wrong_packet','wrong_thread',
    'reentry_uncaught','reentry_caught','validator_failure','initial_callback_failure',
    'final_callback_failure','validator_mutation',
)

@pytest.mark.parametrize('scenario',CASES,ids=CASES)
def test_snapshot_guard(scenario,tmp_path):
    result=tmp_path/'result.json';observed_pgid=None
    env=dict(os.environ)
    for name in('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    with(tmp_path/'child.log').open('xb')as log:
        proc=subprocess.Popen([sys.executable,'-I','-B',str(HERE/'snapshot_child.py'),scenario,str(HERE),str(result)],
            cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            # Immediate ownership uses the previously reviewed71/83 mechanism.
            try:observed_pgid=os.getpgid(proc.pid)
            except ProcessLookupError:pass
            if observed_pgid is not None:assert observed_pgid==proc.pid
            code=proc.wait()  # No healthy-execution deadline.
        except BaseException as primary:
            try:
                try:os.killpg(proc.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                except BaseException as cleanup:
                    if cleanup is not primary:primary.add_note('owned group cleanup: '+repr(cleanup))
            finally:
                try:proc.wait()
                except BaseException as cleanup:
                    if cleanup is not primary:primary.add_note('owned direct wait: '+repr(cleanup))
            raise
    assert code==0,(scenario,(tmp_path/'child.log').read_text()[-8192:])
    data=json.loads(result.read_text())
    assert data['scenario']==scenario and data['assertions_completed']is True
    assert data['recording_full_validation_seam']is True and data['scientific_admission_tested']is False
    assert data['original_validator_identities_restored']is True and data['backend_imported']is False
    assert data['backend_denials']==[]
    assert data['graph_type_count']==13 and data['pid']==proc.pid
    assert type(data['pgid'])is int and data['pgid']==proc.pid
    if observed_pgid is not None:assert data['pgid']==observed_pgid
    try:os.killpg(data['pgid'],0)
    except ProcessLookupError:group_absent=True
    else:group_absent=False
    assert group_absent
    with(tmp_path/'child-terminal.json').open('x')as stream:
        json.dump({'pid':proc.pid,'observed_pgid':observed_pgid,'child_reported_pgid':data['pgid'],
            'exit_code':code,'reaped':True,'group_absent':group_absent,'scenario':scenario,'runtime_scope':'recording guard state only'},stream,sort_keys=True)
        stream.write('\n')
