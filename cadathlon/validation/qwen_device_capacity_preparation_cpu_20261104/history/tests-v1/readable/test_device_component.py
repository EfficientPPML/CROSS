"""One owned child per device component case; parent never imports candidates."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest
from device_cases import CASES

HERE=Path(__file__).resolve().parent

@pytest.mark.parametrize('scenario',CASES,ids=CASES)
def test_device_component(scenario,tmp_path):
    result=tmp_path/'result.json';observed_pgid=None
    env=dict(os.environ)
    for name in('PYTHONPATH','PYTHONHOME','PYTEST_ADDOPTS','PYTEST_PLUGINS'):env.pop(name,None)
    env['PYTHONDONTWRITEBYTECODE']='1'
    with(tmp_path/'child.log').open('xb')as log:
        proc=subprocess.Popen([sys.executable,'-I','-B',str(HERE/'device_child.py'),scenario,str(HERE),str(result)],
            cwd=tmp_path,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
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
    try:os.killpg(proc.pid,0)
    except ProcessLookupError:group_absent=True
    else:group_absent=False
    with(tmp_path/'parent-terminal.json').open('x')as stream:
        json.dump(dict(schema='qwen-device-component-owned-child-v1',pid=proc.pid,owned_session_pgid=proc.pid,
            observed_pgid=observed_pgid,exit_code=code,reaped=True,group_absent=group_absent,
            scenario=scenario,child_result_read=False),stream,sort_keys=True);stream.write('\n')
    assert group_absent
    assert code==0,(scenario,(tmp_path/'child.log').read_text()[-8192:])
    data=json.loads(result.read_text())
    assert data['scenario']==scenario and data['assertions_completed']is True
    assert type(data['pid'])is int and data['pid']==proc.pid
    assert type(data['pgid'])is int and data['pgid']==proc.pid
    if observed_pgid is not None:assert data['pgid']==observed_pgid
    for name in('recording_phase_validation','resource_validator_is_real','original_phase_identities_restored',
            'original_resource_identities_preserved','profile_observer_restored','temporary_jax_removed'):
        assert data[name]is True
    for name in('genuine_backend_imported','genuine_capacity_measured','actual_policy_or_grant','scientific_admission_tested','device_sources_modified'):
        assert data[name]is False
    assert data['backend_denials']==[]
    with(tmp_path/'child-terminal.json').open('x')as stream:
        json.dump(dict(pid=proc.pid,observed_pgid=observed_pgid,child_reported_pgid=data['pgid'],exit_code=code,
            reaped=True,group_absent=group_absent,scenario=scenario,runtime_scope='synthetic device policy/state only'),stream,sort_keys=True)
        stream.write('\n')
