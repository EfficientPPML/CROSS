"""Snapshot/terminal/reaping controls using explicit child-process doubles."""
from pathlib import Path
import json
import os
import pytest
import collect_validator as collector
import component_protocol as p
import component_contract as c

@pytest.mark.parametrize('protected,alias', [('keys',False),('keys',True),('evidence',False),('evidence',True)])
def test_output_cannot_modify_retained_public_roots(tmp_path,protected,alias):
    keys=tmp_path/'keys';evidence=tmp_path/'evidence';keys.mkdir();evidence.mkdir()
    target=keys if protected=='keys'else evidence
    if alias:
        link=tmp_path/'alias';link.symlink_to(target,target_is_directory=True);output=link/'attempt'
    else:output=target/'attempt'
    with pytest.raises(ValueError,match='before publication'):
        collector.collect(output=output,key_workspace=keys,key_evidence=evidence)
    assert not list(keys.iterdir())and not list(evidence.iterdir())


def setup(monkeypatch,*,failure=None):
    events=[];raw={'stub.py':b'pass\n'};expected=p.hashes(raw)
    monkeypatch.setenv('JAX_PLATFORMS','cpu')
    monkeypatch.setattr(p,'sources',lambda:raw)
    monkeypatch.setattr(collector.common,'inspect_sources',lambda expected:(b'profile',b'plan',{'bindings':{'producer_map_sha256':'ab'*32}}))
    def public(*a,**k):events.append('public_gate');return dict(context_id=c.CONTEXT_ID,key_report_sha256=p.KEY_REPORT,key_source_archive_sha256='01'*32)
    monkeypatch.setattr(p,'authenticate_keys',public)
    read=collector.key_gate.read
    settings=dict(mock_test_files=['test_mock.py'],mock_tests=1,installed_test_file='test_installed.py',installed_tests=1)
    def read_control(path,*a,**k):
        if Path(path)==collector.HERE/'validator_preflight_expectations.json':return c.canonical(settings)
        return read(path,*a,**k)
    monkeypatch.setattr(collector.key_gate,'read',read_control)
    live={'group':True}
    class Child:
        def __init__(self,command,**kwargs):
            self.pid=987654321;self.returncode=None;self.command=command;self.wait_calls=0
            live['group']=True
            assert kwargs['start_new_session'] and kwargs['umask']==0o077
            assert (kwargs['cwd']/'stub.py').read_bytes()==b'pass\n'
            self.xml=Path(next(x.split('=',1)[1] for x in command if x.startswith('--junitxml=')))
            events.append(('launch',self.xml.name))
        def wait(self,timeout=None):
            events.append(('wait',timeout));self.wait_calls+=1
            if failure=='cancel' and self.wait_calls==1:raise KeyboardInterrupt('parent wait cancellation')
            self.returncode=1 if failure=='child_nonzero' else -15 if failure=='cancel' else 0
            live['group']=failure=='orphan'
            body='<failure message="failed"/>' if failure=='xml_failure' else ''
            self.xml.write_text('<testsuites><testsuite><testcase name="synthetic">'+body+'</testcase></testsuite></testsuites>')
            return self.returncode
        def poll(self):return self.returncode
    monkeypatch.setattr(collector.subprocess,'Popen',Child)
    def killpg(pid,sig):
        if sig==0:
            if not live['group']:raise ProcessLookupError()
        else:events.append(('kill',pid,int(sig)));live['group']=False
    monkeypatch.setattr(collector.os,'killpg',killpg)
    return events,expected


def test_parent_snapshot_two_independent_children_and_terminal(tmp_path,monkeypatch):
    events,expected=setup(monkeypatch);out=tmp_path/'proof'
    assert collector.collect(output=out,key_evidence=tmp_path/'key_evidence',key_workspace=tmp_path/'keys')==0
    report=json.loads((out/'report.json').read_bytes())
    assert report['passed'] and report['terminal_scope_completed'] and report['all_direct_children_reaped'] and report['all_owned_groups_empty']
    assert report['source_hashes_start']==report['source_hashes_end']==expected
    assert report['source_archive_verified'] and report['no_encrypted_execution']
    assert len(report['children'])==2 and all(x['direct_returncode']==0 and x['direct_child_reaped'] and x['group_empty_verified'] for x in report['children'])
    assert events[0]=='public_gate' and all(x[1] is None for x in events if isinstance(x,tuple) and x[0]=='wait')
    assert not any(x[0]=='kill' for x in events if isinstance(x,tuple))


@pytest.mark.parametrize('failure',['child_nonzero','xml_failure','cancel','orphan'])
def test_parent_failed_child_or_cancellation_never_publishes_success(tmp_path,monkeypatch,failure):
    events,expected=setup(monkeypatch,failure=failure);out=tmp_path/'proof'
    with pytest.raises(KeyboardInterrupt if failure=='cancel' else RuntimeError if failure=='orphan' else ValueError):
        collector.collect(output=out,key_evidence=tmp_path/'key_evidence',key_workspace=tmp_path/'keys')
    report=json.loads((out/'report.json').read_bytes())
    assert report['passed'] is False and report['terminal_scope_completed'] is False
    assert len(report['children'])==1 and report['children'][0]['direct_child_reaped']
    if failure in ('cancel','orphan'):assert any(x[0]=='kill' for x in events if isinstance(x,tuple))
    else:assert not any(x[0]=='kill' for x in events if isinstance(x,tuple))


def test_public_gate_failure_never_spawns_child(tmp_path,monkeypatch):
    events,expected=setup(monkeypatch);out=tmp_path/'proof'
    def fail(*a,**k):raise ValueError('external key report not completed')
    monkeypatch.setattr(p,'authenticate_keys',fail)
    with pytest.raises(ValueError,match='key report'):collector.collect(output=out,key_evidence=tmp_path/'public-evidence',key_workspace=tmp_path/'keys')
    report=json.loads((out/'report.json').read_bytes());assert report['passed'] is False and not report['children']
    assert not any(x[0]=='launch' for x in events if isinstance(x,tuple))


def test_source_snapshot_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):collector.source_snapshot(tmp_path,{'../foreign.py':b'pass'})
    assert not (tmp_path/'foreign.py').exists()
