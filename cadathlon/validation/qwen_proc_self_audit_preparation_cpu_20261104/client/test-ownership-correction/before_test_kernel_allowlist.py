"""24 declared focused cases; uncollected/unexecuted source preparation.

Six exact metadata reads, fifteen spelling/flag denials, three unchanged464
failure controls. All actual candidate role/backend/context access is excluded.
"""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import pytest

HERE=Path(__file__).resolve().parent
SUCCESSOR_CASES=(
    'helper:status','helper:cgroup','helper:mountinfo',
    'builtin:status','builtin:cgroup','builtin:mountinfo',
    'flag:wronly','flag:rdwr','flag:trunc','flag:append','flag:create_excl','flag:directory','flag:path',
    'spelling:bytes','spelling:pathlike','spelling:dot','spelling:double_slash',
    'spelling:explicit_pid','spelling:dir_fd','spelling:unselected_self','spelling:owned_alias')
BASELINE_CASES=('helper:status','helper:cgroup','helper:mountinfo')

def digest(raw):return hashlib.sha256(raw).hexdigest()
def snapshot(root):return {str(p.relative_to(root)):digest(p.read_bytes())for p in root.rglob('*')if p.is_file()}
def absent(group):
    try:os.killpg(group,0)
    except ProcessLookupError:return True
    return False

@pytest.fixture(scope='session')
def roots(tmp_path_factory):
    selected=json.loads((HERE/'selected_tests.json').read_bytes())
    current=HERE/'runtime';current_map=(HERE/'runtime-sources.json').read_bytes()
    assert digest(current_map)==selected['successor_source_manifest_sha256']
    assert snapshot(current)==json.loads(current_map)
    # Reconstruct old464 from the exact current tree plus its sole preserved
    # old source. This is a public-source fixture, not an execution outcome.
    prior=tmp_path_factory.mktemp('old464-source')
    before=(HERE/'before_qwen_client_dependencies.py').read_bytes()
    for name in json.loads(current_map):
        data=before if name=='qwen_client_dependencies.py'else(current/name).read_bytes()
        target=prior/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
    prior_map=(HERE/'parent464-sources.json').read_bytes()
    assert digest(prior_map)==selected['parent_source_manifest_sha256']
    assert snapshot(prior)==json.loads(prior_map)
    yield (current,HERE/'runtime-sources.json',selected['successor_source_manifest_sha256']), (prior,HERE/'parent464-sources.json',selected['parent_source_manifest_sha256'])
    assert snapshot(current)==json.loads(current_map)
    assert snapshot(prior)==json.loads(prior_map)

def run_case(tmp_path,source,case,baseline):
    root,manifest,pin=source;scratch=tmp_path/'owned';scratch.mkdir(mode=0o700)
    receipt=tmp_path/'receipt.json'
    argv=[sys.executable,'-I','-B',str(HERE/'kernel_probe.py'),'--root',str(root),
        '--manifest',str(manifest),'--manifest-sha256',pin,'--case',case,
        '--scratch',str(scratch),'--receipt',str(receipt)]
    if baseline:argv.append('--baseline')
    # Spawn only this declared child; no deadline and all post-Popen operations
    # are inside the ownership try/finally, including live PGID observation.
    with (tmp_path/'stdout.log').open('wb')as stdout,(tmp_path/'stderr.log').open('wb')as stderr:
        child=subprocess.Popen(argv,stdout=stdout,stderr=stderr,start_new_session=True)
        observed_group=None;code=None
        try:
            try:observed_group=os.getpgid(child.pid)
            except ProcessLookupError:pass
            code=child.wait()
        finally:
            if child.poll()is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
    assert code==0,(tmp_path/'stderr.log').read_text()
    report=json.loads(receipt.read_bytes())
    assert report['passed']is True and report['pid']==child.pid
    assert type(report['pgid'])is int and report['pgid']>0
    if observed_group is not None:assert report['pgid']==observed_group
    assert absent(report['pgid'])
    assert report['case']==case and report['baseline']is baseline
    assert report['source_manifest_sha256']==pin
    assert report['original_open_restored']is True and report['audit_inactive']is True
    assert report['forbidden_loaded_modules']==[]
    (tmp_path/'owner.json').write_text(json.dumps(dict(pid=child.pid,pgid=report['pgid'],
        pgid_observed_by_parent=observed_group,returncode=code,waited=True,group_absent=True,
        source_manifest_sha256=pin,receipt_sha256=digest(receipt.read_bytes()),
        synthetic_scope=True,normal_source_readiness=False),sort_keys=True,indent=2)+'\n')

@pytest.mark.parametrize('case',SUCCESSOR_CASES)
def test_successor_kernel_policy(tmp_path,roots,case):
    run_case(tmp_path,roots[0],case,False)

@pytest.mark.parametrize('case',BASELINE_CASES)
def test_old464_alias_controls(tmp_path,roots,case):
    run_case(tmp_path,roots[1],case,True)
