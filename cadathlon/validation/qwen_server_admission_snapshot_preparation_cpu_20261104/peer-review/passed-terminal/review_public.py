"""Exact public terminal readback; no candidate import or test execution."""
import datetime
import hashlib
import json
import os
from pathlib import Path,PurePosixPath
import stat
import tarfile
import xml.etree.ElementTree as ET
R=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v2')
ROOTREVIEW=Path('/tmp/qwen-server-admission-snapshot-root-review-v2-20261104')
OUT=Path('/tmp/qwen-snapshot-pass-terminal-security-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def state(s):return(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def read(p,cap=1<<24):
    s=p.lstat();assert stat.S_ISREG(s.st_mode)and s.st_size<=cap
    fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW)
    try:
        assert state(os.fstat(fd))==state(s);parts=[];total=0
        while True:
            b=os.read(fd,min(65536,cap+1-total))
            if not b:break
            parts.append(b);total+=len(b);assert total<=cap
        assert state(os.fstat(fd))==state(s)
    finally:os.close(fd)
    assert state(p.lstat())==state(s);return b''.join(parts)
def obj(p):return json.loads(read(p))
def inv(root):
    paths=sorted(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return{p.relative_to(root).as_posix():sha(read(p))for p in paths if p.is_file()}
def emit(n,v):(OUT/n).write_text(json.dumps(v,sort_keys=True,indent=2)+'\n')

rt=read(ROOTREVIEW/'terminal-review.json');assert sha(rt)=='fee2eede18c048f15226bb729d891fd079b893e62a838625b61321be12e840eb'
root=json.loads(rt);evidence={n:sha(read(R/n))for n in root['evidence']};assert evidence==root['evidence']
owner=read(ROOTREVIEW/'direct-owner.json');assert sha(owner)==root['direct_owner_receipt_sha256']
owned=json.loads(owner);assert owned['exit_code']==0 and type(owned['exit_code'])is int and owned['direct_session_reaped']is True
assert owned['independent_live_observation']is False
c=obj(R/'run_config.json');manifest=read(R/'sources-before.json');m=json.loads(manifest)
assert len(m)==38 and manifest==(json.dumps(m,sort_keys=True,indent=2)+'\n').encode()
assert sha(manifest)==c['source_manifest_sha256']
assert inv(Path(c['source_root']))==inv(R/'execution')==m
assert read(R/'sources-after.json')==read(R/'execution-after.json')==manifest
with tarfile.open(R/'sources.tar.gz','r:gz')as a:
    members=a.getmembers();assert len(members)==len({x.name for x in members})==38
    assert all(x.isfile()and not Path(x.name).is_absolute()and'..'not in Path(x.name).parts for x in members)
    assert{x.name:sha(a.extractfile(x).read())for x in members}==m
runtime=obj(R/'execution/runtime_sources.json');assert len(runtime)==28 and inv(R/'execution/runtime')==runtime
assert inv(Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104/runtime'))==runtime

xml=ET.fromstring(read(R/'tests.xml'));counts={n:len(list(xml.iter(t)))for n,t in(('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped'))}
assert counts==dict(tests=18,failures=0,errors=0,skipped=0)
assert[dict(classname=x.attrib['classname'],name=x.attrib['name'])for x in xml.iter('testcase')]==c['selected_xml_cases']
audit=obj(R/'worker-audit.json');live=obj(R/'live-observation.json');report=obj(R/'report.json')
assert audit['collected_nodes']==c['selected_nodes']and audit['denied']==audit['loaded_backend']==[]
assert audit['forbidden_roots']==c['forbidden_roots']and audit['numpy_imported']is False
assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
assert type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0 and audit['affinity']==[112,113,114,115]
assert report['passed']is True and report['tests']==counts and report['sources_unchanged']is True
assert report['child_reaped']is True and report['child_group_absent']is True and type(report['child_returncode'])is int and report['child_returncode']==0
for n in('actual_execution_authorized','numpy_imported','model_execution','encrypted_execution','backend_context_construction','actual_key_generation','actual_client_role_execution','actual_proofs_present','full_scientific_admission_tested'):assert report[n]is False

meta=obj(R/'owned-pytest-metadata.json');assert len(meta)==89
regular={n:v for n,v in meta.items()if v['kind']=='regular'};aliases={n:v for n,v in meta.items()if v['kind']=='owned_pytest_alias'}
assert len(regular)==72 and len(aliases)==17
parents=set();raws={};total=0
for name,item in regular.items():
    q=PurePosixPath(name);assert not q.is_absolute()and len(q.parts)==3 and q.parts[0]=='pytest-owned'and'..'not in q.parts
    assert q.name in('child.log','result.json','child-terminal.json','parent-terminal.json')
    p=R/name;s=p.parent.lstat();assert stat.S_ISDIR(s.st_mode)and s.st_uid==os.geteuid();parents.add(p.parent)
    raw=read(p,1<<18);total+=len(raw);assert len(raw)==item['bytes']and sha(raw)==item['sha256'];raws[name]=raw
assert len(parents)==18 and total<=c['owned_temp_max_bytes']
for p in parents:assert set(os.listdir(p))=={'child.log','result.json','child-terminal.json','parent-terminal.json'}
for name,item in aliases.items():
    q=PurePosixPath(name);assert len(q.parts)==2 and q.parts[0]=='pytest-owned'and q.name.endswith('current')
    p=R/name;s=p.lstat();assert stat.S_ISLNK(s.st_mode)and s.st_uid==os.geteuid()
    target=os.readlink(p);assert target==item['target']
    lexical=Path(target)if Path(target).is_absolute()else p.parent/target
    assert lexical in parents and str(lexical.relative_to(R))==item['resolved_target']and'..'not in lexical.parts
    assert state(p.lstat())==state(s)and os.readlink(p)==target
assert set(os.listdir(R/'pytest-owned'))=={p.name for p in parents}|{PurePosixPath(n).name for n in aliases}
assert os.listdir(R/'tmp')==[]

records=obj(R/'test-child-terminals.json');assert len(records)==18 and report['child_records']==records
scenarios=[n.split('[',1)[1][:-1]for n in c['selected_nodes']];ids=[];used_paths=set()
for scenario,row in zip(scenarios,records,strict=True):
    paths={k:row[k]for k in('parent_terminal_path','terminal_path','result_path','log_path')}
    assert all(p in raws for p in paths.values());used_paths.update(paths.values())
    parent=json.loads(raws[paths['parent_terminal_path']]);terminal=json.loads(raws[paths['terminal_path']]);result=json.loads(raws[paths['result_path']])
    assert parent==row['parent_terminal']and terminal==row['terminal']and result==row['result']
    assert sha(raws[paths['parent_terminal_path']])==row['parent_terminal_sha256']and sha(raws[paths['log_path']])==row['log_sha256']
    assert set(parent)=={'schema','pid','owned_session_pgid','observed_pgid','exit_code','reaped','group_absent','scenario','child_result_read','runtime_scope'}
    assert parent['schema']=='qwen-snapshot-owned-child-terminal-v1'and parent['child_result_read']is False
    pid=result['pid'];assert type(pid)is int and pid>0 and type(result['pgid'])is int and result['pgid']==pid;ids.append(pid)
    assert all(type(v)is int and v==pid for v in(parent['pid'],parent['owned_session_pgid'],terminal['pid'],terminal['child_reported_pgid']))
    observed=parent['observed_pgid'];assert observed is None or(type(observed)is int and observed==pid)
    assert terminal['observed_pgid']==observed
    for rec in(parent,terminal):
        assert rec['scenario']==scenario and type(rec['exit_code'])is int and rec['exit_code']==0
        assert rec['reaped']is True and rec['group_absent']is True and rec['runtime_scope']=='recording guard state only'
    assert result['scenario']==scenario and result['assertions_completed']is True and result['original_validator_identities_restored']is True
    assert result['recording_full_validation_seam']is True and result['scientific_admission_tested']is False
    assert result['actual_proof_instances']is False and result['backend_imported']is False and result['backend_denials']==[]
    assert type(result['graph_type_count'])is int and result['graph_type_count']==13
assert len(set(ids))==18 and used_paths==set(regular)
assert report['retained_owned_files']==89 and report['early_parent_records_validated']==18
for role in('outer','worker'):
    pid=live[role+'_pid'];pgid=live[role+'_pgid'];assert type(pid)is int and type(pgid)is int and pid==pgid
    if role=='worker':assert audit['pid']==pid and audit['pgid']==pgid
    ids.append(pid)
absence=[]
for pid in ids:
    for kind,fn in(('pid',os.kill),('pgid',os.killpg)):
        try:fn(pid,0)
        except ProcessLookupError:pass
        else:raise ValueError('recorded process/group is present')
        absence.append({'kind':kind,'value':pid,'absent_now':True})

failed=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
oldmap=obj(failed/'sources-before.json');assert len(oldmap)==37
assert inv(failed/'execution')==inv(Path('/tmp/qwen-server-admission-snapshot-tests-v2-20261104'))==oldmap
retained=Path('/tmp/qwen-snapshot-failed-run-retention-20261104');index=obj(retained/'files.json')
assert sha(read(retained/'files.json'))=='cea9cc7683be38a068ad2008aef8fac4c7a113cb95045055b11ba4066586fdbb'
assert all(sha(read(retained/n))==d for n,d in index.items())
assert read(failed/'tests.xml')==read(retained/'evidence/tests.xml')and len(list(ET.fromstring(read(failed/'tests.xml')).iter('failure')))==18
assert evidence=={n:sha(read(R/n))for n in evidence}
assert inv(Path(c['source_root']))==inv(R/'execution')==m
emit('process-absence.json',dict(observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observations=absence,
    independent_live_observation=False,reviewer_owned_launch_or_reap=False))
emit('review.json',dict(schema='qwen-snapshot-pass-terminal-security-review-v1',passed=True,source_count=38,runtime_count=28,
    ordered_xml=counts,source_manifest_sha256=c['source_manifest_sha256'],source_archive_sha256=c['source_archive_sha256'],
    source_current_execution_archive_before_after_equal=True,evidence=evidence,
    root_terminal_sha256=sha(rt),root_owner_receipt_sha256=sha(owner),
    root_owned_session=owned['session_id'],root_owner_exit=owned['exit_code'],root_owner_terminal_tool=owned['terminal_tool_chunk'],
    reviewer_live_observation=False,reviewer_owned_wait=False,independent_post_exit_absence_checks=40,
    early_parent_records=18,positive_child_records=18,results=18,logs=18,regular_files=72,aliases=17,
    aliases_followed=False,full_case_and_process_joins=True,backend_attempts_or_modules=False,
    failed37_and18_setup_failures_preserved=True,guard_integrated_into_actual_server=False,
    scope='18 recording guard-state cases only; synthetic PublicContract constructor bypass and full-validator seam remain explicit.',
    full_scientific_admission_tested=False,actual_authority=False,no_rerun_or_candidate_import=True))
print(json.dumps({'review':str(OUT/'review.json'),'sha256':sha(read(OUT/'review.json')),'regular_files':72,'aliases':17,'post_exit_absence_checks':40},sort_keys=True))
