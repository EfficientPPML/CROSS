"""Public terminal readback; only fixed32 observation paths beneath pytest-owned.

No candidate imports, reruns, aliases or owned fixture contents are accessed.
The root agent owns the authoritative outer exec/wait receipt.
"""
import hashlib
import json
import os
from pathlib import Path
import stat
import tarfile
import xml.etree.ElementTree as ET

OUT=Path(__file__).parent
PROOF=Path('/tmp/qwen-keygen-proc-self-focused-proof-20261104/preparation-v2')
SOURCE=Path('/tmp/qwen-keygen-proc-self-tests-v2-20261104')
EXPECTED_MAP='bb9bee8a343f45e65fdb344240bdf3b5f7fc66b9ef5167c67909b11e9d81d53d'
EXPECTED_TAR='f6368d99e879e5e9e3629b05a12fc09c4748709ff3c393f7811e606303b9bc56'
H=lambda b:hashlib.sha256(b).hexdigest()
C=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
P=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
refs={}
def public(name):
    p=PROOF/name;raw=p.read_bytes();refs[str(p)]=H(raw);return raw
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():H(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
def absent(pid,group=False):
    assert type(pid)is int and pid>0
    try:(os.killpg if group else os.kill)(pid,0)
    except ProcessLookupError:return True
    return False
def state(s):return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,stat.S_IMODE(s.st_mode),s.st_uid)
def original_observation(path):
    # Exact named file only, with original descriptor/no-follow/type/readback.
    # Neither its siblings nor its nested `owned` directory are enumerated.
    assert str(path.parent).endswith(tuple('test_real_proc_self_audit_boun'+str(i)for i in range(32)))
    parent=path.parent.lstat();assert stat.S_ISDIR(parent.st_mode)
    before=path.lstat();assert stat.S_ISREG(before.st_mode)and 0<before.st_size<=16384
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
    try:
        opened=os.fstat(fd);assert state(opened)==state(before)
        raw=b''
        while len(raw)<=16384:
            part=os.read(fd,min(4096,16385-len(raw)))
            if not part:break
            raw+=part
        assert len(raw)==opened.st_size and state(os.fstat(fd))==state(opened)==state(path.lstat())
    finally:
        original=fd;fd=None;os.close(original)
    return raw,state(before)

raw=public('sources-before.json');expected=json.loads(raw)
assert H(raw)==EXPECTED_MAP and C(expected)==raw and len(expected)==92
assert inventory(SOURCE)==inventory(PROOF/'execution')==expected
for name in('sources-after.json','execution-after.json'):
    assert public(name)==raw
archive=public('sources.tar.gz');assert H(archive)==EXPECTED_TAR
with tarfile.open(PROOF/'sources.tar.gz')as t:
    members=t.getmembers()
    assert len(members)==len({m.name for m in members})==92 and all(m.isfile()for m in members)
    assert {m.name:H(t.extractfile(m).read())for m in members}==expected
runtime_raw=(SOURCE/'runtime_sources.json').read_bytes();runtime=json.loads(runtime_raw)
assert C(runtime)==runtime_raw and H(runtime_raw)=='01e5f8681ff5731d8d19d8f488b692bcf38025b83a0dbb3036b536a3d9f1726e'
assert len(runtime)==85 and all(expected['runtime/'+n]==h for n,h in runtime.items())
config=json.loads(public('run_config.json'))
assert refs[str(PROOF/'run_config.json')]=='ea21fcc6fd3c4d6e72141fd8417218746014f77702d1306fd63d8abd3f9c2634'
assert H(public('executed_driver.py'))=='4a36879a675ea674fdfe9781dbd8a81fbfba797d31d183b9360b7ae8a9478312'
assert H(public('worker_driver.py'))=='e151f7706e792e01c4cddc10029ac03a10dd95e4417042edbd0b23e425d0a8f3'
report=json.loads(public('report.json'));audit=json.loads(public('worker-audit.json'));live=json.loads(public('live-observation.json'))
assert report['passed']is True and report['tests']==dict(tests=32,errors=0,failures=0,skipped=0)
assert report['source_manifest_sha256']==EXPECTED_MAP and report['source_archive_sha256']==EXPECTED_TAR
assert report['sources_unchanged']is True and type(report['child_returncode'])is int and report['child_returncode']==0
assert report['child_reaped']is True and report['child_group_absent']is True
assert report['selected_nodes']==audit['collected_nodes']==config['selected_nodes']
assert len(audit['collected_nodes'])==len(set(audit['collected_nodes']))==32
xml=ET.fromstring(public('tests.xml'))
observed_xml=[{'classname':c.attrib['classname'],'name':c.attrib['name']}for c in xml.iter('testcase')]
assert observed_xml==config['selected_xml_cases']==report['selected_xml_cases']
assert not any(list(xml.iter(n))for n in('failure','error','skipped'))
assert audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported']is False
assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']=={}
assert audit['forbidden_roots']==config['forbidden_roots']
assert audit['affinity']==[112,113,114,115]and audit['pytest_exit_code']==0
assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
assert audit['capacity_evaluated']is False and report['resource_capacity_evaluated']is False
assert report['process_metadata_contents_recorded']is False
assert report['actual_execution_authorized']is False and report['actual_key_generation']is False
kernel=['/proc/self/status','/proc/self/cgroup','/proc/self/mountinfo']
assert audit['own_process_metadata_scope']==config['own_process_metadata_scope']==report['own_process_metadata_scope']==kernel

cases=[node.removesuffix(']').split('[')[1]for node in config['selected_nodes']]
origins={'cleanup_origin':'cleanup_origin.py','keygen_dependencies':'keygen_dependencies.py',
    'keygen_output':'keygen_output.py','public_source_pins':'public_source_pins.py',
    'qwen_file_contract':'qwen_file_contract.py','qwen_finalizer':'qwen_finalizer.py','qwen_public_parse':'qwen_public_parse.py'}
fields={'actual_generation_authorized','backend_attempts','backend_modules','case','denial_count',
    'local_module_origins','passed','permitted_metadata_read_completed','pgid','pid','resource_capacity_evaluated',
    'runtime_sources','schema','scope_restored','source_manifest_sha256','sticky_denial_expected'}
clean=set(cases[:11])|{'body_error'};reads=clean|{'swallowed_then_read'}
rows=[];pids=[];children=OUT/'children';children.mkdir()
for i,case in enumerate(cases):
    source=PROOF/'pytest-owned'/('test_real_proc_self_audit_boun'+str(i))/'child-observation.json'
    raw_obs,file_state=original_observation(source);v=json.loads(raw_obs)
    assert type(v)is dict and set(v)==fields and C(v)==raw_obs
    assert v['schema']=='qwen-proc-self-integration-observation-v1'and v['case']==case
    assert v['passed']is True and v['scope_restored']is True
    assert v['source_manifest_sha256']==H(runtime_raw)and v['runtime_sources']==85
    assert v['local_module_origins']==origins
    assert v['backend_attempts']==v['backend_modules']==[]
    assert v['actual_generation_authorized']is False and v['resource_capacity_evaluated']is False
    assert type(v['pid'])is int and type(v['pgid'])is int and v['pid']==v['pgid']and v['pid']>0
    assert v['sticky_denial_expected']is (case not in clean)
    assert v['permitted_metadata_read_completed']is (case in reads)
    assert type(v['denial_count'])is int and v['denial_count']==(0 if case in clean else 1)
    assert absent(v['pid'])and absent(v['pgid'],True)
    pids.append(v['pid'])
    dest=children/(str(i).zfill(2)+'-'+case);dest.mkdir()
    (dest/'child-observation.json').write_bytes(raw_obs)
    rows.append({'case':case,'source':str(source),'source_state':list(file_state),'sha256':H(raw_obs),
        'retained_path':(dest/'child-observation.json').relative_to(OUT).as_posix(),'pid':v['pid'],'pgid':v['pgid'],
        'current_pid_and_group_absent':True})
assert len(set(pids))==32
for name in('outer_pid','outer_pgid','worker_pid','worker_pgid'):
    assert absent(live[name],name.endswith('pgid'))
assert inventory(SOURCE)==inventory(PROOF/'execution')==expected
(OUT/'runtime_sources.json').write_bytes(runtime_raw)
origin_record={'schema':'qwen-proc-self-observation-origins-v1','declared_paths':rows,
    'runtime_source_manifest_sha256':H(runtime_raw),'local_modules':{n:{'relative_path':p,'sha256':runtime[p]}for n,p in origins.items()},
    'aliases_followed':False,'owned_fixture_directories_read':False,'temporary_subtree_recursive_read':False}
(OUT/'observation_origins.json').write_bytes(P(origin_record))
review={'schema':'qwen-proc-self-security-terminal-review-v1','passed':True,
    'source_count':92,'runtime_sources_unchanged':85,'tests':32,'source_manifest_sha256':EXPECTED_MAP,
    'source_archive_sha256':EXPECTED_TAR,'source_current_execution_start_end_archive_exact':True,
    'exact_ordered_collection_and_XML':True,'backend_numpy_audit_empty':True,
    'child_observations':32,'clean_controls':12,'expected_sticky_denials':20,
    'permitted_own_metadata_read_observations':13,'original_wrapper_restoration':True,
    'unique_child_pid_equals_pgid':True,'current_child_and_outer_worker_processes_absent':True,
    'child_observation_fields_closed_no_proc_contents':True,
    'root_authoritative_terminal_reported':{'launch_tool':'562eca','session_id':67367,'terminal_tool':'74d265',
        'exit_code':0,'reaped':True,'independent_live_observation':False},
    'reviewer_owned_exec_or_reap':False,'reviewer_independent_live_observation':False,
    'evidence_references':refs,'observation_origins_sha256':H(P(origin_record)),
    'candidate_imports_or_tests_rerun':False,'borrowed_or_owned_fixture_bodies_read':False,
    'actual_roles_or_resource_sufficiency_admitted':False}
(OUT/'review.json').write_bytes(P(review))
filemap={p.relative_to(OUT).as_posix():H(p.read_bytes())for p in sorted(OUT.rglob('*'))if p.is_file()}
(OUT/'files.json').write_bytes(P(filemap))
handoff={'schema':'qwen-proc-self32-terminal-handoff-v1','passed':True,'retained_observations':32,
    'review_sha256':H(P(review)),'origins_sha256':H(P(origin_record)),'files_sha256':H(P(filemap)),
    'source_manifest_sha256':EXPECTED_MAP,'source_archive_sha256':EXPECTED_TAR,
    'no_rerun_or_actual_authority':True}
(OUT/'handoff.json').write_bytes(P(handoff))
print(json.dumps(handoff,sort_keys=True))
