"""Independent public proof readback only; no test, producer, or body execution."""
from pathlib import Path
import ast
import hashlib
import json
import os
import tarfile
import xml.etree.ElementTree as ET

PROOF=Path('/tmp/qwen-keygen-authority-boundary-proof-20261104/preparation-v1')
OUT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def document(name,expected=None):
    raw=(PROOF/name).read_bytes()
    if expected is not None:assert sha(raw)==expected,(name,sha(raw))
    return json.loads(raw)
def inventory(root):
    paths=list(root.rglob('*'));assert not any(p.is_symlink()for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
handoff=document('terminal-handoff.json','53037bd05667d42c437fee628e64bef2d358ab75f0ca8cd2d1f0d60b83909592')
owner=document('owner-terminal.json','55faf18e48c75dda9c788d9d0fe5d66865d4189db0c578d7762413fa77e69708')
report=document('report.json','c022c5917868ce0666dcd2024024bedf7a16c2bdcf121587017d4f2b8acbf54f')
config=document('run_config.json');expected=document('sources-before.json');audit=document('worker-audit.json');live=document('live-observation.json')
assert handoff['passed']is True and owner['passed']is True and report['passed']is True
assert handoff['owner_session']==owner['owner_session']==22566
assert handoff['owner_terminal_chunk']==owner['owner_terminal_chunk']=='c646b3'
assert type(owner['owner_tool_exit_code'])is int and owner['owner_tool_exit_code']==0 and owner['owner_reaped']is True
assert handoff['exit_code']==0 and handoff['reaped']is True
for name,digest in owner['files'].items():assert sha((PROOF/name).read_bytes())==digest,name
assert sha((PROOF/'owner-terminal.json').read_bytes())==handoff['owner_receipt_sha256']
assert sha((PROOF/'owner_terminal_review.py').read_bytes())==handoff['owner_script_sha256']
assert len(expected)==39 and compact(expected)==(PROOF/'sources-before.json').read_bytes()
assert sha(compact(expected))==config['source_manifest_sha256']==handoff['source_map_sha256']
current=inventory(Path(config['source_root']));execution=inventory(PROOF/'execution')
assert current==execution==document('sources-after.json')==document('execution-after.json')==expected
archive_raw=(PROOF/'sources.tar.gz').read_bytes()
assert sha(archive_raw)==config['source_archive_sha256']==handoff['source_archive_sha256']=='532c9f51972f851f535f373115022876fa2553af12f21ed84617de83e0f877b2'
with tarfile.open(PROOF/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==39
    assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
    assert {m.name:sha(archive.extractfile(m).read())for m in members}==expected
parent_path=Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json')
parent_raw=parent_path.read_bytes();assert sha(parent_raw)=='d67a5df7aff7cd6560e88bd7731cf124ceefba18f4a2c83c443aa1ad1aa72ff8'
parent=json.loads(parent_raw);assert len(parent)==28 and all(expected[n]==h for n,h in parent.items())
assert (PROOF/'execution/parent28_sources.json').read_bytes()==parent_raw
parent_root=Path('/tmp/qwen-keygen-authority-boundary-20261104')
assert all(sha((parent_root/n).read_bytes())==h for n,h in parent.items())
nodes=config['selected_nodes'];assert len(nodes)==len(set(nodes))==48
assert sha(pretty(nodes))==config['selected_nodes_sha256']
assert sha(pretty(config['selected_xml_cases']))==config['selected_xml_cases_sha256']
xml=ET.parse(PROOF/'tests.xml').getroot()
counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=48,failures=0,errors=0,skipped=0)==report['tests']==owner['tests']
cases=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]
assert cases==config['selected_xml_cases']==report['selected_xml_cases']
assert audit['collected_nodes']==report['selected_nodes']==nodes
assert {name:sum(c['classname']==name for c in cases)for name in ('test_keygen_launch','test_keygen_readiness','test_keygen_retention')}=={'test_keygen_launch':17,'test_keygen_readiness':25,'test_keygen_retention':6}
assert audit['denied']==audit['loaded_backend']==[] and audit['numpy_imported']is False
assert 'numpy'in audit['forbidden_roots']and audit['forbidden_roots']==config['forbidden_roots']
assert audit['runtime_sources_before']==audit['runtime_sources_after']==audit['runtime_module_origins']==config['runtime_source_expectations']==config['runtime_module_origins']=={}
assert type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0
assert audit['affinity']==config['cpu_affinity']==[112,113,114,115]
assert audit['pid']==live['worker_pid']==43440 and audit['pgid']==live['worker_pgid']==43440
assert live['outer_pid']==live['outer_pgid']==43435
assert report['child_returncode']==0 and report['child_reaped']is True and report['child_group_absent']is True
assert owner['child_direct_returncode']==0 and owner['child_reaped']is True
for field in ('actual_execution_authorized','actual_key_generation','backend_context_construction','encrypted_execution','model_execution','numpy_imported'):assert report[field]is False
assert report['synthetic_only']is True and config['synthetic_only']is True
assert report['driver_sha256']==sha((PROOF/'executed_driver.py').read_bytes())
assert report['worker_driver_sha256']==sha((PROOF/'worker_driver.py').read_bytes())
assert report['config_sha256']==sha((PROOF/'run_config.json').read_bytes())==live['config_sha256']
driver=ast.parse((PROOF/'executed_driver.py').read_text())
constants={n.targets[0].id:ast.literal_eval(n.value)for n in driver.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('CONFIG_SHA256','WORKER_SHA256')}
assert constants['CONFIG_SHA256']==report['config_sha256']and constants['WORKER_SHA256']==report['worker_driver_sha256']
race=document('late-live-observation-race.json');assert race['proof_test_failure']is False and race['source_or_proof_mutated']is False
absence={}
for pid,pgid in ((live['outer_pid'],live['outer_pgid']),(live['worker_pid'],live['worker_pgid'])):
    try:os.kill(pid,0)
    except ProcessLookupError:pid_absent=True
    else:pid_absent=False
    try:os.killpg(pgid,0)
    except ProcessLookupError:group_absent=True
    else:group_absent=False
    assert pid_absent and group_absent
    absence[str(pid)]=dict(pid_absent=pid_absent,pgid=pgid,group_absent=group_absent)
result=dict(schema='qwen-keygen-authority-cnn-terminal-review-v1',passed=True,
    review_script_sha256=sha(Path(__file__).read_bytes()),proof_root=str(PROOF),
    evidence_scope='Independent public artifact/source readback; no test or production module execution and no actual body/private/key access.',
    source_count=39,parent28_unchanged=True,current_execution_start_end_tar_equal=True,
    source_manifest_sha256=config['source_manifest_sha256'],source_archive_sha256=sha(archive_raw),
    report_sha256=sha((PROOF/'report.json').read_bytes()),owner_receipt_sha256=sha((PROOF/'owner-terminal.json').read_bytes()),
    handoff_sha256=sha((PROOF/'terminal-handoff.json').read_bytes()),tests=counts,
    exact_collection_xml_order=True,backend_and_numpy_denials_empty=True,numpy_imported=False,
    current_process_absence=absence,
    owner_terminal=dict(session=22566,chunk='c646b3',exit_code=0,reaped=True,
        provenance='Peer-owned tool observation authenticated through the root-pinned owner/handoff; reviewer did not poll this session.'),
    live_observation_scope='Owner reports an earlier successful external observation; later ps race is preserved. No root/reviewer live observation claimed.',
    scope_limits=['Pure synthetic public ModelAdmission/grant/readiness fixtures and335 descriptor dictionaries only.','No genuine ModelAdmission or key generation/RNG/body integrity, resource/security execution, backend context, actual role or downstream authorization inferred.'],
    actual_authority=False)
(OUT/'review.json').write_bytes(pretty(result))
print(json.dumps(dict(passed=True,source_count=39,tests=48,review_sha256=sha((OUT/'review.json').read_bytes())),sort_keys=True))
