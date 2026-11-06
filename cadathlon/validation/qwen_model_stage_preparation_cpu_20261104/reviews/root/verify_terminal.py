"""Root-only review of completed focused proof; never import or rerun it."""
import hashlib,json,os,sys,tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE=Path(__file__).resolve().parent
PROOF=Path('/tmp/qwen-model-stage-boundary-proof-20261104/preparation-v1')
SOURCE=Path('/tmp/qwen-model-stage-aggregate-tests-20261104')
PARENT=Path('/tmp/qwen-model-stage-aggregate-20261104')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}

assert len(sys.argv)==3,'Expected independently received owner and handoff SHA256 pins'
launch=read(HERE/'focused-launch-review.json')
assert sha(HERE/'focused-launch-review.json')=='7565d69255e619bdccf3a1f353b4c328bf82a860910d5f2daa48385a42d1fd69'
pins=dict(launch['pins'],**{
    'report.json':'f11dca76d7b644765f0a26395e2b9b805f096f336190d75bd99f6434aeec3013',
    'owner-terminal.json':sys.argv[1], 'handoff.json':sys.argv[2],
    'worker-audit.json':'03fafa074c4b3243204ac9716610dbbfe2d07594cc026106b3113d7b98e68259',
    'tests.xml':'45da22b7dc0a33ea186b95f52d50655daaa1b1941ca41942a5779609e0f291ed'})
for name,digest in pins.items():assert sha(PROOF/name)==digest,(name,'digest mismatch')
expected=read(PROOF/'sources-before.json');assert len(expected)==38
assert inventory(SOURCE)==inventory(PROOF/'execution')==expected
assert read(PROOF/'sources-after.json')==read(PROOF/'execution-after.json')==expected
parent=read(SOURCE/'parent29_sources.json');assert len(parent)==29
assert inventory(PARENT)==parent==read(Path('/tmp/qwen-model-stage-aggregate-review-sources-v1-20261104.json'))
assert {n:expected[n] for n in parent}==parent
with tarfile.open(PROOF/'sources.tar.gz') as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==38
    assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
    assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==expected
report=read(PROOF/'report.json');owner=read(PROOF/'owner-terminal.json');handoff=read(PROOF/'handoff.json')
config=read(PROOF/'run_config.json');audit=read(PROOF/'worker-audit.json')
live=read(PROOF/'live-observation.json');root_live=read(HERE/'live-observation.json')
assert report['passed'] is report['sources_unchanged'] is report['child_reaped'] is report['child_group_absent'] is True
assert type(report['child_returncode']) is int and report['child_returncode']==0
assert report['source_count']==38 and report['tests']==dict(tests=48,failures=0,errors=0,skipped=0)
assert all(report[k] is False for k in ('actual_execution_authorized','actual_key_generation','backend_context_construction','encrypted_execution','model_execution'))
assert report['synthetic_only'] is True
assert owner['session_id']==75704 and owner['terminal_chunk']=='f20b6d'
assert type(owner['exit_code']) is int and owner['exit_code']==0 and owner['reaped'] is True
assert owner['observed_absence'] and all(v is True for v in owner['observed_absence'].values())
assert owner['observed_processes']==live
assert owner['root_live_observation_sha256']==sha(HERE/'live-observation.json')=='dde4907ad582069ea7099fe5873485756a44230048bc26b207033b85a379118a'
assert root_live['passed'] is True and root_live['live_sha256']==sha(PROOF/'live-observation.json') and root_live['live']==live
assert set(root_live['processes'])=={'4111845','4111846'}
for role in ('outer','worker'):
    pid=live[role+'_pid'];row=root_live['processes'][str(pid)]
    assert row['affinity']==[108,109,110,111] and row['pid']==pid and row['pgid']==live[role+'_pgid']==pid
    assert '/python -I -B ' in row['cmdline']
    assert not (Path('/proc')/str(pid)).exists()
    try:os.killpg(row['pgid'],0)
    except ProcessLookupError:pass
    else:raise AssertionError('owned group still exists')
for field,name in [('source_manifest_sha256','sources-before.json'),('source_archive_sha256','sources.tar.gz'),('config_sha256','run_config.json'),('driver_sha256','executed_driver.py')]:
    assert report[field]==owner[field]==handoff[field]==pins[name]
assert report['worker_driver_sha256']==owner['worker_sha256']==handoff['worker_sha256']==pins['worker_driver.py']
assert owner['report_sha256']==handoff['report_sha256']==pins['report.json']
assert handoff['owner_receipt_sha256']==pins['owner-terminal.json']
assert owner['audit_sha256']==handoff['audit_sha256']==pins['worker-audit.json']
assert owner['tests_xml_sha256']==handoff['tests_xml_sha256']==pins['tests.xml']
assert audit['denied']==audit['loaded_backend']==[] and type(audit['pytest_exit_code']) is int and audit['pytest_exit_code']==0
assert audit['synthetic_only'] is True
assert audit['pid']==live['worker_pid'] and audit['pgid']==live['worker_pgid'] and audit['affinity']==[108,109,110,111]
assert audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']
assert len(config['runtime_source_expectations'])==4 and len(config['runtime_module_origins'])==3
for path,digest in config['runtime_source_expectations'].items():assert sha(Path(path))==digest
assert audit['runtime_module_origins']=={n:dict(file=p,spec=p) for n,p in config['runtime_module_origins'].items()}
assert report['selected_nodes']==audit['collected_nodes']==config['selected_nodes']==read(SOURCE/'selected_model_tests.json')['nodes']
xml=ET.parse(PROOF/'tests.xml').getroot();cases=list(xml.iter('testcase'))
assert len(cases)==48 and all(not list(xml.iter(tag)) for tag in ('failure','error','skipped'))
assert [dict(classname=c.attrib['classname'],name=c.attrib['name']) for c in cases]==config['selected_xml_cases']==report['selected_xml_cases']
result=dict(schema='qwen-model-stage-root-focused-terminal-review-v1',passed=True,source_count=38,unchanged_parent_sources=29,tests=48,
    per_file={'test_model_admission.py':28,'test_held_identity.py':10,'test_model_recording.py':10},pins=pins,
    launch_review_sha256=sha(HERE/'focused-launch-review.json'),root_live_sha256=sha(HERE/'live-observation.json'),
    source_review_sha256=sha(HERE/'test-source-review.json'),script_sha256=sha(Path(__file__)),
    owner_session=75704,owner_terminal_chunk='f20b6d',independent_live_then_absent=True,elapsed_seconds=report['elapsed_s'],
    scope='Synthetic externally pinned evidence, owned tiny NPY identities, recording198/197 matrices and113 gamma traversal/lifetime. No retained body integrity, actual role/backend/HE/calibration or performance.',
    actual_execution_authorized=False)
(HERE/'terminal-review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='pins'},sort_keys=True))
