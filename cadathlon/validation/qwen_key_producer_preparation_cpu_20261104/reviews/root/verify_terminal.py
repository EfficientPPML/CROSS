"""Root admission of completed public synthetic evidence; never rerun tests."""
import hashlib,json,os,tarfile
import xml.etree.ElementTree as ET
from pathlib import Path
HERE=Path(__file__).resolve().parent
PROOF=Path('/tmp/qwen-key-producer-boundary-proof-20261104/preparation-v1')
SOURCE=Path('/tmp/qwen-key-generation-producer-v2-20261104')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()}
launch=read(HERE/'launch-review.json')
pins=dict(launch['pins'],**{'report.json':'23bd0ef77c02bf46cab063ff1e6255981875be1c6391bf8cbb84f672939e5ab1',
    'owner-terminal.json':'231997a225c40566ecb90a8899cea9c5f816e8941bb25712c853983775db7fb4',
    'handoff.json':'53d3676551403d74424d1ca597f6a8649cf2d6b629dbc17fecd35a5aff589824',
    'worker-audit.json':'a4072f8995da03c939d4571c5beb0a8fbe08216e3c01c18a5a15726b0f673a3c',
    'tests.xml':'96ddc308b7865f8489378e7d3d05bdf0c7d73084e355596a8b6cfcb55587f032'})
for name,digest in pins.items():assert sha(PROOF/name)==digest
expected=read(PROOF/'sources-before.json');assert len(expected)==71
assert inventory(SOURCE)==inventory(PROOF/'execution')==expected
assert read(PROOF/'sources-after.json')==read(PROOF/'execution-after.json')==expected
with tarfile.open(PROOF/'sources.tar.gz')as archive:
    members=archive.getmembers();assert len(members)==len({m.name for m in members})==71
    assert all(m.isfile()and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
    assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in members}==expected
old=Path('/tmp/qwen-key-generation-producer-20261104')
assert inventory(old)==read(Path('/tmp/qwen-key-producer-review-sources-v1-20261104.json'))
for name,row in read(SOURCE/'selected_sources.json')['files'].items():assert sha(Path(row['path']))==expected[name]==row['sha256']
report=read(PROOF/'report.json');owner=read(PROOF/'owner-terminal.json');handoff=read(PROOF/'handoff.json')
config=read(PROOF/'run_config.json');audit=read(PROOF/'worker-audit.json')
live=read(PROOF/'live-observation.json');root_live=read(HERE/'live-observation.json')
assert report['passed']is report['sources_unchanged']is report['child_reaped']is report['child_group_absent']is True
assert type(report['child_returncode'])is int and report['child_returncode']==0
assert report['source_count']==71 and report['tests']==dict(tests=30,failures=0,errors=0,skipped=0)
assert all(report[k]is False for k in ('actual_execution_authorized','actual_key_generation','backend_context_construction','encrypted_execution','model_execution'))
assert report['synthetic_only']is True
assert owner['session_id']==72366 and owner['terminal_chunk']=='903bc1'
assert type(owner['exit_code'])is int and owner['exit_code']==0 and owner['reaped']is True
assert all(v is True for v in owner['observed_absence'].values())
assert owner['observed_processes']==live
assert owner['root_live_observation_sha256']==sha(HERE/'live-observation.json')=='9511aa2866a3a2d170e128e2a2cd7074cc2ed6e7ea445ee900bf58f4784b34d6'
assert root_live['self_recorded_live_sha256']==sha(PROOF/'live-observation.json')
for row in root_live['processes']:
    assert row['observed_live']is True and row['affinity']==[108,109,110,111]
    assert row['pid']==live[row['role']+'_pid'] and row['pgid']==live[row['role']+'_pgid']
    assert not(Path('/proc')/str(row['pid'])).exists()
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
assert audit['denied']==audit['loaded_backend']==[] and type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0
assert audit['pid']==live['worker_pid']and audit['pgid']==live['worker_pgid']and audit['affinity']==[108,109,110,111]
assert audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']
for path,digest in config['runtime_source_expectations'].items():assert sha(Path(path))==digest
assert audit['runtime_module_origins']=={n:dict(file=p,spec=p)for n,p in config['runtime_module_origins'].items()}
assert report['selected_nodes']==audit['collected_nodes']==config['selected_nodes']==read(SOURCE/'selected_tests.json')['nodes']
xml=ET.parse(PROOF/'tests.xml').getroot();cases=list(xml.iter('testcase'))
assert len(cases)==30 and all(not list(xml.iter(tag))for tag in ('failure','error','skipped'))
assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in cases]==config['selected_xml_cases']==report['selected_xml_cases']
result=dict(schema='qwen-key-producer-root-terminal-review-v1',passed=True,source_count=71,tests=30,
    per_file={'test_key_store.py':15,'test_key_producer.py':15},pins=pins,
    launch_review_sha256=sha(HERE/'launch-review.json'),root_live_sha256=sha(HERE/'live-observation.json'),
    source_review_sha256=sha(HERE/'successor-source-review.json'),script_sha256=sha(Path(__file__)),
    owner_session=72366,owner_terminal_chunk='903bc1',independent_live_then_absent=True,
    scope='Owned small NPY files and full166/335 trusted recording metadata/order only; no real key generation/security/capacity/backend/model/HE/actual role.',
    actual_execution_authorized=False)
(HERE/'terminal-review.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k!='pins'},sort_keys=True))
