"""Independent public source/result/terminal readback; no runtime execution."""
from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_bytes())
source=Path('/tmp/qwen-client-stage-glue-v2-20261104')
proof=Path('/tmp/qwen-client-stage-boundary-proof-20261104/development-v1')
reviews=Path(__file__).parent
pins={
    proof/'report.json':'c13117ffac80ff05cf7a27dcdd514a24c4d61aa3fe1ae75808c83b6c93419130',
    proof/'owner-terminal.json':'0047eb894b2ee284d23dbd89c4cb8eb8b7ad19548f141e1db9d45b504e6ffb0b',
    proof/'handoff.json':'f43a6fc4868c06b84c618a032d4fe914f543db802f1bc389c1e28ee7d99fe02d',
    proof/'sources.tar.gz':'afc12d6b4e649aba105fd6a4de2c31aed4edad383eac6feaa9ceb927770510ab',
    proof/'sources-before.json':'7837d7c4d2a3d23ca77c7d3ff58052b45b4cda309f5cb6c234afb08fa5772f1b',
    reviews/'source-review.json':'cd8d830f917dddb1dee01126ca9c71779446431fabb7afb73e662f6ff24376ec',
    reviews/'launch-review.json':'f3ec8bd3a7f4ec0e13bc768c1b543c10fa4da474525b66017f1524479caa21be',
    reviews/'live-observation.json':'a25f4e3d499ada93cf1f3e9bda32157cdc721cc76bd05ba5647192423f766f5c',
}
for path,digest in pins.items():assert sha(path)==digest,path
report=load(proof/'report.json');owner=load(proof/'owner-terminal.json');handoff=load(proof/'handoff.json')
expected=load(proof/'sources-before.json');assert len(expected)==373
assert expected==load(proof/'sources-after.json')==load(proof/'execution-after.json')
for root in (source,proof/'execution'):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    assert {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()}==expected
with tarfile.open(proof/'sources.tar.gz')as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==373 and all(m.isfile()for m in members)
    assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()for m in members}==expected
parent=load(Path('/tmp/qwen-client-stage-glue-review-sources-v1-20261104.json'))
assert len(parent)==370 and all(expected[n]==h for n,h in parent.items())
assert {p.relative_to(Path('/tmp/qwen-client-stage-glue-20261104')).as_posix():sha(p)for p in Path('/tmp/qwen-client-stage-glue-20261104').rglob('*')if p.is_file()}==parent
launch=load(reviews/'launch-review.json')
for field in ('source_manifest_sha256','source_archive_sha256','driver_sha256','worker_driver_sha256'):
    assert report[field]==launch[field]==handoff[field]
assert sha(proof/'executed_driver.py')==sha(Path('/tmp/qwen-client-stage-boundary-driver-20261104.py'))==report['driver_sha256']
assert sha(proof/'worker_driver.py')==report['worker_driver_sha256']
cases=list(ET.parse(proof/'tests.xml').getroot().iter('testcase'))
assert len(cases)==19 and all(c.find(k)is None for c in cases for k in ('failure','error','skipped'))
nodes=['test_client_stage.py::'+c.attrib['name']for c in cases]
assert nodes==launch['selected_nodes']==report['selected_nodes']==load(source/'client_test_origin.json')['selected_nodes']
assert report['tests']==dict(tests=19,failures=0,errors=0,skipped=0)
assert all(report[k]is True for k in ('passed','sources_unchanged','synthetic_only','child_reaped','child_group_absent'))
assert all(report[k]is False for k in ('actual_role_execution','model_execution','encrypted_execution'))
assert type(report['child_returncode'])is int and report['child_returncode']==0
audit=load(proof/'worker-audit.json')
assert audit['denied']==audit['loaded_backend']==[]and audit['pytest_exit_code']==0
assert audit['synthetic_only']is True and audit['actual_backend_imported']is False
assert audit['affinity']==[108,109,110,111]and audit['pid']==audit['pgid']==3971745
assert owner['session_id']==30266 and owner['terminal_chunk']=='a84cf9'and owner['exit_code']==0 and owner['reaped']is True
assert owner['report_sha256']==sha(proof/'report.json')and owner['root_live_observation_sha256']==sha(reviews/'live-observation.json')
assert all(value is True for value in owner['observed_absence'].values())
assert owner['observed_processes']==load(proof/'live-observation.json')
live=load(reviews/'live-observation.json')
assert live['owner_session']==30266 and live['self_recorded_live_sha256']==sha(proof/'live-observation.json')
assert [row['pid']for row in live['observed']]==[3971740,3971745]
for row in live['observed']:
    pid=row['pid'];assert pid==row['pgid']and row['affinity']==[108,109,110,111]
    assert not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('process group present',pid))
assert handoff['actual_entry_enabled']is False and handoff['actual_execution_authorized']is False
out=reviews/'terminal-review.json';assert not out.exists()
result=dict(schema='qwen-client-stage-root-terminal-review-v1',passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pins={str(p):h for p,h in pins.items()},source_count=373,parent370_unchanged=True,tests=19,failures_errors_skips=0,exact_nodes=True,backend_audit_empty=True,driver_sha256=report['driver_sha256'],worker_driver_sha256=report['worker_driver_sha256'],worker_audit_sha256=sha(proof/'worker-audit.json'),tests_xml_sha256=sha(proof/'tests.xml'),elapsed_s=report['elapsed_s'],owner_terminal='30266 / a84cf9 exit0 and reaped',root_live_then_absent_pids_groups=[3971740,3971745],script_sha256=sha(Path(__file__)),scope='Nineteen source-bound recording client input/output boundaries with owned full-shape synthetic files. Original positions/one-twoL49 inputs; all38 authentication before private; default decoder argument policy and immutable full-slot snapshots; original consumer/callback cleanup and failure propagation. Context/key maps/security/codecs/signals and producer receipts are recording fixtures. No actual HE, generated keys, model/learned/ranked accuracy, active-role or calibrated performance claim.')
out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,path=str(out),sha256=sha(out),sources=373,tests=19)))
