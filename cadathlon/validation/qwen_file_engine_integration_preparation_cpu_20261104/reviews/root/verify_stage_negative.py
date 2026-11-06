"""Authenticate preserved recording-stage negatives; no tests/runtime executed."""
from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
source=Path('/tmp/qwen-stage-boundary-negative-20261104')
proof=Path('/tmp/qwen-stage-boundary-negative-proof-20261104/development-v1')
report=load(proof/'report.json');live=load(proof/'live.json')
expected=load(proof/'sources-before.json')
assert len(expected)==365 and expected==load(proof/'sources-after.json')==load(proof/'execution-after.json')
for name,digest in expected.items():
    rel=Path(name);assert not rel.is_absolute()and '..'not in rel.parts
    for tree in (source,proof/'execution'):assert sha((tree/rel).read_bytes())==digest
archive=proof/'sources.tar.gz';assert sha(archive.read_bytes())==report['source_archive_sha256']=='11b997fe67c8fa019289e51c88c007ffd83625fe72663b4932359d4b8d550691'
with tarfile.open(archive)as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==365 and all(m.isfile()for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==expected
assert report['passed']is False and type(report['exit_code'])is int and report['exit_code']==1
assert report['child_reaped']is True and report['child_group_absent']is True and report['source_unchanged']is True
cases=list(ET.parse(proof/'tests.xml').getroot().iter('testcase'))
assert len(cases)==12 and all(c.find('error')is None and c.find('skipped')is None for c in cases)
failed=[c.attrib['name']for c in cases if c.find('failure')is not None]
assert failed==[
    'test_close_accessor_replacement_is_rejected_and_original_cleanup_runs',
    'test_handle_mutation_after_finish_is_rejected_before_key_retirement',
    'test_factory_mutation_during_cleanup_is_rejected_before_loader_return',
]
assert all('DID NOT RAISE' in c.find('failure').text for c in cases if c.find('failure')is not None)
assert live['wrapper_pid']==live['wrapper_pgid']==3905012
assert live['worker_pid']==live['worker_pgid']==3905032
for pid in (3905012,3905032):
    assert not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('group exists',pid))
stage=source/'qwen_stage_composition.py'
assert stage.read_bytes()==Path('/tmp/qwen-organizer-file-engine-integration-20261104/qwen_stage_composition.py').read_bytes()
out=Path(__file__).with_name('stage-negative-review.json');assert not out.exists()
result=dict(schema='qwen-stage-negative-root-review-v1',verified=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_count=365,stage_sha256=sha(stage.read_bytes()),test_sha256=sha((source/'test_stage_boundaries.py').read_bytes()),source_archive_sha256=sha(archive.read_bytes()),report_sha256=sha((proof/'report.json').read_bytes()),live_receipt_sha256=sha((proof/'live.json').read_bytes()),owner_terminal='54890/405f14 exit1 and reaped; public recorded process IDs independently absent',test_count=12,expected_failures=failed,passing_controls=9,script_sha256=sha(Path(__file__).read_bytes()),scope='Confirmed the original joined recording glue returned success after three bounded callback/identity mutations. Synthetic335 key maps and backend/context doubles plus owned synthetic38 response files only. No real key/model/CT bodies, HE/backend/context execution or positive integration admission. Separate successor pending.')
out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),verified=True,failed=3,controls=9,stage_sha256=result['stage_sha256'])))
