from pathlib import Path
import datetime,hashlib,json,os,tarfile,xml.etree.ElementTree as ET
sha=lambda b:hashlib.sha256(b).hexdigest()
root=Path('/tmp/qwen-server-admission-snapshot-root-review-20261104')
proof=Path('/tmp/qwen-server-admission-snapshot-focused-proof-20261104/preparation-v1')
retained=Path('/tmp/qwen-snapshot-failed-run-retention-20261104')
assert sha((retained/'review.json').read_bytes())=='0bb3fb900f912c57c4cb39ebc81030d4151a8850284abc6d9fe3a70f8ad206dd'
index=(retained/'files.json').read_bytes();assert sha(index)=='cea9cc7683be38a068ad2008aef8fac4c7a113cb95045055b11ba4066586fdbb'
def inventory(p):
 assert not any(q.is_symlink()for q in p.rglob('*'))
 return {q.relative_to(p).as_posix():sha(q.read_bytes())for q in p.rglob('*')if q.is_file()}
actual=inventory(retained);actual.pop('files.json');assert actual==json.loads(index)and len(actual)==44
pre=json.loads((root/'preexecution-review.json').read_bytes());cfg=json.loads((proof/'run_config.json').read_bytes())
for n,h in pre['harness_pins'].items():assert sha((proof/n).read_bytes())==h
before=(proof/'sources-before.json').read_bytes();expected=json.loads(before)
assert len(expected)==37 and inventory(Path(cfg['source_root']))==inventory(proof/'execution')==expected
assert (proof/'sources-after.json').read_bytes()==(proof/'execution-after.json').read_bytes()==before
assert sha((proof/'sources.tar.gz').read_bytes())==pre['source_archive_sha256']
with tarfile.open(proof/'sources.tar.gz')as tf:
 members=tf.getmembers();assert len(members)==len({m.name for m in members})==37
 assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
xml=ET.parse(proof/'tests.xml').getroot();counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
assert counts==dict(tests=18,failures=18,errors=0,skipped=0)
assert [dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]==cfg['selected_xml_cases']
audit=json.loads((proof/'worker-audit.json').read_bytes());live=json.loads((proof/'live-observation.json').read_bytes())
assert audit['pytest_exit_code']==1 and audit['collected_nodes']==cfg['selected_nodes']and audit['denied']==audit['loaded_backend']==[]
assert audit['numpy_imported']is False and audit['affinity']==[112,113,114,115]
assert audit['pid']==audit['pgid']==live['worker_pid']==live['worker_pgid']
for pid in(live['outer_pid'],live['worker_pid']):
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise AssertionError('outer or worker remains')
for pgid in set((live['outer_pgid'],live['worker_pgid'])):
 try:os.killpg(pgid,0)
 except ProcessLookupError:pass
 else:raise AssertionError('outer or worker group remains')
missing=['report.json','owned-pytest-metadata.json','test-child-terminals.json']
assert not any((proof/n).exists()for n in missing)
logs=list((retained/'child-logs').rglob('child.log'));assert len(logs)==18
for p in logs:
 text=p.read_text();assert 'snapshot_child.py", line 63'in text and 'ValueError: public source state changed'in text and '__post_init__'in text
record=dict(schema='snapshot-root-failed-terminal-v1',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),evidence_verified=True,test_outcome='FAIL',counts=counts,guard_exercised=False,source_count=37,source_current_execution_before_after_archive_equal=True,retention_root=str(retained),retention_index_sha256=sha(index),retention_review_sha256=sha((retained/'review.json').read_bytes()),retained_indexed_files=44,owned_child_logs=18,child_pid_pgid_receipts_available=False,child_group_absence_proved=False,missing_outputs=missing,outer_worker_pid_and_pgid_absent=True,independent_live_observation=False,direct_owner=dict(session_id=97824,launch_tool_chunk='32b53a',terminal_tool_chunk='9b71cb',exit_code=1,direct_session_reaped=True,late_observation_tool='4ed067'),cause='All18 children rejected synthetic PublicContract construction before recording seam/guard; collector then rejected missing18 terminal records. Original failure retained.',script_sha256=sha(Path(__file__).read_bytes()))
out=root/'failed-terminal-review.json'
with out.open('x')as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(dict(path=str(out),sha256=sha(out.read_bytes()),evidence_verified=True,test_outcome='FAIL',counts=counts)))
