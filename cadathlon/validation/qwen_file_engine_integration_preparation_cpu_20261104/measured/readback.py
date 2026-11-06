from pathlib import Path
import hashlib,json,os,tarfile,xml.etree.ElementTree as ET,datetime
s=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104');p=Path('/tmp/qwen-stage-integration-successor-proof-20261104/development-v1');e=p/'execution'
h=lambda b:hashlib.sha256(b).hexdigest()
expected=json.loads(Path('/tmp/qwen-stage-integration-v2-sources-20261104.json').read_bytes())
assert len(expected)==366
for root in (s,e):
 observed={q.relative_to(root).as_posix():h(q.read_bytes())for q in sorted(root.rglob('*'))if q.is_file()};assert observed==expected
for n in ('sources-before.json','sources-after.json','execution-after.json'):assert json.loads((p/n).read_bytes())==expected
with tarfile.open(p/'sources.tar.gz','r:gz')as t:
 members=t.getmembers();assert len(members)==366 and all(m.isfile()for m in members)
 assert {m.name:h(t.extractfile(m).read())for m in members}==expected
x=ET.parse(p/'tests.xml');assert len(x.findall('.//testcase'))==12 and all(not x.findall('.//'+k)for k in ('failure','error','skipped'))
r=json.loads((p/'report.json').read_bytes());assert r['passed']is True and r['exit_code']==0 and r['child_reaped']is True
live=json.loads((p/'live.json').read_bytes());absence={}
for k in ('wrapper_pid','worker_pid','wrapper_pgid','worker_pgid'):
 v=live[k]
 try:(os.killpg if k.endswith('pgid')else os.kill)(v,0)
 except ProcessLookupError:absence[k]=True
 else:raise AssertionError(k+' still present')
assert h((p/'run.py').read_bytes())==json.loads((p/'driver-before.json').read_bytes())['sha256']
owner=dict(schema='qwen-stage-boundary-owner-terminal-v1',scope='Owned synthetic files and recording-only key/context/engine seams; no actual backend, model, key or ciphertext bodies',source_root=str(s),execution_root=str(e),source_count=366,session_id=68008,launch_tool_chunk='7293e0',terminal_tool_chunk='17d040',exit_code=0,reaped=True,source_manifest_sha256=h(Path('/tmp/qwen-stage-integration-v2-sources-20261104.json').read_bytes()),source_archive_sha256=h((p/'sources.tar.gz').read_bytes()),report_sha256=h((p/'report.json').read_bytes()),xml_sha256=h((p/'tests.xml').read_bytes()),driver_sha256=h((p/'run.py').read_bytes()),live=live,process_absence=absence,process_scope='IDs self-recorded during run; owner checked absence after authoritative wait. No external live observation claimed.',compaction_note='Launch output was hidden by context compaction; session68008 was recovered from the original tool receipt and authoritatively polled once. No duplicate launch.',test_count=12,failed_errors_skipped=0,checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(p/'owner-terminal.json').write_text(json.dumps(owner,sort_keys=True,indent=2)+'\n')
(p/'readback.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({n:h((p/n).read_bytes())for n in ('report.json','owner-terminal.json','tests.xml','run.py','readback.py','sources.tar.gz')},sort_keys=True))
