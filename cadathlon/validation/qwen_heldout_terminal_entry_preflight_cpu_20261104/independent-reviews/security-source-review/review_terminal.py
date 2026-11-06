"""Public-only heldout successor preflight verification; no descriptor following."""
import hashlib,json,os,tarfile
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
R=Path('/tmp/qwen-heldout1088-runtime-preflight-v5-20261104')
SOURCE=Path('/tmp/qwen-heldout1088-runtime-v5-20261104')
OWNER=Path('/tmp/qwen-heldout1088-runtime-preflight-v5-owner-terminal-20261104.json')
OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
raw=(R/'source_manifest.json').read_bytes();expected=json.loads(raw)
rraw=(R/'report.json').read_bytes();r=json.loads(rraw);wraw=(R/'worker/report.json').read_bytes();w=json.loads(wraw);owner=load(OWNER)
assert sha(raw)=='9985aad6205af33a429fba3802bac415397973418e568ea3414d61197f03deeb' and len(expected)==171
assert sha(rraw)=='c4b4a7df248fe3b24d4c9d8e0f8cf4c99fcfb6bdaeae8e25aa9c136cff2ab31f'==owner['report_sha256']
assert sha(wraw)=='e3a449af0a13a070e4bda85436de1b0dfa7a51b6ecaa0080a561b9d8ee00dfd2'==r['worker_report_sha256']==owner['worker_report_sha256']
assert r['worker']==w
for row in (r,w):
 for flag in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
  assert row[flag]is True,flag
 assert row['audit_denials']==[] and 'error_type'not in row
 assert row['source_hashes_start']==row['source_hashes_end']==expected
 assert row['source_manifest_sha256']==sha(raw)
 assert row['bootstrap_origins']=={'__main__':'heldout_entry.py'}
 assert row['model_forward']is False and row['coefficient_selection']is False
assert r['actual_heldout_stage']is False and w['validation_inputs_read']is False
assert r['child_reaped']is True and r['child_group_absent']is True and type(r['child_returncode'])is int and r['child_returncode']==0
assert r['child_pid']==r['child_pgid']==w['pid']
assert type(w['tests'])is int and w['tests']==563
for k in ('errors','failures','skipped','pytest_returncode'):assert type(w[k])is int and w[k]==0
assert w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla']
xml=ET.fromstring((R/'worker/tests.xml').read_bytes());suites=xml.findall('testsuite');assert len(suites)==1
assert int(suites[0].get('tests'))==563 and len(xml.findall('.//testcase'))==563
for k in ('failures','errors','skipped'):assert int(suites[0].get(k))==0
assert not xml.findall('.//failure') and not xml.findall('.//error') and not xml.findall('.//skipped')
for base in (SOURCE,R/'execution_source'):
 assert {p.relative_to(base).as_posix():sha(p.read_bytes())for p in base.rglob('*')if p.is_file()}==expected
archive=(R/'sources.tar.gz').read_bytes();assert sha(archive)==r['source_archive']['sha256']==owner['source_archive_sha256']=='417f2c15b85f0f3bd17d071a6a6a4bddd734e3fee1e37d812e1bd1d901b2d7c6'
assert len(archive)==r['source_archive']['bytes']
with tarfile.open(R/'sources.tar.gz','r:gz')as tf:
 members=tf.getmembers();assert len(members)==171 and all(m.isfile()for m in members) and len({m.name for m in members})==171
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
assert owner['session']==97252 and owner['exit_code']==0 and owner['reaped']is True
assert owner['actual_model_validation_or_private_body_reads']is False
processes=[]
def absent(group):
 try:os.killpg(group,0)
 except ProcessLookupError:return True
 return False
for name,report in (('collector',r),('worker',w)):
 row=owner[name];assert row['pid']==report['pid'] and row['absent']is True
 assert not Path('/proc',str(row['pid'])).exists() and absent(row['pgid'])
 processes.append(dict(role=name,pid=row['pid'],pgid=row['pgid'],pid_absent_at_review=True,group_absent_at_review=True))
receipts=sorted(p for p in (R/'worker/synthetic').glob('*/child_receipt.json')if not p.parent.is_symlink());assert len(receipts)==82
for p in receipts:
 row=load(p);assert row['reaped']is True and row['group_absent']is True
 assert not Path('/proc',str(row['pid'])).exists() and absent(row['pgid'])
origin=load(SOURCE/'heldout_terminal_origin.json');assert len(origin['parent167_sources'])==167
assert sum(expected[n]==h for n,h in origin['parent167_sources'].items())==164
assert all(expected[n]==h for n,h in origin['inherited139_sources'].items())
assert (R/'report.json').read_bytes()==rraw and (R/'worker/report.json').read_bytes()==wraw
result=dict(schema='qwen-heldout-terminal-entry-preflight-independent-readback-v1',created_utc=datetime.now(timezone.utc).isoformat(),
 scope='Public preflight source/report/XML/process readback only. No test rerun or actual model/calibration/validation/row0/private-journal body read.',proof_root=str(R),source_root=str(SOURCE),
 report_sha256=sha(rraw),worker_report_sha256=sha(wraw),source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),source_count=171,
 current_execution_start_end_archive_equal=True,synthetic_tests=563,installed_import_denials=10,xml_errors=0,xml_failures=0,xml_skipped=0,
 all139_inherited_exact=True,parent167_unchanged_count=164,
 owner_tool_terminal=dict(session=97252,exit_code=0,reaped=True,evidence_sha256=sha(OWNER.read_bytes()),scope='Authoritative owner receipt; not an independent second wait.'),
 current_process_observations=processes,focused_subprocess_receipts_checked=82,focused_subprocesses_absent_at_review=True,
 finding='PASS for isolated preparation proof. No actual scientific result, execution grant, or old-result reclassification.',tests_rerun=False,actual_private_body_read=False)
p=OUT/'terminal-readback.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p.read_bytes()),'sources':171,'tests':563,'installed_denials':10}))
