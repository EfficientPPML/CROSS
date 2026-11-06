"""Independent public preflight readback; never follows external/private descriptors."""
import hashlib,json,os,tarfile
from pathlib import Path
from datetime import datetime,timezone
import xml.etree.ElementTree as ET
R=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-preflight-v1-20261104')
SOURCE=Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
HERE=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
report_raw=(R/'report.json').read_bytes();r=json.loads(report_raw)
worker_raw=(R/'worker/report.json').read_bytes();w=json.loads(worker_raw)
owner=load(R/'terminal_observation.json');raw=(R/'source_manifest.json').read_bytes();expected=json.loads(raw)
assert sha(report_raw)=='c847b64d3cada0add4071fafe5c6ce79f2b70b0db6da7abcde41adf48f853ab9'
assert sha(raw)=='45cd92473d9ba558288651d59d4f089c7fb9ba37650802848cad24687beaa437'
assert len(expected)==141 and owner['source_count']==141
assert sha(worker_raw)==r['worker_report_sha256']==owner['worker_report_sha256']
assert r['worker']==w
for row in (r,w):
 for flag in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
  assert row[flag] is True,flag
 assert row['audit_denials']==[] and 'error_type' not in row
 assert row['source_hashes_start']==row['source_hashes_end']==expected
 assert row['source_manifest_sha256']==sha(raw)
 assert row['bootstrap_origins']=={'__main__':'entry.py'}
assert r['actual_candidate_stage'] is False and r['model_forward'] is False and r['coefficient_selection'] is False
assert r['child_reaped'] is True and r['child_group_absent'] is True and type(r['child_returncode'])is int and r['child_returncode']==0
assert w['pid']==r['child_pid']==r['child_pgid']
assert type(w['tests'])is int and w['tests']==781
for k in ('errors','failures','skipped','pytest_returncode'):assert type(w[k])is int and w[k]==0
assert w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla']
tree=ET.fromstring((R/'worker/tests.xml').read_bytes());suites=tree.findall('testsuite');assert len(suites)==1
assert int(suites[0].get('tests'))==781 and len(tree.findall('.//testcase'))==781
for k in ('failures','errors','skipped'):assert int(suites[0].get(k))==0
assert not tree.findall('.//failure') and not tree.findall('.//error') and not tree.findall('.//skipped')
for base in (SOURCE,R/'execution_source'):
 assert {p.relative_to(base).as_posix():sha(p.read_bytes())for p in base.rglob('*')if p.is_file()}==expected
archive_raw=(R/'sources.tar.gz').read_bytes();assert len(archive_raw)==r['source_archive']['bytes']
assert sha(archive_raw)==r['source_archive']['sha256']==owner['source_archive_sha256']=='04edc1012ef8d503b878979c21cdeb997c2dd6e098a9407e8d59d9328ddb6c12'
with tarfile.open(R/'sources.tar.gz','r:gz')as tf:
 members=tf.getmembers();assert len(members)==141 and all(m.isfile()for m in members)
 assert len({m.name for m in members})==141
 assert {m.name:sha(tf.extractfile(m).read())for m in members}==expected
assert owner['session']==29701 and owner['exit_code']==0 and owner['reaped'] is True
assert owner['report_sha256']==sha(report_raw) and owner['passed'] is True
for k in ('actual_candidate_outcome_read','actual_model_execution','actual_private_body_read','new_execution_grant'):assert owner[k] is False
processes=[]
def absent(pid):
 try:os.killpg(pid,0)
 except ProcessLookupError:return True
 return False
for row in owner['observed_owned_processes']:
 assert not Path('/proc',str(row['pid'])).exists() and absent(row['pgid'])
 processes.append(dict(pid=row['pid'],pgid=row['pgid'],pid_absent_at_review=True,group_absent_at_review=True))
assert {row['pid']for row in processes}=={r['pid'],w['pid']}
receipts=sorted((R/'worker/synthetic').glob('*/child_receipt.json'))
assert len(receipts)==42
for p in receipts:
 row=load(p);assert row['reaped'] is True and row['group_absent'] is True
 assert not Path('/proc',str(row['pid'])).exists() and absent(row['pgid'])
old=load(Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json'))
assert sorted(n for n in old if expected[n]!=old[n])==['entry.py','preflight_expectations.json']
assert len(set(expected)-set(old))==2
assert (R/'report.json').read_bytes()==report_raw and (R/'worker/report.json').read_bytes()==worker_raw
result=dict(schema='qwen-candidate-terminal-preflight-independent-readback-v1',created_utc=datetime.now(timezone.utc).isoformat(),
 scope='Public source/report/XML/process metadata readback only. No test rerun, actual candidate result, private journal, calibration/token/model/validation body read.',
 proof_root=str(R),source_root=str(SOURCE),source_count=141,source_manifest_sha256=sha(raw),report_sha256=sha(report_raw),worker_report_sha256=sha(worker_raw),source_archive_sha256=sha(archive_raw),
 current_execution_start_end_archive_equal=True,synthetic_tests=781,installed_import_denials=10,xml_failures=0,xml_errors=0,xml_skipped=0,
 parent137_exact=True,entry_and_count_metadata_only_parent_overlays=True,all_six_numerical_sources_unchanged=True,
 owner_tool_terminal=dict(session=29701,exit_code=0,reaped=True,evidence_sha256=sha((R/'terminal_observation.json').read_bytes()),scope='Authoritative owner receipt, not an independent second wait.'),
 current_process_observations=processes,focused_subprocess_receipts_checked=42,focused_subprocesses_absent_at_review=True,
 finding='PASS for isolated preparation proof; no actual execution grant or old-result reclassification.',
 actual_candidate_outcome_read=False,actual_private_body_read=False,tests_rerun=False)
p=HERE/'terminal-readback.json';p.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'receipt':str(p),'sha256':sha(p.read_bytes()),'source_count':141,'tests':781,'installed_denials':10}))
