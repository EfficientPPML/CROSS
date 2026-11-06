from pathlib import Path
import hashlib,json,os,tarfile
P=Path('/tmp/qwen-expanded-prefix-observer-actual-v1-20261104')
LAB=Path('/tmp/qwen-expanded-prefix-observer-runtime-20261104')
OUT=P/'root-review'
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((P/'report.json').read_bytes());w=json.loads((P/'worker/report.json').read_bytes())
expected=r['source_hashes_start'];assert len(expected)==52
assert r['passed'] is False and r['initial_selection_admissible'] is False
assert r['child_returncode']==1 and r['child_reaped'] is True and r['child_group_absent'] is True
assert w['passed'] is False and w['complete'] is False and w['error_type']=='ImportError'
assert w['initial_selection_admissible'] is False and w['calibration_reauthenticated'] is True and w['private_directory_handle_closed'] is True
assert r['worker']==w and r['worker_report_sha256']==sha((P/'worker/report.json').read_bytes())
for row in (r,w):
 assert all(row[k] is True for k in ('cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
 assert row['source_hashes_start']==row['source_hashes_end']==expected and row['audit_denials']==[]
for tree in (LAB,P/'execution_source'):
 files={q.relative_to(tree).as_posix():q for q in tree.rglob('*') if q.is_file()}
 assert set(files)==set(expected) and not any(q.is_symlink() for q in tree.rglob('*'))
 assert {n:sha(q.read_bytes()) for n,q in files.items()}==expected
with tarfile.open(P/'sources.tar.gz') as t:
 members=t.getmembers();assert len(members)==len({x.name for x in members})==52 and all(x.isfile() for x in members)
 assert {x.name:sha(t.extractfile(x).read()) for x in members}==expected
archive_hash=sha((P/'sources.tar.gz').read_bytes());assert archive_hash==r['source_archive']['sha256']
pids=[r['pid'],r['child_pid']];assert r['child_pid']==w['pid']
assert not any(Path('/proc',str(pid)).exists() for pid in pids)
try:os.killpg(r['child_pgid'],0)
except ProcessLookupError:pass
else:raise AssertionError('child group exists')
private=Path('/tmp/qwen-expanded-prefix-observer-private-v1-20261104')
assert private.is_dir() and list(private.iterdir())==[]
trace=(P/'stderr.log').read_text()
assert 'from transformers import Qwen3ForCausalLM' in trace and 'find_spec("tensorflow")' in trace
assert trace.rstrip().endswith('ImportError: backend import forbidden in this observer phase')
assert not (P/'stdout.log').read_bytes()
source=(LAB/'model_runtime.py').read_text()
assert source.index('from transformers import Qwen3ForCausalLM')<source.index('model, loading = Qwen3ForCausalLM.from_pretrained')
assert w['model_forward'] is True
OUT.mkdir(mode=0o700)
result=dict(review_passed=True,actual_observer_passed=False,session=68077,authoritative_outer_exit_code=1,authoritative_outer_reaped=True,
 report_sha256=sha((P/'report.json').read_bytes()),worker_report_sha256=sha((P/'worker/report.json').read_bytes()),source_archive_sha256=archive_hash,
 source_count=52,current_execution_start_end_archive_equal=True,independently_absent_pids=pids,child_exit_code=1,child_group_absent=True,
 model_constructed=False,pretrained_forward_completed=False,private_journal_entries=0,calibration_reauthenticated=True,
 worker_model_forward_field=True,worker_model_forward_field_is_premature_stage_intent_not_execution_evidence=True,
 failure='Transformers optional TensorFlow availability probe rejected by strict forbidden import finder before model construction',
 historical_source_and_failure_unchanged=True,no_selection_admission=True,no_coefficients_or_heldout_validation=True,
 review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_text(json.dumps(result,indent=2)+'\n')
(OUT/'review.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps(result))
