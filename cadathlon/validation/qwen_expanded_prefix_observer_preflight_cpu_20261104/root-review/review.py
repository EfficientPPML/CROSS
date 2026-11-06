"""Read-only public proof review, then create a separately scoped observer grant.

Run only after the owner reports authoritative preflight exit0/reaping. This
script never reads calibration tokens, validation, model bodies or outcomes.
"""
from pathlib import Path
import argparse,hashlib,json,os,tarfile,xml.etree.ElementTree as ET

def sha(b):return hashlib.sha256(b).hexdigest()
def read_json(path):return json.loads(path.read_bytes())
def inventory(root):
 result={}
 for p in root.rglob('*'):
  assert not p.is_symlink(),str(p)
  if p.is_file():result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
  else:assert p.is_dir()
 return result

def main():
 parser=argparse.ArgumentParser()
 parser.add_argument('--preflight-report-sha256',required=True)
 parser.add_argument('--source-manifest-sha256',required=True)
 parser.add_argument('--preflight-session',type=int,required=True)
 args=parser.parse_args()
 lab=Path('/tmp/qwen-expanded-prefix-observer-runtime-20261104')
 proof=Path('/tmp/qwen-expanded-prefix-observer-preflight-v2-20261104')
 manifest=Path('/tmp/qwen-expanded-prefix-observer-sources-v1-20261104.json')
 destination=Path('/tmp/qwen-expanded-prefix-observer-root-review-20261104')
 actual=Path('/tmp/qwen-expanded-prefix-observer-actual-v1-20261104')
 private=Path('/tmp/qwen-expanded-prefix-observer-private-v1-20261104')
 raw=manifest.read_bytes();expected=json.loads(raw)
 assert sha(raw)==args.source_manifest_sha256
 assert 40<=len(expected)<=256
 r=read_json(proof/'report.json')
 assert sha((proof/'report.json').read_bytes())==args.preflight_report_sha256
 assert all(r.get(k)is True for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
 assert r['actual_observer_stage'] is False and r['model_forward'] is False and r['coefficient_selection'] is False and r['audit_denials']==[]
 assert r['source_hashes_start']==r['source_hashes_end']==expected and r['source_manifest_sha256']==args.source_manifest_sha256
 assert r['child_returncode']==0 and r['child_reaped'] is True and r['child_group_absent'] is True
 w=read_json(proof/'worker/report.json')
 assert r['worker']==w and r['worker_report_sha256']==sha((proof/'worker/report.json').read_bytes())
 for k in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):assert w.get(k)is True,k
 assert w['source_hashes_start']==w['source_hashes_end']==expected and w['audit_denials']==[]
 assert w['source_manifest_sha256']==args.source_manifest_sha256
 expectations=read_json(lab/'preflight_expectations.json')
 assert w['tests']==expectations['synthetic_tests']>=198
 assert w['pytest_returncode']==w['failures']==w['errors']==w['skipped']==0
 assert len(w['installed_import_denials'])==expectations['installed_import_denials']==8
 assert set(w['installed_import_denials'])=={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors'}
 xml=ET.parse(proof/'worker/tests.xml');assert len(xml.findall('.//testcase'))==w['tests']
 assert not any(xml.findall('.//'+k) for k in ('failure','error','skipped'))
 for tree in (lab,proof/'execution_source'):assert inventory(tree)==expected,str(tree)
 archive=proof/'sources.tar.gz';archive_hash=sha(archive.read_bytes());assert archive_hash==r['source_archive']['sha256']
 with tarfile.open(archive) as tar:
  members=tar.getmembers();assert len(members)==len({m.name for m in members})==len(expected)
  assert all(m.isfile() and 0<m.size<=8<<20 for m in members)
  contents={m.name:tar.extractfile(m).read() for m in members}
 assert {n:sha(v) for n,v in contents.items()}==expected
 assert json.loads(contents['execution_authorization.json'])=={'execute':False}
 assert sha(contents['public/protocol.json'])=='eea8e4b38b4c2719e63a0aa13028150564901305e9e5f1f9d0f2985c821cb70f'
 assert sha(contents['public/candidate3_manifest.json'])=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
 reviewed=read_json(Path('/tmp/qwen-expanded-prefix-observer-transformer-review-20261104/review-v3.json'))
 assert all(expected[n]==h for group in ('source_hashes','lifecycle_source_hashes') for n,h in reviewed[group].items())
 pids=[r['pid'],r['child_pid'],w['pid']]
 assert r['child_pid']==w['pid']==r['child_pgid']
 assert not any(Path('/proc',str(pid)).exists() for pid in pids)
 try:os.killpg(r['child_pgid'],0)
 except ProcessLookupError:pass
 else:raise AssertionError('preflight child group alive')
 ext=json.loads(contents['external_inputs.json'])
 assert ext['calibration']['sha256']=='80eeaefad423167ee0710797d7eb76d8cf3a422e6f3b2e6069932d682835963b'
 assert ext['calibration']['bytes']==1221283
 assert not actual.exists() and not private.exists()
 capacity=os.statvfs('/tmp');available=capacity.f_bavail*capacity.f_frsize
 assert available>8*(1<<30)
 report=dict(passed=True,scope='Public/source and preflight review; no actual calibration/model/validation reads',
  preflight_report_sha256=args.preflight_report_sha256,source_manifest_sha256=args.source_manifest_sha256,
  source_archive_sha256=archive_hash,source_files=len(expected),tests=w['tests'],installed_import_denials=8,
  current_execution_start_end_archive_equal=True,preflight_session=args.preflight_session,
  outer_exit0_reaped='Security owner authoritative tool receipt; root independently verifies PID/group absence',
  independently_absent_pids=pids,reviewed_numerical_sources_match=True,
  calibration_sha256=ext['calibration']['sha256'],calibration_body_opened=False,model_body_opened=False,
  proposed_scope='native/exact3200 and frozen candidate3 free64 only; no new coefficient selection, heldout validation, row0, tokenizer or HE',
  private_journals=True,initial_ranges_native_exact_only=True,baseline_ranges_separate=True,
  no_healthy_deadline=True,cpu_affinity=[112,113,114,115],available_tmp_bytes_point_observation=available,
  limitations=['Python/source integrity audit is not an OS sandbox or binary reproducibility image.',
              'Initial completeness and numerical gates remain unmeasured until the separate actual run.'],
  review_script_sha256=sha(Path(__file__).read_bytes()))
 destination.mkdir(mode=0o700)
 review_raw=(json.dumps(report,indent=2,allow_nan=False)+'\n').encode()
 (destination/'review.json').write_bytes(review_raw)
 grant=dict(schema='qwen-expanded-prefix-observer-grant-v1',execute=True,scope='initial-native-exact3200-baseline-free64-only',
  source_manifest_sha256=args.source_manifest_sha256,protocol_sha256=ext['producer']['protocol_sha256'],
  baseline_manifest_sha256=ext['baseline_manifest_sha256'],producer=ext['producer'],calibration=ext['calibration'],
  model_pins_sha256=sha(contents['public/model_pins.json']),private_output=str(private),evidence_output=str(actual),
  model_forward=True,tokenizer_execution=False,coefficient_selection=False,validation_read=False,
  cpu_affinity=[112,113,114,115],torch_threads=4,torch_interop_threads=1,
  preflight_report_sha256=args.preflight_report_sha256,independent_review_sha256=sha(review_raw))
 grant_raw=(json.dumps(grant,indent=2,allow_nan=False)+'\n').encode()
 (destination/'launch.json').write_bytes(grant_raw)
 (destination/'review.py').write_bytes(Path(__file__).read_bytes())
 print(json.dumps({'review':str(destination/'review.json'),'review_sha256':sha(review_raw),
  'grant':str(destination/'launch.json'),'grant_sha256':sha(grant_raw),'sources':len(expected),'tests':w['tests']}))

if __name__=='__main__':main()
