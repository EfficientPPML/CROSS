"""Portable public-only preflight verifier; no model or private input reads."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parent
REPORT='b8d609ab7b18836fab60ef19c4f518a03a00ab2facd7f1f0ca33585c4c9eb4a0'
ARCHIVE='85f266a199bfad48e230211e62f7294ddcf73d0a3a7ab9cecfdae634568365e0'
MANIFEST='6e43df5161eb1f50a4b4728606dcf7437eca9fb62f348fee1e0dba7a7fa0db7f'

def sha(data):return hashlib.sha256(data).hexdigest()
def load(path):return json.loads(path.read_bytes())
def main():
 p=argparse.ArgumentParser();p.add_argument('--current',type=Path);args=p.parse_args()
 inventory=load(ROOT/'artifact_hashes.json')
 actual={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
 assert actual==inventory,'artifact inventory differs'
 base=ROOT/'preflight-v2';report=load(base/'report.json');worker=load(base/'worker/report.json')
 assert sha((base/'report.json').read_bytes())==REPORT
 assert sha((base/'sources.tar.gz').read_bytes())==ARCHIVE
 assert sha((base/'source_manifest.json').read_bytes())==MANIFEST
 expected=load(base/'source_manifest.json');assert len(expected)==52
 with tarfile.open(fileobj=io.BytesIO((base/'sources.tar.gz').read_bytes()),mode='r:gz') as t:
  members=t.getmembers();assert len(members)==len(expected) and all(m.isfile() and m.size<=8*1024*1024 for m in members)
  assert sum(m.size for m in members)<=32*1024*1024
  bodies={m.name:t.extractfile(m).read() for m in members}
  assert {n:sha(b) for n,b in bodies.items()}==expected
 for r in (report,worker):
  for name in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):
   assert r[name] is True,name
  assert r['source_hashes_start']==r['source_hashes_end']==expected and r['audit_denials']==[]
 assert report['child_returncode']==0 and report['child_reaped'] is True and report['child_group_absent'] is True
 assert report['worker_report_sha256']==sha((base/'worker/report.json').read_bytes())
 assert report['worker']==worker
 assert worker['tests']==226 and worker['failures']==worker['errors']==worker['skipped']==0 and worker['pytest_returncode']==0
 assert worker['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors']
 tree=ET.parse(base/'worker/tests.xml');assert len(tree.findall('.//testcase'))==226
 assert not any(tree.findall('.//'+tag) for tag in ('failure','error','skipped'))
 terminal=load(ROOT/'terminal.json');assert terminal['session_id']==17127 and terminal['authoritative_exit_code']==0
 assert terminal['report_sha256']==REPORT and terminal['archive_sha256']==ARCHIVE
 review=load(ROOT/'peer-review/review-v3.json')
 for name,digest in review['source_hashes'].items():assert expected[name]==digest,(name,digest)
 assert json.loads(bodies['execution_authorization.json'])=={'execute':False}
 root_review=load(ROOT/'root-review/review.json');grant=load(ROOT/'root-review/launch.json')
 assert sha((ROOT/'root-review/review.json').read_bytes())=='558b61467cc6da447dec22b7a664eb3cd0de2124e4216a187b008fc2cc8a59ed'
 assert sha((ROOT/'root-review/launch.json').read_bytes())=='b84d95d509bbd624fe6fe33921094794adc80fa8245d8bd1e488e6b267805726'
 assert root_review['passed'] is True and root_review['preflight_report_sha256']==REPORT
 assert grant['independent_review_sha256']==sha((ROOT/'root-review/review.json').read_bytes())
 assert grant['source_manifest_sha256']==MANIFEST and grant['preflight_report_sha256']==REPORT
 assert grant['execute'] is True and grant['coefficient_selection'] is False and grant['validation_read'] is False
 adapter=load(ROOT/'native-adapter-development/report.json')
 assert adapter['passed'] is True and adapter['source_count']==9
 assert sha((ROOT/'native-adapter-development/sources.tar.gz').read_bytes())==adapter['source_archive_sha256']
 assert len(ET.parse(ROOT/'native-adapter-development/tests.xml').findall('.//testcase'))==62
 if args.current:
  current={str(p.relative_to(args.current)):sha(p.read_bytes()) for p in args.current.rglob('*') if p.is_file()}
  assert current==expected,'current source differs'
 print('PASS: 52 exact sources, 226 synthetic tests, 8 backend denials; no actual observer execution.')
if __name__=='__main__':main()
