"""Verify public import/preparation evidence without model or private bodies."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
MANIFEST='8131c5ac045f1cf274a493bbc14278e330c0a8e057079edc4f10f9a23dd9bf91'
PINS={
 'preflight-v5':('cc65c93236162f553acc136f7a97e29f80bdb222687e9c61ab3a4a4b29b2bc77','de0b7086cc103b13bd69faf98991e0936957ae6ea77eeadab832704ad552cee4',44184),
 'import-smoke-v5':('d39f76292878ca56197ac12a015dd388f0de37c86094260f3210503ad7f8e2e5','a5e1389590028c98b5b429bea302dc2f079f7a8b00c5ac1304b3a8938ed08ba9',55494),
}
def sha(b):return hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_bytes())
def archive(base):
 m=load(base/'source_manifest.json')
 with tarfile.open(fileobj=io.BytesIO((base/'sources.tar.gz').read_bytes()),mode='r:gz') as t:
  members=t.getmembers();assert len(members)==len(m) and all(x.isfile() and x.size<=8*1024*1024 for x in members)
  assert sum(x.size for x in members)<=32*1024*1024
  blobs={x.name:t.extractfile(x).read() for x in members}
 assert {n:sha(b) for n,b in blobs.items()}==m
 return m,blobs

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--current',type=Path);a=parser.parse_args()
 actual={str(p.relative_to(ROOT)):sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'}
 assert actual==load(ROOT/'artifact_hashes.json')
 common=None
 for name,(report_pin,tar_pin,session) in PINS.items():
  base=ROOT/name;r=load(base/'report.json');w=load(base/'worker/report.json');m,blobs=archive(base)
  assert len(m)==61 and sha((base/'source_manifest.json').read_bytes())==MANIFEST
  assert sha((base/'report.json').read_bytes())==report_pin and sha((base/'sources.tar.gz').read_bytes())==tar_pin
  if common is not None:assert m==common
  common=m
  for q in (r,w):
   assert all(q[f]is True for f in ('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
   assert q['source_hashes_start']==q['source_hashes_end']==m and q['audit_denials']==[]
  assert r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
  assert r['worker']==w and r['worker_report_sha256']==sha((base/'worker/report.json').read_bytes())
  term=load(base/'owner_terminal.json');assert term['session_id']==session and term['authoritative_exit_code']==0
  if name=='preflight-v5':
   assert w['tests']==274 and w['failures']==w['errors']==w['skipped']==w['pytest_returncode']==0 and len(w['installed_import_denials'])==8
   tree=ET.parse(base/'worker/tests.xml');assert len(tree.findall('.//testcase'))==274
   assert not any(tree.findall('.//'+k) for k in ('failure','error','skipped'))
  else:
   assert w['import_smoke_completed']is True and w['model_class_imported']is True
   assert w['model_constructed']is False and w['model_forward']is False and w['tokenizer_execution']is False
   assert w['constructor_attempts']==w['forbidden_backends_loaded']==w['denied_access_diagnostics']==[]
   scope=w['import_probe_scope'];assert scope['restored']is True and scope['find_spec_restored']is True
   assert scope['source_hashes_before']==scope['source_hashes_after'] and len(scope['source_hashes_before'])==3
   assert w['backend_sources_before']==w['backend_sources_after'] and len(w['backend_sources_before'])==14
 for name,count,session in [('failed-smoke57',57,31741),('failed-smoke59',59,14747),('failed-smoke61',61,40549)]:
  base=ROOT/name;r=load(base/'report.json');m,_=archive(base);term=load(base/'owner_terminal.json')
  assert len(m)==count and r['passed']is False and r['child_returncode']==1 and r['child_reaped']is True and r['child_group_absent']is True
  assert term['session_id']==session and term['authoritative_exit_code']==1
  assert term['report_sha256']==sha((base/'report.json').read_bytes()) and term['archive_sha256']==sha((base/'sources.tar.gz').read_bytes())
 overlay=load(ROOT/'final_overlay.json');composed=dict(overlay['unchanged_parent']);composed.update({n:v['after'] for n,v in overlay['changed_parent'].items()});composed.update(overlay['added']);assert composed==common
 for n,h in load(ROOT/'peer-review/review-runtime-v5-smoke.json')['source_hashes'].items():assert common[n]==h
 review=load(ROOT/'root-review/review.json');grant=load(ROOT/'root-review/launch.json')
 assert sha((ROOT/'root-review/review.json').read_bytes())=='58c787c89acd29fda3a112c50f3f0f3ff9adea347b0c8a90a342b1d4e90e08a6'
 assert sha((ROOT/'root-review/launch.json').read_bytes())=='e0068125a3a271880a69231d5108fe9591ae10489e05f06d043646d8be192704'
 assert review['passed'] is True and grant['source_manifest_sha256']==MANIFEST
 assert grant['schema']=='qwen-expanded-prefix-observer-grant-v2' and grant['execute']is True
 assert grant['backend_environment']=={'USE_TF':'0','USE_FLAX':'0','USE_TORCH':'1','USE_TORCH_XLA':'0','FORCE_TF_AVAILABLE':'0'}
 if a.current:assert {str(p.relative_to(a.current)):sha(p.read_bytes()) for p in a.current.rglob('*') if p.is_file()}==common
 print('PASS: exact61, 274 synthetic tests/8 denials, actual Qwen class import only; prior three smoke failures preserved.')
if __name__=='__main__':main()
