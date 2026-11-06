"""Verify public evidence/source bytes only; never follow private descriptors."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
CURRENT=Path('/tmp/qwen-heldout1088-runtime-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
def load(name):return json.loads((HERE/name).read_bytes())
def main(current=False):
 index=load('SHA256SUMS.json');actual={p.relative_to(HERE).as_posix()for p in HERE.rglob('*')if p.is_file()}
 assert not any(p.is_symlink()for p in HERE.rglob('*'))
 assert actual==set(index)|{'SHA256SUMS.json'}
 for name,digest in index.items():assert sha((HERE/name).read_bytes())==digest,name
 raw=(HERE/'proof/source_manifest.json').read_bytes();expected=json.loads(raw)
 assert len(expected)==167 and sha(raw)=='cce2630028708f41e79614bb367039fd146fc924471bfd674ade1737f9c6017e'
 rraw=(HERE/'proof/report.json').read_bytes();wraw=(HERE/'proof/worker/report.json').read_bytes();r=json.loads(rraw);w=json.loads(wraw)
 assert sha(rraw)=='1575f36e4e647d35f5685a626c8b9fb6ba4eb530ed08ab4aa8d3615cc42ffa81'
 assert sha(wraw)=='28fcb81720105b7018976a956748e3552e9d2cc83f6b4b7b7a9891d99e82e6d4'
 for row in(r,w):
  for key in('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'):assert row[key]is True,key
  assert row['audit_denials']==[]and 'error_type'not in row
  assert row['source_hashes_start']==row['source_hashes_end']==expected
  assert row['source_manifest_sha256']==sha(raw)
 assert r['worker']==w and r['worker_report_sha256']==sha(wraw)
 assert type(r['child_returncode'])is int and r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True
 assert r['actual_heldout_stage']is False and r['model_forward']is False
 assert w['tests']==478 and w['failures']==w['errors']==w['skipped']==w['pytest_returncode']==0
 assert w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla']
 x=ET.parse(HERE/'proof/worker/tests.xml');assert len(x.findall('.//testcase'))==478
 assert not any(x.findall('.//'+k)for k in('failure','error','skipped'))
 archive=(HERE/'proof/sources.tar.gz').read_bytes();assert sha(archive)=='6cab10a38f4074bea8089d50d3544c6e462c79bb309cb44fc5c408ce3dd4d991'
 assert sha(archive)==r['source_archive']['sha256']and len(archive)==r['source_archive']['bytes']
 sources={};total=0
 with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as t:
  members=t.getmembers();assert len(members)==167
  for member in members:
   assert member.isfile()and member.name in expected and member.name not in sources
   cap=(64 if member.name=='candidate_proof/selector-assessments.json'else 8)<<20
   assert 0<member.size<=cap
   total+=member.size;assert total<=96<<20
   data=t.extractfile(member).read();assert sha(data)==expected[member.name]
   sources[member.name]=data
 assert set(sources)==set(expected)
 parent=json.loads(sources['parent139_sources.json']);assert len(parent)==139 and all(expected[n]==h for n,h in parent.items())
 config=json.loads(sources['heldout_preflight_expectations.json']);assert config['synthetic_tests']==478 and config['installed_import_denials']==10
 assert json.loads(sources['execution_authorization.json'])=={'execute':False}
 terminal=load('proof/terminal_observation.json');assert terminal['passed']is True and terminal['tool_session']==62369 and terminal['tool_exit_code']==0 and terminal['tool_reaped']is True
 assert terminal['observed_processes']=={'collector':{'pid':2938023,'ppid':16918,'pgid':2938023},'worker':{'pid':2938141,'ppid':2938023,'pgid':2938141}}
 assert set(terminal['processes'].values())=={'absent'}and terminal['process_observation_tool_chunk']=='6f9d12'
 if current:
  assert not any(p.is_symlink()for p in CURRENT.rglob('*'))
  assert {p.relative_to(CURRENT).as_posix():sha(p.read_bytes())for p in CURRENT.rglob('*')if p.is_file()}==expected
 print(json.dumps({'passed':True,'indexed_public_files':len(index),'source_count':167,'parent139_unchanged':True,'synthetic_tests':478,'installed_import_denials':10,'current_checked':current,'private_descriptors_followed':False},sort_keys=True))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',action='store_true');main(p.parse_args().current)
