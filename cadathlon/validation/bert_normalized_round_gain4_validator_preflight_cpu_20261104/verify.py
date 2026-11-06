"""Portable public preparation proof; does not imply secure startup admission."""
from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--current',type=Path);args=p.parse_args()
 pins=json.loads((ROOT/'artifact_hashes.json').read_bytes())
 for n,h in pins.items():assert sha((ROOT/n).read_bytes())==h,n
 r=json.loads((ROOT/'source-bound-v1/report.json').read_bytes())
 assert sha((ROOT/'source-bound-v1/report.json').read_bytes())=='858c241d00a324ca7de08f0c448bd7f6e55911ef27353d58b3fa869923b8b93c'
 assert r['execute']is False and r['stage']=='validate'and not r.get('error')and not r['cleanup_errors']
 for n in ['passed','source_files_unchanged','source_archive_verified','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','terminal_scope_completed','termination_scope_completed','no_encrypted_execution','no_private_or_ciphertext_payload_read']:assert r[n]is True,n
 expected=r['source_hashes_start'];assert len(expected)==181 and expected==r['source_hashes_end']
 archive=ROOT/'source-bound-v1/sources.tar.gz';assert sha(archive.read_bytes())==r['source_archive_sha256']=='bd08f5b998b915acb36d5a51cf46dad668e7f42139374ba93dc8a18b726fe691'
 with tarfile.open(archive)as z:
  members=z.getmembers();assert len(members)==181 and len({m.name for m in members})==181 and sum(m.size for m in members)<=16<<20
  assert all(m.isfile()and 0<=m.size<=8<<20 and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
  raw={m.name:z.extractfile(m).read()for m in members};assert {n:sha(b)for n,b in raw.items()}==expected
 old=json.loads(raw['producer_sources.json']);assert len(old)==163 and sha(raw['producer_sources.json'])=='d8533cc0c770683691fb01fb078977a0e248812696ca7548edc537af2d93cbec'
 changed={n for n in old if old[n]!=expected[n]}
 assert changed=={'runtime/stage/client_keys.py','runtime/stage/key_contexts.py','source_inventory.json','execution_authorization.json','gain4_component_overlays.json'}
 assert expected['component_protocol.py']==old['component_protocol.py']=='17dc37314a939da70c9055e8d3e75404e23628d8c9dd603fb6aceafb6f12819e'
 assert json.loads(raw['execution_authorization.json'])=={'ready':False}
 auth=json.loads(raw['validator_authorization.json']);assert auth['ready']is False and auth['execution_authorized']is False
 bound=json.loads(raw['producer_bindings.json']);assert bound['server_report_sha256']is None and bound['server_root_review_sha256']is None
 for name,count in [('mock_tests',528),('installed_denials',10)]:
  cases=ET.parse(ROOT/('source-bound-v1/'+name+'.xml')).findall('.//testcase')
  assert len(cases)==count and all(c.find(n)is None for c in cases for n in ['failure','error','skipped'])
  assert r[name]==dict(tests=count,failures=0,errors=0,skipped=0,exit_code=0)
 assert len(r['children'])==2
 for c in r['children']:assert c['direct_returncode']==0 and all(c[n]is True for n in ['direct_child_reaped','group_empty_verified','cleanup_completed'])and not c['cleanup_errors']
 if args.current:
  for n,h in expected.items():assert sha((args.current/n).read_bytes())==h,n
 print(json.dumps(dict(verified_preparation=True,sources=181,tests=528,installed_tests=10,actual_execution_authorized=False,known_startup_source_guard_gap=True,current_checked=args.current is not None)))
if __name__=='__main__':main()
