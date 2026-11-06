"""Portable public preparation proof; no actual validation authorized."""
from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--current',type=Path);args=p.parse_args()
 pins=json.loads((ROOT/'artifact_hashes.json').read_bytes())
 for n,h in pins.items():assert sha((ROOT/n).read_bytes())==h,n
 r=json.loads((ROOT/'source-bound-v1/report.json').read_bytes())
 assert sha((ROOT/'source-bound-v1/report.json').read_bytes())=='ec43f2d92a51d6b66de2629b877c9fe46c1000fc60ea60735a6ec636d86c57c0'
 assert r['execute']is False and r['stage']=='validate'and not r.get('error')and not r['cleanup_errors']
 for n in ['passed','source_files_unchanged','source_archive_verified','all_direct_children_reaped','all_owned_groups_empty','process_cleanup_completed','terminal_scope_completed','termination_scope_completed','no_encrypted_execution','no_private_or_ciphertext_payload_read']:assert r[n]is True,n
 expected=r['source_hashes_start'];assert len(expected)==185 and expected==r['source_hashes_end']
 archive=ROOT/'source-bound-v1/sources.tar.gz';assert sha(archive.read_bytes())==r['source_archive_sha256']=='2c6c2d8a2cf6de65ee1c19103e54706dac2d851c897b80aba22deead7614c185'
 with tarfile.open(archive)as z:
  members=z.getmembers();assert len(members)==185 and len({m.name for m in members})==185 and sum(m.size for m in members)<=16<<20
  assert all(m.isfile()and 0<=m.size<=8<<20 and not Path(m.name).is_absolute()and'..'not in Path(m.name).parts for m in members)
  raw={m.name:z.extractfile(m).read()for m in members};assert {n:sha(b)for n,b in raw.items()}==expected
 old=json.loads(raw['producer_sources.json']);assert len(old)==163 and sha(raw['producer_sources.json'])=='d8533cc0c770683691fb01fb078977a0e248812696ca7548edc537af2d93cbec'
 changed={n for n in old if old[n]!=expected[n]}
 assert changed=={'runtime/stage/client_keys.py','runtime/stage/key_contexts.py','source_inventory.json','execution_authorization.json','gain4_component_overlays.json'}
 assert expected['component_protocol.py']==old['component_protocol.py']=='17dc37314a939da70c9055e8d3e75404e23628d8c9dd603fb6aceafb6f12819e'
 assert json.loads(raw['execution_authorization.json'])=={'ready':False}
 auth=json.loads(raw['validator_authorization.json']);assert auth['ready']is False and auth['execution_authorized']is False
 bound=json.loads(raw['producer_bindings.json']);assert bound['server_report_sha256']is None and bound['server_root_review_sha256']is None
 for name,count in [('mock_tests',543),('installed_denials',10)]:
  cases=ET.parse(ROOT/('source-bound-v1/'+name+'.xml')).findall('.//testcase')
  assert len(cases)==count and all(c.find(n)is None for c in cases for n in ['failure','error','skipped'])
  assert r[name]==dict(tests=count,failures=0,errors=0,skipped=0,exit_code=0)
 assert len(r['children'])==2
 for c in r['children']:assert c['direct_returncode']==0 and all(c[n]is True for n in ['direct_child_reaped','group_empty_verified','cleanup_completed'])and not c['cleanup_errors']
 assert sha((ROOT/'external_source_manifest.json').read_bytes())=='aa5e823a116a9d64353b9112e9e09ba03641cd2acd420f4ed8d7b8e3fac2fa8c'
 assert json.loads((ROOT/'external_source_manifest.json').read_bytes())==expected
 fresh=r['fresh_source_guard'];assert fresh['passed']is True and fresh['fresh']is True and fresh['source_count']==185 and fresh['manifest_sha256']==sha((ROOT/'external_source_manifest.json').read_bytes())
 if args.current:
  for n,h in expected.items():assert sha((args.current/n).read_bytes())==h,n
 print(json.dumps(dict(verified_preparation=True,sources=185,tests=543,installed_tests=10,actual_execution_authorized=False,known_historical_startup_gap_repaired=True,current_checked=args.current is not None)))
if __name__=='__main__':main()
