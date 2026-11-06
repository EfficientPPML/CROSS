import argparse,hashlib,json,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root,current=None):
 files=json.loads((root/'artifact_sha256.json').read_bytes());assert all(sha(root/n)==h for n,h in files.items())
 r=json.loads((root/'measured/report.json').read_bytes());assert r['passed']and r['source_files_unchanged']and r['source_archive_verified']and r['cleanup_completed']and r['backends_imported']is False
 expected=r['source_hashes_before'];assert len(expected)==38 and expected==r['source_hashes_after']
 assert sha(root/'measured/sources.tar.gz')==r['source_archive_sha256']
 with tarfile.open(root/'measured/sources.tar.gz','r:gz')as t:
  members=t.getmembers();assert len(members)==38 and all(m.isfile()and Path(m.name).suffix in('.py','.json','.gz','.md')for m in members)
  actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in members};assert actual==expected
  keys=json.loads(t.extractfile('inputs/actual_stage1/public_manifest.json').read());profile=json.loads(t.extractfile('inputs/bsgs/p60_profile_and_refresh.json').read())
 matrices=r['matrices'];assert len(matrices)==74 and len({m['name']for m in matrices})==74
 dense=set()
 for m in matrices:
  assert m['tensor_sha256']==m['checkpoint_tensor_sha256']and m['file_sha256_before']==m['file_sha256_after']
  active=m['active_indices'];assert len(active)==m['active_diagonal_count']and active==sorted(set(active))
  n1=m['n1'];babies=sorted({d%n1 for d in active}-{0});giants=sorted({d//n1*n1 for d in active},reverse=True)
  shifts=[b-a for a,b in zip([0]+babies,babies)]+[a-b for a,b in zip(giants,giants[1:])]+([min(giants)]if giants else [])
  required=sorted({v%65536 for v in shifts if v%65536});assert required==m['required_keys'];dense.update(required)
 assert dense==set(r['dense_keys'])=={1,32,64,61760,64064,64768}
 graph=set(r['fixed_groups']['attention'])|set(r['fixed_groups']['norm'])|dense
 refresh=set(profile['refresh_plan']['rotations']);available=set(keys['rotations']);union=graph|refresh
 assert len(graph)==62 and len(refresh)==106 and len(union)==118 and len(available)==130
 assert graph==set(r['graph_keys'])and refresh==set(r['refresh_keys'])and union==set(r['required_keys'])and not(union-available)
 assert r['missing_keys']==[]and r['actual_key_store_covers']and len(available-union)==12 and 262143 in refresh
 assert r['checkpoint_sha256_before']==r['checkpoint_sha256_after']=='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
 tests=json.loads((root/'tests/report.json').read_bytes());assert tests['passed']and tests['returncode']==0 and tests['reaped']and len(tests['source_hashes_before'])==39 and tests['source_hashes_before']==tests['source_hashes_after']
 assert sha(root/'tests/sources.tar.gz')==tests['source_archive_sha256']
 with tarfile.open(root/'tests/sources.tar.gz','r:gz')as t:assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in t.getmembers()}==tests['source_hashes_before']
 cases=list(ET.parse(root/'tests/tests.xml').getroot().iter('testcase'));assert len(cases)==15 and not any(c.find(n)is not None for c in cases for n in ('error','failure','skipped'))
 term=json.loads((root/'terminal.json').read_bytes());assert term['actual_authoritatively_reaped']and term['actual_observed_exit_code']==0 and term['producer_pid_absent']
 old=json.loads((root/'history/actual-v1/report.json').read_bytes());assert old['passed']is False and old['source_files_unchanged']is False
 if current is not None:assert all(sha(current/n)==h for n,h in expected.items())
 return dict(passed=True,actual_audit_sources=38,test_proof_sources=39,tests=15,actual_public_matrices=74,graph_keys=62,refresh_keys=106,total_required=118,available=130,missing=[],current_checked=current is not None,no_HE_claim=True)
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=HERE);parser.add_argument('--current',type=Path);a=parser.parse_args();print(json.dumps(verify(a.archive,a.current),sort_keys=True))
