"""Offline/current integrity check for the isolated public mocked candidate."""
import argparse,hashlib,json,tarfile
from pathlib import Path
import xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(root,current=None):
 artifacts=json.loads((root/'artifact_sha256.json').read_bytes())
 assert all(sha(root/n)==h for n,h in artifacts.items()),'artifact mismatch'
 report=json.loads((root/'measured/report.json').read_bytes())
 expected=report['source_hashes_before']
 assert report['passed']is True and len(expected)==251 and expected==report['source_hashes_after']
 assert report['source_files_unchanged']and report['source_archive_verified']and report['source_archive_unchanged']and report['all_started_children_reaped']
 tar=root/'measured/sources.tar.gz';assert sha(tar)==report['source_archive_sha256']
 with tarfile.open(tar,'r:gz')as archive:
  members=archive.getmembers();assert len(members)==251 and all(m.isfile()and Path(m.name).suffix in('.py','.json','.md')for m in members)
  actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest()for m in members}
 assert actual==expected
 for name,count in [('engine',37),('prepared',49),('native-wrapper',9)]:
  cases=list(ET.parse(root/('measured/'+name+'.xml')).getroot().iter('testcase'))
  assert len(cases)==count and not any(c.find(x)is not None for c in cases for x in ('failure','error','skipped'))
  group=next(g for g in report['groups']if g['name']==name);assert group['returncode']==0 and group['reaped']and group['passed']
 samples=list((root/'measured/signal-cases').glob('*.json'));assert len(samples)==9
 for path in samples:
  item=json.loads(path.read_bytes());r=item['report'];assert item['reaped']and item['returncode']in(129,130,143)and r['validation_passed']and r['operation_passed']is False
  assert r['native_thread_count']>1 and r['original_prepared_owners_closed']and r['provider_closed']and r['refresh_clear_count']==5 and r['handler_state_restored']and r['main_mask_restored']
 terminal=json.loads((root/'terminal.json').read_bytes());assert terminal['observed_exit_code']==0 and terminal['authoritatively_reaped']and terminal['all_producer_pids_absent']
 assert terminal['source_report_sha256']==sha(root/'measured/report.json')
 if current is not None:assert all(sha(current/n)==h for n,h in expected.items()),'current source mismatch'
 return dict(passed=True,source_files=251,focused_checks=95,separate_groups=[37,49,9],current_checked=current is not None,scope='public/mock integrity only; historical combined67 failure retained')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=HERE);parser.add_argument('--current',type=Path);args=parser.parse_args()
 print(json.dumps(verify(args.archive,args.current),sort_keys=True))
