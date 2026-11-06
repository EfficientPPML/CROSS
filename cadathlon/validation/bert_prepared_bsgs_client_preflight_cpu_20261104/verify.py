"""Portable public source/proof verifier; never imports archived runtime."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
REPORT='c695723f88ced9e66a1dc134bc0277e7e50e4f07de463659bfb6f278814637b2'
ARCHIVE='c6ad40b952aa1a4ed9f3fbb789b68e0e663802d6dc70a9e0b157fce8e0f44e72'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read_json(p):return json.loads(p.read_bytes())
def source_tar(raw,expected=None,depth=0):
 assert depth<=6
 with tarfile.open(fileobj=io.BytesIO(raw))as t:
  rows=t.getmembers();assert len(rows)==len({x.name for x in rows})and sum(x.size for x in rows)<=32<<20
  actual={}
  for x in rows:
   p=Path(x.name);assert x.isfile()and not p.is_absolute()and '..'not in p.parts and 0<=x.size<=16<<20
   b=t.extractfile(x).read();assert len(b)==x.size;actual[x.name]=sha(b)
   if x.name.endswith('.tar.gz'):source_tar(b,depth=depth+1)
   else:assert p.suffix in ('.py','.json','.md')or x.name.endswith('.py.txt')
 if expected is not None:assert actual==expected
 return actual

def verify(current=None):
 manifest=read_json(HERE/'files.sha256.json');files={p.relative_to(HERE).as_posix():p for p in HERE.rglob('*')if p.is_file()and p.name!='files.sha256.json'and '__pycache__'not in p.parts}
 assert set(files)==set(manifest)
 for n,p in files.items():assert not p.is_symlink()and sha(p.read_bytes())==manifest[n],n
 p=HERE/'measured';r=read_json(p/'report.json');i=read_json(p/'installed.json');terminal=read_json(p/'terminal_observation.json')
 assert sha((p/'report.json').read_bytes())==REPORT and sha((p/'sources.tar.gz').read_bytes())==ARCHIVE
 expected=r['source_hashes_start'];assert len(expected)==115 and expected==r['source_hashes_end']==read_json(p/'source_manifest.json')
 source_tar((p/'sources.tar.gz').read_bytes(),expected)
 assert all(r[k]is True for k in ('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','source_archive_verified','all_started_children_reaped'))
 assert r['actual_HE']is False and r['actual_client_decode']is False
 cases=list(ET.parse(p/'tests.xml').iter('testcase'));assert len(cases)==r['test_count']==198
 assert not any(x.findall(k)for x in cases for k in ('failure','error','skipped'))
 assert all(i[k]is True for k in ('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','all_forbidden_entries_restored','allowed_client_entries_unchanged','optional_completion_api_verified'))
 assert all(i[k]is False for k in ('actual_client_decode','actual_contexts','actual_keys','payload_reads','private_reads','native_model'))
 assert i['source_hashes_start']==i['source_hashes_end']==expected and i['denied_count']==len(i['denied_entries'])==21 and len(set(i['denied_entries']))==21
 assert i['source_guard_start']['fresh']is True and i['source_guard_start']['parent_source_count']==89
 assert len(i['source_guard_end']['loaded_local_modules'])==40
 for row in i['source_guard_end']['loaded_local_modules'].values():assert expected[row['path']]==row['sha256']
 with tarfile.open(p/'sources.tar.gz')as t:
  parent=json.loads(t.extractfile('client_89_sources.json').read());overlays=i['source_guard_start']['overlays']
  assert len(parent)==89 and len(overlays)==2
  assert {n:expected[n]for n in parent}=={n:overlays.get(n,h)for n,h in parent.items()}
  for name in ('file_stages.py','request_lifecycle.py'):assert expected['baseline/'+name]==parent['runtime/stage/'+name]
 assert terminal['session_id']==28968 and terminal['observed_exit_code']==0 and terminal['session_reaped']is True and terminal['collector_absent']is True
 assert terminal['report_sha256']==REPORT and terminal['source_archive_sha256']==ARCHIVE
 for row in terminal['children']:
  assert row['absent']is True;process=row['process'];assert process['direct_returncode']==0 and not process['cleanup_errors']
  assert all(process[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
 failed=HERE/'history/umask-fixture-failure-v1';old=read_json(failed/'report.json');assert not old['passed']and old['source_hashes_start']==old['source_hashes_end']
 source_tar((failed/'sources.tar.gz').read_bytes(),old['source_hashes_start'])
 oldcases=list(ET.parse(failed/'tests.xml').iter('testcase'));assert len(oldcases)==198 and sum(bool(x.findall('failure'))for x in oldcases)==1
 assert read_json(failed/'terminal.json')['exit_code']==1
 negative=HERE/'history/output-ownership-negative-v1';oldh=read_json(negative/'before.json');assert oldh==read_json(negative/'after.json');source_tar((negative/'sources.tar.gz').read_bytes(),oldh)
 oldcases=list(ET.parse(negative/'tests.xml').iter('testcase'));assert len(oldcases)==4 and all(x.findall('failure')for x in oldcases)
 independent=HERE/'history/independent-output-retest';source_tar((independent/'sources.tar.gz').read_bytes(),read_json(independent/'before.json')['sources'])
 assert len(list(ET.parse(independent/'tests.xml').iter('testcase')))==4 and not any(x.findall(k)for x in ET.parse(independent/'tests.xml').iter('testcase')for k in ('failure','error','skipped'))
 if current is not None:assert {n:sha((Path(current)/n).read_bytes())for n in expected}==expected
 return dict(passed=True,files=len(files),sources=115,parent=89,overlays=2,tests=198,installed_denials=21,current_checked=current is not None,actual_HE=False)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args();print(json.dumps(verify(a.current)))
