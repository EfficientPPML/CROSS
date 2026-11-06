from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
def load(p):return json.loads(p.read_bytes())
def unpack(p):
 raw=p.read_bytes()
 with tarfile.open(fileobj=io.BytesIO(raw))as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})and sum(m.size for m in ms)<=16<<20
  data={}
  for m in ms:
   n=Path(m.name);assert m.isfile()and not n.is_absolute()and '..'not in n.parts and m.size<=4<<20
   assert n.suffix in('.py','.json','.md','.txt','.xml')or m.name in('.pytest_cache/.gitignore','.pytest_cache/CACHEDIR.TAG','.pytest_cache/v/cache/nodeids')
   b=t.extractfile(m).read();assert len(b)==m.size;data[m.name]=b
 return data
p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args()
manifest=load(ROOT/'files.sha256.json');files={f.relative_to(ROOT).as_posix():f for f in ROOT.rglob('*')if f.is_file()and f.name!='files.sha256.json'and '__pycache__'not in f.parts};assert set(manifest)==set(files);assert all(h(f.read_bytes())==manifest[n]for n,f in files.items())
for f in files.values():
 if f.name.endswith('.tar.gz'):unpack(f)
def xml(path,tests,failures=0):
 tree=ET.parse(path);cases=list(tree.iter('testcase'));assert len(cases)==tests;assert len(list(tree.iter('failure')))==failures;assert not list(tree.iter('error'))and not list(tree.iter('skipped'))
def proof(path,count,tests,session,report_sha,tar_sha,map_sha):
 r=load(path/'report.json');w=load(path/'worker/report.json');term=load(path/'terminal_observation.json');expected=load(path/'source_manifest.json');data=unpack(path/'sources.tar.gz')
 assert h((path/'report.json').read_bytes())==report_sha and h((path/'sources.tar.gz').read_bytes())==tar_sha and h((path/'source_manifest.json').read_bytes())==map_sha
 assert len(expected)==len(data)==count and {n:h(b)for n,b in data.items()}==expected==r['source_hashes_start']==r['source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']
 assert r['worker']==w and r['worker_report_sha256']==h((path/'worker/report.json').read_bytes())and r['source_archive']['sha256']==tar_sha
 for row in(r,w):
  assert all(row[n]is True for n in('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
  assert row['source_manifest_sha256']==map_sha and not row['audit_denials']
 assert r['actual_tokenizer_stage']is False and r['model_forward']is False and r['coefficient_selection']is False
 assert r['child_returncode']==0 and r['child_reaped']is True and r['child_group_absent']is True and r['child_pid']==w['pid']
 assert w['tests']==tests and w['pytest_returncode']==w['failures']==w['errors']==w['skipped']==0
 assert w['installed_import_denials']==['torch','transformers','jax','jaxlib','jaxite_word','tensorflow'];xml(path/'worker/tests.xml',tests)
 assert term['session_id']==session and term['outer_returncode']==0 and all(term[n]is True for n in('outer_reaped','collector_absent','child_absent','group_absent'))and term['report_sha256']==report_sha and term['source_archive_sha256']==tar_sha
 return expected,data
current,raw=proof(ROOT/'measured',33,91,79760,'e8f3e1ab64e2f1bc17273ee7cbac74b5c4c7cfc42fe3ee547b4112207f0fed31','5447e799a8662338ddcfb4de0cdb02cc60e4a1a45c257f3af2e9b5b8b7dbe097','99f60b4b4401ca246364127984af69df739f1fd0c4afa96e5e962558d44cde54')
old,_=proof(ROOT/'history/preflight32',32,86,3015,'f8b9a28569b1586483e94e4abd3057bc41031a2f02c1e9196eb4c7ea48e9ad51','39436d03cf7c3b3c8a2661efea71fe533f939414c417fc590136a7228de890f5','91650139ae02a6efd4b549bf9f73c99a310113545f367f262d794cc423abad72')
assert len([n for n in current if n.startswith('.pytest_cache/')])==4 and not any(n.endswith(('.pyc','.pyo'))for n in current)
for name,expected,total,passing in(('history/startup32',old,4,4),('history/alias32-negative',old,3,1),('reviews/startup33',current,4,4),('reviews/alias33',current,3,3)):
 q=ROOT/name;r=load(q/'report.json');data=unpack(q/'sources.tar.gz');assert {n:h(b)for n,b in data.items()}==expected==r['source_hashes_before']==r['source_hashes_after'];assert h((q/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
 assert len(r['cases'])==total and sum(c.get('passed',c.get('requirement_passed',False))for c in r['cases'])==passing
 for c in r['cases']:assert c['direct_child_waited']and c['pid_absent']and c['process_group_absent']and c['outputs_absent']
xml(ROOT/'history/lifecycle-negative-v1/tests.xml',4,4);xml(ROOT/'history/lifecycle-successor-v1/tests.xml',4);xml(ROOT/'history/lifecycle-negative-nested-v1/tests.xml',1,1);xml(ROOT/'history/lifecycle-successor-v2/tests.xml',5)
q=ROOT/'history/lifecycle-successor-v2';r=load(q/'report.json');data=unpack(q/'sources.tar.gz');assert len(data)==35 and {n:h(b)for n,b in data.items()}==r['source_hashes_before']==r['source_hashes_after'];assert all(data[n]==raw[n]for n in current);assert r['passed']and r['direct_returncode']==0 and r['direct_wait_completed']and r['pid_and_group_absent']
if a.current:assert {n:h((a.current/n).read_bytes())for n in current}==current
print(json.dumps(dict(passed=True,files=len(files),sources=33,synthetic_tests=91,installed_import_denials=6,independent_startup_checks=7,independent_cleanup_checks=5,current=a.current is not None,actual_inputs=False)))
