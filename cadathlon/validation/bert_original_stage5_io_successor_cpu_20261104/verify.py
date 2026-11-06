from pathlib import Path
import argparse,ast,hashlib,io,json,tarfile,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent
REPORT='a95c198760c9a07a5362f0decdd608e7f8294de51e938cde00f328bff6bba3b2'
ARCHIVE='72c5e02bad85a0759cfe381ef5aa4deda7d51b360e6d1fe3197dd88dc04b0eb9'
h=lambda b:hashlib.sha256(b).hexdigest()
def source(raw):
 with tarfile.open(fileobj=io.BytesIO(raw))as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})and sum(m.size for m in ms)<=32<<20
  result={}
  for m in ms:
   p=Path(m.name);assert m.isfile()and not p.is_absolute()and '..'not in p.parts and m.size<=16<<20
   b=t.extractfile(m).read();assert len(b)==m.size
   assert p.suffix in ('.py','.json','.md')or m.name.endswith('.tar.gz')
   if m.name.endswith('.tar.gz'):source(b)
   result[m.name]=b
 return result
p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args()
manifest=json.loads((ROOT/'files.sha256.json').read_bytes());files={f.relative_to(ROOT).as_posix():f for f in ROOT.rglob('*')if f.is_file()and f.name!='files.sha256.json'and '__pycache__'not in f.parts};assert set(files)==set(manifest)
assert all(h(f.read_bytes())==manifest[n]for n,f in files.items())
for f in files.values():
 if f.name.endswith('.tar.gz'):source(f.read_bytes())
d=ROOT/'measured';r=json.loads((d/'report.json').read_bytes());i=json.loads((d/'installed.json').read_bytes());term=json.loads((d/'terminal_observation.json').read_bytes());expected=r['source_hashes_start']
assert h((d/'report.json').read_bytes())==REPORT and h((d/'sources.tar.gz').read_bytes())==ARCHIVE
raw=source((d/'sources.tar.gz').read_bytes());assert len(raw)==122 and {n:h(b)for n,b in raw.items()}==expected==r['source_hashes_end']==json.loads((d/'source_manifest.json').read_bytes())
assert all(r[n]is True for n in ('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','source_archive_verified','all_started_children_reaped'))
assert r['actual_HE']is False and r['actual_client_decode']is False
cases=list(E.parse(d/'tests.xml').iter('testcase'));assert len(cases)==r['test_count']==279 and not any(c.findall(n)for c in cases for n in ('error','failure','skipped'))
assert all(i[n]is True for n in ('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','all_forbidden_entries_restored','allowed_client_entries_unchanged','default_completion_api_verified'))
assert i['denied_count']==len(set(i['denied_entries']))==21 and i['source_hashes_start']==i['source_hashes_end']==expected
assert all(i[n]is False for n in ('actual_client_decode','actual_contexts','actual_keys','payload_reads','private_reads','native_model'))
assert len(i['source_guard_end']['loaded_local_modules'])==40
parent=json.loads(raw['client_89_sources.json']);overlays=i['source_guard_start']['overlays'];assert len(parent)==89 and len(overlays)==7
assert {n:expected[n]for n in parent}=={n:overlays.get(n,v)for n,v in parent.items()}
assert raw['runtime/stage/file_stages.py']==raw['baseline/file_stages.py'] and len(json.loads(raw['stage4_source_manifest.json']))==217
assert 'combined_server_gate.py'not in raw
for name in ('record_completed','retire_completed','_paths'):
 def get(blob):return next(ast.dump(n)for n in ast.parse(blob).body if isinstance(n,ast.FunctionDef)and n.name==name)
 assert get(raw['runtime/stage/request_lifecycle.py'])==get(raw['baseline/request_lifecycle.py'])
assert term['session_id']==29062 and term['observed_exit_code']==0 and term['session_reaped']and term['collector_absent']
for row in term['children']:
 assert row['absent'];q=row['process'];assert q['direct_returncode']==0 and all(q[n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))and not q['cleanup_errors']
neg=ROOT/'history/active-reader-negative';v=json.loads((neg/'report.json').read_bytes());assert v['tests']==13 and v['failures']==7 and v['errors']==0 and v['exit_code']==1 and v['direct_child_waited'];assert {n:h(b)for n,b in source((neg/'sources.tar.gz').read_bytes()).items()}==v['source_hashes_before']==v['source_hashes_after']
if a.current:assert {n:h((a.current/n).read_bytes())for n in expected}==expected
print(json.dumps(dict(passed=True,files=len(files),sources=122,tests=279,installed_denials=21,current=a.current is not None,actual_client=False)))
