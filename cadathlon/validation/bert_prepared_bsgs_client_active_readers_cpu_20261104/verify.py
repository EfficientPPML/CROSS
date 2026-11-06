from pathlib import Path
import argparse,ast,hashlib,io,json,tarfile,xml.etree.ElementTree as E
ROOT=Path(__file__).resolve().parent
REPORT='a3fe3c4e767aaa36c24d59182048247df9c4ccf8cdb1f26977bac25676bbe08c'
ARCHIVE='7fb01ae5ba8d52788b7f46077202a43aeaf2dc69a9ff3072c187b321a7c512bf'
h=lambda b:hashlib.sha256(b).hexdigest()
def source(raw):
 with tarfile.open(fileobj=io.BytesIO(raw))as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})and sum(m.size for m in ms)<=32<<20
  result={}
  for m in ms:
   p=Path(m.name);assert m.isfile()and not p.is_absolute()and '..'not in p.parts and m.size<=16<<20
   b=t.extractfile(m).read();assert len(b)==m.size
   assert p.suffix in('.py','.json','.md')or m.name.endswith(('.tar.gz','.py.txt'))
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
raw=source((d/'sources.tar.gz').read_bytes());assert len(raw)==125 and {n:h(b)for n,b in raw.items()}==expected==r['source_hashes_end']==json.loads((d/'source_manifest.json').read_bytes())
assert all(r[n]is True for n in('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','source_archive_verified','all_started_children_reaped'))
assert r['actual_HE']is False and r['actual_client_decode']is False
cases=list(E.parse(d/'tests.xml').iter('testcase'));assert len(cases)==r['test_count']==317 and not any(c.findall(n)for c in cases for n in('error','failure','skipped'))
assert all(i[n]is True for n in('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','all_forbidden_entries_restored','allowed_client_entries_unchanged','optional_completion_api_verified'))
assert i['denied_count']==len(set(i['denied_entries']))==21 and i['source_hashes_start']==i['source_hashes_end']==expected
assert h((d/'installed.json').read_bytes())==r['installed_report_sha256']
assert all(i[n]is False for n in('actual_client_decode','actual_contexts','actual_keys','payload_reads','private_reads','native_model'))
assert len(i['source_guard_end']['loaded_local_modules'])==40
for row in i['source_guard_end']['loaded_local_modules'].values():assert row['sha256']==expected[row['path']]
parent89=json.loads(raw['client_89_sources.json']);overlays=i['source_guard_start']['overlays'];assert len(parent89)==89 and len(overlays)==8
assert {n:expected[n]for n in parent89}=={n:overlays.get(n,v)for n,v in parent89.items()}
port=json.loads((ROOT/'provenance/port.json').read_bytes());parents={}
for directory,count,reportkey,tarkey in(('parent117',117,'parent_report_sha256','parent_source_archive_sha256'),('reader_origin122',122,'original_reader_report_sha256','original_reader_source_archive_sha256')):
 q=ROOT/'provenance'/directory;pr=json.loads((q/'report.json').read_bytes());assert h((q/'report.json').read_bytes())==port[reportkey]and h((q/'sources.tar.gz').read_bytes())==port[tarkey]
 prraw=source((q/'sources.tar.gz').read_bytes());assert len(prraw)==count and {n:h(b)for n,b in prraw.items()}==pr['source_hashes_start']==pr['source_hashes_end']==json.loads((q/'source_manifest.json').read_bytes());parents[directory]=prraw
old=parents['parent117'];orig=parents['reader_origin122'];changed={n:dict(before=h(old[n]),after=h(raw[n]))for n in old if raw[n]!=old[n]};added={n:h(raw[n])for n in raw.keys()-old.keys()};assert changed==port['changed']and added==port['added']and len(changed)==10 and len(added)==8
for n in('key_gate.py','input_gate.py','runtime/stage/transport.py'):assert raw[n]==orig[n]
for n in('combined_server_gate.py','separate_public_readback.py','separate_client_readback.py','runtime/stage/file_stages.py','runtime/stage/request_lifecycle.py','runtime/stage/client_keys.py','runtime/stage/key_contexts.py','runtime/stage/file_stage_io.py'):assert raw[n]==old[n]
def bootstrap(b):return next(ast.dump(n)for n in ast.parse(b).body if isinstance(n,ast.FunctionDef)and n.name=='_cli_bootstrap')
for role in('worker','collect'):assert bootstrap(raw['separate_client_'+role+'.py'])==bootstrap(orig['original_client_'+role+'.py'])
assert term['session_id']==24184 and term['observed_exit_code']==0 and term['session_reaped']and term['collector_absent']
assert term['report_sha256']==REPORT and term['source_archive_sha256']==ARCHIVE
for row in term['children']:
 assert row['absent'];q=row['process'];assert q['direct_returncode']==0 and all(q[n]is True for n in('direct_child_reaped','group_empty_verified','cleanup_completed'))and not q['cleanup_errors']
neg=ROOT/'history/active-reader-negative';v=json.loads((neg/'report.json').read_bytes());assert v['tests']==13 and v['failures']==7 and v['errors']==0 and v['exit_code']==1 and v['direct_child_waited'];assert {n:h(b)for n,b in source((neg/'sources.tar.gz').read_bytes()).items()}==v['source_hashes_before']==v['source_hashes_after']
if a.current:assert {n:h((a.current/n).read_bytes())for n in expected}==expected
print(json.dumps(dict(passed=True,files=len(files),sources=125,tests=317,installed_denials=21,parent117_unchanged=107,current=a.current is not None,actual_client=False)))
