"""Independent public capsule readback after root's own original terminal admission."""
import ast
import hashlib
import json
import tarfile
from datetime import datetime, timezone
from pathlib import Path

BASE=Path('/tmp/p3-original-six-final-public-20261105')
OUT=Path('/tmp/p3-original-six-final-root-readback-20261105.json')
sha=lambda b:hashlib.sha256(b).hexdigest()
assert not OUT.exists()
index_raw=(BASE/'artifact_hashes.json').read_bytes()
assert sha(index_raw)=='1fa64987df2fcc08e3704ece6c260b6de1531455765a796ebc46f902e1b45b33'
index=json.loads(index_raw);assert len(index)==56
assert {p.relative_to(BASE).as_posix() for p in BASE.rglob('*') if p.is_file()}==set(index)|{'artifact_hashes.json'}
raw={}
for name,r in index.items():
 assert not Path(name).is_absolute() and '..' not in Path(name).parts
 p=BASE/name;assert p.is_file() and not p.is_symlink()
 b=p.read_bytes();assert sha(b)==r['sha256'] and len(b)==r['bytes'];raw[name]=b
get=lambda n:json.loads(raw[n])
origins=get('origins.json')
for name,r in origins.items():
 assert sha(raw[name])==r['sha256'] and len(raw[name])==r['bytes']
 assert Path(r['path']).read_bytes()==raw[name]
admission=get('root/terminal-admission.json')
assert sha(raw['root/terminal-admission.json'])=='8724aaa62574c9346affd06797517569d018ee6a8a89332a29484e6d04f3ddd9'
assert admission['passed'] is True and admission['complete'] is True and admission['terminal_tool_chunk']=='84c601'
assert admission['report_sha256']==sha(raw['raw/report.json'])
report=get('raw/report.json');summary=get('summary.json')
assert report['passed'] is True and report['complete'] is True and 'active_case' not in report
assert summary['cases']==admission['cases'] and summary['passed_cases']==6 and summary['failed_cases']==0
assert summary['parent_elapsed_s']==report['elapsed_s']==admission['elapsed_s']==3369.480570556014
assert summary['independent_decrypted_logit_recomputation'] is False and summary['new_numerical_execution'] is False
sources=get('preparation/sources.json');aux=get('preparation/auxiliary.json')
assert len(sources)==128 and len(aux)==1
assert sources==get('raw/source-start.json')==get('raw/source-end.json')==report['source_hashes_start']==report['source_hashes_end']
for i in range(1,7):
 root='raw/test'+str(i)+'/'
 assert {n.removeprefix(root) for n in raw if n.startswith(root)}=={'process.json','source-after.json','stderr.log','stdout.json','time.txt'}
 process=get(root+'process.json');result=get(root+'stdout.json')
 assert process==report['cases']['test'+str(i)+'.in'] and result==process['result']
 assert process['passed'] is True and process['returncode']==0 and get(root+'source-after.json')==sources
 for n in raw:
  if n.startswith(root):assert sha(raw[n])==admission['case_file_hashes'][n.removeprefix('raw/')]
archives={}
for name,expected in [('sources/actual-128.tar.gz',sources),('sources/prepared-129.tar.gz',sources|aux)]:
 with tarfile.open(BASE/name,'r:gz') as t:
  members=t.getmembers();assert len(members)==len(expected) and {m.name for m in members}==set(expected)
  for m in members:
   assert m.isfile() and not m.issym() and not m.islnk()
   b=t.extractfile(m).read();assert sha(b)==expected[m.name]
   assert archives.setdefault(m.name,b)==b
audit=get('source-audit.json');assert len(audit['current_sources'])==129
for r in audit['current_sources']:
 assert sha(Path(r['original_path']).read_bytes())==sha(Path(r['prepared_path']).read_bytes())==r['sha256']==(sources|aux)[r['name']]
for label,r in audit['source_functions'].items():
 name,fun=label.rsplit(':',1);tree=ast.parse(archives[name]);node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fun]
 assert r==dict(line=node.lineno,end_line=node.end_lineno,AST_sha256=sha(ast.dump(node,include_attributes=False).encode()))
capture=get('capture.json')
for flag in ('candidate_import_or_execution','key_body_read','new_process_or_key_path_observation','packager_direct_wait_authority','packager_polled_original_session'):assert capture[flag] is False
for name,b in raw.items():assert (BASE/name).read_bytes()==b
assert (BASE/'artifact_hashes.json').read_bytes()==index_raw
r=dict(schema='root-final-p3-six-public-capsule-readback-v1',utc=datetime.now(timezone.utc).isoformat(),passed=True,
 indexed_files=56,exact_origin_copies=len(origins),actual_sources=128,prepared_sources=129,source_definitions=len(audit['source_functions']),
 cases=6,original_terminal_session=81235,original_terminal_chunk='84c601',original_root_admission_sha256=sha(raw['root/terminal-admission.json']),
 capsule_index_sha256=sha(index_raw),parent_elapsed_s=report['elapsed_s'],new_process_poll_or_cleanup=False,
 independent_decrypted_logit_recomputation=False,candidate_rerun=False,root_owns_prior_terminal_admission=True,
 verifier_path=str(Path(__file__)),verifier_sha256=sha(Path(__file__).read_bytes()))
OUT.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(path=str(OUT),sha256=sha(OUT.read_bytes()),indexed_files=56,origins=len(origins),source_definitions=r['source_definitions'],passed=True)))
