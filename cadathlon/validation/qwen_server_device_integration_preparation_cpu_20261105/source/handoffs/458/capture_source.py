import ast,gzip,hashlib,io,json,tarfile
from pathlib import Path
root=Path('/tmp/qwen-full28-device-focused-tests-20261104')
out=Path('/tmp/qwen-full28-device-focused-test-handoff-20261104');out.mkdir(exist_ok=False)
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canon=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
expected=json.loads((root/'runtime_sources.json').read_bytes());assert len(expected)==403
parent=Path('/tmp/qwen-server-device-integration-runtime-v4-20261104')
assert all((root/'runtime'/n).read_bytes()==(parent/n).read_bytes()for n in expected)
assert {p.relative_to(root/'runtime').as_posix():sha(p.read_bytes())for p in(root/'runtime').rglob('*')if p.is_file()}==expected
selection=json.loads((root/'selected_tests.json').read_bytes());nodes=json.loads((root/'accepted_plan/nodes.json').read_bytes())['nodes']
assert selection['count']==59 and selection['nodes']==[v['node']for v in nodes]and selection['cases']==[v['case']for v in nodes]
assert len(set(selection['nodes']))==59
helpers=json.loads((root/'helper_selection.json').read_bytes());assert len(helpers)==52
assert all((root/n).is_file()for n in helpers.values())
assert all(not n.startswith('encrypted_')and n not in('full_contract','support')for n in helpers)
python_rows={}
for p in root.glob('*.py'):
 tree=ast.parse(p.read_bytes(),filename=str(p));python_rows[p.name]=sha(ast.dump(tree,include_attributes=False).encode())
assert sha((root/'cases_packet.py').read_bytes())=='bccf7091699ca5161a234c9511c05e31e3e671de53c409b27097df981cf526e3'
assert sha((root/'cases_process.py').read_bytes())=='d252c794b0cb0363e0e601c6cbb1366ad3b81f6dcebaedb324116dd31bf47546'
assert sha((root/'cases_loader_engine.py').read_bytes())=='722e882c25df34d6e6f4404822073eacaf5619c97733a697cac7bb06d04bfa2c'
static=dict(schema='full28-device-static-source-review-v1',passed=True,parent403_exact=True,static_nodes=59,child_count=59,record_count=236,fixture_count=14,helper_selection_count=52,python_ast_sha256=python_rows,candidate_imports=0,collection_runs=0,test_runs=0,backend_or_capacity_probes=0)
(root/'static_source_review.json').write_bytes(canon(static))
files={}
for p in sorted(root.rglob('*')):
 assert not p.is_symlink()
 if p.is_dir():continue
 assert p.is_file()and p.suffix!='.pyc'and '__pycache__'not in p.parts
 files[p.relative_to(root).as_posix()]=sha(p.read_bytes())
rawmap=canon(files);map_path=Path('/tmp/qwen-full28-device-focused-test-sources-20261104.json');assert not map_path.exists();map_path.write_bytes(rawmap)
(out/'sources.json').write_bytes(rawmap)
with(out/'sources.tar.gz').open('xb')as sink:
 with gzip.GzipFile(fileobj=sink,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT)as archive:
   for name,h in files.items():
    raw=(root/name).read_bytes();assert sha(raw)==h
    info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(raw))
with tarfile.open(out/'sources.tar.gz')as a:
 assert {m.name:sha(a.extractfile(m).read())for m in a.getmembers()}==files
assert {p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}==files
handoff=dict(schema='full28-device-focused-source-handoff-v1',status='HELD_SOURCE_ONLY_NOT_COLLECTED_OR_EXECUTED',source_root=str(root),source_count=len(files),source_map=str(map_path),source_manifest_sha256=sha(rawmap),source_archive=str(out/'sources.tar.gz'),source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),source_archive_bytes=(out/'sources.tar.gz').stat().st_size,parent_count=403,parent_map_sha256=sha((root/'runtime_sources.json').read_bytes()),parent_unchanged=True,static_cases=59,selected_tests_sha256=files['selected_tests.json'],static_nodes=selection['nodes'],prospective_owned_children=59,prospective_regular_record_files=236,prospective_fixture_metadata_files=14,actual_entry_ready=False,scope='Python source/metadata recording boundaries; no genuine NumPy/backend/capacity/body/actual role',imports_collection_execution=False,new_source_hashes={n:h for n,h in files.items()if not n.startswith('runtime/')},runtime_map_encoding='compact sorted JSON + newline',full_map_encoding='compact sorted JSON + newline')
(out/'handoff.json').write_bytes(canon(handoff));(out/'static-review.json').write_bytes(canon(static));(out/'capture_source.py').write_bytes(Path(__file__).read_bytes())
print(json.dumps({k:handoff[k]for k in('source_count','source_manifest_sha256','source_archive_sha256','source_archive_bytes','parent_map_sha256','selected_tests_sha256')},indent=2));print('handoff',sha((out/'handoff.json').read_bytes()))
