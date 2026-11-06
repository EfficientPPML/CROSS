import ast,difflib,gzip,hashlib,io,json,shutil,tarfile
from pathlib import Path
old=Path('/tmp/qwen-full28-device-focused-tests-20261104');new=Path('/tmp/qwen-full28-device-focused-tests-v2-20261105');out=Path('/tmp/qwen-full28-device-focused-test-handoff-v2-20261105')
sha=lambda b:hashlib.sha256(b).hexdigest();canon=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def inventory(root):
 assert not any(p.is_symlink()for p in root.rglob('*'))
 return{p.relative_to(root).as_posix():sha(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
parent_path=Path('/tmp/qwen-full28-device-focused-test-sources-20261104.json');parent=parent_path.read_bytes();before=json.loads(parent)
assert inventory(old)==before and len(before)==458
assert not new.exists()and not out.exists();shutil.copytree(old,new);out.mkdir()
source=(old/'owned_child.py').read_bytes();text=source.decode();line='            import importlib\n';assert text.count(line)==1
fixed=text.replace(line,'');(new/'owned_child.py').write_text(fixed)
old_ast=ast.parse(text);new_ast=ast.parse(fixed)
old_main=next(n for n in old_ast.body if isinstance(n,ast.FunctionDef)and n.name=='main')
inner=[n for n in ast.walk(old_main)if isinstance(n,ast.Import)and any(a.name=='importlib'for a in n.names)];assert len(inner)==1
class Remove(ast.NodeTransformer):
 def visit_Import(self,node):
  if node is inner[0]:return None
  return node
normalized=Remove().visit(old_ast)
assert ast.dump(normalized,include_attributes=False)==ast.dump(new_ast,include_attributes=False)
assert any(isinstance(n,ast.Import)and any(a.name=='importlib.util'for a in n.names)for n in new_ast.body)
peer=Path('/tmp/qwen-full28-device-failed-security-peer-20261105/review.json')
patch=''.join(difflib.unified_diff(text.splitlines(True),fixed.splitlines(True),fromfile='failed458/owned_child.py',tofile='unrun459/owned_child.py'))
history=dict(schema='full28-device-focused-import-binding-successor-v1',parent_root=str(old),parent_source_count=458,parent_map_sha256=sha(parent),parent_archive_sha256='3a88579485a582dcaefbf4ded73c4c29fe33bed05aeaf8b3cebd7cf1cb68a86e',
 predecessor_run=dict(proof_root='/tmp/qwen-full28-device-focused-proof-20261104/preparation-v2',root_owned_session=20699,launch_tool='f209b7',terminal_tool='2d61ff',collector_exit_code=70,worker_exit_code=1,tests=59,failures=59,passes=0,child_exit_code=70,child_results=0),
 failure='Redundant inner import importlib makes main importlib a local variable; earlier importlib.util is unbound before source helper or candidate execution.',
 secondary_collector_failure='Strict success-only record inspector then raises FileNotFoundError because no normal child result exists; source correction does not claim to repair that separate reporting path.',
 correction='Delete only the redundant inner import; use original module-level import importlib.util binding. Entire other AST identical.',
 before_sha256=sha(source),after_sha256=sha(fixed.encode()),runtime403_unchanged=True,static59_nodes_unchanged=True,
 unchanged_predecessor_metadata='Inherited static_source_review.json/origin.json and authoring histories remain original458 source-stage records; this successor handoff supplies current459 identity.',
 independent_negative_review=dict(path=str(peer),sha256=sha(peer.read_bytes())),source_patch=patch,
 actual_body_or_backend_tested=False,imports_collection_execution=False,successor_status='HELD_UNRUN_SOURCE_ONLY')
(new/'failed458_import_binding_successor.json').write_bytes(canon(history))
files=inventory(new);assert len(files)==459
assert[n for n,h in before.items()if files[n]!=h]==['owned_child.py']
assert set(files)-set(before)=={'failed458_import_binding_successor.json'}
assert{n:h for n,h in files.items()if n.startswith('runtime/')}=={n:h for n,h in before.items()if n.startswith('runtime/')}
selection=(new/'selected_tests.json').read_bytes();assert selection==(old/'selected_tests.json').read_bytes()
for p in new.glob('*.py'):ast.parse(p.read_bytes(),filename=str(p))
manifest=canon(files);map_path=Path('/tmp/qwen-full28-device-focused-test-sources-v2-20261105.json');assert not map_path.exists();map_path.write_bytes(manifest);(out/'sources.json').write_bytes(manifest)
with(out/'sources.tar.gz').open('xb')as sink:
 with gzip.GzipFile(fileobj=sink,mode='wb',mtime=0,filename='')as gz:
  with tarfile.open(fileobj=gz,mode='w',format=tarfile.PAX_FORMAT)as archive:
   for name,h in files.items():
    raw=(new/name).read_bytes();assert sha(raw)==h;info=tarfile.TarInfo(name);info.size=len(raw);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(raw))
with tarfile.open(out/'sources.tar.gz')as a:assert{m.name:sha(a.extractfile(m).read())for m in a.getmembers()}==files
(out/'before_owned_child.py.txt').write_bytes(source);(out/'source.patch').write_text(patch)
(out/'failed-peer-review.json').write_bytes(peer.read_bytes());(out/'prepare_source.py').write_bytes(Path(__file__).read_bytes())
handoff=dict(schema='full28-device-focused-source-successor-handoff-v2',status='HELD_UNRUN_SOURCE_ONLY',source_root=str(new),source_count=459,source_map=str(map_path),source_manifest_sha256=sha(manifest),source_archive=str(out/'sources.tar.gz'),source_archive_sha256=sha((out/'sources.tar.gz').read_bytes()),parent_source_count=458,parent_map_sha256=sha(parent),parent_current_preserved=inventory(old)==before,runtime_count=403,runtime_unchanged=True,static_cases=59,selected_tests_sha256=sha(selection),code_overlays={'owned_child.py':dict(before=sha(source),after=sha(fixed.encode()))},additions=['failed458_import_binding_successor.json'],ast_delta='only remove one function-local redundant import importlib',failed_history_sha256=files['failed458_import_binding_successor.json'],imports_collection_execution=False,actual_authority=False)
(out/'handoff.json').write_bytes(canon(handoff));assert inventory(old)==before and inventory(new)==files
print(json.dumps({k:handoff[k]for k in('source_count','source_manifest_sha256','source_archive_sha256','selected_tests_sha256')},indent=2));print('owned_child',sha(fixed.encode()));print('handoff',sha((out/'handoff.json').read_bytes()))
