import ast,difflib,hashlib,json,shutil,tarfile
from pathlib import Path
old=Path('/tmp/qwen-full28-device-focused-proof-20261104/preparation-v1');new=old.parent/'preparation-v2'
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def inventory(root):
 assert not any(p.is_symlink()for p in root.rglob('*'))
 return {p.relative_to(root).as_posix():sha(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
before=inventory(old);assert not new.exists();shutil.copytree(old,new)
text=(old/'inspect_children.py').read_text()
left="""            target=os.readlink(child);assert len(target)<=4096 and not Path(target).is_absolute()and '..'not in Path(target).parts
            named=child.parent/target;resolved=named.resolve(strict=True)
            assert named==resolved and resolved.parent==base and resolved.is_dir()and not resolved.is_symlink()
            prefix=child.name[:-len('current')]
            assert resolved.name.startswith(prefix)and resolved.name[len(prefix):].isdigit()
            after=child.lstat();assert(info.st_dev,info.st_ino,info.st_mode)==(after.st_dev,after.st_ino,after.st_mode)and os.readlink(child)==target
"""
right="""            target=os.readlink(child)
            # The only selected target is one lexical sibling basename. Never
            # resolve an alias or follow a target symlink during validation.
            assert type(target)is str and 0<len(target)<=255 and target not in('.', '..')
            assert '/'not in target and '\\\\'not in target and Path(target).name==target
            named=child.parent/target;assert named.parent==base and named!=child
            target_info=named.lstat()
            assert stat.S_ISDIR(target_info.st_mode)and target_info.st_uid==os.geteuid()
            prefix=child.name[:-len('current')]
            assert named.name.startswith(prefix)and named.name[len(prefix):].isdigit()
            after=child.lstat()
            assert(info.st_dev,info.st_ino,info.st_mode,info.st_uid)==(after.st_dev,after.st_ino,after.st_mode,after.st_uid)
            assert os.readlink(child)==target
"""
assert text.count(left)==1;text=text.replace(left,right)
(new/'inspect_children.py').write_text(text);inspector_sha=sha(text.encode())
config=json.loads((old/'run_config.json').read_bytes())
for key in('proof_root','owned_tmpdir','pytest_basetemp'):config[key]=config[key].replace(str(old),str(new))
raw=pretty(config);(new/'run_config.json').write_bytes(raw);config_sha=sha(raw)
driver=(old/'executed_driver.py').read_text();oldhandoff=json.loads((old/'handoff.json').read_bytes())
assert oldhandoff['config_sha256'] in driver and oldhandoff['inspector_sha256'] in driver
driver=driver.replace(oldhandoff['config_sha256'],config_sha).replace(oldhandoff['inspector_sha256'],inspector_sha)
(new/'executed_driver.py').write_text(driver)
proposal=json.loads((old/'launch-proposal.json').read_bytes());proposal['argv']=[s.replace(str(old),str(new))for s in proposal['argv']]
proposal.update(driver_sha256=sha(driver.encode()),config_sha256=config_sha,inspector_sha256=inspector_sha)
(new/'launch-proposal.json').write_bytes(pretty(proposal))
# Retain v1 history independently; do not relabel the original source as run.
history=new/'v1-alias-correction';history.mkdir()
for name in('inspect_children.py','executed_driver.py','run_config.json','launch-proposal.json','handoff.json','driver_provenance.json'):
 (history/(name+'.txt')).write_bytes((old/name).read_bytes())
for name in('inspect_children.py','executed_driver.py','run_config.json','launch-proposal.json'):
 (history/(name+'.patch')).write_text(''.join(difflib.unified_diff((old/name).read_text().splitlines(True),(new/name).read_text().splitlines(True),fromfile='unrun-preparation-v1/'+name,tofile='unrun-preparation-v2/'+name)))
provenance=dict(schema='full28-focused-harness-alias-successor-v2',prior_root=str(old),prior_handoff_sha256=sha((old/'handoff.json').read_bytes()),prior_all_file_map=before,
 reason='Root source review found alias target resolve/is_dir followed before rejection, contradicting lstat/readlink-only evidence scope.',
 correction='One lexical sibling basename plus named.lstat directory/UID/prefix validation; original alias readlink/lstat identity preserved; no target resolve or followed directory tests.',
 changed_execution_files=['inspect_children.py','executed_driver.py','run_config.json'],worker_unchanged=True,execution_source458_unchanged=True,
 source_map_archive_unchanged=True,selected_nodes59_unchanged=True,owned_popen_wait_finally_unchanged=True,prior_v1_status='HELD_UNRUN_SOURCE_ONLY',
 current_status='HELD_UNRUN_SOURCE_ONLY',no_imports_collection_tests=True)
(new/'driver_provenance.json').write_bytes(pretty(provenance))
# Preserve inherited authoring script as history; this source is the v2 builder.
(new/'prepare_harness.py').write_bytes(Path(__file__).read_bytes())
for name in('inspect_children.py','executed_driver.py','worker_driver.py'):ast.parse((new/name).read_bytes())
# Mechanically exclude target-following calls from the symlink branch only.
tree=ast.parse(text);function=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='inspect')
loop=next(n for n in function.body if isinstance(n,ast.For));branch=next(n for n in loop.body if isinstance(n,ast.If))
assert not any(isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr in('resolve','is_dir','is_symlink','stat')for n in ast.walk(branch))
expected=json.loads((old/'sources-before.json').read_bytes());assert len(expected)==458
assert inventory(new/'execution')==inventory(old/'execution')==expected
assert(new/'sources-before.json').read_bytes()==(old/'sources-before.json').read_bytes()
assert(new/'sources.tar.gz').read_bytes()==(old/'sources.tar.gz').read_bytes()
with tarfile.open(new/'sources.tar.gz')as archive:assert {m.name:sha(archive.extractfile(m).read())for m in archive.getmembers()}==expected
assert(new/'worker_driver.py').read_bytes()==(old/'worker_driver.py').read_bytes()
def owned(text):return text.split("    with (out/'tests.log').open('xb')as log:\n",1)[1].split('    current,after=inventory(source),inventory(execution)',1)[0]
assert owned(driver)==owned((old/'executed_driver.py').read_text())
assert inventory(old)==before
pins={p.relative_to(new).as_posix():sha(p.read_bytes())for p in sorted(new.rglob('*'))if p.is_file()and 'execution'not in p.relative_to(new).parts and p.name!='handoff.json'}
handoff=dict(oldhandoff);handoff.update(proof_root=str(new),execution_root=str(new/'execution'),driver_sha256=sha(driver.encode()),inspector_sha256=inspector_sha,config_sha256=config_sha,
 proposal_sha256=sha((new/'launch-proposal.json').read_bytes()),pins=pins,predecessor_handoff_sha256=sha((old/'handoff.json').read_bytes()),correction='lexical sibling basename and lstat-only target metadata',imports_collection_runs=0)
(new/'handoff.json').write_bytes(pretty(handoff))
print(json.dumps({k:handoff[k]for k in('driver_sha256','worker_sha256','inspector_sha256','config_sha256','proposal_sha256')},indent=2));print('handoff',sha((new/'handoff.json').read_bytes()));print('original preparation-v1 all files unchanged; source458/tar/map unchanged; no imports/collection/tests')
