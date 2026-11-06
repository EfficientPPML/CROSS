"""Stdlib-only source closure capture; no candidate imports or test execution."""
from pathlib import Path
import ast,hashlib,json,tarfile,io
ROOT=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v2-20261104')
OLD=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-20261104')
OUT=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-v2-handoff-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def raw(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def put(p,b):
    with p.open('xb')as f:f.write(b)
parent_raw=Path('/tmp/qwen-toy5-actual-checkpoint-runtime-handoff-20261104/sources.json').read_bytes()
assert sha(parent_raw)=='b787490da5ad804dacdeca7c19cbb39a99868ab80ea762a607e1a6457cfb3180'
parent=json.loads(parent_raw);assert len(parent)==382
files={}
for p in sorted(ROOT.rglob('*')):
    assert not p.is_symlink()
    if p.is_file():files[p.relative_to(ROOT).as_posix()]=sha(p.read_bytes())
assert len(files)==385
changed=sorted(n for n,pin in parent.items()if files[n]!=pin)
assert changed==sorted(['q5_ciphertext_owner.py','q5_actual_process.py','SOURCE_DESIGN.md','RESOURCE_STATUS.json'])
added=sorted(set(files)-set(parent))
assert added==sorted(['owner_correction_history.json','provenance/owner_correction_ciphertext.patch','provenance/owner_correction_setup.patch'])
assert all(sha((OLD/n).read_bytes())==pin for n,pin in parent.items())
assert all(files[n]==pin for n,pin in parent.items()if n.startswith(('parent/','runtime/jaxite_word/','common/')))
assert len([n for n in files if n.startswith('parent/')])==250
assert len([n for n in files if n.startswith('runtime/jaxite_word/')])==34
for n in files:
    if n.endswith('.py'):ast.parse((ROOT/n).read_bytes(),filename=n)
def nodes(path):return {n.name:n for n in ast.parse(path.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
def dump(n):return ast.dump(n,include_attributes=False)
for name,allowed in [('q5_actual_process.py',{'_EncryptedInput'}),('q5_ciphertext_owner.py',{'CiphertextOwner','InputOwner'})]:
    old=nodes(OLD/name);new=nodes(ROOT/name);assert set(old)==set(new)
    assert all(dump(v)==dump(new[k])for k,v in old.items()if k not in allowed)
    for cls in allowed:
        om={n.name:n for n in old[cls].body if isinstance(n,ast.FunctionDef)}
        nm={n.name:n for n in new[cls].body if isinstance(n,ast.FunctionDef)}
        changed_methods={k for k in om if dump(om[k])!=dump(nm[k])}
        wanted={'__init__','check','load','close'}if cls=='_EncryptedInput'else {'load','close'}if cls=='CiphertextOwner'else {'close'}
        assert changed_methods==wanted,(cls,changed_methods)
        assert set(nm)-set(om)==({'_retain_returned'}if cls=='_EncryptedInput'else set())
oldres=json.loads((OLD/'RESOURCE_STATUS.json').read_bytes());newres=json.loads((ROOT/'RESOURCE_STATUS.json').read_bytes())
assert all(newres[k]==v for k,v in oldres.items()if k!='required_independent_review')
assert newres['required_independent_review'][:-2]==oldres['required_independent_review']
OUT.mkdir();put(OUT/'sources.json',raw(files));put(OUT/'parent382-sources.json',parent_raw)
for src,name in [('/tmp/build_q5_owner_correction_20261104.py','build_source.py'),(__file__,'capture_source.py')]:put(OUT/name,Path(src).read_bytes())
with tarfile.open(OUT/'sources.tar.gz','w:gz')as archive:
    for n in files:
        b=(ROOT/n).read_bytes();info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o644;info.mtime=0
        archive.addfile(info,io.BytesIO(b))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as archive:
    assert {m.name:sha(archive.extractfile(m).read())for m in archive.getmembers()}==files
review=dict(schema='qwen-toy5-owner-correction-static-readback-v1',source_only=True,passed=True,source_count=385,
 parent382_current_unchanged=True,parent250_exact=True,core34_exact=True,common12_exact=True,
 overlays=changed,additions=added,unchanged_parent_files=378,all_python_ast_parse=True,
 unrelated_top_level_and_methods_ast_exact=True,resource_facts_unchanged=True,
 candidate_imports=False,collection=False,tests=False,device_or_capacity_probes=False,body_reads=False,
 actual_entry_and_runtime_denials_unchanged=True,
 scope='source delta/closure only; no runtime/barrier correctness or numerical proof')
put(OUT/'source-review.json',raw(review))
handoff=dict(schema='qwen-toy5-owner-correction-handoff-v1',root=str(ROOT),source_count=385,
 source_map_sha256=sha((OUT/'sources.json').read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
 source_review_sha256=sha((OUT/'source-review.json').read_bytes()),parent_source_count=382,parent_source_map_sha256=sha(parent_raw),
 code_hashes={n:files[n]for n in ('q5_ciphertext_owner.py','q5_actual_process.py')},
 changed=changed,added=added,actual_runtime_issuer=False,actual_execution=False,candidate_imports=False,tests=False,
 next_scope='Separate source-only RuntimeAdmission/policy/owner-hook/terminal implementation under plan453599dc; not part of385')
put(OUT/'handoff.json',raw(handoff));print(json.dumps(handoff,sort_keys=True))
