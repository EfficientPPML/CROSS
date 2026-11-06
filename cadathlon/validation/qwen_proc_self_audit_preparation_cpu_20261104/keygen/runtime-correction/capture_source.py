"""Source-only kernel-policy successor capture; no import or probe."""
from pathlib import Path
import ast,difflib,hashlib,json,tarfile
PARENT=Path('/tmp/qwen-keygen-actual-runtime-v2-20261104')
ROOT=Path('/tmp/qwen-keygen-actual-runtime-v3-20261104')
OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
compact=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def inv(r):
    paths=sorted(r.rglob('*'));assert not any(p.is_symlink() for p in paths)
    return {p.relative_to(r).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}
parent_raw=Path('/tmp/qwen-keygen-actual-runtime-v2-handoff-20261104/sources.json').read_bytes()
parent=json.loads(parent_raw);assert sha(parent_raw)=='28782d1927a4334c778dc8429d92637776bfe6a97a6a42e0d25f99cc82f35699'
assert inv(PARENT)==parent
current=inv(ROOT);assert len(current)==85 and set(current)==set(parent)
assert [n for n in parent if parent[n]!=current[n]]==['keygen_dependencies.py']
before=(PARENT/'keygen_dependencies.py').read_bytes();after=(ROOT/'keygen_dependencies.py').read_bytes()
bt=ast.parse(before);at=ast.parse(after)
old={n.name:n for n in bt.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
new={n.name:n for n in at.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert set(new)-set(old)=={'_literal_proc_self'}
assert {n for n in old if ast.dump(old[n],include_attributes=False)!=ast.dump(new[n],include_attributes=False)}=={'AccessScope'}
om={n.name:n for n in old['AccessScope'].body if isinstance(n,ast.FunctionDef)}
nm={n.name:n for n in new['AccessScope'].body if isinstance(n,ast.FunctionDef)}
assert set(om)==set(nm)
assert {n for n in om if ast.dump(om[n],include_attributes=False)!=ast.dump(nm[n],include_attributes=False)}=={'access','install'}
(OUT/'parent85-sources.json').write_bytes(parent_raw)
(OUT/'before_keygen_dependencies.py').write_bytes(before)
(OUT/'keygen_dependencies.patch').write_text(''.join(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='frozen85-v2/keygen_dependencies.py',tofile='kernel-successor85-v3/keygen_dependencies.py')))
(OUT/'sources.json').write_bytes(compact(current))
with tarfile.open(OUT/'sources.tar.gz','w:gz')as tar:
    for name in current:tar.add(ROOT/name,arcname=name,recursive=False)
with tarfile.open(OUT/'sources.tar.gz')as tar:
    members=tar.getmembers();assert len(members)==85 and all(m.isfile()for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==current
scope=('Only three exact observed strings /proc/self/status,cgroup,mountinfo get readonly alias handling. '
    'os.open rejects bytes/PathLike/noncanonical/relative/dirfd forms before normalization; direct builtin audit can only attest the observed event string. '
    'Allowed flags subset O_CLOEXEC|O_NOFOLLOW|O_NONBLOCK; resolution must equal current /proc/PID/leaf, but explicit PID callers remain unselected. '
    'Protected checks and all other aliases remain rejected. TLS carries the original literal decision to the original matching audit event. '
    'No current proc read, imports, tests, capacity measurement or actual grant. Prior61 proof is unchanged and does not cover this new integration.')
(OUT/'correction.json').write_bytes(pretty(dict(schema='qwen-keygen85-kernel-source-correction-v1',parent_root=str(PARENT),
    parent_manifest_sha256=sha(parent_raw),changed_only=['keygen_dependencies.py'],scope=scope,
    source_only=True,actual_entry_ready=False,actual_execution_authorized=False,tests=False,imports=False,probes=False)))
(OUT/'README.md').write_text('# Narrow keygen kernel metadata successor\n\n'+scope+'\n\nThe runtime inventory remains85 with84 exact parent files. No phase, arithmetic, owner, key format, grant or source-prerequisite semantics change. The external patch and correction preserve the old source finding. New targeted test authoring is separate; no inherited suite is selected by this capture.\n')
h=dict(schema='qwen-keygen85-kernel-source-handoff-v1',source_root=str(ROOT),source_count=85,
    source_manifest_sha256=sha((OUT/'sources.json').read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    module_sha256=sha(after),parent_manifest_sha256=sha(parent_raw),changed_only=['keygen_dependencies.py'],
    source_only=True,imports=False,tests=False,probes=False,actual_execution_authorized=False)
(OUT/'handoff.json').write_bytes(pretty(h))
assert inv(PARENT)==parent and inv(ROOT)==current
print(json.dumps(dict(**h,handoff_sha256=sha((OUT/'handoff.json').read_bytes())),sort_keys=True))
