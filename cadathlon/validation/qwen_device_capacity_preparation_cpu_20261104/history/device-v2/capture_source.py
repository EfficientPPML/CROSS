"""Source-only identity-successor capture; no candidate import or test."""
import ast
import difflib
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

OLD=Path('/tmp/qwen-server-device-admission-source-20261104')
ROOT=Path('/tmp/qwen-server-device-admission-source-v2-20261104')
OUT=Path('/tmp/qwen-server-device-admission-handoff-v2-20261104')
PRIOR=Path('/tmp/qwen-server-device-admission-handoff-20261104')
PEER=Path('/tmp/qwen-device-component-transformers-review-20261104/review.json')
def canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(b):return hashlib.sha256(b).hexdigest()
def emit(p,v):p.write_bytes(canonical(v))
def inv(r):
    assert all(p.is_file()and not p.is_symlink()for p in r.iterdir())
    return{p.name:sha(p.read_bytes())for p in sorted(r.iterdir())}
old=json.loads((PRIOR/'sources.json').read_bytes());assert inv(OLD)==old and len(old)==5
assert sha(PEER.read_bytes())=='2965d5612888d945b1717c4f6bba43a0d473655bae67b83589b484fe5a25a752'
before=inv(ROOT);assert set(before)==set(old)
assert {n for n in old if old[n]!=before[n]}=={'API.md','qwen_device_capacity.py'}
delta=''.join(''.join(difflib.unified_diff((OLD/n).read_text().splitlines(True),(ROOT/n).read_text().splitlines(True),fromfile='v1/'+n,tofile='v2/'+n))for n in('qwen_device_capacity.py','API.md'))
(OUT/'source.patch').write_text(delta)
emit(ROOT/'successor_history.json',dict(schema='qwen-device-policy-identity-successor-v2',
    parent_root=str(OLD),parent_manifest_sha256=sha((PRIOR/'sources.json').read_bytes()),
    parent_archive_sha256=sha((PRIOR/'sources.tar.gz').read_bytes()),parent_sources=old,
    peer_finding={'path':str(PEER),'sha256':sha(PEER.read_bytes()),'scope':'source-only local retained-policy identity gap; no actual bypass/probe claim'},
    code_overlays=['qwen_device_capacity.py'],documentation_overlays=['API.md'],additions=['successor_history.json'],
    retained_unchanged=['qwen_device_policy.py','source_refs.json','provenance.json'],
    inherited_provenance_json_is_historical_v1=True,
    changes=['retain original policy dictionary and three original field objects before authentication/callbacks',
        'compare dictionary/field identities before and after reauthentication and hook calls',
        'authenticate and invoke hooks using retained original inputs',
        'sticky failure on even-equal identity replacement, retaining original inputs'],
    nested_phase_graph_is_not_newly_snapshotted=True,actual_authority_bypass_claim=False,
    candidate_imports=False,tests=False,probes=False,policy_instances=False,actual_grants=False,root_server_edits=False))
blobs={p.name:p.read_bytes()for p in sorted(ROOT.iterdir())};source_map={n:sha(b)for n,b in blobs.items()}
assert len(source_map)==6
for n,b in blobs.items():
    if n.endswith('.py'):ast.parse(b,filename=n)
tree=ast.parse(blobs['qwen_device_capacity.py'])
func={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
for n in('_capture_policy','_policy_guard','_reauthenticate'):assert n in func
oldtree=ast.parse((OLD/'qwen_device_capacity.py').read_bytes());oldfunc={n.name:n for n in oldtree.body if isinstance(n,ast.FunctionDef)}
changed=[n for n in oldfunc if ast.dump(oldfunc[n],include_attributes=False)!=ast.dump(func[n],include_attributes=False)]
assert set(changed)=={'_caller','_context','_hook','_host','issue_initialization','_jax_binding','_device_guard','observe_capacity'}
# All policy-input field access now occurs in capture/guard using the retained
# dictionary. No later hook or observation rereads dataclass field attributes.
assert not any(isinstance(n,ast.Attribute)and n.attr in('phase','blobs','expected_pins')and isinstance(n.value,ast.Subscript)
    and isinstance(n.value.slice,ast.Constant)and n.value.slice.value=='policy'for n in ast.walk(tree))
emit(OUT/'sources.json',source_map)
with(OUT/'sources.tar.gz').open('wb')as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as compressed:
        with tarfile.open(mode='w',fileobj=compressed,format=tarfile.PAX_FORMAT)as archive:
            for n,b in blobs.items():
                info=tarfile.TarInfo(n);info.size=len(b);info.mode=0o644;info.mtime=0;archive.addfile(info,io.BytesIO(b))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers();assert len(members)==6 and all(m.isfile()for m in members)
    assert{m.name:sha(archive.extractfile(m).read())for m in members}==source_map
assert inv(OLD)==old and inv(ROOT)==source_map
emit(OUT/'static-review.json',dict(schema='qwen-device-v2-source-static-review',passed=True,source_count=6,
    old5_unchanged=True,pure_policy_bytes_unchanged=True,new_functions=['_capture_policy','_policy_guard','_reauthenticate'],
    changed_existing_functions=changed,source_map_sha256=sha((OUT/'sources.json').read_bytes()),
    archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    no_candidate_imports=True,no_tests_or_probes=True,no_policy_instances_or_grants=True,
    scope='author static AST/map/archive readback only; independent successor review pending'))
emit(OUT/'handoff.json',dict(schema='qwen-device-policy-identity-source-handoff-v2',root=str(ROOT),source_count=6,
    source_map_sha256=sha((OUT/'sources.json').read_bytes()),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
    capacity_sha256=source_map['qwen_device_capacity.py'],policy_sha256=source_map['qwen_device_policy.py'],
    static_review_sha256=sha((OUT/'static-review.json').read_bytes()),patch_sha256=sha((OUT/'source.patch').read_bytes()),
    old5_unchanged=True,held_unexecuted=True,candidate_imports=False,tests=False,device_probes=False,actual_authority=False))
print(json.dumps({'handoff_sha256':sha((OUT/'handoff.json').read_bytes()),'map_sha256':sha((OUT/'sources.json').read_bytes()),
    'archive_sha256':sha((OUT/'sources.tar.gz').read_bytes()),'capacity_sha256':source_map['qwen_device_capacity.py'],
    'static_review_sha256':sha((OUT/'static-review.json').read_bytes())},sort_keys=True))
