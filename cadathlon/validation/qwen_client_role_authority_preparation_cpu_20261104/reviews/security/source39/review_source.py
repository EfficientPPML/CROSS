"""Source-only reviewer capture; imports no candidate or test modules."""
from pathlib import Path
import ast
import hashlib
import json
import tarfile

ROOT=Path('/tmp/qwen-client-role-authority-20261104')
HAND=Path('/tmp/qwen-client-role-authority-source-handoff-20261104')
OUT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
pretty=lambda v:(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
raw=(HAND/'sources.json').read_bytes();sources=json.loads(raw)
assert sha(raw)=='da3b9014385331b2efe3b1633cf331f44e69a9853feab687c41183e5b09154d9'
assert len(sources)==39
current={p.relative_to(ROOT).as_posix():sha(p.read_bytes())for p in sorted(ROOT.rglob('*'))if p.is_file()}
assert current==sources
assert not any(p.is_symlink()for p in ROOT.rglob('*'))
with tarfile.open(HAND/'sources.tar.gz')as tar:
    members=tar.getmembers()
    assert len(members)==len({m.name for m in members})==39 and all(m.isfile()for m in members)
    assert {m.name:sha(tar.extractfile(m).read())for m in members}==sources
parent=json.loads((ROOT/'authority28_v2_sources.json').read_bytes())
assert len(parent)==28 and all(sources[n]==h for n,h in parent.items())
assert {p.relative_to(Path('/tmp/qwen-keygen-authority-boundary-v2-20261104')).as_posix():sha(p.read_bytes())for p in Path('/tmp/qwen-keygen-authority-boundary-v2-20261104').rglob('*')if p.is_file()}==parent
target=['client_role_admission.py','client_role_common.py','client_server_terminal.py','client_role_source_pins.py',
        'qwen_file_contract.py','keygen_authority.py','model_admission.py','CLIENT_AUTHORITY_DESIGN.md','client_source_selection.json']
pins_ast=ast.parse((ROOT/'client_role_source_pins.py').read_text())
values={n.targets[0].id:ast.literal_eval(n.value)for n in pins_ast.body if isinstance(n,ast.Assign)}
assert len(values['FIXED_CLIENT_SOURCES'])==93 and len(values['ACTUAL_CLIENT_MODULES'])==14
for name in target:
    if name.endswith('.py'):ast.parse((ROOT/name).read_text())
for name in ['client_role_admission.py','client_server_terminal.py']:
    tree=ast.parse((ROOT/name).read_text())
    for node in tree.body:
        if isinstance(node,ast.FunctionDef)and node.name.startswith('create_actual_'):
            assert len(node.body)==1 and isinstance(node.body[0],ast.Raise)
result=dict(schema='qwen-client-authority-security-source-review-v1',passed=True,source_only=True,
    source_manifest_sha256=sha(raw),source_archive_sha256=sha((HAND/'sources.tar.gz').read_bytes()),
    source_count=39,exact_parent28=True,fixed_prerequisites=93,required_actual_names=14,
    pins={n:sources[n]for n in target},
    findings=[],conclusions=[
        'Conditional role proof binds the concrete generation intent/source but no later keygen grant digest; it grants no immediate body/context authority.',
        'Conditional and phase gates reauthenticate ModelAdmission for the exact selected client consumer; phase reauthenticates GeneratedKeyReadiness and matches original generation intent/map/archive and role grant.',
        'Separate server execution map is bound by exact envelope/outer/root evidence; historical inner producer fields remain request provenance, not server execution identity.',
        'Closed schemas, exact integer checks and canonical-byte comparison retain bool-versus-int distinctions and bounded immutable public inputs.',
        'Actual client source requires93 fixed prerequisites plus14 actual names; current pure39 lacks the full actual implementation and its entry stubs deny.',
        'No concrete blocker found in this bounded source review; public proof bytes and expected pins remain independently trusted caller inputs, not filesystem or process proof by themselves.'
    ],limitations=['No tests, imports or body/current-store reads','No actual grant or phase execution','Actual policy-byte admission, source resolution, owned-parent/body validation and dependency/resource enforcement remain future process work'],
    review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_bytes(pretty(result))
print(json.dumps(dict(passed=True,receipt_sha256=sha(pretty(result)),sources=39,parent=28),sort_keys=True))
