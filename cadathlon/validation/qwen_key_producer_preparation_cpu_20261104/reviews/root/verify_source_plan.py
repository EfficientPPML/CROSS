"""Root public source/AST verification only; never import candidate code."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
HELD = Path('/tmp/qwen-key-generation-producer-20261104')
PLAN = Path('/tmp/qwen-key-writer-unbuffered-plan-20261104')
MAP = Path('/tmp/qwen-key-producer-review-sources-v1-20261104.json')
SECURITY = Path('/tmp/qwen-key-producer-security-review-20261104/review.json')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(path): return json.loads(path.read_bytes())

assert sha(MAP.read_bytes()) == 'd072e2815aaa6ecb7dc3d0853381c7888c13e44f4fac82ba8d7ec1067f2dcede'
expected = read(MAP)
assert len(expected) == 64
assert not any(p.is_symlink() for p in HELD.rglob('*'))
assert {p.relative_to(HELD).as_posix(): sha(p.read_bytes())
        for p in HELD.rglob('*') if p.is_file()} == expected
selected = read(HELD/'selected_sources.json')['files']
assert len(selected) == 57
for name, row in selected.items():
    assert sha(Path(row['path']).read_bytes()) == expected[name] == row['sha256']
assert sha(SECURITY.read_bytes()) == 'fc9f6144966d26de4192fc533de0418cb5e3ef2bc7dd163cc244280853576d25'
for name, digest in read(PLAN/'files.sha256.json').items():
    assert sha((PLAN/name).read_bytes()) == digest
refs = read(PLAN/'source_refs.json')
for path, digest in refs['refs'].items():
    assert sha(Path(path).read_bytes()) == digest
assert sha((PLAN/'PLAN.md').read_bytes()) == 'd0d8b77de62f289e2a1bf75da06e5807786ff5290ef1dfb4acfa23eb4e3f4437'
assert sha((PLAN/'proposed.patch.txt').read_bytes()) == refs['patch_sha256'] == '7a23c09af7983ebed08b0749c9e47b57148198ada9621963a8392dcd100253ad'
# Read only literal replacement text from the author's proposal builder.
literal = {}
for node in ast.parse((PLAN/'make_proposal.py').read_bytes()).body:
    if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
        if node.targets[0].id in ('helper', 'anchor', 'before', 'after'):
            literal[node.targets[0].id] = ast.literal_eval(node.value)
old = (HELD/'qwen_key_store.py').read_text()
assert old.count(literal['anchor']) == old.count(literal['before']) == 1
proposed = old.replace(literal['anchor'], literal['helper']+literal['anchor']).replace(literal['before'], literal['after'])
assert sha(proposed.encode()) == refs['proposed_store_sha256'] == '2e60908c9773debaa85a5d325df1367013ffb99c6ddead2f587dafca78efb0bd'
tree = ast.parse(proposed)
sink = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == '_NpySink')
assert not sink.bases
assert {n.name for n in sink.body if isinstance(n, ast.FunctionDef)} == {'__init__', '_live', 'write', 'invalidate'}
for filename, entry in [('qwen_key_store.py', 'create_actual_key_store'), ('qwen_key_producer.py', 'create_actual_key_producer')]:
    fn = next(n for n in ast.parse((HELD/filename).read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == entry)
    assert len(fn.body) == 1 and isinstance(fn.body[0], ast.Raise)
    assert fn.body[0].exc.func.id == 'PermissionError'
report = dict(schema='qwen-key-producer-root-source-plan-review-v1', source_count=64,
    selected_exact_copies=57, held_source_map_sha256=sha(MAP.read_bytes()),
    security_source_review_sha256=sha(SECURITY.read_bytes()),
    plan_sha256=sha((PLAN/'PLAN.md').read_bytes()), patch_sha256=refs['patch_sha256'],
    proposed_store_sha256=refs['proposed_store_sha256'],
    script_sha256=sha(Path(__file__).read_bytes()),
    source_plan_accepted=True, tests_executed=0, actual_execution_authorized=False,
    scope='Public source/AST only. New successor source and focused tests may be authored; execution awaits separate source/test review.',
    disposition='Replace uncertain buffered writer handoff with synchronous bounded sink; original raw descriptor remains sole owner. No demonstrated old ordinary-flush corruption. Byte compatibility and fault behavior remain untested.')
(HERE/'source-plan-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
