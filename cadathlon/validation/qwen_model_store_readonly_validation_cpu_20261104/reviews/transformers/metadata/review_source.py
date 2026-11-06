"""Source-only peer readback; never imports or invokes acquisition code."""
import ast
import difflib
import hashlib
import json
from pathlib import Path

OLD = Path('/tmp/qwen-model-store-metadata-acquisition-20261104')
NEW = Path('/tmp/qwen-model-store-metadata-acquisition-v2-20261104')
OUT = Path('/tmp/qwen-model-metadata-transformers-review-20261104')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_index(root):
    rows = json.loads((root/'files.json').read_bytes())
    for name, expected in rows.items():
        assert Path(name).name == name
        assert sha(root/name) == expected
    return rows | {'files.json': sha(root/'files.json')}


old_map = source_index(OLD)
new_map = source_index(NEW)
assert len(old_map) == 6 and len(new_map) == 8
assert old_map['acquire_expectations.py'] == '81d7d829b539ab2d837a2c2465c8ef0ca9919e6f3ab02988b018bd8abba0c043'
assert new_map['acquire_expectations.py'] == '8346d29375ead391fedf40c2da18fd2241c4e246052211ff37986741b4556285'
assert new_map['files.json'] == 'ffb6ae04bf51302b265698fcf4c6d708fde0fa9823de607ad7b5c600c5868a15'
assert old_map['planned_names.json'] == new_map['planned_names.json']
old_text = (OLD/'acquire_expectations.py').read_text()
new_text = (NEW/'acquire_expectations.py').read_text()
actual_patch = ''.join(difflib.unified_diff(old_text.splitlines(True), new_text.splitlines(True),
                                          fromfile='held81d7/acquire_expectations.py',
                                          tofile='successor/acquire_expectations.py'))
assert actual_patch == (NEW/'ownership.patch.txt').read_text()

old_tree = ast.parse(old_text)
new_tree = ast.parse(new_text)
dump = lambda node: ast.dump(node, include_attributes=False)
old_defs = {n.name:n for n in old_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
new_defs = {n.name:n for n in new_tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
assert old_defs.keys() == new_defs.keys()
changed = sorted(n for n in old_defs if dump(old_defs[n]) != dump(new_defs[n]))
assert changed == ['run', 'write_new']
outside_old = [n for n in old_tree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]
outside_new = [n for n in new_tree.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]
assert [dump(n) for n in outside_old] == [dump(n) for n in outside_new]
boundary = '    # Only a fresh evidence leaf is created; existing parent may have protected siblings.\n'
def prefix(text):
    return text[text.index('def run(args):'):text.index(boundary)]
assert prefix(old_text) == prefix(new_text)
for name in ('run',):
    calls = [n for n in ast.walk(new_defs[name]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id == 'observe']
    assert len(calls) == 2
    assert all(len(n.args)==1 and isinstance(n.args[0],ast.Name) and n.args[0].id=='rows' for n in calls)
provenance_old = next(n.value for n in ast.walk(old_defs['run']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='provenance' for t in n.targets))
provenance_new = next(n.value for n in ast.walk(new_defs['run']) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='provenance' for t in n.targets))
assert dump(provenance_old) == dump(provenance_new)

old_pins = json.loads((OLD/'source_pins.json').read_bytes())
new_pins = json.loads((NEW/'source_pins.json').read_bytes())
assert {k:v for k,v in old_pins.items() if k!='script'} == {k:v for k,v in new_pins.items() if k!='script'}
assert new_pins['script'] == dict(path=str(NEW/'acquire_expectations.py'),sha256=new_map['acquire_expectations.py'])
proposal = json.loads((NEW/'proposal.json').read_bytes())
assert proposal['actual_execution_authorized'] is False and proposal['executed'] is False
assert proposal['current_store_metadata_acquired'] is False
assert proposal['source_pins_sha256'] == new_map['source_pins.json']
assert proposal['script_sha256'] == new_map['acquire_expectations.py']
assert '-I' in proposal['argv'] and '-B' in proposal['argv']
assert source_index(OLD) == old_map and source_index(NEW) == new_map

report = dict(schema='qwen-model-metadata-ownership-peer-source-review-v1',passed=True,
    reviewer='transformers',scope='source-only ownership delta and unchanged metadata acquisition surface',
    original_root=str(OLD),successor_root=str(NEW),original_sources=old_map,successor_sources=new_map,
    changed_functions=changed,exact_patch_verified=True,planned_names_and_public_inputs_unchanged=True,
    fixed_roots_catalog_model_output_constants_unchanged=True,two_metadata_passes_and_prepublication_source_rechecks_unchanged=True,
    expected_store_schema_and_provenance_schema_unchanged=True,
    checks=[
      'Original held and named output parent are compared by device/inode/mode/UID before mkdir and through final publication.',
      'Parent owner must be root or current UID; group/other writable parent requires sticky bit. Ordinary existing /tmp with protected siblings is allowed.',
      'Fresh directory and exclusive output files require current UID; original held/named identities and UIDs are checked through publication/readback.',
      'No new borrowed-directory/file open or enumeration: unchanged observe/inventory/ancestors use known named lstat only; output additions inspect only owned output and parent.',
      'Shared cleanup-origin helpers, detached single-close wrapper, audit body-open denial and terminal70 boundary are unchanged.',
      'Hashes in expected stores remain declared catalog hashes, not recomputed body digests; output is not model or execution admission.'
    ],
    unresolved_concrete_blockers=[],candidate_imported=False,candidate_invoked=False,help_invoked=False,
    tests_or_collection=False,current_store_stat=False,borrowed_body_read=False,grant_issued=False,
    limits=['No runtime or metadata acquisition executed by reviewer.',
            'Two matching lstat passes are not an atomic snapshot; later original-held readers must reject changed expectations.',
            'Root must authorize the exact command and independently admit normal exit0/reap and output digests before use.'],
    review_script_sha256=sha(Path(__file__)))
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,review_sha256=sha(OUT/'review.json'),changed_functions=changed,source_only=True)))
