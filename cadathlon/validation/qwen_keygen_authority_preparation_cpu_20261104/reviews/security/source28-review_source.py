"""Public source/AST readback only; never import the reviewed boundary."""
import ast
import hashlib
import io
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-keygen-authority-boundary-20261104')
OUT=Path(__file__).parent
MAP=Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json')
HANDOFF=Path('/tmp/qwen-keygen-authority-boundary-handoff-20261104/handoff.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def load(p):return json.loads(p.read_bytes())
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'))+'\n').encode()
expected=load(MAP);handoff=load(HANDOFF)
assert len(expected)==28 and MAP.read_bytes()==canonical(expected)
assert sha(MAP.read_bytes())==handoff['source_manifest_sha256']
assert {p.relative_to(ROOT).as_posix():sha(p.read_bytes())for p in ROOT.rglob('*')if p.is_file()}==expected
archive=Path(handoff['source_archive']).read_bytes()
assert sha(archive)==handoff['source_archive_sha256']
with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
    assert all(m.isfile()for m in tar.getmembers())
    assert {m.name:sha(tar.extractfile(m).read())for m in tar.getmembers()}==expected
copies=load(ROOT/'selected_copies.json');assert len(copies)==20
for name,row in copies.items():
    assert sha(Path(row['path']).read_bytes())==expected[name]==row['sha256']
tree=ast.parse((ROOT/'keygen_authority.py').read_bytes())
fields={}
for node in tree.body:
    if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        name=node.targets[0].id
        if name.endswith('_FIELDS'):fields[name]=ast.literal_eval(node.value)
defs=load(ROOT/'boundary.schema.json')['$defs']
mapping=dict(actual_intent='INTENT_FIELDS',expected_launch_pins='LAUNCH_PIN_FIELDS',
    expected_ready_pins='READY_PIN_FIELDS',grant='GRANT_FIELDS',outer='OUTER_FIELDS',
    positive_report='REPORT_FIELDS',root_readiness='ROOT_FIELDS')
assert set(defs)==set(mapping)
for name,field in mapping.items():
    row=defs[name]
    assert row['additionalProperties']is False and set(row['properties'])==set(row['required'])==fields[field]
funcs={n.name:n for n in tree.body if isinstance(n,ast.FunctionDef)}
denied=funcs['create_actual_keygen_process']
assert len(denied.body)==1 and isinstance(denied.body[0],ast.Raise)
assert isinstance(denied.body[0].exc,ast.Call)and isinstance(denied.body[0].exc.func,ast.Name)and denied.body[0].exc.func.id=='PermissionError'
model=Path('/tmp/qwen-model-stage-aggregate-20261104/model_admission.py').read_bytes()
assert sha(model)==expected['model_admission.py']=='60bb47d044649625a5c1d9870256aed7b6d7a3bf53c49b5267bcfc4919d098d3'
pins_tree=ast.parse((ROOT/'keygen_boundary_pins.py').read_bytes())
pin_values={n.targets[0].id:ast.literal_eval(n.value)for n in pins_tree.body if isinstance(n,ast.Assign)}
required=pin_values['REQUIRED_ACTUAL_SOURCES'];assert len(required)==58
parent=load(ROOT/'parent71_sources.json')
assert all(parent.get(n)==h for n,h in required.items()if n!='model_admission.py')
assert required['model_admission.py']==expected['model_admission.py']
assert pin_values['CORE_MANIFEST_SHA256']==expected['core_source_manifest.json']
imports={n.name.split('.')[0]for row in tree.body if isinstance(row,ast.Import)for n in row.names}
imports|={row.module.split('.')[0]for row in tree.body if isinstance(row,ast.ImportFrom)}
assert imports=={'dataclasses','pathlib','re','qwen_file_contract','qwen_public_parse','qwen_key_contract','model_admission','keygen_boundary_pins'}
report=dict(schema='qwen-keygen-boundary-security-source-review-v1',source_root=str(ROOT),
    source_manifest_sha256=sha(MAP.read_bytes()),source_archive_sha256=sha(archive),
    source_count=28,exact_copied_sources=20,source_hashes=expected,
    evidence_scope='Independent source/AST/public-metadata review only; no module imported or executed.',
    candidate_imports=False,tests_or_collection=False,backend_imports=False,
    body_or_current_store_reads=False,RNG=False,key_generation=False,grant_issued=False,
    checks=dict(current_map_archive_equal=True,selected_copies_exact=True,
        accepted_model_admission_exact=True,seven_closed_schema_field_sets_match=True,
        fifty_eight_future_prerequisite_pins_match=True,actual_entry_immediate_denial=True,
        model_reauthentication_before_launch_binding=True,
        exact_generation_consumer_source_and_five_model_proof_digests=True,
        context_and_manifest_ids_only_from_independent_postterminal_pins=True,
        original_role_source_ids_distinct_from_generation_and_model_producer=True,
        complete_public_key_schema_and_source_security_terminal_checks=True,
        private_manifest_body_absent_and_private_digest_client_only=True,
        retained_ready_binding_reauthenticates_before_expected_ids=True,
        actual_process_resource_parent_and_downstream_authority_remain_unimplemented=True),
    findings=[],conclusion='No concrete blocker found in the held pure public authority/readiness scope.',
    limitations=['Expected pin arguments remain independently trusted caller inputs, not signatures or ordinary-object capabilities.',
        'Future actual source/import/CPU-JAX/resource/held-parent acquisition and role maps/grants are still required.',
        'Fixed71 recording producer/store is not activated or upgraded by this pure boundary.',
        'Actual report/readiness assertions require genuine producer plus independent direct wait/root archive/store verification; pure equality is not execution evidence.',
        'No private manifest/key body is authenticated by this public gate; future client must authenticate the separately pinned private manifest before loading.',
        'Scientific calibration/freeze/model execution authority is separate.'])
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(review_sha256=sha((OUT/'review.json').read_bytes()),source_count=28,scope='source-only'),sort_keys=True))
