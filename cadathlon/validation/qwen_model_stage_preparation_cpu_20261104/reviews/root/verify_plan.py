"""Read/replay public metadata only; no tensor paths or candidate imports."""
import ast
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PLAN = Path('/tmp/qwen-model-stage-aggregate-plan-20261104')
def sha(raw): return hashlib.sha256(raw).hexdigest()
def read(p): return json.loads(p.read_bytes())
def canon(v): return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

index = read(PLAN/'files.json')['files']
assert len(index) == 10
for name, digest in index.items():
    assert sha((PLAN/name).read_bytes()) == digest
assert index['PLAN.md'] == '29140aa94ae5a32d33c93fc134d443beb173287a60f7782c11fea9276a4f680c'
pins = read(PLAN/'evidence_pins.json')['pins']
for row in pins.values():
    assert sha(Path(row['path']).read_bytes()) == row['sha256']
script = (PLAN/'derive_public_metadata.py').read_bytes()
assert sha(script) == '72c9f24df77812ee616bde0b4c99eb1a4c8826fcfe96e734f214eb3a1399c039'
tree = ast.parse(script)
imports = {n.name for node in ast.walk(tree) if isinstance(node,ast.Import) for n in node.names}
imports |= {node.module for node in ast.walk(tree) if isinstance(node,ast.ImportFrom)}
assert imports == {'pathlib','ast','hashlib','json','math'}
copy = HERE/'replay/derive_public_metadata.py'
copy.write_bytes(script)
run = subprocess.run([sys.executable,'-I','-B',str(copy)],check=True,capture_output=True,text=True)
for name in ('unadmitted_content.json','unadmitted_catalog.json','unadmitted_aggregate_manifest.json','derivation.json'):
    assert (HERE/'replay'/name).read_bytes() == (PLAN/name).read_bytes()
content = read(PLAN/'unadmitted_content.json')
catalog = read(PLAN/'unadmitted_catalog.json')
manifest = read(PLAN/'unadmitted_aggregate_manifest.json')
status = read(PLAN/'derivation.json')
assert sha(canon(content)) == manifest['model_id'] == 'ee3daf8b07141134466d02a6e1c141365ac4e0f0117d62f887c416a050b04bfb'
assert sha(canon(catalog)) == manifest['catalog_sha256'] == '6faa4630169f055b853a62a64088952e2ead48dc2ace5b61b6aa438c0106a0c0'
assert manifest['content'] == content and manifest['catalog'] == catalog
assert len(content['matrices']) == len(catalog['matrices']) == 198
assert len(content['gammas']) == len(catalog['gammas']) == 113
physical = {(row['family'],row['descriptor']['file']):row['descriptor'] for row in catalog['matrices'].values()}
assert len(physical) == 197
assert sum(math.prod(row['shape'])*8 for row in physical.values()) == 4767875072
assert sum(math.prod(row['shape'])*8 for row in catalog['gammas'].values()) == 524288
assert set(content) == {'schema','checkpoint_sha256','config_sha256','matrices','gammas','logical_matrices','physical_matrices','gamma_vectors'}
schema = read(PLAN/'aggregate_manifest.schema.json')
assert schema['additionalProperties'] is False
assert set(schema['required']) == set(schema['properties']) == set(manifest)
assert all(rule == {'const':manifest[name]} for name,rule in schema['properties'].items())
admission = read(PLAN/'admission.schema.json')
assert admission['additionalProperties'] is False
assert set(admission['required']) == set(admission['properties']) == {'expected_pins','validation_report','validation_outer','root_review'}
assert status['actual_authorization'] is status['body_integrity_admitted'] is False
assert status['parameter_body_reads'] == status['runtime_imports'] == status['tests_run'] == 0
report = dict(schema='qwen-model-aggregate-root-plan-review-v1',public_plan_review_passed=True,
    plan_sha256=index['PLAN.md'],plan_index_sha256=sha((PLAN/'files.json').read_bytes()),
    public_evidence_pins_verified=len(pins),replay_returncode=run.returncode,
    model_id=manifest['model_id'],catalog_sha256=manifest['catalog_sha256'],
    aggregate_manifest_sha256=sha((PLAN/'unadmitted_aggregate_manifest.json').read_bytes()),
    script_sha256=sha(Path(__file__).read_bytes()),source_or_public_metadata_only=True,
    actual_execution_authorized=False,body_integrity_admitted=False,
    required_clarification='Pure authenticator has no archive bytes argument; independently trusted root/launch review must authenticate archive bytes/members and pure gate binds that receipt and exact digest.',
    next='Review source selection and held-identity acquisition hook before implementation; no tests or body reads authorized.')
(HERE/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
