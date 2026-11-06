"""Verify only the clarified public plan and declared source references."""
import hashlib,json
from pathlib import Path
OLD=Path('/tmp/qwen-model-stage-aggregate-plan-20261104')
NEW=Path('/tmp/qwen-model-stage-aggregate-plan-v2-20261104')
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
for root in (OLD,NEW):
    for name,digest in read(root/'files.json')['files'].items():assert sha(root/name)==digest
selection=read(NEW/'source_selection.json')
assert len(selection['base_modules'])==8 and len(selection['base_public_metadata'])==12
for group in ('base_modules','base_public_metadata'):
    for row in selection[group].values():assert sha(Path(row['path']))==row['sha256']
for name in ('unadmitted_content.json','unadmitted_catalog.json','unadmitted_aggregate_manifest.json','aggregate_manifest.schema.json','derivation.json'):
    assert (OLD/name).read_bytes()==(NEW/name).read_bytes()
old=read(OLD/'admission.schema.json');new=read(NEW/'admission.schema.json')
review=new['properties']['root_review']
assert review['properties'].pop('source_archive_members_verified')=={'const':True}
review['required'].remove('source_archive_members_verified')
assert new['description'] == old['description'] + ' The pure gate never opens an archive: independently trusted root review verifies bounded archive bytes/members; the pure gate binds its pinned digest, source map and receipt flag.'
new['description']=old['description']
assert new==old
report=dict(schema='qwen-model-aggregate-root-clarification-review-v1',passed=True,
    parent_review_sha256=sha(HERE/'review.json'),plan_sha256=sha(NEW/'PLAN.md'),
    plan_index_sha256=sha(NEW/'files.json'),selection_sha256=sha(NEW/'source_selection.json'),
    held_identity_plan_sha256=sha(NEW/'HELD_IDENTITY.md'),
    schema_sha256=sha(NEW/'admission.schema.json'),base_sources=8,public_metadata=12,
    content_catalog_manifest_unchanged=True,only_schema_change='trusted root archive member verification flag plus explanatory description',
    script_sha256=sha(Path(__file__)),source_implementation_authorized=True,
    test_execution_authorized=False,actual_execution_authorized=False,
    scope='New isolated authenticator/validator source plus declared held-identity overlay only; old closures unchanged. Source review precedes tests or body reads.')
(HERE/'clarification-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(report,sort_keys=True))
