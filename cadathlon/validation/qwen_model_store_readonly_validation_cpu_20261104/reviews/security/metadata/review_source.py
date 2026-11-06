"""Static/public-only review. Never import or execute acquisition code."""
import ast
import hashlib
import json
from pathlib import Path

ROOT=Path('/tmp/qwen-model-store-metadata-acquisition-20261104')
OUT=Path(__file__).parent
def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path): return json.loads(path.read_bytes())
def canonical(value): return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

index=load(ROOT/'files.json')
assert {n:sha((ROOT/n).read_bytes())for n in index}==index
pins=load(ROOT/'source_pins.json');proposal=load(ROOT/'proposal.json')
script=(ROOT/'acquire_expectations.py').read_bytes()
assert sha(script)==proposal['script_sha256']==pins['script']['sha256']
assert sha((ROOT/'source_pins.json').read_bytes())==proposal['source_pins_sha256']
assert proposal['executed']is False and proposal['current_store_metadata_acquired']is False
public={}
for name,row in pins['public_inputs'].items():
    # Only three explicitly pinned source/public JSON documents are followed.
    p=Path(row['path']);raw=p.read_bytes()
    assert sha(raw)==row['sha256'] and len(raw)<=row['maximum_bytes']
    public[name]=json.loads(raw)
catalog=public['catalog'];aggregate=public['aggregate'];layout=public['layout_reference']['public_parameters']
assert sha(canonical(catalog))=='6faa4630169f055b853a62a64088952e2ead48dc2ace5b61b6aa438c0106a0c0'
assert aggregate['catalog']==catalog and sha(canonical(aggregate['content']))==aggregate['model_id']
tree=ast.parse(script)
constants={}
for node in tree.body:
    if isinstance(node,ast.Assign)and len(node.targets)==1 and isinstance(node.targets[0],ast.Name):
        if node.targets[0].id in ('ROOTS','GAMMA_NAME','OUTPUT'):
            constants[node.targets[0].id]=ast.literal_eval(node.value)
roots=constants['ROOTS'];gamma=constants['GAMMA_NAME']
assert roots==pins['expected_roots']==proposal['root_paths']
assert roots['qkv']==layout['qkv_directory'] and roots['decoder']==layout['decoder_directory']
assert roots['gamma']+'/'+gamma==layout['gamma_path'] and catalog['gamma_container_sha256']==layout['gamma_sha256']
physical={}
for row in catalog['matrices'].values():
    d=row['descriptor'];key=(row['family'],d['file'])
    selected=(d['shape'],d['file_sha256'],d['sha256'])
    if key in physical: assert physical[key]==selected
    physical[key]=selected
assert len(catalog['matrices'])==198 and len(physical)==197 and len(catalog['gammas'])==113
names={family:sorted(name for fam,name in physical if fam==family)for family in ('qkv','decoder')}
names['gamma']=[gamma]
planned=load(ROOT/'planned_names.json')
assert planned['files']==names and planned['current_metadata_observed']is False
assert [len(names[n])for n in ('qkv','decoder','gamma')]==[84,113,1]
cleanup=Path('/tmp/qwen-model-stage-aggregate-20261104/cleanup_origin.py')
cleanup_raw=cleanup.read_bytes();assert sha(cleanup_raw)=='4543b26bf1168e29bc21375c2f80a9aa99648205a9fd9aea6d5a8059c6d18175'
def functions(t):return {n.name:n for n in t.body if isinstance(n,ast.FunctionDef)}
f=functions(tree);shared=functions(ast.parse(cleanup_raw))
for name in ('is_cleanup','protect_body','mark','select'):
    assert ast.dump(f[name],include_attributes=False)==ast.dump(shared[name],include_attributes=False)
imports={n.name.split('.')[0]for row in tree.body if isinstance(row,ast.Import)for n in row.names}
imports|={row.module.split('.')[0]for row in tree.body if isinstance(row,ast.ImportFrom)}
assert imports=={'contextlib','pathlib','argparse','hashlib','json','math','os','stat','sys','time'}
for name in ('observe','ancestors'):
    os_calls=[n.func.attr for n in ast.walk(f[name])if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and isinstance(n.func.value,ast.Name)and n.func.value.id=='os']
    assert os_calls and set(os_calls)=={'lstat'}
    assert not any(isinstance(n,ast.Attribute)and n.attr in ('open','read','read_bytes','scandir','glob','rglob','iterdir')for n in ast.walk(f[name]))
assert sum(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='observe'for n in ast.walk(f['run']))==2
report=dict(schema='qwen-metadata-acquisition-security-source-review-v1',
    source_only=True,acquisition_executed=False,candidate_imports=False,tests_executed=False,
    borrowed_metadata_inspected=False,borrowed_body_reads=False,
    files_index_sha256=sha((ROOT/'files.json').read_bytes()),source_hashes=index,
    public_inputs=pins['public_inputs'],shared_cleanup_sha256=sha(cleanup_raw),
    checks=dict(exact_held_files=True,three_roots_from_pinned_public_provenance=True,
        exact_names_84_113_1=True,canonical_ExpectedStoreIdentity_field_shape=True,
        fixed_hashes_copied_not_recomputed=True,lstat_only_known_borrowed_names=True,
        two_observation_calls=True,no_borrowed_directory_harvest=True,
        symlink_and_duplicate_identity_rejection=True,extent_matches_accepted_consumer=True,
        shared_cleanup_four_functions_AST_exact=True,exclusive_original_descriptor_write_readback_fsync=True,
        final_escaped_failure_maps70=True,no_integrity_or_read_grant_claim=True),
    finding=dict(id='explicit_output_ownership',severity='narrow source-contract gap',
        observation='Parent/leaf/file identity and modes are checked, but parent UID/sticky trust and fresh leaf/file UID are not explicitly enforced.',
        recommendation='In a separate source successor add parent uid0-or-euid and sticky-or-nonshared trust checks, fresh directory/file euid checks, and preserve those checks through final publication.',
        runtime_failure_demonstrated=False),
    conclusion='Fixed metadata derivation/access and terminal scope align. One explicit output-ownership check remains recommended before eventual metadata execution; no current store observation was made.',
    limits=['Repeated named metadata is not an atomic or body-integrity proof.','Read denial hook is supplemental trusted-source protection, not an OS sandbox.','Independent actual exit0/reap/group absence and output hashes remain prerequisites for root use.'])
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(review_sha256=sha((OUT/'review.json').read_bytes()),source=sha(script),known_names=sum(map(len,names.values())),scope='public source/AST only'),sort_keys=True))
