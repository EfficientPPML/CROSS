"""Stdlib-only public source/tar/AST review; never import or collect the target."""
import ast,hashlib,json,tarfile
from pathlib import Path
ROOT=Path('/tmp/qwen-keygen-actual-runtime-tests-20261104')
HAND=Path('/tmp/qwen-keygen-actual-runtime-test-handoff-20261104')
PARENT=Path('/tmp/qwen-keygen-actual-runtime-v2-20261104')
OUT=Path(__file__).parent
sha=lambda b:hashlib.sha256(b).hexdigest()
can=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
raw=(HAND/'sources.json').read_bytes();manifest=json.loads(raw)
assert sha(raw)=='59d8d58cd42d6beb8265b632b9c08277de2d913f5f0190364ca2d40689e3d8d3'
assert len(manifest)==97
before={n:sha((ROOT/n).read_bytes())for n in manifest}
assert before==manifest
assert {p.relative_to(ROOT).as_posix()for p in ROOT.rglob('*')if p.is_file()}==set(manifest)
assert not any(p.is_symlink()for p in ROOT.rglob('*'))
parent=json.loads((ROOT/'parent85_sources.json').read_bytes())
assert len(parent)==85
for name,pin in parent.items():assert manifest['runtime/'+name]==pin==sha((PARENT/name).read_bytes())
tar_raw=(HAND/'sources.tar.gz').read_bytes()
assert sha(tar_raw)=='11a7c027a38452239fd164520e0789423054147f01aae38953cbfc54c576999f'
with tarfile.open(HAND/'sources.tar.gz')as archive:
    members=archive.getmembers();assert all(m.isfile() for m in members)
    archived={m.name:sha(archive.extractfile(m).read())for m in members}
assert archived==manifest
selection=json.loads((ROOT/'selected_tests.json').read_bytes());handoff=json.loads((HAND/'handoff.json').read_bytes())
nodes=[];counts={}
for name in selection['selected_files']:
    tree=ast.parse((ROOT/name).read_bytes());file_nodes=[]
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef)or not node.name.startswith('test_'):continue
        values=[None]
        for deco in node.decorator_list:
            if isinstance(deco,ast.Call)and isinstance(deco.func,ast.Attribute)and deco.func.attr=='parametrize':
                assert len(values)==1 and values[0]is None
                values=ast.literal_eval(deco.args[1]);assert all(type(v)is str for v in values)
        file_nodes.extend(name+'::'+node.name+(''if v is None else '['+v+']')for v in values)
    counts[name]=len(file_nodes);nodes.extend(file_nodes)
assert nodes==selection['selected_nodes']==handoff['selected_nodes'] and len(nodes)==len(set(nodes))==61
assert counts==selection['per_file']==handoff['per_file']
xml=[{'classname':n.split('::')[0][:-3],'name':n.split('::')[1]}for n in nodes]
assert xml==selection['selected_xml_cases']==handoff['selected_xml_cases']
assert selection['collected']==selection['executed']==0 and selection['actual_execution_authorized']is False
assert selection['cpu_proposal']==[112,113,114,115]
assert sha((ROOT/'selected_tests.json').read_bytes())==handoff['selected_tests_sha256']
assert json.loads((ROOT/'runtime/keygen_entry_status.json').read_bytes())['actual_entry_ready']is False
for name in ('keygen_entry_status.json','keygen_static_policy.json'):
    old=(ROOT/'history'/name).read_bytes();new=(ROOT/'runtime'/name).read_bytes()
    assert old!=new and json.loads(old)==json.loads(new)
after={n:sha((ROOT/n).read_bytes())for n in manifest};assert after==before
report=dict(schema='qwen-keygen-runtime-focused-peer-source-review-v1',passed=True,source_only=True,
 source_root=str(ROOT),source_count=97,source_manifest_sha256=sha(raw),source_archive_sha256=sha(tar_raw),
 handoff_sha256=sha((HAND/'handoff.json').read_bytes()),source_hashes=before,parent85_exact=True,
 static_node_count=61,per_file=counts,static_selected_nodes=nodes,source_before_after_exact=True,
 reviewer_script_sha256=sha(Path(__file__).read_bytes()),
 source_read_targets=['test_keygen_output.py','test_keygen_policy.py','test_keygen_generation.py','test_keygen_entry.py','fixtures_keygen.py','conftest.py','TEST_SCOPE.md','test_origin.json','runtime/keygen_output.py','runtime/keygen_process.py','runtime/keygen_dependencies.py','runtime/keygen_entry.py','runtime/qwen_finalizer.py'],
 findings=[],
 observations=[
 'External immutable parent negatives precede open; real newly owned parent replacement checks rejection before leaf creation and original descriptor retirement.',
 'Policy bytes are independently repinned after semantic mutation; backend origin and capacity observations are explicit fake mappings/kernel observations, not actual readiness.',
 'Actual generate body uses recorded168 calls and335 metadata-only writes; nested degree2 Python-list secret identity is checked; real objects are tiny newly owned directories/manifests/FDs.',
 'Construction/body/accessor/cleanup cases exercise original captured owners; body identity and later cleanup-only cancellation priority have distinct assertions.',
 'Entry source/expected-pin tests deliberately stub later authority types and do not claim positive complete authority admission.',
 'One future actual isolated CLI case exercises default denial with exact nested85 compact source map; communicate failure cleanup kills only its owned group then direct-waits, without a healthy deadline.',
 'Three declared run-flow cases temporarily install explicit ModuleType jax/numpy doubles inside monkeypatch.context and restore them before the autouse forbidden-origin check; all real security/generation/capacity/signal seams are disclosed.',
 'Report publication is observed before late audit/close failures; terminal_status must return70, while clean recording flow returns0.'],
 limitations=['No imports, pytest collection, test execution or runtime API calls by this review.',
 'No actual backend, NumPy, RNG, body access, security/capacity acquisition, generation, model integrity or execution grant.',
 'Source reasoning finds no concrete scope blocker; execution and a reviewed independent harness remain necessary.',
 'No claim that these cases execute the older pure authority positive suite or real key/header/residue authentication.'])
(OUT/'review.json').write_bytes(can(report))
print(json.dumps(dict(passed=True,source_count=97,static_nodes=61,review_sha256=sha(can(report)),imports=False,tests=False)))
