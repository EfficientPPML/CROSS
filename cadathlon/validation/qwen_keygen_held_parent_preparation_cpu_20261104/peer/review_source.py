"""Public source/AST readback only; no candidate import or test execution."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

OUT=Path('/tmp/qwen-held-parent-tests-transformers-review-20261104')
ROOT=Path('/tmp/qwen-key-generation-held-parent-tests-20261104')
PARENT=Path('/tmp/qwen-key-generation-held-parent-20261104')
MAP=Path('/tmp/qwen-keygen-held-parent-test-sources-v1-20261104.json')
PMAP=Path('/tmp/qwen-keygen-held-parent-sources-v1-20261104.json')
HANDOFF=Path('/tmp/qwen-keygen-held-parent-test-handoff-20261104/handoff.json')
TAR=Path('/tmp/qwen-keygen-held-parent-test-handoff-20261104/sources.tar.gz')
def sha(raw):return hashlib.sha256(raw).hexdigest()
assert sha(MAP.read_bytes())=='50625b443ad26a88907e37ed901618b55fe3f5370ae96504b49dbaa0ca6974a1'
assert sha(PMAP.read_bytes())=='7ffa419573da6100462fbc7d8ab778975c11d37fba4e551bd86c3b92d448a4b9'
assert sha(HANDOFF.read_bytes())=='ec6a8e28c5c16442d0212fc6f02dfe7e51418eafe5193cec694d4ed167417591'
assert sha(TAR.read_bytes())=='93961105185d6c22566f8fbd315f392a75e6bdfc114dc7937c3721279d17ccbc'
expected=json.loads(MAP.read_bytes());parent=json.loads(PMAP.read_bytes());handoff=json.loads(HANDOFF.read_bytes())
assert len(expected)==83 and len(parent)==74
def read_tree(root):
    result={}
    for path in root.rglob('*'):
        assert not path.is_symlink(),path
        if path.is_file():result[str(path.relative_to(root))]=sha(path.read_bytes())
        else:assert path.is_dir(),path
    return result
assert read_tree(ROOT)==expected
assert read_tree(PARENT)==parent
assert {n[8:]:h for n,h in expected.items()if n.startswith('runtime/')}==parent
added=set(expected)-{'runtime/'+n for n in parent}
assert added=={'conftest.py','parent_test_helpers.py','test_parent_admission.py','test_parent_policy.py','test_parent_lifetime.py','TEST_SCOPE.md','test_origin.json','READY_TESTS.json','selected_parent_tests.json'}
archive={}
with tarfile.open(TAR,'r:gz')as tar:
    for member in tar.getmembers():
        assert member.isfile() and not member.issym() and not member.islnk(),member.name
        assert member.name not in archive,member.name
        archive[member.name]=sha(tar.extractfile(member).read())
assert archive==expected
selection=json.loads((ROOT/'selected_parent_tests.json').read_bytes())
files=['test_parent_admission.py','test_parent_policy.py','test_parent_lifetime.py']
assert selection['selected_files']==handoff['selected_files']==files
nodes=[];xml=[];counts={}
for name in files:
    tree=ast.parse((ROOT/name).read_bytes(),filename=name);file_nodes=[]
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef)or not fn.name.startswith('test_'):continue
        params=[]
        for dec in fn.decorator_list:
            assert isinstance(dec,ast.Call) and isinstance(dec.func,ast.Attribute)and dec.func.attr=='parametrize'
            assert len(dec.args)==2 and not dec.keywords
            assert isinstance(ast.literal_eval(dec.args[0]),str)
            values=ast.literal_eval(dec.args[1]);assert isinstance(values,list)
            assert not params;params=values
        suffixes=['']if not params else['['+str(v)+']'for v in params]
        for suffix in suffixes:
            item=fn.name+suffix;file_nodes.append(name+'::'+item)
            xml.append(dict(classname=name[:-3],name=item))
    counts[name]=len(file_nodes);nodes.extend(file_nodes)
assert counts=={'test_parent_admission.py':18,'test_parent_policy.py':5,'test_parent_lifetime.py':11}
assert len(nodes)==34 and len(set(nodes))==34
assert nodes==selection['ordered_nodes']==handoff['ordered_nodes']
assert xml==selection['ordered_xml_cases']==handoff['ordered_xml_cases']
for name in added:
    if name.endswith('.py'):ast.parse((ROOT/name).read_bytes(),filename=name)
assert expected['runtime/qwen_key_store.py']=='96a20067b0034a4f9ecd394e2b56e29fe5389c996505230701d9cbd883a2d67d'
assert handoff['tests_executed']==0 and handoff['collection_executed']is False
assert handoff['numpy_needed_in_later_tiny_tests']is True
assert handoff['signal_installation_selected']is False
assert read_tree(ROOT)==expected and read_tree(PARENT)==parent
report=dict(schema='qwen-held-parent-test-source-peer-review-v1',passed=True,
    scope='Source/AST/archive review only; no candidate or NumPy imports, collection, tests, borrowed-store metadata/body reads, backend, RNG or actual grant.',
    source_root=str(ROOT),source_manifest_sha256=sha(MAP.read_bytes()),source_archive_sha256=sha(TAR.read_bytes()),
    handoff_sha256=sha(HANDOFF.read_bytes()),source_count=83,parent_exact=74,new_files=sorted(added),
    static_counts=counts,static_selected_nodes=nodes,tests_executed=0,
    checks=dict(current_source_exact=True,parent_current_and_runtime_subtree_exact=True,archive_members_exact=True,
      static_node_and_xml_order_exact=True,inherited_tests_unselected=True,source_before_after_exact=True,
      metadata_denial_before_open_asserted=True,external_identity_before_mkdir_asserted=True,
      genuine_small_numpy_wire_parity_asserted=True,original_fd_retirement_and_primary_identity_asserted=True),
    source_findings=[],
    limits=[
      'Metadata parent expectations come only from owned temporary fixtures, not real grants.',
      'UID/current-user fields use explicit stat and geteuid seams; no privilege/chown claim.',
      'The largest numeric fixture is sixteen uint32 values; no full166/335 generation or capacity claim.',
      'nullcontext is the selected shield; no fresh signal installation coverage.',
      'The uncertain-close case calls real close then raises; controlled ownership seam, not a demonstrated OS close failure.',
      'Source conftest provides backend discovery/preload guards; later harness needs independent source/audit/terminal review.',
      'Test assertions have not been executed by this reviewer.'],
    reviewed_pins={n:expected[n]for n in sorted(added|{'runtime/qwen_key_store.py','runtime/qwen_finalizer.py','runtime/cleanup_origin.py','runtime/key_descriptor_io.py','runtime/qwen_key_contract.py'})},
    reviewer_script_sha256=sha(Path(__file__).read_bytes()))
raw=(json.dumps(report,sort_keys=True,indent=2)+'\n').encode()
with (OUT/'review.json').open('xb')as f:f.write(raw)
print(json.dumps(dict(passed=True,source_count=83,parent_exact=74,static_nodes=34,receipt_sha256=sha(raw))))
