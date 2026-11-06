"""Capture authored focused test source using stdlib/AST only."""
from pathlib import Path
import ast,gzip,hashlib,io,json,tarfile

SOURCE=Path('/tmp/qwen-key-generation-held-parent-tests-20261104')
PARENT=Path('/tmp/qwen-key-generation-held-parent-20261104')
PARENT_MAP=Path('/tmp/qwen-keygen-held-parent-sources-v1-20261104.json')
MAP=Path('/tmp/qwen-keygen-held-parent-test-sources-v1-20261104.json')
ROOT_REVIEW=Path('/tmp/qwen-key-generation-actual-root-review-20261104/held-parent74-source-review.json')
NUMPY_REFERENCE=Path('/tmp/qwen-key-producer-boundary-proof-20261104/preparation-v1/run_config.json')
OUT=Path(__file__).parent
FILES=['test_parent_admission.py','test_parent_policy.py','test_parent_lifetime.py']
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canonical=lambda value:(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def inventory(root):
    paths=sorted(root.rglob('*'))
    assert all(not p.is_symlink()and(p.is_file()or p.is_dir())for p in paths)
    return{p.relative_to(root).as_posix():sha(p.read_bytes())for p in paths if p.is_file()}

def main():
    parent_raw=PARENT_MAP.read_bytes();parent=json.loads(parent_raw)
    assert sha(parent_raw)=='7ffa419573da6100462fbc7d8ab778975c11d37fba4e551bd86c3b92d448a4b9'
    assert len(parent)==74 and inventory(PARENT)==inventory(SOURCE/'runtime')==parent
    assert sha(ROOT_REVIEW.read_bytes())=='7a0c6adc40c51b7a89e9cf43f5fe3d4cdc6d8b603aae8a955bfb06f059787206'
    nodes=[];counts={}
    for filename in FILES:
        tree=ast.parse((SOURCE/filename).read_bytes(),filename=filename);before=len(nodes)
        for node in tree.body:
            if not isinstance(node,ast.FunctionDef)or not node.name.startswith('test_'):continue
            assert len(node.decorator_list)<=1
            if node.decorator_list:
                mark=node.decorator_list[0]
                assert isinstance(mark,ast.Call)and ast.unparse(mark.func)=='pytest.mark.parametrize'
                assert len(mark.args)==2 and mark.keywords==[]
                values=ast.literal_eval(mark.args[1])
                assert type(values)is list and values and all(type(v)in(str,bool)for v in values)
                nodes.extend(filename+'::'+node.name+'['+str(v)+']'for v in values)
            else:nodes.append(filename+'::'+node.name)
        counts[filename]=len(nodes)-before
    assert list(counts.values())==[18,5,11]and len(nodes)==len(set(nodes))==34
    xml=[dict(classname=Path(n.split('::')[0]).stem,name=n.split('::')[1])for n in nodes]
    selected=dict(schema='qwen-held-parent-static-test-selection-v1',selected_files=FILES,ordered_nodes=nodes,ordered_xml_cases=xml,
        counts=counts,total=34,source='static AST expansion only',collection_executed=False)
    (SOURCE/'selected_parent_tests.json').write_bytes(canonical(selected))
    numpy_ref=json.loads(NUMPY_REFERENCE.read_bytes())
    numerical={k:numpy_ref[k]for k in('runtime_source_expectations','runtime_module_origins')}
    assert len(numerical['runtime_source_expectations'])==len(numerical['runtime_module_origins'])==3
    for path,digest in numerical['runtime_source_expectations'].items():assert sha(Path(path).read_bytes())==digest
    origin=dict(schema='qwen-held-parent-focused-test-origin-v1',parent_root=str(PARENT),parent_count=74,
        parent_source_manifest_path=str(PARENT_MAP),parent_source_manifest_sha256=sha(parent_raw),all_parent_bytes_exact=True,
        root_source_review_path=str(ROOT_REVIEW),root_source_review_sha256=sha(ROOT_REVIEW.read_bytes()),
        runtime_layout='exact74 under runtime/',inherited_tests_selected=False,
        actual_constructor_compatibility='new required expected_parent supplied explicitly by tiny owned fixtures; historical calls unchanged/unselected',
        new_python_files=['conftest.py','parent_test_helpers.py',*FILES],new_metadata_files=['TEST_SCOPE.md','test_origin.json','selected_parent_tests.json','READY_TESTS.json'],
        numpy_source_reference_path=str(NUMPY_REFERENCE),numpy_source_reference_sha256=sha(NUMPY_REFERENCE.read_bytes()),
        proposed_numerical_expectations=numerical,numpy_imported_during_authoring=False,
        allowed_later_numpy_scope='deterministic16-value uint32 arrays and tiny NPY/manifest serialization only',
        stat_seams='parent UID fields and one effective-UID1001 fixture only; actual owned directories/modes/FDs',
        signal_seam='nullcontext; no fresh signal-installation claim',
        root_review_before_collection_required=True,body_or_capacity_inventory=False,actual_keygen=False)
    (SOURCE/'test_origin.json').write_bytes(canonical(origin))
    ready=dict(schema='qwen-held-parent-test-source-ready-v1',source_held=True,static_selected_count=34,
        group_counts=counts,all74_parent_bytes_exact=True,new_test_authoring_only=True,candidate_imported=False,
        numpy_imported=False,collection_executed=False,tests_executed=0,actual_grant=False,
        future_cpu_affinity_proposal=[112,113,114,115],inherited71_test_collection_forbidden=True)
    (SOURCE/'READY_TESTS.json').write_bytes(canonical(ready))
    expected=inventory(SOURCE)
    assert len(expected)==83
    assert all(expected['runtime/'+n]==digest for n,digest in parent.items())
    assert all(n.startswith('runtime/')or n in origin['new_python_files']+origin['new_metadata_files']for n in expected)
    parsed=[]
    for name in expected:
        if name.endswith('.py'):ast.parse((SOURCE/name).read_bytes(),filename=name);parsed.append(name)
    raw=canonical(expected);MAP.write_bytes(raw);(OUT/'sources.json').write_bytes(raw)
    with (OUT/'sources.tar.gz').open('xb')as target:
        with gzip.GzipFile(fileobj=target,mode='wb',filename='',mtime=0)as zipped:
            with tarfile.open(fileobj=zipped,mode='w')as archive:
                for name in sorted(expected):
                    data=(SOURCE/name).read_bytes();m=tarfile.TarInfo(name);m.size=len(data);m.mode=0o644;m.mtime=0
                    archive.addfile(m,io.BytesIO(data))
    with tarfile.open(OUT/'sources.tar.gz')as archive:
        members=archive.getmembers();assert len(members)==len({m.name for m in members})==83
        assert all(m.isfile()for m in members)
        assert{m.name:sha(archive.extractfile(m).read())for m in members}==expected
    assert inventory(SOURCE)==expected and inventory(PARENT)==parent
    handoff=dict(schema='qwen-held-parent-focused-test-source-handoff-v1',source_root=str(SOURCE),source_count=83,
        source_manifest_path=str(MAP),source_manifest_sha256=sha(raw),source_archive_path=str(OUT/'sources.tar.gz'),
        source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),parent74_unchanged=True,parent_source_manifest_sha256=sha(parent_raw),
        selected_files=FILES,ordered_nodes=nodes,ordered_xml_cases=xml,group_counts=counts,static_expected_count=34,
        runtime_subtree='runtime',parsed_python_files=parsed,selection_method='stdlib AST expansion; not pytest collection',
        numerical_expectations=numerical,forbidden_backend_roots=['flax','jax','jaxite_word','jaxlib','libtpu','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers'],
        numpy_authoring_import=False,numpy_needed_in_later_tiny_tests=True,signal_installation_selected=False,
        backend_imports=False,candidate_imports=False,collection_executed=False,tests_executed=0,
        current_body_or_capacity_inventory=False,actual_keygen=False,actual_grant=False,
        cpu_affinity_proposal=[112,113,114,115],capture_script_sha256=sha(Path(__file__).read_bytes()),
        root_source_and_harness_review_required_before_collection=True)
    (OUT/'handoff.json').write_bytes(canonical(handoff))
    print(json.dumps(dict(source_count=83,parent74_unchanged=True,static_count=34,group_counts=counts,
        source_manifest_sha256=sha(raw),source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),handoff_sha256=sha((OUT/'handoff.json').read_bytes())),sort_keys=True))

if __name__=='__main__':main()
