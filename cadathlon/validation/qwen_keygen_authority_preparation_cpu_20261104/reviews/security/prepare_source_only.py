"""Stdlib-only source review and harness finalization; never imports candidate code."""
from pathlib import Path
import ast
import hashlib
import json
import shutil
import tarfile

SOURCE = Path('/tmp/qwen-keygen-authority-tests-20261104')
HANDOFF = Path('/tmp/qwen-keygen-authority-test-source-handoff-20261104/handoff.json')
MAP = Path('/tmp/qwen-keygen-authority-test-sources-v1-20261104.json')
ARCHIVE = Path('/tmp/qwen-keygen-authority-test-source-handoff-20261104/sources.tar.gz')
PARENT = Path('/tmp/qwen-keygen-authority-boundary-20261104')
PARENT_MAP = Path('/tmp/qwen-keygen-authority-boundary-sources-v1-20261104.json')
REFERENCE = Path('/tmp/qwen-model-stage-boundary-proof-20261104/preparation-v1')
OUT = Path('/tmp/qwen-keygen-authority-boundary-proof-20261104/preparation-v1')
REVIEW = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
compact = lambda v: (json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
pretty = lambda v: (json.dumps(v, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()


def inventory(root):
    paths = sorted(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    assert all(p.is_file() or p.is_dir() for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in paths if p.is_file()}


def static_nodes(filename):
    tree = ast.parse((SOURCE / filename).read_bytes(), filename=filename)
    names = []
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef) or not function.name.startswith('test_'):
            continue
        parameters = []
        for decorator in function.decorator_list:
            assert isinstance(decorator, ast.Call)
            assert ast.unparse(decorator.func) == 'pytest.mark.parametrize'
            assert len(decorator.args) == 2 and not decorator.keywords
            assert ast.literal_eval(decorator.args[0]) == 'case'
            parameters = ast.literal_eval(decorator.args[1])
            assert type(parameters) is list and parameters and all(type(x) is str for x in parameters)
        names.extend([filename+'::'+function.name+'['+p+']' for p in parameters]
                     if parameters else [filename+'::'+function.name])
    return names


def replace_once(text, before, after):
    assert text.count(before) == 1, before
    return text.replace(before, after)


def main():
    raw = MAP.read_bytes(); expected = json.loads(raw)
    assert sha(raw) == '3b2ddbcde655dc3485a36b160ed97f166ef6aad69512dd824bc409545a13db48'
    assert raw == compact(expected) and len(expected) == 39
    handoff_raw = HANDOFF.read_bytes(); handoff = json.loads(handoff_raw)
    assert handoff['source_map'] == expected and handoff['source_root'] == str(SOURCE)
    assert inventory(SOURCE) == expected
    parent_raw = PARENT_MAP.read_bytes(); parent = json.loads(parent_raw)
    assert sha(parent_raw) == 'd67a5df7aff7cd6560e88bd7731cf124ceefba18f4a2c83c443aa1ad1aa72ff8'
    assert len(parent) == 28 and inventory(PARENT) == parent
    assert all(expected.get(n) == h for n,h in parent.items())
    assert sha(ARCHIVE.read_bytes()) == '532c9f51972f851f535f373115022876fa2553af12f21ed84617de83e0f877b2'
    with tarfile.open(ARCHIVE) as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 39
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert {m.name:sha(archive.extractfile(m).read()) for m in members} == expected
    files = handoff['selected_files']
    assert files == ['test_keygen_launch.py','test_keygen_readiness.py','test_keygen_retention.py']
    nodes = sum((static_nodes(n) for n in files), [])
    assert nodes == handoff['ordered_nodeids'] and len(nodes) == len(set(nodes)) == 48
    assert [sum(n.startswith(f+'::') for n in nodes) for f in files] == [17,25,6]
    xml = [dict(classname=Path(n.split('::')[0]).stem, name=n.split('::')[1]) for n in nodes]
    assert handoff['ordered_xml_cases'] == [dict(file=n.split('::')[0],name=n.split('::')[1]) for n in nodes]
    parsed = {n:ast.parse((SOURCE/n).read_bytes(), filename=n) for n in expected if n.endswith('.py')}
    forbidden = ['flax','jax','jaxite_word','jaxlib','libtpu','numpy','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers']
    imports = set()
    for tree in parsed.values():
        for item in ast.walk(tree):
            if isinstance(item, ast.Import): imports.update(a.name.split('.')[0] for a in item.names)
            elif isinstance(item, ast.ImportFrom) and item.module: imports.add(item.module.split('.')[0])
    assert not set(forbidden) & imports
    assert 'model_admission' in imports and 'keygen_authority' in imports
    assert not OUT.exists()
    OUT.mkdir(parents=True)
    execution = OUT/'execution'; execution.mkdir()
    for name in expected:
        target = execution/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((SOURCE/name).read_bytes())
    assert inventory(execution) == expected
    (OUT/'sources-before.json').write_bytes(raw)
    shutil.copyfile(ARCHIVE,OUT/'sources.tar.gz')
    scope = ('Exactly48 pure public-byte tests:17 launch/model/source,25 generated-readiness,6 retained-state/actual-entry. '
             'All28 parent files unchanged. Synthetic ModelAdmission, grants, process/identity claims and335 key descriptor dictionaries only. '
             'No genuine model/key/ciphertext/token/private body input, ndarray allocation, RNG, backend or NumPy import, '
             'actual role, capacity check or execution authority. Import guards are process assertions, not an OS filesystem sandbox.')
    config = dict(schema='qwen-keygen-authority-focused-proof-v1', source_root=str(SOURCE), proof_root=str(OUT),
        source_count=39, source_manifest_sha256=sha(raw), source_manifest_encoding='canonical sorted compact JSON plus newline',
        source_archive_sha256=sha(ARCHIVE.read_bytes()), selected_files=files, selected_nodes=nodes,
        selected_nodes_sha256=sha(pretty(nodes)), selected_xml_cases=xml, selected_xml_cases_sha256=sha(pretty(xml)),
        synthetic_only=True, actual_execution_authorized=False, runtime_source_expectations={}, runtime_module_origins={},
        forbidden_roots=forbidden, scope=scope, cpu_affinity=[112,113,114,115])
    config_raw = pretty(config); (OUT/'run_config.json').write_bytes(config_raw)
    original_worker = (REFERENCE/'worker_driver.py').read_text()
    original_driver = (REFERENCE/'executed_driver.py').read_text()
    assert sha(original_worker.encode()) == 'f26cf16ec3ad1fca60ea93c7fb6e07059f597123e86662e23dc0a7f437cda91b'
    assert sha(original_driver.encode()) == '27003446b2b49976d4dd6f067fdfae63a42d229538ec4f189d5d42a148f732fc'
    worker = original_worker.replace('recording model-stage proof','pure key-authority proof').replace('{108, 109, 110, 111}','{112, 113, 114, 115}')
    worker = replace_once(worker,"forbidden = {'jax', 'jaxlib', 'jaxite_word', 'torch', 'transformers',\n             'tensorflow', 'safetensors', 'libtpu'}",'forbidden = set('+repr(forbidden)+')')
    # Literal guard plus externally pinned exact list, with no imported numeric origins.
    worker = replace_once(worker,'denied, collected = [], []',"assert sorted(forbidden) == config['forbidden_roots']\nassert runtime_expected == {} and config['runtime_module_origins'] == {}\ndenied, collected = [], []")
    worker = replace_once(worker,'runtime_module_origins=module_origins,',"runtime_module_origins=module_origins, forbidden_roots=sorted(forbidden),\n    numpy_imported=any(n == 'numpy' or n.startswith('numpy.') for n in sys.modules),")
    (OUT/'worker_driver.py').write_text(worker)
    driver = original_driver.replace('model-stage','key-authority').replace('qwen-key-authority-focused-proof-v1','qwen-keygen-authority-focused-proof-v1')
    driver = driver.replace('{108, 109, 110, 111}','{112, 113, 114, 115}').replace('[108, 109, 110, 111]','[112, 113, 114, 115]').replace('108-111','112-115')
    driver = replace_once(driver,"CONFIG_SHA256 = 'da8269a2002ebeb55cb8084b27e26c592d95aee53e1f2c78cd168242c64a4137'",'CONFIG_SHA256 = '+repr(sha(config_raw)))
    driver = replace_once(driver,"WORKER_SHA256 = 'f26cf16ec3ad1fca60ea93c7fb6e07059f597123e86662e23dc0a7f437cda91b'",'WORKER_SHA256 = '+repr(sha(worker.encode())))
    driver = replace_once(driver,'expected = json.loads(manifest)',"expected = json.loads(manifest)\n    assert manifest == (json.dumps(expected, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\\n').encode()")
    driver = replace_once(driver,"and audit['denied'] == audit['loaded_backend'] == []", "and audit['denied'] == audit['loaded_backend'] == []\n              and audit['forbidden_roots'] == config['forbidden_roots']\n              and audit['numpy_imported'] is False")
    driver = replace_once(driver,'actual_execution_authorized=False, model_execution=False, encrypted_execution=False,', 'actual_execution_authorized=False, numpy_imported=audit[\'numpy_imported\'], model_execution=False, encrypted_execution=False,')
    (OUT/'executed_driver.py').write_text(driver)
    ast.parse(worker); ast.parse(driver)
    # Verify the corrected immediate post-Popen ownership branch is structurally identical.
    def ownership(tree):
        for node in ast.walk(ast.parse(tree)):
            if isinstance(node, ast.Try) and any(isinstance(n,ast.Call) and ast.unparse(n.func)=='os.getpgid' for n in ast.walk(node)):
                return ast.dump(node,include_attributes=False)
        raise AssertionError('missing immediate owned-child block')
    assert ownership(driver) == ownership(original_driver)
    env = dict(PATH='/usr/bin:/bin',HOME='/home/jianming_gatech',LANG='C.UTF-8',TZ='UTC',
               PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    argv = ['taskset','-c','112-115','env','-i',*[k+'='+v for k,v in env.items()],
            '/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(OUT/'executed_driver.py')]
    shared = dict(source_count=39, source_manifest_sha256=sha(raw), source_archive_sha256=sha(ARCHIVE.read_bytes()),
        config_sha256=sha(config_raw),driver_sha256=sha(driver.encode()),worker_sha256=sha(worker.encode()),
        selected_nodes_sha256=sha(pretty(nodes)),selected_xml_cases_sha256=sha(pretty(xml)),
        launch_authorized=False,tests_collected=0,tests_executed=0,tests_declared=48)
    proposal = dict(schema='qwen-keygen-authority-focused-launch-proposal-v1',**shared,argv=argv,environment=env,scope=scope)
    (OUT/'launch-proposal.json').write_bytes(pretty(proposal))
    provenance = dict(schema='qwen-keygen-authority-focused-driver-provenance-v1',**shared,
        parent_driver_path=str(REFERENCE/'executed_driver.py'),parent_driver_sha256=sha(original_driver.encode()),
        parent_worker_path=str(REFERENCE/'worker_driver.py'),parent_worker_sha256=sha(original_worker.encode()),
        changes=['Exact39 source and static48 nodes/ordered XML names; canonical compact source-map assertion.',
                 'CPU112–115 and pure-key-authority schema/scope/diagnostics.',
                 'NumPy plus all declared backend roots forbidden; numeric runtime sources/origins explicitly empty.',
                 'Record/assert forbidden roots and NumPy absence; corrected owned-child try/kill/direct-wait AST unchanged.'],
        supplied_handoff_path=str(HANDOFF),supplied_handoff_sha256=sha(handoff_raw))
    (OUT/'driver_provenance.json').write_bytes(pretty(provenance))
    report = dict(schema='qwen-keygen-authority-harness-source-review-v1',passed=True,scope='Source-only; no imports/collection/tests/body reads/RNG',
        source_map_sha256=sha(raw),source_archive_sha256=sha(ARCHIVE.read_bytes()),parent28_map_sha256=sha(parent_raw),
        all_parent28_exact=True,current_source_matches=True,execution_copy_matches=True,archive_members_match=True,
        static_nodes=nodes,static_group_counts=[17,25,6],static_count=48,selected_files=files,
        meaningful_assertion_findings=['Mutations rebuild surrounding proof digests, so semantic gates are reached.',
            'ModelAdmission is constructed without eager authentication; tested launch reauthenticates model scope and consumer map.',
            'Full335 descriptor metadata and literal166 keys are checked; no arrays or actual key bodies created.',
            'Normal-zero/reap/group/root proof requirements and retained-byte reauthentication are tested.',
            'Actual factory denial occurs before poison argument inspection; pure readiness I/O/RNG seam is blocked.'],
        limits=['No runtime results or actual independently trusted authority are claimed.',
                'The harness guards backend/NumPy imports; source-reviewed test closure supplies the pure I/O scope, not an OS sandbox.',
                'Installed pytest/interpreter binary closure is not newly authenticated by this narrow inherited harness.'],
        direct_import_roots=sorted(imports),forbidden_roots=forbidden,numeric_sources_and_origins_empty=True,
        corrected_owned_child_ast_unchanged=True,helper_source_sha256=sha(Path(__file__).read_bytes()),
        harness={n:sha((OUT/n).read_bytes()) for n in ('executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json','driver_provenance.json')})
    assert inventory(SOURCE) == expected and inventory(PARENT) == parent and inventory(execution) == expected
    (REVIEW/'review.json').write_bytes(pretty(report))
    (OUT/'source-review.json').write_bytes(pretty(report))
    print(json.dumps(dict(prepared=str(OUT),source_count=39,static_count=48,review_sha256=sha(pretty(report)),**report['harness']),sort_keys=True))


if __name__ == '__main__': main()
