"""Source-only peer capture. No candidate import, collection, child or body IO."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path('/tmp/qwen-model-store-readonly-tests-v2-20261104')
MAP=Path('/tmp/qwen-model-store-readonly-test-sources-v2-20261104.json')
ARCHIVE=Path('/tmp/qwen-model-store-readonly-test-handoff-v2-20261104/sources.tar.gz')
PROOF=Path('/tmp/qwen-model-store-readonly-focused-proof-20261104/preparation-v1')
OUT=Path('/tmp/qwen-readonly-focused-transformers-review-20261104')
sha=lambda raw:hashlib.sha256(raw).hexdigest()
dump=lambda node:ast.dump(node,include_attributes=False)
pretty=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def current(root):
    rows={}
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink()
        if path.is_file(): rows[path.relative_to(root).as_posix()]=sha(path.read_bytes())
        else: assert path.is_dir()
    return rows


def archive_map(path):
    with tarfile.open(path,'r:gz')as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})
        assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
        return {m.name:sha(archive.extractfile(m).read())for m in members}


expected=json.loads(MAP.read_bytes())
assert sha(MAP.read_bytes())=='319b4503b63635f4ba9ddf675a97a2a481fe6b67df503919f91bec3ea356057d'
assert sha(ARCHIVE.read_bytes())=='e8747c20a234cd1272d16a43a7fbc2c93b5555136e48d174843697c1cedf419e'
assert len(expected)==54 and current(ROOT)==current(PROOF/'execution')==archive_map(ARCHIVE)==expected
assert (PROOF/'sources-before.json').read_bytes()==MAP.read_bytes()
assert sha((PROOF/'sources.tar.gz').read_bytes())==sha(ARCHIVE.read_bytes())
runtime=json.loads((ROOT/'runtime_sources.json').read_bytes())
assert len(runtime)==42 and sha((ROOT/'runtime_sources.json').read_bytes())=='b73e669b0cc2bc34d7930f2a8a4c620be719fe663dfa6ae7b12df7cae22c2bc0'
assert current(ROOT/'runtime')==runtime
for name,h in runtime.items():
    assert sha((Path('/tmp/qwen-model-store-readonly-v2-20261104')/name).read_bytes())==h

previous_root=Path('/tmp/qwen-model-store-readonly-tests-20261104')
previous=current(previous_root)
assert previous.keys()==expected.keys()
changed=sorted(n for n in expected if previous[n]!=expected[n])
assert changed==['README.md','test_provenance.json','tests/support.py']
before=ast.parse((previous_root/'tests/support.py').read_text())
after=ast.parse((ROOT/'tests/support.py').read_text())
before_defs={n.name:n for n in before.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
after_defs={n.name:n for n in after.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert before_defs.keys()==after_defs.keys()
assert [n for n in before_defs if dump(before_defs[n])!=dump(after_defs[n])]==['child']
assert [dump(n)for n in before.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]==[dump(n)for n in after.body if not isinstance(n,(ast.FunctionDef,ast.ClassDef))]
child_source=ast.get_source_segment((ROOT/'tests/support.py').read_text(),after_defs['child'])
assert 'finally:'in child_source and 'try:proc.wait()'in child_source
assert 'start_new_session=True'in child_source and 'observed=os.getpgid(proc.pid)'in child_source
assert all(not n.args and not n.keywords for n in ast.walk(after_defs['child'])if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)and n.func.attr=='wait')

history=json.loads((ROOT/'test_provenance.json').read_bytes())['history']
for name,h in history.items():
    assert expected[name]==h
    original=Path('/tmp/qwen-model-store-readonly-20261104')/name.split('source40-')[1].removesuffix('.txt')
    assert sha(original.read_bytes())==h

selection=json.loads((ROOT/'selected_tests.json').read_bytes())
nodes=[];xml=[];counts={}
for name in selection['selected_files']:
    tree=ast.parse((ROOT/name).read_text(),filename=name);count=0
    for node in tree.body:
        if not isinstance(node,ast.FunctionDef)or not node.name.startswith('test_'):continue
        ids=None
        for decorator in node.decorator_list:
            assert isinstance(decorator,ast.Call)and isinstance(decorator.func,ast.Attribute)and decorator.func.attr=='parametrize'
            assert ids is None
            ids=ast.literal_eval(next(k.value for k in decorator.keywords if k.arg=='ids'))
            assert isinstance(decorator.args[1],ast.List)and len(decorator.args[1].elts)==len(ids)
            assert all(type(value)is str for value in ids)
        cases=[node.name]if ids is None else[node.name+'['+value+']'for value in ids]
        count+=len(cases)
        nodes.extend(name+'::'+value for value in cases)
        xml.extend(dict(classname=name.removesuffix('.py').replace('/','.'),name=value)for value in cases)
    counts[name]=count
assert len(nodes)==len(set(nodes))==selection['static_count']==62
assert nodes==selection['selected_nodes']and xml==selection['selected_xml_cases']
assert list(counts.values())==[30,9,23]

config_raw=(PROOF/'run_config.json').read_bytes();config=json.loads(config_raw)
driver=(PROOF/'executed_driver.py').read_bytes();worker=(PROOF/'worker_driver.py').read_bytes()
driver_tree=ast.parse(driver);worker_tree=ast.parse(worker)
constants={n.targets[0].id:ast.literal_eval(n.value)for n in driver_tree.body
           if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('CONFIG_SHA256','WORKER_SHA256')}
assert constants==dict(CONFIG_SHA256=sha(config_raw),WORKER_SHA256=sha(worker))
assert config['source_manifest_sha256']==sha(MAP.read_bytes())and config['source_archive_sha256']==sha(ARCHIVE.read_bytes())
assert config['selected_nodes']==nodes and config['selected_xml_cases']==xml
assert config['selected_nodes_sha256']==sha(pretty(nodes))and config['selected_xml_cases_sha256']==sha(pretty(xml))
assert config['selected_files']==selection['selected_files']and config['source_count']==54
assert config['synthetic_only']is True and config['actual_execution_authorized']is False
assert selection['cores']==[108,109,110,111]
proposal=json.loads((PROOF/'launch-proposal.json').read_bytes())
assert proposal['driver_sha256']==sha(driver)and proposal['worker_sha256']==sha(worker)and proposal['config_sha256']==sha(config_raw)
assert proposal['launch_authorized']is False and proposal['actual_role_authorized']is False
assert '-I'in proposal['argv']and '-B'in proposal['argv']and '108-111'in proposal['argv']
for path,h in config['runtime_source_expectations'].items():assert sha(Path(path).read_bytes())==h
assert len(config['runtime_source_expectations'])==4 and len(config['runtime_module_origins'])==3
assert current(ROOT)==current(PROOF/'execution')==expected

report=dict(schema='qwen-readonly-focused-peer-source-review-v1',passed=True,
    scope='Source/test/harness review only; no collection, execution, backend imports, current model inventory or bodies',
    source_root=str(ROOT),source_count=54,source_manifest_sha256=sha(MAP.read_bytes()),
    source_archive_sha256=sha(ARCHIVE.read_bytes()),runtime42_exact=True,
    source_current_execution_archive_equal=True,test_count_static=62,static_group_counts=counts,
    selection_sha256=expected['selected_tests.json'],test_source_pins={n:h for n,h in expected.items()if n.startswith('tests/')},
    old40_history_pins=history,source_only_v1_delta=changed,
    driver_sha256=sha(driver),worker_sha256=sha(worker),config_sha256=sha(config_raw),
    launch_proposal_sha256=sha((PROOF/'launch-proposal.json').read_bytes()),
    findings=[
      'Meaningful old40 controls use exact origins/run AST against the same controlled module metadata/recording dependencies; no old source is selected as actual positive role.',
      'Child helper successor changes only child(): cleanup attempts direct wait in finally even after group cleanup failure, preserves primary, records observed fresh process group and checks absence; no healthy deadline.',
      'Full198/197+113 traversal is explicit recording admission/resource/signal/audit/catalog/reader/observation scope; actual traversal, retained callbacks, shared Finalizer and Output remain selected.',
      'Separate real small-FD observation and empty-root mismatch cases exercise actual selected metadata/ownership functions before body hash/header; they are not whole model integrity proof.',
      'Harness binds source54/current/execution/archive, immutable runtime42, exact selected collection/XML order, zero errors/skips, backend audit, direct child exit/reap/group absence and four NumPy source/three origin observations.',
      'Worker uses isolated no-bytecode Python, explicit tests, fixed root/confcutdir, cache disabled and plugin autoload disabled; actual role/resource checks remain recording or negative entry paths.'
    ],
    unresolved_concrete_blockers=[],candidate_imported=False,tests_collected=False,tests_executed=False,
    current_store_stat=False,model_or_private_body_reads=False,launch_granted=False,
    limitations=['NumPy Python source/origin checks do not establish a complete binary dependency closure.',
                 'Source-only review establishes no runtime pass, acquisition, actual ModelAdmission or key/body authorization.'],
    review_script_sha256=sha(Path(__file__).read_bytes()))
(OUT/'review.json').write_bytes(pretty(report))
print(json.dumps(dict(passed=True,source_count=54,static_cases=62,review_sha256=sha((OUT/'review.json').read_bytes()),source_only=True)))
