"""Independent source/AST/hash review only. Does not import candidate/test code."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import sys
import tarfile

HERE = Path(__file__).parent
SOURCE = Path('/tmp/qwen-client-role-authority-tests-v2-20261104')
HANDOFF = Path('/tmp/qwen-client-role-authority-test-handoff-v2-20261104')
OLD = Path('/tmp/qwen-client-role-authority-tests-20261104')
PARENT = Path('/tmp/qwen-client-role-authority-20261104')
BASE = Path('/tmp/qwen-keygen-authority-boundary-proof-20261104/preparation-v1')
PROOF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v1')
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pretty = lambda obj: (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False)+'\n').encode()
def read(path): return json.loads(path.read_bytes())
def inventory(root):
    paths=list(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in sorted(paths) if p.is_file()}
def archived(path):
    with tarfile.open(path) as t:
        members=t.getmembers()
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        assert len(members)==len({m.name for m in members})
        return {m.name:sha(t.extractfile(m).read()) for m in members}
def literal_assignment(tree,name):
    for n in tree.body:
        if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets):
            return ast.literal_eval(n.value)
    raise AssertionError(name)
def ownership_try(path):
    tree=ast.parse(path.read_text())
    found=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.With):
            for i,s in enumerate(n.body[:-1]):
                if isinstance(s,ast.Assign) and isinstance(s.value,ast.Call) and isinstance(s.value.func,ast.Attribute) and s.value.func.attr=='Popen':
                    following=n.body[i+1]
                    assert isinstance(following,ast.Try)
                    found.append(ast.dump(following,include_attributes=False))
    assert len(found)==1
    return found[0]

source_map=read(HANDOFF/'sources.json')
assert sha((HANDOFF/'sources.json').read_bytes())=='597ca4c89e66a22608d28d2e67d9234bac37bfee40d84b04139ac727309dee18'
assert (HANDOFF/'sources.json').read_bytes()==pretty(source_map)
assert len(source_map)==53 and inventory(SOURCE)==source_map
assert archived(HANDOFF/'sources.tar.gz')==source_map
assert sha((HANDOFF/'sources.tar.gz').read_bytes())=='d568676329e12379f6b480980ab68d346928945de90832f151bb2eca76e329ea'
parent_map=read(SOURCE/'parent39_sources.json')
assert len(parent_map)==39 and inventory(PARENT)==parent_map
assert all(source_map[n]==h for n,h in parent_map.items())
assert source_map['client_role_source_pins.py']=='2fdd3fa16ea1e461cb9eac85c344a654d005723327d9d6ee26f3061bfd2d3d18'
changes=read(SOURCE/'PREEXECUTION_CORRECTIONS.json')
old_map=inventory(OLD)
assert old_map==changes['parent52_map'] and len(old_map)==52
actual_changes={n:{'before':old_map[n],'after':source_map[n]} for n in old_map if source_map[n]!=old_map[n]}
assert actual_changes==changes['changed']
assert set(source_map)-set(old_map)=={'PREEXECUTION_CORRECTIONS.json'}
selection=read(SOURCE/'selected_client_tests.json')
nodes=[];counts={}
for filename in selection['selected_files']:
    tree=ast.parse((SOURCE/filename).read_text()); local=[]
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('test_'): continue
        values=[]
        for dec in reversed(fn.decorator_list):
            assert isinstance(dec,ast.Call) and isinstance(dec.func,ast.Attribute) and dec.func.attr=='parametrize'
            assert type(ast.literal_eval(dec.args[0])) is str and len(dec.args)==2 and not dec.keywords
            vs=ast.literal_eval(dec.args[1]);assert type(vs) is list and all(type(v) in (str,bool,int) for v in vs)
            values.append(vs)
        names=[fn.name] if not values else [fn.name+'['+'-'.join(map(str,row))+']' for row in itertools.product(*values)]
        local.extend(filename+'::'+name for name in names)
    counts[filename]=len(local); nodes.extend(local)
assert counts==selection['per_file']=={'test_client_conditional.py':29,'test_client_phase.py':33,'test_client_server.py':33}
assert len(nodes)==95 and nodes==selection['selected_nodes']
xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1]) for n in nodes]
assert xml==selection['selected_xml_cases']

handoff=read(PROOF/'handoff.json'); config=read(PROOF/'run_config.json'); proposal=read(PROOF/'launch-proposal.json')
for name,field in [('executed_driver.py','driver_sha256'),('worker_driver.py','worker_sha256'),('run_config.json','config_sha256'),('launch-proposal.json','launch_proposal_sha256')]:
    assert sha((PROOF/name).read_bytes())==handoff[field]
assert inventory(PROOF/'execution')==source_map==read(PROOF/'sources-before.json')==archived(PROOF/'sources.tar.gz')
assert sha((PROOF/'sources-before.json').read_bytes())==handoff['source_manifest_sha256']==config['source_manifest_sha256']
assert sha((PROOF/'sources.tar.gz').read_bytes())==handoff['source_archive_sha256']==config['source_archive_sha256']
assert nodes==config['selected_nodes'] and xml==config['selected_xml_cases']
assert config['selected_files']==selection['selected_files']
assert config['runtime_source_expectations']==config['runtime_module_origins']=={}
assert config['forbidden_roots']==['flax','jax','jaxite_word','jaxlib','libtpu','numpy','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers']
assert proposal['argv'][proposal['argv'].index('-c')+1]=='108-111'
assert '-I' in proposal['argv'] and '-B' in proposal['argv']
driver=(PROOF/'executed_driver.py').read_text(); worker=(PROOF/'worker_driver.py').read_text()
assert literal_assignment(ast.parse(driver),'CONFIG_SHA256')==handoff['config_sha256']
assert literal_assignment(ast.parse(driver),'WORKER_SHA256')==handoff['worker_sha256']
assert ownership_try(PROOF/'executed_driver.py')==ownership_try(BASE/'executed_driver.py')
assert "os.sched_getaffinity(0) == {108, 109, 110, 111}" in driver and "os.sched_getaffinity(0) == {108, 109, 110, 111}" in worker
assert "audit['numpy_imported'] is False" in driver
assert "audit['denied'] == audit['loaded_backend'] == []" in driver
assert "assert runtime_expected == {} and config['runtime_module_origins'] == {}" in worker
assert worker.index('sys.addaudithook(audit)')<worker.index('import pytest')
assert worker.index('sys.meta_path.insert(0, DenyBackend())')<worker.index('import pytest')
assert "if collected != config['selected_nodes']:" in worker
assert "--confcutdir=" in worker and "no:cacheprovider" in worker
assert not (PROOF/'tests.xml').exists() and not (PROOF/'report.json').exists()
known=[]
if config['cpu_affinity']!=[108,109,110,111]: known.append('run_config.cpu_affinity is stale; driver/worker/proposal enforce 108-111')
if config['source_manifest_encoding']!='canonical sorted indent=2 JSON plus newline': known.append('run_config source_manifest_encoding label is stale; actual bytes and driver use sorted indent=2 plus newline')
v1=Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v1')
assert PROOF.name=='preparation-v2'
v1_map=inventory(v1);v2_map=inventory(PROOF)
assert set(v1_map)==set(v2_map)
changed={n for n in v1_map if v1_map[n]!=v2_map[n]}
assert changed=={'run_config.json','executed_driver.py','launch-proposal.json','handoff.json'}
old_config=read(v1/'run_config.json');expected_config=dict(old_config)
expected_config.update(cpu_affinity=[108,109,110,111],proof_root=str(PROOF),source_manifest_encoding='canonical sorted indent=2 JSON plus newline')
assert config==expected_config
old_driver=(v1/'executed_driver.py').read_text()
assert driver==old_driver.replace(sha((v1/'run_config.json').read_bytes()),sha((PROOF/'run_config.json').read_bytes()))
assert known==[]
report=dict(schema='qwen-client-authority-security-source-test-review-v1',scope='source/AST/hash only; no candidate imports, collection, tests, body reads, RNG or actual execution',
    source_root=str(SOURCE),source_manifest_sha256=sha((HANDOFF/'sources.json').read_bytes()),source_count=53,all39_unchanged=True,
    source_archive_sha256=sha((HANDOFF/'sources.tar.gz').read_bytes()),static_count=95,per_file=counts,ordered_static_nodes_sha256=sha(pretty(nodes)),
    correction_review=[
        'Ancestor-input uses a nonroot canonical parent and overlap-specific rejection.',
        'Inner-source negative uses the fixture actual server map and requires request-provenance rejection.',
        'Envelope/root mutation hooks have private deep copies; model-proof negative requires root-links rejection.',
        'String nan scale is accurately labeled a type rejection, not a nonfinite numeric JSON case.'
    ],
    semantic_scope='Positive synthetic chains and repinned authority/link/type/retained-object negatives; no real store/private-read ordering claim.',
    harness_root=str(PROOF),harness_pins={n:sha((PROOF/n).read_bytes()) for n in ['executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json','handoff.json']},
    original_post_popen_cleanup_ast_exact=True,forbidden_roots=config['forbidden_roots'],runtime_origins_empty=True,
    selected_response_root_successor=False,known_metadata_findings=known,additional_concrete_source_blockers=[],
    launch_recommendation='withheld pending corrected harness metadata and root authorization' if known else 'no source-level blocker for root-reviewed exact synthetic proposal; no execution authorization issued',
    tests_executed=False,candidate_imports=False,v2_changed_files=sorted(changed),v1_preserved=True)
target=HERE/('review-'+PROOF.name+'.json');target.write_bytes(pretty(report))
print(json.dumps(dict(review=str(target),sha256=sha(target.read_bytes()),static_count=95,known_findings=known),sort_keys=True))
