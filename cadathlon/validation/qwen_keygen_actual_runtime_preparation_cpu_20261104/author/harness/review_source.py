"""Read-only stdlib source/harness verification, never candidate import or collection."""
from pathlib import Path
import ast
import hashlib
import itertools
import json
import tarfile

OUT = Path('/tmp/qwen-keygen-actual-focused-proof-20261104/preparation-v1')
BASE = Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2')
REVIEW = Path(__file__).parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
pretty = lambda value: (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact = lambda value: (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def inv(root):
    paths=sorted(root.rglob('*'))
    assert not any(p.is_symlink() for p in paths)
    return {p.relative_to(root).as_posix():sha(p.read_bytes()) for p in paths if p.is_file()}
hraw=(OUT/'handoff.json').read_bytes()
assert sha(hraw)=='d212a3e7221f78d86f37ca1bdd0a7130ef334b50a2f6a84eae1fb1d767e2b1f3'
h=json.loads(hraw);craw=(OUT/'run_config.json').read_bytes();c=json.loads(craw)
assert sha(craw)==h['config_sha256'] and craw==pretty(c)
mraw=(OUT/'sources-before.json').read_bytes();m=json.loads(mraw)
assert mraw==compact(m) and sha(mraw)==h['source_manifest_sha256']
assert len(m)==97 and inv(Path(h['source_root']))==inv(OUT/'execution')==m
parent=json.loads((OUT/'execution/parent85_sources.json').read_bytes())
assert len(parent)==85 and inv(OUT/'execution/runtime')==parent
with tarfile.open(OUT/'sources.tar.gz') as t:
    members=t.getmembers()
    assert len(members)==len({x.name for x in members})==97
    assert all(x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in members)
    assert {x.name:sha(t.extractfile(x).read()) for x in members}==m
assert sha((OUT/'sources.tar.gz').read_bytes())==h['source_archive_sha256']
nodes=[]
for filename in c['selected_files']:
    tree=ast.parse((OUT/'execution'/filename).read_text())
    for fn in tree.body:
        if not isinstance(fn,ast.FunctionDef) or not fn.name.startswith('test_'):continue
        axes=[]
        for d in reversed(fn.decorator_list):
            assert isinstance(d,ast.Call) and isinstance(d.func,ast.Attribute) and d.func.attr=='parametrize'
            assert len(d.args)==2 and not d.keywords
            axis=ast.literal_eval(d.args[1]);assert type(axis)is list;axes.append(axis)
        suffixes=[''] if not axes else ['['+'-'.join(map(str,row))+']' for row in itertools.product(*axes)]
        nodes.extend(filename+'::'+fn.name+s for s in suffixes)
assert nodes==c['selected_nodes'] and len(nodes)==len(set(nodes))==61
xml=[dict(classname=n.split('::')[0][:-3],name=n.split('::')[1]) for n in nodes]
assert xml==c['selected_xml_cases']
assert sha(pretty(nodes))==c['selected_nodes_sha256'] and sha(pretty(xml))==c['selected_xml_cases_sha256']
driver=(OUT/'executed_driver.py').read_text();worker=(OUT/'worker_driver.py').read_text()
assert sha(driver.encode())==h['driver_sha256'] and sha(worker.encode())==h['worker_sha256']
dt=ast.parse(driver);wt=ast.parse(worker)
bindings={n.targets[0].id:ast.literal_eval(n.value) for n in dt.body
          if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ('CONFIG_SHA256','WORKER_SHA256')}
assert bindings==dict(CONFIG_SHA256=h['config_sha256'],WORKER_SHA256=h['worker_sha256'])
def owner(tree):
    found=[]
    for n in ast.walk(tree):
        if isinstance(n,ast.With) and any(isinstance(x,ast.Assign) and isinstance(x.value,ast.Call)
             and isinstance(x.value.func,ast.Attribute) and x.value.func.attr=='Popen' for x in n.body):
            assert len(n.body)==2 and isinstance(n.body[1],ast.Try)
            found.append(ast.dump(n,include_attributes=False))
    assert len(found)==1
    return found[0]
assert owner(dt)==owner(ast.parse((BASE/'executed_driver.py').read_text()))
expected_worker=(BASE/'worker_driver.py').read_text().replace('{108, 109, 110, 111}','{112, 113, 114, 115}').replace('pure client-authority proof','synthetic keygen-runtime proof')
assert worker==expected_worker
assert c['cpu_affinity']==[112,113,114,115]
assert c['runtime_source_expectations']==c['runtime_module_origins']=={}
assert c['forbidden_roots']==sorted(['flax','jax','jaxite_word','jaxlib','libtpu','numpy','safetensors','tensorflow','tokenizers','torch','torch_xla','transformers'])
assert "assert manifest == compact_map(expected)" in driver
assert "(out / 'sources-after.json').write_bytes(compact_map(current))" in driver
assert "(out / 'execution-after.json').write_bytes(compact_map(after))" in driver
assert "audit['denied'] == audit['loaded_backend'] == []" in driver
assert "if collected != config['selected_nodes']:" in worker
assert "raise SystemExit(int(status) if not denied and not loaded else 70)" in worker
assert worker.index('sys.addaudithook(audit)')<worker.index('import pytest')
assert worker.index('sys.meta_path.insert(0, DenyBackend())')<worker.index('import pytest')
praw=(OUT/'launch-proposal.json').read_bytes();p=json.loads(praw)
assert sha(praw)==h['launch_proposal_sha256']
assert p['execution_authorized'] is False and p['healthy_job_deadline'] is False and p['automatic_retry'] is False
assert p['argv']==['/usr/bin/env','-i','PATH=/usr/bin:/bin','LANG=C.UTF-8','TZ=UTC','PYTHONDONTWRITEBYTECODE=1','PYTEST_DISABLE_PLUGIN_AUTOLOAD=1','OPENBLAS_NUM_THREADS=4','OMP_NUM_THREADS=4','MKL_NUM_THREADS=4','/usr/bin/taskset','-c','112-115','/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(OUT/'executed_driver.py')]
for name in ('driver','worker','config'):assert p[name+'_sha256']==h[name+'_sha256']
assert c['synthetic_only'] is True and c['actual_execution_authorized'] is False
assert c['genuine_backend_imports_authorized'] is False and c['capacity_acquisition_authorized'] is False
assert 'ModuleType' in c['scope'] and 'terminal module presence' in c['scope']
assert not any((OUT/name).exists() for name in ('report.json','tests.xml','tests.log','worker-audit.json','live-observation.json'))
selected={name:sha((OUT/name).read_bytes()) for name in ['handoff.json','executed_driver.py','worker_driver.py','run_config.json','launch-proposal.json','driver_provenance.json','README.md','sources-before.json','sources.tar.gz','source-handoff.json','selected-tests.json','executed_driver.py.patch','worker_driver.py.patch']}
report=dict(schema='qwen-keygen97-focused-harness-static-review-v1',passed=True,source_only=True,
    candidate_imports=False,collection=False,test_execution=False,body_reads=False,capacity_acquisition=False,
    source_count=97,parent_runtime_exact=85,static_selected=61,cpu=[112,113,114,115],
    compact_manifest_verified=True,exact_source_execution_archive=True,post_Popen_ownership_AST_identical=True,
    NumPy_forbidden=True,runtime_origins={},temporary_ModuleType_scope_disclosed=True,
    genuine_backend_imports_claimed=False,no_final_backend_modules_required=True,
    root_review_and_separate_launch_authorization_pending=True,proof_root=str(OUT),files=selected)
(REVIEW/'review.json').write_bytes(pretty(report))
print(json.dumps(dict(passed=True,review_sha256=sha((REVIEW/'review.json').read_bytes()),handoff_sha256=sha(hraw),source_count=97,static_nodes=61),sort_keys=True))
