"""Stdlib source-only capture; does not import the proposed component."""
import ast
import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT=Path('/tmp/qwen-server-device-admission-source-20261104')
OUT=Path('/tmp/qwen-server-device-admission-handoff-20261104')
SITE=Path('/home/jianming_gatech/miniconda3/envs/jaxite/lib/python3.13/site-packages')
DESIGN=Path('/tmp/qwen-server-device-capacity-design-20261104')
SERVER=Path('/tmp/qwen-server-actual-runtime-20261104')

def canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def digest(v):return hashlib.sha256(v).hexdigest()
def emit(path,value):path.write_bytes(canonical(value))
def record(path):
    raw=path.read_bytes()
    return {'path':str(path),'sha256':digest(raw),'bytes':len(raw)}

refs=[DESIGN/'DESIGN.md',DESIGN/'review.json',DESIGN/'source_refs.json',DESIGN/'files.json',DESIGN/'py_device.revision.cc',
    SITE/'jax/__init__.py',SITE/'jax/_src/xla_bridge.py',SITE/'jax/_src/cloud_tpu_init.py',SITE/'jax/_src/hardware_utils.py',
    SITE/'jax/_src/stages.py',SITE/'jaxlib/xla_client.py',SITE/'jaxlib/version.py',SITE/'libtpu/__init__.py',
    SERVER/'qwen_server_admission.py',SERVER/'qwen_server_resource.py',SERVER/'qwen_server_process.py',
    SERVER/'qwen_server_dependencies.py',SERVER/'runtime/common/encrypted_primitives.py']
before={str(p):record(p)for p in refs}
emit(ROOT/'source_refs.json',dict(schema='qwen-device-component-source-references-v1',references=before,
    upstream_url='https://raw.githubusercontent.com/jax-ml/jax/dc4af684446e65ee98eee8090e1ff6f8211473fb/jaxlib/py_device.cc',
    references_are_source_only=True,installed_revision_is_not_binary_reproducibility_proof=True,
    server_references_are_individual_source_observations_not_a_frozen_server_closure=True))
emit(ROOT/'provenance.json',dict(schema='qwen-device-component-provenance-v1',
    status='unselected_source_only_unexecuted',new_runtime=['qwen_device_policy.py','qwen_device_capacity.py'],
    copies=[],overlays=[],runtime_imports=False,tests_authored=False,tests_collected=False,tests_run=False,
    backend_imports=False,device_probes=False,capacity_measurements=False,borrowed_body_reads=False,
    policy_instances=False,numeric_defaults=False,actual_grants=False,root_server_edits=False,
    trust_boundary='independently reviewed pinned native backend; Python audit is not an arbitrary-Python/native syscall sandbox',
    dependency_mode='references only; actual selected phase/resource/process helpers must join the future source closure',
    pending=['independently factual device/access/allocator/workspace policies and exclusive lease',
        'five genuine source-fixed process hooks and their ownership review',
        'complete source selection, dependency and audit transition',
        'separate focused source/tests review and explicit actual authority'],
    declared_limitations=['single visible addressable local JAX TPU device and single process in this first API',
        'allocator bytes_limit minus bytes_in_use is not global free memory or a reservation',
        'one-use process/thread state guard does not contain arbitrary interpreter/global/native mutation',
        'stage bounds are reviewed external additional peaks, not inferred from file sizes or host RAM',
        'native partial owner retention and lease retirement remain original process-owner duties'],
    authoring_history={'syntax_draft':'/tmp/qwen-server-device-admission-authoring-history-20261104/capacity.syntax-draft.py.txt',
        'syntax_draft_was_unheld':True,'candidate_code_was_not_imported':True,
        'subsequent_ast_parse':'446ef5 exit 0; source parsing only'}))

names=('API.md','provenance.json','qwen_device_capacity.py','qwen_device_policy.py','source_refs.json')
assert sorted(p.name for p in ROOT.iterdir())==list(names)
blobs={n:(ROOT/n).read_bytes()for n in names}
trees={n:ast.parse(b,filename=n)for n,b in blobs.items()if n.endswith('.py')}
policy_tree=trees['qwen_device_policy.py'];capacity_tree=trees['qwen_device_capacity.py']
constants={node.targets[0].id:ast.literal_eval(node.value)for node in policy_tree.body
    if isinstance(node,ast.Assign)and isinstance(node.targets[0],ast.Name)}
assert len(constants['ROLES'])==10 and len(constants['STAGES'])==4 and len(constants['HOOKS'])==5
actual=next(n for n in capacity_tree.body if isinstance(n,ast.FunctionDef)and n.name=='create_actual_server')
assert len(actual.body)==1 and isinstance(actual.body[0],ast.Raise)
assert isinstance(actual.body[0].exc,ast.Call)and actual.body[0].exc.func.id=='PermissionError'
assert not any(isinstance(n,(ast.Import,ast.ImportFrom))and any(a.name.split('.')[0]in{'jax','jaxlib','libtpu','numpy'}for a in n.names)for t in trees.values()for n in ast.walk(t))
source_map={n:digest(b)for n,b in blobs.items()}
emit(OUT/'sources.json',source_map)
with(OUT/'sources.tar.gz').open('wb')as raw:
    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0)as compressed:
        with tarfile.open(mode='w',fileobj=compressed,format=tarfile.PAX_FORMAT)as archive:
            for name,data in blobs.items():
                info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
                archive.addfile(info,io.BytesIO(data))
with tarfile.open(OUT/'sources.tar.gz','r:gz')as archive:
    members=archive.getmembers();assert sorted(m.name for m in members)==list(names)
    assert all(m.isfile()and digest(archive.extractfile(m).read())==source_map[m.name]for m in members)
assert {str(p):record(p)for p in refs}==before
assert {n:digest((ROOT/n).read_bytes())for n in names}==source_map
review=dict(schema='qwen-device-component-static-capture-v1',passed=True,
    checks=['exact five source names and before/end hashes','two modules AST parsed without import',
        'ten document roles, four stages, five source-fixed hook names','actual server factory immediate denial',
        'no backend or NumPy imports in component AST','archive names/member bytes equal canonical source map',
        '18 public source references unchanged during capture'],source_count=len(source_map),
    source_manifest_sha256=digest((OUT/'sources.json').read_bytes()),source_archive_sha256=digest((OUT/'sources.tar.gz').read_bytes()),
    roots_are_unmodified=True,candidate_imports=False,tests=False,device_or_capacity_probes=False,
    backend_or_borrowed_body_reads=False,actual_execution_authorized=False,
    review_scope='author source/static capture; not independent peer review, a runtime test, complete server closure, or factual device-policy admission')
emit(OUT/'static-review.json',review)
handoff=dict(schema='qwen-device-component-source-handoff-v1',source_root=str(ROOT),
    source_map=record(OUT/'sources.json'),source_archive=record(OUT/'sources.tar.gz'),
    static_review=record(OUT/'static-review.json'),capture_script=record(Path(__file__)),
    runtime_pins={n:source_map[n]for n in trees},api_sha256=source_map['API.md'],
    source_count=len(source_map),scope='source only, conditional and unselected; no imports/tests/probes/policy instances/actual authority',
    current_complete_capacity_denial_is_unchanged=True,
    construction='authenticate → issue_initialization → initialize_once → ordered observe_capacity/require_current → retire')
emit(OUT/'handoff.json',handoff)
print(json.dumps({'handoff':record(OUT/'handoff.json'),'review':record(OUT/'static-review.json'),
    'source_map':record(OUT/'sources.json'),'archive':record(OUT/'sources.tar.gz'),'runtime_pins':handoff['runtime_pins']},sort_keys=True))
