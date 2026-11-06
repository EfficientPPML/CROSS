"""Static harness review; does not import or execute driver, worker or tests."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile
P=Path('/tmp/qwen-server-snapshot-integration-focused-proof-20261104/preparation-v1')
O=Path(__file__).parent
H=lambda raw:hashlib.sha256(raw).hexdigest()
def load(name):return json.loads((P/name).read_bytes())
def scan(root):
    out={}
    for p in root.rglob('*'):
        assert not p.is_symlink()
        if p.is_file():out[p.relative_to(root).as_posix()]=H(p.read_bytes())
    return out
handoff=load('handoff.json')
assert H((P/'handoff.json').read_bytes())=='ad7b69f004da28a838094861bd384bc8759556951991252bff0f6691a86dedfe'
for name,digest in handoff['files'].items():assert H((P/name).read_bytes())==digest,name
config=load('run_config.json');proposal=load('launch-proposal.json');source=load('source-handoff.json')
expected=load('sources-before.json');raw=(P/'sources-before.json').read_bytes()
assert raw==(json.dumps(expected,sort_keys=True,separators=(',',':'))+'\n').encode()
assert H(raw)==config['source_manifest_sha256']==source['source_manifest_sha256']
assert len(expected)==404 and scan(Path(config['source_root']))==scan(P/'execution')==expected
runtime={n[8:]:v for n,v in expected.items()if n.startswith('runtime/')}
assert len(runtime)==391 and runtime==config['runtime_source_expectations']
archive={}
with tarfile.open(P/'sources.tar.gz')as tar:
    for member in tar.getmembers():
        assert member.isfile()and member.name not in archive and not member.name.startswith('/')and '..'not in Path(member.name).parts
        with tar.extractfile(member)as stream:archive[member.name]=H(stream.read())
assert archive==expected
assert config['selected_nodes']==source['static_nodeids']and len(config['selected_nodes'])==36
assert config['selected_files']==['test_snapshot_integration.py']
assert config['selected_xml_cases']==[dict(classname='test_snapshot_integration',name=n.split('::')[1])for n in source['static_nodeids']]
pretty=lambda value:(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
assert H(pretty(config['selected_nodes']))==config['selected_nodes_sha256']
assert H(pretty(config['selected_xml_cases']))==config['selected_xml_cases_sha256']
assert config['cpu_affinity']==[112,113,114,115]and config['runtime_module_origins']=={}
assert 'numpy'in config['forbidden_roots']and 'jax'in config['forbidden_roots']
assert config['expected_regular_child_files']==144 and config['expected_owned_public_fixture_files']==6
assert config['expected_parent_records']==config['expected_child_records']==36
assert config['synthetic_only']is True and config['actual_execution_authorized']is False
assert proposal['argv']==handoff['argv']
assert proposal['argv'][:2]==['env','-i']and proposal['argv'][-4:]==['/tmp/cross-pretrained-validation-env/bin/python','-I','-B',str(P/'executed_driver.py')]
assert proposal['argv'][proposal['argv'].index('taskset')+1:proposal['argv'].index('taskset')+3]==['-c','112-115']
trees={name:ast.parse((P/name).read_bytes(),filename=name)for name in('executed_driver.py','worker_driver.py','before_executed_driver.py','before_worker_driver.py')}
def constants(tree):
    return {n.targets[0].id:ast.literal_eval(n.value)for n in tree.body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('CONFIG_SHA256','WORKER_SHA256')}
pins=constants(trees['executed_driver.py'])
assert pins==dict(CONFIG_SHA256=H((P/'run_config.json').read_bytes()),WORKER_SHA256=H((P/'worker_driver.py').read_bytes()))
def ownership(tree):
    main=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='main')
    with_log=next(n for n in main.body if isinstance(n,ast.With)and n.body and isinstance(n.body[0],ast.Assign)and isinstance(n.body[0].value,ast.Call)and ast.unparse(n.body[0].value.func)=='subprocess.Popen')
    assert len(with_log.body)==2 and isinstance(with_log.body[1],ast.Try)
    return ast.dump(with_log.body[1],include_attributes=False)
assert ownership(trees['executed_driver.py'])==ownership(trees['before_executed_driver.py'])
baseline=Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1')
assert (P/'before_executed_driver.py').read_bytes()==(baseline/'executed_driver.py').read_bytes()
assert (P/'before_worker_driver.py').read_bytes()==(baseline/'worker_driver.py').read_bytes()
for name in('report.json','worker-audit.json','tests.xml','tests.log','live-observation.json','tmp','pytest-owned'):assert not (P/name).exists(),name
report=dict(schema='qwen-server-snapshot-integration-harness-source-peer-review-v1',
    conclusion='No concrete blocker found in the held unrun harness delta.',harness_root=str(P),
    handoff_sha256=H((P/'handoff.json').read_bytes()),driver_sha256=H((P/'executed_driver.py').read_bytes()),
    worker_sha256=H((P/'worker_driver.py').read_bytes()),config_sha256=H((P/'run_config.json').read_bytes()),
    launch_proposal_sha256=H((P/'launch-proposal.json').read_bytes()),source_manifest_sha256=H(raw),source_archive_sha256=H((P/'sources.tar.gz').read_bytes()),
    source_count=404,runtime_count=391,static_count=36,current_execution_archive_equal=True,
    exact_original_device50_owned_popen_try=True,canonical_compact_source_map=True,
    review=[
        'Full driver/worker and declared patches read. Config/source/archive are checked before Popen; worker validates nested391 and exact ordered collection before any fixture.',
        'Original device50 immediate Popen ownership, abnormal group cleanup and unconditional direct wait are unchanged; no healthy deadline. Outer launch and child use isolated -I -B, sanitized environment and CPU112-115.',
        'NumPy and backend imports/discovery remain forbidden in pytest worker; no invented installed-module origin table. Source-owned child doubles and definition-only entry are explicitly disclosed.',
        'Inspector requires36 unique case joins, each with original early parent record, positive terminal, result and log:144 regular artifacts. Six <=1024-byte owned public fixture files are tied exactly to process cases.',
        'Pytest aliases are bounded metadata records using lstat/readlink plus owned canonical target checks; os.walk does not follow them. Alias count is not assumed. All owned metadata is bounded by256 entries/16MiB.',
        'PASS requires exact XML order, zero failure/error/skip, matching collected nodes, runtime/source current/execution/archive equality, empty audit/backend sets and owned child/group absence.',
        'Success-only inspector may raise on missing failed-child result/positive records. The early parent records/logs remain evidence; no complete report is promised for setup failures.'
    ],candidate_imports=False,collection=False,tests=False,backend_or_capacity_probes=False,borrowed_body_reads=False,launch_authorized=False)
(O/'harness-review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(json.dumps({'harness_review_sha256':H((O/'harness-review.json').read_bytes()),'static_count':36,'unrun':True},sort_keys=True))
