"""Public source/evidence review only: no candidate or NumPy imports/rerun."""
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile
import xml.etree.ElementTree as ET

HERE=Path(__file__).parent
BASE=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104')
ROOT_REVIEW=Path('/tmp/qwen-toy5-checkpoint-root-review-v2-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(path):return json.loads(path.read_bytes())
def inventory(path):
    assert not any(p.is_symlink()for p in path.rglob('*'))
    return{str(p.relative_to(path)):sha(p.read_bytes())for p in path.rglob('*')if p.is_file()}
versions=[]
for version,root in [('preparation-v1',Path('/tmp/qwen-toy5-checkpoint-tests-20261104')),('preparation-v2',Path('/tmp/qwen-toy5-checkpoint-tests-v2-20261104'))]:
    proof=BASE/version;config=read(proof/'run_config.json');mapping=read(proof/'sources-before.json')
    assert len(mapping)==261 and sha((proof/'sources-before.json').read_bytes())==config['source_manifest_sha256']
    assert read(proof/'sources-after.json')==read(proof/'execution-after.json')==mapping
    assert inventory(root)==inventory(proof/'execution')==mapping
    archive=(proof/'sources.tar.gz').read_bytes();assert sha(archive)==config['source_archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz')as tar:
        members=tar.getmembers();assert len(members)==261 and all(m.isfile()for m in members)
        assert{m.name:sha(tar.extractfile(m).read())for m in members}==mapping
    xml=ET.parse(proof/'tests.xml').getroot()
    counts={n:len(list(xml.iter(tag)))for n,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
    rows=[dict(classname=n.attrib['classname'],name=n.attrib['name'])for n in xml.iter('testcase')]
    audit=read(proof/'worker-audit.json');report=read(proof/'report.json')
    assert rows==config['selected_xml_cases']==report['selected_xml_cases']
    assert audit['collected_nodes']==config['selected_nodes']==report['selected_nodes']
    assert len(rows)==48 and counts==report['tests']
    assert audit['denied']==audit['loaded_backend']==[]and audit['numpy_imported']is True
    assert audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']and len(config['runtime_source_expectations'])==4
    assert audit['runtime_module_origins']=={n:dict(file=p,spec=p)for n,p in config['runtime_module_origins'].items()}
    assert report['sources_unchanged']is True and report['child_reaped']is True and report['child_group_absent']is True
    versions.append(dict(proof=proof,source=root,config=config,mapping=mapping,counts=counts,audit=audit,report=report))
old,new=versions
assert old['counts']==dict(tests=48,failures=6,errors=0,skipped=0)and old['report']['passed']is False and old['report']['child_returncode']==1
assert new['counts']==dict(tests=48,failures=0,errors=0,skipped=0)and new['report']['passed']is True and new['report']['child_returncode']==0
changed=[n for n in old['mapping']if old['mapping'][n]!=new['mapping'][n]]
assert changed==['q5_test_fixtures.py']
assert sum(n.startswith('runtime/')for n in new['mapping'])==250
old_fixture=(old['source']/'q5_test_fixtures.py').read_bytes();new_fixture=(new['source']/'q5_test_fixtures.py').read_bytes()
def indices(blob):
    tree=ast.parse(blob)
    return ast.literal_eval(next(n.value for n in tree.body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='EXPECTED_SELECTED_TRACE_INDICES'for t in n.targets)))
assert indices(old_fixture)==indices(new_fixture)and len(indices(new_fixture))==47
assert 'layer_phases' not in old_fixture.decode()
assert "for layer in range(config['num_hidden_layers'])for phase in ('attention','MLP')"in new_fixture.decode()
assert "(kind=='phase' and label in layer_phases)"in new_fixture.decode()
correction=read(new['proof']/'source-correction.json')
assert correction['unchanged_files']==260 and correction['runtime250_unchanged']is True
phase_counts=correction['public_nominal_trace_phase_counts']
six={f'packed Qwen layer {i}: {name} ready'for i in range(3)for name in('attention','MLP')}
assert {k for k,v in phase_counts.items()if v==1}==six
assert phase_counts['bounded homomorphic refresh starting']==phase_counts['bounded homomorphic refresh complete']==19
frozen=new['source']/'runtime/reference/frozen/cadathlon/common'
graph=(frozen/'encrypted_packed_transformer.py').read_text();bounded=(frozen/'encrypted_bounded_arithmetic.py').read_text()
assert "f'packed Qwen layer {layer}: attention ready'"in graph and "f'packed Qwen layer {layer}: MLP ready'"in graph
assert "'bounded homomorphic refresh starting'"in bounded and "'bounded homomorphic refresh complete'"in bounded
for n in('executed_driver.py','worker_driver.py','run_config.json'):
    a=(old['proof']/n).read_bytes();b=(new['proof']/n).read_bytes()
    diff=''.join(difflib.unified_diff(a.decode().splitlines(True),b.decode().splitlines(True),fromfile='failed-preparation-v1/'+n,tofile='unrun-preparation-v2/'+n)).encode()
    assert diff==(new['proof']/(n+'.patch')).read_bytes()
worker=(new['proof']/'worker_driver.py').read_text()
reverse=worker.replace("assert config['pytest_config_path'] == '/dev/null'\n",'').replace("['-q', '-c', '/dev/null', '-p'","['-q', '-p'")
assert reverse==(old['proof']/'worker_driver.py').read_text()
def defs(path):
    return{n.name:ast.dump(n,include_attributes=False)for n in ast.parse(path.read_bytes()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
assert defs(old['proof']/'executed_driver.py')==defs(new['proof']/'executed_driver.py')
root_raw=(ROOT_REVIEW/'terminal-review.json').read_bytes();root= json.loads(root_raw)
assert root['passed']is True and root['counts']==new['counts']
for n,h in root['evidence'].items():assert sha((new['proof']/n).read_bytes())==h
prior=read(new['proof']/'predecessor-root-terminal-review.json')
assert prior['passed']is False and prior['counts']==old['counts']and prior['passed_testcases']==42
for n,h in prior['evidence'].items():assert sha((old['proof']/n).read_bytes())==h
owner_raw=(ROOT_REVIEW/'direct-owner.json').read_bytes();owner=json.loads(owner_raw)
assert sha(owner_raw)==root['direct_owner_receipt_sha256']
assert owner['session_id']==32409 and owner['terminal_tool_chunk']=='d48ad9'and owner['launch_tool_chunk']=='aa8679'
assert owner['exit_code']==0 and owner['direct_session_reaped']is True
live=read(new['proof']/'live-observation.json');assert new['audit']['pid']==live['worker_pid']and new['audit']['pgid']==live['worker_pgid']
assert root['outer_and_worker_pid_and_group_absent']is True and root['independent_live_observation']is False
for v in versions:assert inventory(v['source'])==inventory(v['proof']/'execution')==v['mapping']
report={
 'schema':'q5-fixture-v2-public-source-terminal-peer-review-v1','passed':True,
 'source_manifest_sha256':new['config']['source_manifest_sha256'],'source_archive_sha256':new['config']['source_archive_sha256'],
 'fixture_before_sha256':sha(old_fixture),'fixture_after_sha256':sha(new_fixture),'unchanged_files':260,'runtime250_unchanged':True,'expected47_indices_unchanged':True,
 'source_selection':'Six exact original attention/MLP layer-end labels exclude the38 refresh start/complete bookkeeping phase records; input/output/refresh_before/refresh_after/final-normalization selectors unchanged. No expected numeric index used as selector.',
 'worker_delta':'Only explicit -c /dev/null and its exact config assertion; driver function AST unchanged apart from external constant pins.',
 'terminal_counts':new['counts'],'preserved_predecessor_counts':old['counts'],'predecessor_passed_cases':42,
 'ordered48_xml_audit_exact':True,'zero_backend_audit_denials':True,'genuine_numpy_reported':True,
 'root_terminal_receipt_sha256':sha(root_raw),'owner_receipt_sha256':sha(owner_raw),'owner_session':32409,'owner_terminal_tool_chunk':'d48ad9','owner_exit_code':0,'owner_reaped':True,
 'outer_worker_process_absence':'Verified in bound independent root terminal evidence; this peer did not claim a live observation or perform a new process probe.',
 'source_evidence_current_before_end_execution_archive_exact':True,
 'scope':'Public toy NumPy arithmetic and synthetic refresh metadata; no prior140 result transfer, real encrypted capture, first encrypted-divergence, HE/model/backend execution or capacity proof.',
 'candidate_imported':False,'numpy_imported_by_reviewer':False,'tests_rerun':False,'body_reads':False,'execution_authorized':False,
 'review_script_sha256':sha(Path(__file__).read_bytes()),
}
out=HERE/'review.json'
with out.open('xb')as f:f.write((json.dumps(report,sort_keys=True,indent=2)+'\n').encode())
print(json.dumps({'passed':True,'receipt':str(out),'sha256':sha(out.read_bytes()),'scope':'source/public terminal only'}))
