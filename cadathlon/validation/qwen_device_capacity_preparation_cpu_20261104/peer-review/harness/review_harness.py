import ast, hashlib, json, pathlib, tarfile
P=pathlib.Path('/tmp/qwen-device-component-focused-proof-20261104/preparation-v1'); O=pathlib.Path(__file__).parent
H=lambda x:hashlib.sha256(x).hexdigest()
J=lambda p:json.loads(p.read_bytes())
hraw=(P/'handoff.json').read_bytes();assert H(hraw)=='b46829a7876dc45f445d36dbb05271ca579548ec3c26fed2016d441f4aa350a9';h=json.loads(hraw)
for n,v in h['files'].items():assert not(P/n).is_symlink() and H((P/n).read_bytes())==v
craw=(P/'run_config.json').read_bytes();c=json.loads(craw);assert H(craw)==h['config_sha256']
sraw=(P/'sources-before.json').read_bytes();m=json.loads(sraw)
assert sraw==(json.dumps(m,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
assert H(sraw)==c['source_manifest_sha256']==h['source_manifest_sha256'] and len(m)==50
assert H((P/'sources.tar.gz').read_bytes())==c['source_archive_sha256']==h['source_archive_sha256']
def current(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():H(p.read_bytes())for p in root.rglob('*')if p.is_file()}
source=pathlib.Path(c['source_root']);assert current(source)==current(P/'execution')==m
with tarfile.open(P/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==50 and all(x.isfile()for x in members)
    assert {x.name:H(t.extractfile(x).read())for x in members}==m
runtime={k[8:]:v for k,v in m.items()if k.startswith('runtime/')};assert len(runtime)==34 and runtime==c['runtime_source_expectations']
cases=ast.literal_eval(next(n.value for n in ast.parse((source/'device_cases.py').read_text()).body if isinstance(n,ast.Assign)))
nodes=['test_device_component.py::test_device_component['+x+']'for x in cases]
assert len(nodes)==30 and nodes==c['selected_nodes']==J(source/'selected_tests.json')['static_nodeids']
canonical=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
assert H(canonical(nodes))==c['selected_nodes_sha256']
assert c['selected_xml_cases']==[dict(classname='test_device_component',name='test_device_component['+x+']')for x in cases]
assert H(canonical(c['selected_xml_cases']))==c['selected_xml_cases_sha256']
assert c['cpu_affinity']==[112,113,114,115] and c['runtime_module_origins']=={} and 'numpy'in c['forbidden_roots']
assert c['expected_regular_child_files']==120 and c['expected_child_records']==c['expected_parent_records']==30
assert c['synthetic_only']is True and c['actual_execution_authorized']is False
assert h['argv'][0:2]==['env','-i'] and h['argv'][-3:-1]==['-I','-B']
driver=ast.parse((P/'executed_driver.py').read_text());before=ast.parse((P/'before_executed_driver.py').read_text())
worker=ast.parse((P/'worker_driver.py').read_text())
assign={n.targets[0].id:ast.literal_eval(n.value)for n in driver.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id in('CONFIG_SHA256','WORKER_SHA256')}
assert assign==dict(CONFIG_SHA256=H(craw),WORKER_SHA256=H((P/'worker_driver.py').read_bytes()))
def popen_with(tree):
    return next(n for n in ast.walk(tree)if isinstance(n,ast.With)and any(isinstance(x,ast.Call)and isinstance(x.func,ast.Attribute)and x.func.attr=='Popen'for x in ast.walk(n)))
assert ast.dump(popen_with(driver),include_attributes=False)==ast.dump(popen_with(before),include_attributes=False)
# Source-only ownership and selection trace. No evaluated candidate AST/code.
text=(P/'worker_driver.py').read_text(); assert text.index('sys.addaudithook(audit)')<text.index('import pytest')<text.index('status=pytest.main(')
assert "if collected!=config['selected_nodes']"in text and 'def pytest_collection_finish' in text
assert not any((P/n).exists()for n in('report.json','tests.xml','worker-audit.json','tmp','pytest-owned'))
report=dict(schema='qwen-device-focused-harness-security-source-review-v1',handoff_sha256=H(hraw),source_manifest_sha256=H(sraw),source_archive_sha256=c['source_archive_sha256'],driver_sha256=h['driver_sha256'],worker_sha256=h['worker_sha256'],config_sha256=H(craw),source_count=50,runtime_count=34,static_count=30,owned_Popen_cleanup_AST_exact=True,disposition='no concrete source blocker in requested harness scope',reviewed_files=h['files'],conclusions=['Current source/execution/archive50 and runtime34 exact; indent2 sorted newline map encoding matches driver before/end checks.','Exact static30 order/XML and source-fixed prefixture collection check; sanitized -I -B worker on112-115, backend/NumPy discovery and import denied, no artificial genuine origins.','Worker remains backend-free; each fresh child separately allows only exact source-owned temporary JAX-shaped fixture and restores it. Genuine native APIs/capacity are not exercised.','Immediate outer Popen ownership with fresh session, original exception preservation, directwait/reap without deadline and postexit group absence preserved AST-identically from admitted snapshot harness.','Collector requires120 regular owned child files:30 early parent terminal,30 positive terminal,30 result,30 log; typed PID/PGID/exit/reap/absence, scope/restoration and real-resource counter joins.','Owned pytest aliases bounded and recorded through lstat/readlink without traversal; variable alias count is honest.','Child local module origins are completed source-bound assertions, not a separately collected origin table. Harness refuses a clean PASS when child receipts/results are missing; source does not guarantee final report on incomplete execution.'],candidate_imports=False,collection=False,execution=False,backend_imports=False,capacity_or_body_probes=False,actual_authority=False)
(O/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n');print(H((O/'review.json').read_bytes()))
