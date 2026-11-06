import ast, hashlib, json, pathlib, tarfile
ROOT=pathlib.Path('/tmp/qwen-device-component-tests-20261104')
HAND=pathlib.Path('/tmp/qwen-device-component-test-handoff-20261104')
OUT=pathlib.Path(__file__).parent
H=lambda b:hashlib.sha256(b).hexdigest()
hraw=(HAND/'handoff.json').read_bytes(); h=json.loads(hraw)
assert H(hraw)=='2d454c100eae3984010b051ac1097d2fd9ad57804d7e780fa3a443aedddbb5f7'
mraw=pathlib.Path(h['source_manifest']).read_bytes(); m=json.loads(mraw)
assert H(mraw)==h['source_manifest_sha256'] and m==h['source_hashes'] and len(m)==49
actual={p.relative_to(ROOT).as_posix():H(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file()}
assert not any(p.is_symlink() for p in ROOT.rglob('*')) and actual==m
archive=HAND/'sources.tar.gz'; assert H(archive.read_bytes())==h['source_archive_sha256']
with tarfile.open(archive) as t:
    members=t.getmembers(); assert len(members)==49 and all(x.isfile() for x in members)
    assert {x.name:H(t.extractfile(x).read()) for x in members}==m
runtime=json.loads((ROOT/'runtime_sources.json').read_bytes()); assert len(runtime)==34
assert {k[8:]:v for k,v in m.items() if k.startswith('runtime/')}==runtime
copies=json.loads((ROOT/'copied_sources.json').read_bytes()); assert len(copies)==36
for name, row in copies.items(): assert m[name]==row['sha256']==H(pathlib.Path(row['path']).read_bytes())
cases=ast.literal_eval(next(n.value for n in ast.parse((ROOT/'device_cases.py').read_text()).body if isinstance(n,ast.Assign) and n.targets[0].id=='CASES'))
assert len(cases)==30 and len(set(cases))==30
nodes=['test_device_component.py::test_device_component['+x+']' for x in cases]
assert nodes==h['static_nodeids']==json.loads((ROOT/'selected_tests.json').read_bytes())['static_nodeids']
child=ast.parse((ROOT/'device_child.py').read_text())
fn=next(n for n in ast.walk(child) if isinstance(n,ast.FunctionDef) and n.name=='retained_failed')
called={n.func.attr for n in ast.walk(fn) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
assert 'issue_initialization' in called and not({'initialize_once','observe_capacity'}&called)
report={'schema':'qwen-device-tests-security-source-review-v1','source_root':str(ROOT),'handoff_sha256':H(hraw),'source_manifest_sha256':H(mraw),'source_archive_sha256':h['source_archive_sha256'],'source_count':49,'runtime_count':34,'copied_exact_count':36,'static_count':30,'source_hashes':m,'disposition':'source scope coherent; one proof coverage gap retained','finding':{'kind':'proof coverage; no runtime failure claim','target':'device_child.py retained_failed','description':'Failed-state/private original retention and repeat issuance are checked, but original permit is not retried after restoring the original fields/hooks/device and benign fixture behavior. Repeat issuance alone is always denied after a clean one-use issuance.','narrow_successor':'Restore only fixture mutations and benign hook/statistics values, then require the original permit otherwise-valid public transition to reject without new hook, initialization, or sampling calls; never reset component state/registry.'},'source_read_conclusions':['Real resource authenticate_policy/logical_facts are copied and observed through original code-object profiling, not replaced. Two phase check methods are explicitly recording seams.','Synthetic JAX-shaped owned module/process hooks and policy values are disclosed; no installed backend, real device selection or capacity sufficiency established.','Closed 30 cases include repinned semantic document negatives, original policy/dict/field retention, hook/descriptor substitution, stage ordering, initialization failure, caught reentry and missing/invalid stats.','Parent immediately owns subprocess, waits/reaps, records early parent terminal before child-success assertions; child restores recording seams and checks source/module absence.'], 'candidate_imports':False,'collection':False,'tests_executed':False,'backend_imports':False,'body_or_capacity_reads':False,'actual_authority':False}
(OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
print(H((OUT/'review.json').read_bytes()))
