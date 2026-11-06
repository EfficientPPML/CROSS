"""Unrun harness successor only. No candidate imports, collection or execution."""
import ast
import difflib
import hashlib
import json
from pathlib import Path
import tarfile

BASE=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')
OUT=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v2')
SOURCE=Path('/tmp/qwen-toy5-checkpoint-tests-v2-20261104')
MAP=Path('/tmp/qwen-toy5-checkpoint-test-sources-v2-20261104.json')
SH=Path('/tmp/qwen-toy5-checkpoint-test-source-handoff-v2-20261104')
ROOT_REVIEW=Path('/tmp/qwen-toy5-checkpoint-root-review-20261104/terminal-review.json')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def save(name,raw):
    if not isinstance(raw,bytes):raw=raw.encode()if isinstance(raw,str)else canonical(raw)
    with(OUT/name).open('xb')as f:f.write(raw)
def change(text,before,after):
    assert text.count(before)==1,before
    return text.replace(before,after)
old_handoff=json.loads((BASE/'handoff.json').read_bytes())
for name,pin in old_handoff['files'].items():assert sha((BASE/name).read_bytes())==pin
old_pins={name:sha((BASE/name).read_bytes())for name in ['executed_driver.py','worker_driver.py','run_config.json','handoff.json','sources-before.json','sources-after.json','execution-after.json','sources.tar.gz','report.json','tests.log','tests.xml','worker-audit.json','live-observation.json']}
assert sha(ROOT_REVIEW.read_bytes())=='c04da9fe0ca28314435d1efd7755f5162ea6e1a31f625f880f7413685e036f5d'
assert old_pins['report.json']=='3b8576c1f84298b752a0f3f61aeb43cee228ca6ceba161e347ff0f63e3677d3c'
raw=MAP.read_bytes();expected=json.loads(raw)
assert raw==compact(expected)and len(expected)==261
assert sha(raw)=='1136d1a47ed36c2ae2723bb8f3c05eca2f64a18b7ae9de3533c6ea1073b64d7a'
execution=OUT/'execution';execution.mkdir()
for name,pin in expected.items():
    p=SOURCE/name;assert p.is_file()and not p.is_symlink();data=p.read_bytes();assert sha(data)==pin
    q=execution/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(data)
save('sources-before.json',raw)
archive=(SH/'sources.tar.gz').read_bytes()
assert sha(archive)=='442ab2df032b451c9e067a49e046d26c661fa967b70aa026e117d20b874f9463'
save('sources.tar.gz',archive)
with tarfile.open(OUT/'sources.tar.gz')as tar:assert {m.name:sha(tar.extractfile(m).read())for m in tar}==expected
config=json.loads((BASE/'run_config.json').read_bytes())
config.update(source_root=str(SOURCE),proof_root=str(OUT),source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),pytest_basetemp=str(OUT/'pytest-owned'),pytest_config_path='/dev/null',pytest_config_policy='Explicit -c /dev/null; no pytest configuration discovery.',source_correction_sha256=sha((SH/'correction.json').read_bytes()))
selection=json.loads((SOURCE/'selected_tests.json').read_bytes())
assert config['selected_nodes']==selection['ordered_nodes']and len(config['selected_nodes'])==48
assert config['source_count']==261 and config['cpu_affinity']==[108,109,110,111]
assert config['source_manifest_encoding']=='canonical sorted compact JSON plus newline'
save('run_config.json',config);config_sha=sha((OUT/'run_config.json').read_bytes())
before_worker=(BASE/'worker_driver.py').read_text()
worker=change(before_worker,"status = pytest.main(['-q', '-p', 'no:cacheprovider', '--rootdir=' + str(source),", "assert config['pytest_config_path'] == '/dev/null'\nstatus = pytest.main(['-q', '-c', '/dev/null', '-p', 'no:cacheprovider', '--rootdir=' + str(source),")
save('worker_driver.py',worker);worker_sha=sha(worker.encode())
before_driver=(BASE/'executed_driver.py').read_text();driver=before_driver
for name,new in [('CONFIG_SHA256',config_sha),('WORKER_SHA256',worker_sha)]:
    node=next(n for n in ast.parse(driver).body if isinstance(n,ast.Assign)and len(n.targets)==1 and isinstance(n.targets[0],ast.Name)and n.targets[0].id==name)
    driver=change(driver,name+' = '+repr(ast.literal_eval(node.value)),name+' = '+repr(new))
save('executed_driver.py',driver);driver_sha=sha(driver.encode())
def owned(text):
    main=next(n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)and n.name=='main')
    return ast.dump(next(n for n in main.body if isinstance(n,ast.With)),include_attributes=False)
assert owned(driver)==owned(before_driver)
for name,text in [('executed_driver.py',driver),('worker_driver.py',worker)]:ast.parse(text,filename=name)
for name,before,after in [('executed_driver.py',before_driver,driver),('worker_driver.py',before_worker,worker),('run_config.json',(BASE/'run_config.json').read_text(),(OUT/'run_config.json').read_text())]:
    save(name+'.patch',''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='failed-preparation-v1/'+name,tofile='unrun-preparation-v2/'+name)))
save('source-handoff.json',(SH/'handoff.json').read_bytes())
save('source-correction.json',(SH/'correction.json').read_bytes())
save('predecessor-root-terminal-review.json',ROOT_REVIEW.read_bytes())
save('driver_provenance.json',dict(schema='q5-focused-harness-v2-source-provenance-v1',base=str(BASE),predecessor_pins=old_pins,predecessor_root_terminal_sha256=sha(ROOT_REVIEW.read_bytes()),predecessor_result=dict(tests=48,passed=42,failed=6,exit_code=1,owner='root'),changes=['Source/proof/basetemp paths and corrected261 map/archive pins','Explicit -c /dev/null in pytest argument vector with matching configuration metadata','Resulting config/worker/driver pins; original owned-Popen block unchanged'],source_count=261,static_cases=48,immediate_Popen_ownership_AST_exact=True,failed_proof_preserved=True,collection=False,candidate_imports=False,execution=False,automatic_retry=False,actual_grant=False))
proposal=json.loads((BASE/'launch-proposal.json').read_bytes())
proposal.update(driver_sha256=driver_sha,worker_sha256=worker_sha,config_sha256=config_sha,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),pytest_config_path='/dev/null')
proposal['argv'][-1]=str(OUT/'executed_driver.py')
assert proposal['authorized']is False and proposal['executed']is False
save('launch-proposal.json',proposal)
save('README.md','''# Q5 focused harness preparation-v2 — UNRUN

This successor selects the exact corrected261/48 sources. Only the fixture's semantic layer-phase selector changed; all250 runtime files and all48 test assertions, including independent47 trace indices, remain unchanged. The original preparation-v1 failed42PASS/6FAIL because its fixture also captured38 refresh-start/complete implementation phases. Its sources, execution copy, harness and complete failed evidence remain unchanged; the root terminal receipt is preserved here by exact bytes.

The pytest argument vector now explicitly includes `-c /dev/null`, preventing configuration discovery. The driver changes only its two derived config/worker pins. The immediate-Popen ownership block remains AST-identical to v1 and the admitted pure53v2 model; abnormal group cleanup always attempts direct wait, and there is no healthy deadline. CPU108–111, sanitized environment, Python-I-B, plugin suppression, exact48 prefixture selection, compact source maps, NumPy4 source hashes/root-only origin and backend denials are unchanged. Fresh own pytest basetemp stays outside execution and has the same128-file/16MiB post-run retention bound, not OS confinement.

No successor imports, collection, arithmetic, probes, tests or role execution occurred. This is not a retry grant. Root review and a separate explicit authorization remain required. No prior140 PASS or first encrypted-divergence claim transfers.
''')
for name,pin in old_pins.items():assert sha((BASE/name).read_bytes())==pin
for name,pin in expected.items():assert sha((SOURCE/name).read_bytes())==sha((execution/name).read_bytes())==pin
files={p.name:sha(p.read_bytes())for p in OUT.iterdir()if p.is_file()}
handoff=dict(schema='q5-focused-harness-v2-held-handoff-v1',proof_root=str(OUT),source_root=str(SOURCE),source_count=261,static_cases=48,source_manifest_sha256=sha(raw),source_archive_sha256=sha(archive),driver_sha256=driver_sha,worker_sha256=worker_sha,config_sha256=config_sha,proposal_sha256=files['launch-proposal.json'],files=files,argv=proposal['argv'],immediate_Popen_ownership_AST_exact=True,predecessor_failed_proof_unchanged=True,explicit_no_pytest_config='/dev/null',source_current_execution_archive_equal=True,candidate_imports=False,collection=False,execution=False,probes=False,capacity_observations=False,actual_execution_authorized=False,ready_for_source_review_only=True)
save('handoff.json',handoff)
print(json.dumps({k:v for k,v in handoff.items()if k not in ('files','argv')},sort_keys=True))
