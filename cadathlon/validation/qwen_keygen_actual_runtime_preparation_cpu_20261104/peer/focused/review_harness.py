"""Static public harness review; target drivers/tests are never imported or run."""
import ast,hashlib,json,tarfile
from pathlib import Path
D=Path('/tmp/qwen-keygen-actual-focused-proof-20261104/preparation-v1');OUT=Path(__file__).parent
OLD=Path('/tmp/qwen-client-authority-boundary-proof-20261104/preparation-v2')
sha=lambda b:hashlib.sha256(b).hexdigest()
pretty=lambda x:(json.dumps(x,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
compact=lambda x:(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
h=json.loads((D/'handoff.json').read_bytes());c=json.loads((D/'run_config.json').read_bytes());p=json.loads((D/'launch-proposal.json').read_bytes())
for name,key in [('run_config.json','config_sha256'),('executed_driver.py','driver_sha256'),('worker_driver.py','worker_sha256'),('launch-proposal.json','launch_proposal_sha256')]:assert sha((D/name).read_bytes())==h[key]
raw=(D/'sources-before.json').read_bytes();m=json.loads(raw)
assert raw==compact(m) and sha(raw)==c['source_manifest_sha256'] and len(m)==97
for root in (Path(c['source_root']),D/'execution'):
    assert {x.relative_to(root).as_posix():sha(x.read_bytes())for x in root.rglob('*')if x.is_file()}==m
    assert not any(x.is_symlink()for x in root.rglob('*'))
with tarfile.open(D/'sources.tar.gz')as t:
    members=t.getmembers();assert len(members)==97 and all(x.isfile()for x in members)
    assert {x.name:sha(t.extractfile(x).read())for x in members}==m
assert sha((D/'sources.tar.gz').read_bytes())==h['source_archive_sha256']==c['source_archive_sha256']
selection=json.loads((Path(c['source_root'])/'selected_tests.json').read_bytes())
for n in ('selected_files','selected_nodes','selected_xml_cases'):assert c[n]==selection[n]
assert len(c['selected_nodes'])==61 and c['cpu_affinity']==h['cpu_affinity']==[112,113,114,115]
assert c['source_manifest_encoding']=='canonical sorted compact JSON plus newline'
assert c['runtime_source_expectations']==c['runtime_module_origins']=={}
assert 'numpy'in c['forbidden_roots'] and 'jax'in c['forbidden_roots']
for n in ('selected_nodes','selected_xml_cases'):assert sha(pretty(c[n]))==c[n+'_sha256']
assert p['argv'][:2]==['/usr/bin/env','-i']and '112-115'in p['argv']and '-I'in p['argv']and '-B'in p['argv']
assert p['healthy_job_deadline']is False and p['automatic_retry']is False and p['execution_authorized']is False
new=(D/'executed_driver.py').read_text();old=(OLD/'executed_driver.py').read_text()
def ownership(text):
    tree=ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node,ast.With):
            for i,item in enumerate(node.body):
                if isinstance(item,ast.Assign)and isinstance(item.value,ast.Call)and isinstance(item.value.func,ast.Attribute)and item.value.func.attr=='Popen':
                    assert isinstance(node.body[i+1],ast.Try)
                    return ast.dump(node.body[i+1],include_attributes=False)
    raise AssertionError('ownership try missing')
assert ownership(new)==ownership(old)
assert 'manifest == compact_map(expected)'in new
assert "(out / 'sources-after.json').write_bytes(compact_map(current))"in new
assert "(out / 'execution-after.json').write_bytes(compact_map(after))"in new
worker=(D/'worker_driver.py').read_text()
assert "runtime_expected == {} and config['runtime_module_origins'] == {}"in worker
assert "collected != config['selected_nodes']"in worker and 'pytest_collection_finish'in worker
assert "status = pytest.main"in worker and "'--import-mode=importlib'"in worker
assert 'sys.addaudithook(audit)'in worker and 'DenyBackend()'in worker
for term in ("audit['pytest_exit_code'] == 0","audit['pid'] == child.pid","audit['pgid'] == child_pgid","audit['affinity'] == [112, 113, 114, 115]","audit['numpy_imported'] is False"):
    assert term in new
pins={x.name:sha(x.read_bytes())for x in D.iterdir()if x.is_file()}
review=dict(schema='qwen-keygen-runtime-focused-harness-peer-review-v1',passed=True,source_only=True,source_count=97,static_nodes=61,
 proof_root=str(D),pins=pins,source_manifest_sha256=sha(raw),source_archive_sha256=h['source_archive_sha256'],
 prior_source_review_sha256=sha((OUT/'review.json').read_bytes()),review_script_sha256=sha(Path(__file__).read_bytes()),
 compact_source_map_and_terminal_maps=True,selection_hashes_pretty_separately_declared=True,empty_genuine_runtime_origins=True,
 cpu_affinity=[112,113,114,115],exact_post_Popen_ownership_AST_preserved=True,imports_collection_tests=False,execution_authorized=False,
 observations=['Exact61 collection is checked before fixtures; ordered XML/collection/return code/worker PID/PGID and affinity are independently checked by parent.',
 'Genuine NumPy/backend import and discovery are denied, with sticky denial and final loaded-name veto; temporary reviewed ModuleType doubles are restored before final audit and are not historical real-import evidence.',
 'All post-Popen observations/publication are inside owned try; exceptional group cleanup attempts direct wait, and healthy wait has no deadline.',
 'Config/worker pins and exact97 current/execution/archive are bound before launch; parent85 remains nested and exact.'],
 limitations=['No target import, collection, test execution, actual backend/body/capacity/RNG or launch grant.',
 'An initial source-read command requested nonexistent generic driver.py/worker.py names; immediately corrected to executed_driver.py/worker_driver.py. No candidate code ran or changed.'])
(OUT/'harness-review.json').write_bytes(pretty(review));print(json.dumps(dict(passed=True,sha256=sha(pretty(review)),tests=False)))
