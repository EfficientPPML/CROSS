"""Public frozen-source launch review for nineteen recording-only cases."""
from pathlib import Path
import datetime,hashlib,json,tarfile
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_bytes())
source=Path('/tmp/qwen-client-stage-glue-v2-20261104')
proof=Path('/tmp/qwen-client-stage-boundary-proof-20261104/development-v1')
manifest=Path('/tmp/qwen-client-stage-glue-sources-v2-20261104.json')
assert sha(manifest)=='7837d7c4d2a3d23ca77c7d3ff58052b45b4cda309f5cb6c234afb08fa5772f1b'
expected=load(manifest);assert len(expected)==373
parent=load(Path('/tmp/qwen-client-stage-glue-review-sources-v1-20261104.json'))
assert len(parent)==370 and all(expected[n]==h for n,h in parent.items())
assert set(expected)-set(parent)=={'client_parent370_manifest.json','test_client_stage.py','client_test_origin.json'}
for root in (source,proof/'execution'):
    assert {p.relative_to(root).as_posix():sha(p)for p in root.rglob('*')if p.is_file()}==expected
assert (proof/'sources-before.json').read_bytes()==manifest.read_bytes()
assert sha(proof/'sources.tar.gz')=='afc12d6b4e649aba105fd6a4de2c31aed4edad383eac6feaa9ceb927770510ab'
with tarfile.open(proof/'sources.tar.gz')as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==373 and all(m.isfile()for m in members)
    assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()for m in members}==expected
driver=Path('/tmp/qwen-client-stage-boundary-driver-20261104.py')
assert sha(driver)=='2a59ca397137e2bdce8286f4fcac0815d0f0eea0830bb38962316cbad215fb25'
assert driver.read_bytes()==(proof/'executed_driver.py').read_bytes()
assert sha(proof/'worker_driver.py')=='7b4cdc5e9726db658e1945df15366c2f54d67a397b11d842e590f106b3100d7f'
prep=load(proof/'preparation.json')
assert prep['driver_sha256']==sha(driver)and prep['worker_driver_sha256']==sha(proof/'worker_driver.py')
assert prep['manifest_sha256']==sha(manifest)and prep['archive_sha256']==sha(proof/'sources.tar.gz')
assert prep['executed']==0 and prep['selected']==19
origin=load(source/'client_test_origin.json');assert len(origin['selected_nodes'])==19
assert origin['selected_test_files']==['test_client_stage.py']and origin['runtime_changes']==[]
assert expected['client_stage.py']=='e822bb2d77c8310875458c4a7821e9e9d48df977508757dbd6736de7909945dd'
assert expected['test_client_stage.py']=='dc83cf4cc15e49850532de108a2e8df9ce87d1b3b4b30dc263111018d8c44295'
assert not (proof/'tests.log').exists()and not (proof/'report.json').exists()
out=Path(__file__).with_name('launch-review.json');assert not out.exists()
result=dict(schema='qwen-client-boundary-launch-review-v1',passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(source),proof_root=str(proof),source_count=373,parent370_unchanged=True,source_manifest_sha256=sha(manifest),source_archive_sha256=sha(proof/'sources.tar.gz'),driver_sha256=sha(driver),worker_driver_sha256=sha(proof/'worker_driver.py'),runtime_sha256=expected['client_stage.py'],test_sha256=expected['test_client_stage.py'],selected_nodes=origin['selected_nodes'],expected_tests=19,cpu_affinity=[108,109,110,111],scope='One source-bound synthetic-only client boundary proof, using owned physical-shape files and recording key/context/security/signal/codec seams. Synthetic producer receipts are test fixtures. Actual backend imports denied; actual roles/keygen/model/private/body paths unexecuted. Collection log is planning evidence only, with no invented authoritative terminal. No inherited suite rerun or genuine HE grant.',script_sha256=sha(Path(__file__)))
out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,path=str(out),sha256=sha(out),sources=373,tests=19)))
