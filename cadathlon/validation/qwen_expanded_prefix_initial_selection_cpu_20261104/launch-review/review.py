"""Pin completed initial/public proofs and publish one initial-selection grant.

No private snapshot, token/model body, validation or coefficient selection here.
"""
from pathlib import Path
import hashlib,json,os,sys
SOURCE=Path('/tmp/qwen-expanded-prefix-selector-admission-preflight-v2-20261104/execution_source')
MAP=Path('/tmp/qwen-expanded-prefix-selector-admission-v2-sources-20261104.json')
OUT=Path('/tmp/qwen-expanded-prefix-selector-root-launch-20261104')
ACTUAL=Path('/tmp/qwen-expanded-prefix-selector-actual-v1-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def raw(path,pin):
    data=Path(path).read_bytes();assert sha(data)==pin;return data
def document(path,pin):return json.loads(raw(path,pin))
manifest_raw=raw(MAP,'f3d0bf68c503c2baba2cc34b7806c152efae153bbc180a4ebb85aaeae2d833f0')
manifest=json.loads(manifest_raw)
files={p.relative_to(SOURCE).as_posix():sha(p.read_bytes())for p in SOURCE.rglob('*')if p.is_file()}
assert files==manifest and len(files)==36 and not any(p.is_symlink()for p in SOURCE.rglob('*'))
preflight_pin='804fea6b0fbacaaaa226d50962e3e535ce6bdff62e0348c9febc896bfa120c3d'
review_pin='8b227233d673dfc41d5fbb997125e6a6c564250e8f501da519c805184168740f'
preflight=document('/tmp/qwen-expanded-prefix-selector-admission-preflight-v2-20261104/report.json',preflight_pin)
review=document('/tmp/qwen-expanded-prefix-selector-root-preflight-review-20261104/review.json',review_pin)
assert preflight['passed']is review['passed']is True and preflight['tests']==review['tests']==150
assert review['source_manifest_sha256']==sha(manifest_raw)and review['report_sha256']==preflight_pin
assert review['source_archive_sha256']==preflight['source_archive_sha256']and review['outer_exit_code']==0
assert review['outer_reaped']is review['process_groups_absent']is True
locations={
 'parent':('/tmp/qwen-expanded-prefix-observer-actual-v2-20261104/report.json','bc77c79e104af3f2f7c0da709bbf5c9319483ce7ad9a1dbb7cb3b8002510afb3'),
 'worker':('/tmp/qwen-expanded-prefix-observer-actual-v2-20261104/worker/report.json','58005996a1348ac96fc50aee270429cf29f27280675140ae294603943df3d68d'),
 'source_archive':('/tmp/qwen-expanded-prefix-observer-actual-v2-20261104/sources.tar.gz','d4ccebcca8691b78a00eb3af58cbb67e213e286038c112d7f85550484303791c'),
 'root_review':('/tmp/qwen-expanded-prefix-observer-terminal-root-review-20261104/review.json','edc9a06ef42108d8ccdefdf5a805c262c0a0f85c5d4ce900b3ec8b13a7feccad'),
 'reconstruction_script':('/tmp/qwen-initial-terminal-root-preflight-20261104/execution/root_qwen_initial3200_terminal_review_20261104.py','3690b5710a8f019f529062303289d415fad99d85bb579ff426e66ec9633ff2e3'),
 'root_reconstruction_outer_receipt':('/tmp/qwen-expanded-prefix-observer-terminal-outer-root-20261104/outer_receipt.json','120e6df818f6ea2207fa8c7b40aeaa34dcc709376294de909ba3438256c90644')}
data={name:raw(path,pin)for name,(path,pin)in locations.items()}
evidence={name:dict(path=path,sha256=pin,bytes=len(data[name]))for name,(path,pin)in locations.items()}
# Only authenticated frozen source is imported; its public-byte authenticator
# performs no IO or coefficient selection.
sys.path.insert(0,str(SOURCE))
import initial_proof as proof
from selection_grant import authenticate_grant
pins=proof.ProofPins(evidence['parent']['sha256'],evidence['worker']['sha256'],evidence['source_archive']['sha256'],
    evidence['root_review']['sha256'],(SOURCE/'inputs/observer_sources.json').read_bytes(),
    evidence['reconstruction_script']['sha256'],evidence['root_reconstruction_outer_receipt']['sha256'])
binding=proof.authenticate_initial(data['parent'],data['worker'],data['source_archive'],data['root_review'],pins=pins,root_outer_bytes=data['root_reconstruction_outer_receipt'])
assert binding.execution_authorized is False and binding.source_manifest_sha256==proof.SOURCE_MANIFEST_SHA256
assert not ACTUAL.exists()and not ACTUAL.is_symlink()and ACTUAL.resolve()==ACTUAL
for path,_ in locations.values():
    other=Path(path).parent
    assert ACTUAL!=other and ACTUAL not in other.parents and other not in ACTUAL.parents
grant=dict(schema='qwen-expanded-initial-selector-grant-v1',ready=True,execute_authorized=True,candidate_ordinal=1,
    source_manifest_sha256=sha(manifest_raw),observer_source_manifest_sha256=proof.SOURCE_MANIFEST_SHA256,
    protected_manifest_sha256=proof.BASELINE_SHA256,protocol_sha256=proof.PROTOCOL_SHA256,gate_bounds_sha256=proof.GATE_BOUNDS_SHA256,
    output_directory=str(ACTUAL),model_execution=False,token_or_calibration_body_reads=False,validation_body_reads=False,encrypted_execution=False,
    preflight_report_sha256=preflight_pin,independent_preflight_review_sha256=review_pin,initial_evidence=evidence)
grant_raw=(json.dumps(grant,indent=2,allow_nan=False)+'\n').encode()
authorization=authenticate_grant(grant_raw,sha(grant_raw),source_manifest_sha256=sha(manifest_raw),output_directory=str(ACTUAL))
assert authorization.grant_sha256==sha(grant_raw)
assert {p.relative_to(SOURCE).as_posix():sha(p.read_bytes())for p in SOURCE.rglob('*')if p.is_file()}==manifest
OUT.mkdir(mode=0o700)
review=dict(passed=True,scope='Public completed3200/root reconstruction and full150 selector admission review; one initial-only grant. No snapshot/token/model/validation read or coefficient selection.',
    source_manifest_sha256=sha(manifest_raw),preflight_report_sha256=preflight_pin,independent_preflight_review_sha256=review_pin,
    initial_proof_sha256=binding.proof_sha256,initial_evidence=evidence,initial_contexts=3200,native_calls=3232,baseline_free_contexts=64,
    initial_selection_ordinal=1,maximum_new_candidates=3,root_reconstruction_outer_independently_verified=True,
    private_snapshot_opened=False,calibration_or_model_body_opened=False,validation_read=False,coefficient_selection_performed=False,
    intended_cpu_affinity=[108,109,110,111],intended_numeric_threads=1,no_healthy_deadline=True,
    review_script_sha256=sha(Path(__file__).read_bytes()),grant_sha256=sha(grant_raw))
for name,value in (('review.json',(json.dumps(review,indent=2,allow_nan=False)+'\n').encode()),('launch.json',grant_raw),('review.py',Path(__file__).read_bytes())):
    fd=os.open(OUT/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(value)
print(json.dumps(dict(review=str(OUT/'review.json'),review_sha256=sha((OUT/'review.json').read_bytes()),grant=str(OUT/'launch.json'),grant_sha256=sha(grant_raw),initial_proof_sha256=binding.proof_sha256)))
