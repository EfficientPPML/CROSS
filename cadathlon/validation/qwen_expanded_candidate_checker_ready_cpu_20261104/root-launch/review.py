"""Independent public proof review and a single actual calibration grant.

No calibration, validation, journal, snapshot or model body is opened here.
"""
from pathlib import Path
import hashlib, json, os, stat, sys, tarfile, xml.etree.ElementTree as ET

LAB=Path('/tmp/qwen-expanded-candidate-checker-ready-v2-20261104')
P=Path('/tmp/qwen-expanded-candidate-checker-ready-preflight-v2-20261104')
MAP=Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json')
OUT=Path('/tmp/qwen-expanded-candidate-checker-root-launch-20261104')
ACTUAL=Path('/tmp/qwen-expanded-candidate-checker-actual-v1-20261104')
PRIVATE=Path('/tmp/qwen-expanded-candidate-checker-private-v1-20261104')
MANIFEST='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
REPORT='8b8cfc6b99948a48229695b1325e91178fc6fcddaea0bad168f3b908b2533695'
ARCHIVE='7909d32548c7eea8974a9fdc21a508028d231716ef5c8cf6ed474b40a0d72901'
PEER=Path('/tmp/qwen-ready-checker-cnn-review-20261104/review.json')
PEER_SHA='e1a944017715838dd227a0915d1e621f6549922fac37710036a7b5896b44faeb'

def sha(raw): return hashlib.sha256(raw).hexdigest()
def load(path): return json.loads(path.read_bytes())
def inventory(root):
    result={}
    for p in root.rglob('*'):
        assert not p.is_symlink(), str(p)
        if p.is_file(): result[p.relative_to(root).as_posix()]=sha(p.read_bytes())
        else: assert p.is_dir(), str(p)
    return result
def jsonbytes(value): return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write(path,raw):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW|os.O_CLOEXEC,0o600)
    with os.fdopen(fd,'wb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())

assert sys.flags.isolated and sys.dont_write_bytecode
expected=load(MAP)
assert sha(MAP.read_bytes())==MANIFEST and len(expected)==139
assert (P/'source_manifest.json').read_bytes()==MAP.read_bytes()
assert inventory(LAB)==inventory(P/'execution_source')==expected
r=load(P/'report.json'); w=load(P/'worker/report.json'); outer=load(P/'owner_terminal.json')
assert sha((P/'report.json').read_bytes())==REPORT
assert r['worker']==w and sha((P/'worker/report.json').read_bytes())==r['worker_report_sha256']
flags=('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged',
       'terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash')
for row in (r,w):
    assert all(row.get(k) is True for k in flags)
    assert row['source_hashes_start']==row['source_hashes_end']==expected
    assert row['source_manifest_sha256']==MANIFEST and row['audit_denials']==[]
assert r['actual_candidate_stage'] is False and r['model_forward'] is False and r['coefficient_selection'] is False
assert type(r['child_returncode']) is int and r['child_returncode']==0
assert r['child_reaped'] is True and r['child_group_absent'] is True
assert w['tests']==738 and w['pytest_returncode']==w['failures']==w['errors']==w['skipped']==0
assert set(w['installed_import_denials'])=={'torch','transformers','jax','jaxlib','jaxite_word','tensorflow','tokenizers','safetensors','flax','torch_xla'}
expectations=load(LAB/'preflight_expectations.json')
assert expectations==dict(synthetic_tests=738,installed_import_denials=10,excluded_tests=['test_entry.py'],
    exclusion_reason='Exact historical observer bootstrap tests target its old helper closure. New test_checker_entry.py reproduces all12 probes for the actual checker entry.')
xml=ET.parse(P/'worker/tests.xml')
assert len(xml.findall('.//testcase'))==738 and not any(xml.findall('.//'+k) for k in ('failure','error','skipped'))
assert outer['session_id']==33399 and type(outer['exit_code']) is int and outer['exit_code']==0
assert all(outer[k] is True for k in ('reaped','producer_pids_absent','child_group_absent'))
assert outer['report_sha256']==REPORT and outer['source_archive_sha256']==ARCHIVE and outer['source_manifest_sha256']==MANIFEST
assert outer['pids']==[r['pid'],w['pid']] and r['child_pid']==r['child_pgid']==w['pid']
for pid in outer['pids']:
    assert not Path('/proc',str(pid)).exists()
    try: os.killpg(pid,0)
    except ProcessLookupError: pass
    else: raise AssertionError('preflight group still exists')
assert sha((P/'sources.tar.gz').read_bytes())==ARCHIVE==r['source_archive']['sha256']
with tarfile.open(P/'sources.tar.gz') as tar:
    members=tar.getmembers()
    assert len(members)==len({m.name for m in members})==139
    assert all(m.isfile() and m.name in expected and 0<m.size<=(64<<20 if m.name=='candidate_proof/selector-assessments.json' else 8<<20) for m in members)
    assert sum(m.size for m in members)<=96<<20
    blobs={m.name:tar.extractfile(m).read() for m in members}
assert {n:sha(b) for n,b in blobs.items()}==expected
assert json.loads(blobs['execution_authorization.json'])=={'execute':False}
assert sha(PEER.read_bytes())==PEER_SHA
peer=load(PEER)
assert peer['passed'] is True and peer['unresolved_concrete_findings']==[]
assert peer['reviewed_sources_before']==peer['reviewed_sources_after']
assert all(expected[n]==h for n,h in peer['reviewed_sources_after'].items())
old=load(Path('/tmp/qwen-expanded-candidate-checker-preflight-v2-20261104/source_manifest.json'))
six=('checker_runtime.py','strict_metrics.py','candidate_domains.py','test_checker_runtime.py','test_strict_metrics.py','test_candidate_domains.py')
assert all(expected[n]==old[n]==peer['numerical_six'][n]['parent115'] for n in six)
initial=load(Path('/tmp/qwen-expanded-prefix-observer-preflight-v5-20261104/source_manifest.json'))
stable=('arithmetic.py','reference.py','observed_reference.py','model_runtime.py','native_observer.py','ranges.py','workload.py','backend_environment.py','backend_sources.json','import_sources.json','import_smoke.py')
assert all(expected[n]==initial[n] for n in stable)
assert len([n for n in expected if n.startswith('candidate_proof/')])==17
assert expected['candidate_proof/selector-root-review.json']=='6c98b64cd506cddeb5f4bd64767e5b7e1483ba5d28ca04a442cd3d7b4fa960de'
assert expected['candidate_proof/selector-launch-review.json']=='50601486fdbf17b2a685316b2fb92d8ebccb9d9a3c95c2c5be36ab6e94fb13b1'
ext=json.loads(blobs['external_inputs.json'])
protected=tuple(Path(p) for p in ext['protected_paths']+json.loads(blobs['checker_protected_paths.json'])['paths'])
denials=[]
def audit(event,args):
    reason=None
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        path=Path(os.path.abspath(os.fsdecode(args[0]))).resolve()
        if any(path==p or p in path.parents for p in protected) or str(path).startswith('/home/jianming_gatech/cross_validation_assets/'):
            reason='private/token/model body'
    elif event.startswith('socket.') or event in ('subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.posix_spawnp'):
        reason='network or child execution'
    elif event=='import' and args and str(args[0]).split('.')[0] in {'torch','transformers','jax','jaxlib','tensorflow','flax','torch_xla','tokenizers','safetensors'}:
        reason='model backend'
    if reason:
        denials.append(reason)
        raise PermissionError(reason)
sys.addaudithook(audit)
sys.path.insert(0,str(P/'execution_source'))
import checker_admission, input_gate
binding=checker_admission.require_candidate(blobs)
external=input_gate.inspect(blobs)
assert binding.execution_authorized is False and binding.ordinal==1
assert binding.manifest_sha256=='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
assert binding.initial_proof_sha256=='2b4478125fa4162d1f9184c421213bdfc571f6e33bd0c56580fd00b3b4bbbf76'
assert external['calibration']==ext['calibration'] and external['calibration']['sha256']=='80eeaefad423167ee0710797d7eb76d8cf3a422e6f3b2e6069932d682835963b'
assert not any(n.split('.')[0] in {'torch','transformers','jax','jaxlib','tokenizers','safetensors'} for n in sys.modules)
assert inventory(LAB)==inventory(P/'execution_source')==expected and denials==[]
assert not any(p.exists() or p.is_symlink() for p in (OUT,ACTUAL,PRIVATE))
runtime=json.loads(blobs['checker_runtime_policy.json'])
assert runtime==dict(schema='qwen-expanded-candidate-runtime-policy-v1',cores=[172,173,174,175],torch_threads=4,torch_interop_threads=1,candidate_budget=3,measured_contexts=3200,native_forwards=3232,validation_enabled=False)
capacity=os.statvfs('/tmp'); available=capacity.f_bavail*capacity.f_frsize
assert available>8<<30
receipt=dict(schema='qwen-candidate-checker-root-launch-review-v1',passed=True,
    scope='Public source/proof review only; no actual candidate/model/private/token/validation body reads',
    preflight_report_sha256=REPORT,source_manifest_sha256=MANIFEST,source_archive_sha256=ARCHIVE,
    sources=139,tests=738,installed_import_denials=10,current_execution_start_end_archive_equal=True,
    preflight_session=33399,owner_terminal_sha256=sha((P/'owner_terminal.json').read_bytes()),outer_exit_code=0,outer_reaped=True,
    independently_absent_pids=outer['pids'],process_groups_absent=True,peer_review_sha256=PEER_SHA,
    numerical_six_unchanged=True,initial_observer_runtime_eleven_unchanged=True,public_proof_blobs=17,
    candidate_manifest_sha256=binding.manifest_sha256,candidate_binding_sha256=binding.binding_sha256,
    initial_proof_sha256=binding.initial_proof_sha256,selection_report_sha256=binding.selection_proof_sha256,
    audit_denials=denials,actual_model_or_private_reads=False,public_admission_numpy_dependency_imported=True,
    available_tmp_bytes_point_observation=available,
    root_review_development_corrections=['Initial review assertion omitted the existing explicit legacy-test exclusion metadata; draft preserved.',
        'Initial root audit also denied NumPy imported by the public selector admission dependency; allowed NumPy while retaining all model/body/network/child denials; draft preserved.'],
    runtime_policy=runtime,no_healthy_deadline=True,
    limitations=['Source and Python audit integrity is not an OS sandbox.','Candidate calibration, heldout1088, row0 and HE outcomes remain unmeasured.','Byte caps and available storage observations are not peak-memory guarantees.'],
    review_script_sha256=sha(Path(__file__).read_bytes()))
review_raw=jsonbytes(receipt)
grant=dict(schema='qwen-expanded-candidate-checker-grant-v1',execute=True,
    scope='one-frozen-candidate-own3200-calibration-only',candidate_ordinal=1,
    candidate_manifest_sha256=binding.manifest_sha256,candidate_binding_sha256=binding.binding_sha256,
    initial_report_sha256=binding.initial_report_sha256,initial_proof_sha256=binding.initial_proof_sha256,
    selection_report_sha256=binding.selection_proof_sha256,selector_outer_sha256=binding.selector_outer_sha256,
    source_manifest_sha256=MANIFEST,protocol_sha256=binding.protocol_sha256,calibration=external['calibration'],
    model_pins_sha256=sha(blobs['public/model_pins.json']),runtime_policy=runtime,
    backend_environment={'USE_TF':'0','USE_FLAX':'0','USE_TORCH':'1','USE_TORCH_XLA':'0','FORCE_TF_AVAILABLE':'0'},
    evidence_output=str(ACTUAL),private_output=str(PRIVATE),model_forward=True,tokenizer_execution=False,
    coefficient_selection=False,validation_read=False,encrypted_execution=False,candidate_budget=3,
    preflight_report_sha256=REPORT,independent_review_sha256=sha(review_raw))
OUT.mkdir(mode=0o700)
write(OUT/'review.py',Path(__file__).read_bytes())
write(OUT/'review.json',review_raw)
grant_raw=jsonbytes(grant);write(OUT/'launch.json',grant_raw)
print(json.dumps(dict(passed=True,review_sha256=sha(review_raw),grant_sha256=sha(grant_raw),sources=139,tests=738)))
