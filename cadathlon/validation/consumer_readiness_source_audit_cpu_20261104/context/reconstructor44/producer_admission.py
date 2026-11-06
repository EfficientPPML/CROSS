"""Pure external-pin admission for one candidate1 reconstruction.

No filesystem access. Public blobs are supplied by the separately authorized
root wrapper. Passing this function does not authorize validation or revision.
"""
from dataclasses import dataclass
import hashlib
import io
from pathlib import PurePosixPath
import tarfile
from contract import SOURCE_MANIFEST_SHA256,PROTOCOL_SHA256
from verify_records import parse
from verify_statistics import canonical,same,fields,integer,digest

NAMES=('source_manifest','source_archive','parent','worker','producer_outer','producer_grant','candidate_manifest','model_pins','reconstruction_grant')
OUTER_SCHEMA='qwen-candidate3200-producer-terminal-v1'

def sha(raw):return hashlib.sha256(raw).hexdigest()
def absolute(path):
    if type(path)is not str or not path.startswith('/')or str(PurePosixPath(path))!=path or '..'in PurePosixPath(path).parts:raise ValueError('Canonical absolute path required')
    return path
def identity(value):
    if type(value)is not list or len(value)!=2:raise ValueError('Exact inode pair required')
    for item in value:integer(item,0,1<<64)
    return value

def verify_archive(raw,expected):
    if type(raw)is not bytes or not 0<len(raw)<=96<<20:raise ValueError('Bounded source archive required')
    seen=set();total=0
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz')as archive:
        for member in archive:
            name=member.name
            if name not in expected or name in seen or str(PurePosixPath(name))!=name or name.startswith('/')or '..'in PurePosixPath(name).parts or not member.isfile():raise ValueError('Exact source archive inventory required')
            limit=64<<20 if name=='candidate_proof/selector-assessments.json'else 32<<20 if name.startswith('candidate_proof/')else 8<<20
            total+=member.size
            if not 0<=member.size<=limit or total>96<<20:raise ValueError('Source archive size limit')
            stream=archive.extractfile(member)
            if stream is None:raise ValueError('Missing source body')
            with stream:data=stream.read(limit+1)
            if len(data)!=member.size or sha(data)!=expected[name]:raise ValueError('Source archive byte mismatch')
            seen.add(name)
    if seen!=set(expected):raise ValueError('Missing source archive member')

@dataclass(frozen=True,slots=True)
class AdmittedAttempt:
    public_blobs:tuple
    expected_pins:tuple
    calibration_bytes:bytes
    private_path:str
    private_identity:tuple
    manifest_bytes:bytes
    worker_bytes:bytes
    producer_passed:bool
    reconstruction_source_manifest_sha256:str
    reconstruction_source_archive_sha256:str
    reconstruction_script_sha256:str

def admit(public,expected):
    """Authenticate all public roles before permitting the private reader."""
    fields(public,NAMES);fields(expected,NAMES)
    for name in NAMES:
        raw=public[name]
        limit=96<<20 if name=='source_archive'else 64<<20
        if type(raw)is not bytes or not 0<len(raw)<=limit or sha(raw)!=digest(expected[name]):raise ValueError('External public proof pin differs: '+name)
    same(expected['source_manifest'],SOURCE_MANIFEST_SHA256)
    source=parse(public['source_manifest'])
    if type(source)is not dict or len(source)!=139:raise ValueError('Exact139 producer sources required')
    for name,pin in source.items():
        digest(pin)
        if type(name)is not str or name.startswith('/')or '..'in PurePosixPath(name).parts or str(PurePosixPath(name))!=name:raise ValueError('Canonical source member required')
    same(source.get('candidate_proof/candidate-manifest.json'),expected['candidate_manifest'])
    same(source.get('public/model_pins.json'),expected['model_pins'])
    same(source.get('protocol.json'),PROTOCOL_SHA256)
    verify_archive(public['source_archive'],source)
    parent=parse(public['parent']);worker=parse(public['worker']);outer=parse(public['producer_outer']);grant=parse(public['producer_grant']);reconstruction=parse(public['reconstruction_grant'])
    passed=worker.get('passed')
    if type(passed)is not bool:raise ValueError('Typed scientific result required')
    code=0 if passed else 1
    ofields=('schema','report_sha256','worker_report_sha256','source_manifest_sha256','source_archive_sha256','candidate_manifest_sha256','grant_sha256','exit_code','reaped','process_group_absent','worker_exit_code','worker_reaped','worker_group_absent')
    fields(outer,ofields)
    ofixed=dict(schema=OUTER_SCHEMA,report_sha256=expected['parent'],worker_report_sha256=expected['worker'],source_manifest_sha256=expected['source_manifest'],source_archive_sha256=expected['source_archive'],candidate_manifest_sha256=expected['candidate_manifest'],grant_sha256=expected['producer_grant'],exit_code=code,reaped=True,process_group_absent=True,worker_exit_code=code,worker_reaped=True,worker_group_absent=True)
    same(outer,ofixed)
    common_flags=('complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash')
    for report in(parent,worker):
        for name in common_flags:same(report.get(name),True)
        same(report.get('source_manifest_sha256'),expected['source_manifest']);same(report.get('source_hashes_start'),source);same(report.get('source_hashes_end'),source)
        same(report.get('audit_denials'),[]);same(report.get('passed'),passed)
        if 'error_type'in report:raise ValueError('Infrastructure failure cannot admit private scientific replay')
    same(parent.get('worker'),worker);same(parent.get('worker_report_sha256'),expected['worker'])
    same(parent.get('source_archive',{}).get('sha256'),expected['source_archive'])
    same(parent.get('child_returncode'),code);same(parent.get('child_reaped'),True);same(parent.get('child_group_absent'),True);same(parent.get('actual_candidate_stage'),True)
    for name in('calibration_reauthenticated','private_directory_handle_closed','private_outputs_retained','external_inputs_never_removed','model_asset_postchecks_completed','model_cleanup_completed','native_capture_cleanup_completed','ledger_complete','all_length_invariance','model_loaded'):
        same(worker.get(name),True)
    for name in('coefficient_selection_performed','encrypted_execution','validation_inputs_read','tokenizer_execution','contains_tokens_logits_or_failure_text'):same(worker.get(name),False)
    same(worker.get('candidate_ordinal'),1);same(worker.get('expected_entries'),3200);same(worker.get('retained_entries'),3200)
    same(worker.get('manifest_sha256'),expected['candidate_manifest']);same(worker.get('grant_sha256'),expected['producer_grant']);same(worker.get('protocol_sha256'),PROTOCOL_SHA256)
    for report in(parent,worker):same(report.get('candidate_validation_admissible'),passed)
    if not passed:
        for report in(parent,worker):same(report.get('failed_attempt_terminal_candidate'),True)
        same(worker.get('revision_eligible_ledger'),True)
    for name,value in dict(schema='qwen-expanded-candidate-checker-grant-v1',execute=True,candidate_ordinal=1,candidate_manifest_sha256=expected['candidate_manifest'],source_manifest_sha256=expected['source_manifest'],protocol_sha256=PROTOCOL_SHA256,model_pins_sha256=expected['model_pins'],validation_read=False,coefficient_selection=False,encrypted_execution=False).items():same(grant.get(name),value)
    cal=grant.get('calibration');fields(cal,('path','sha256','bytes','identity','parent_identity','mode','parent_mode'))
    if PurePosixPath(absolute(cal['path'])).name!='calibration_inputs.json':raise ValueError('Calibration-only input required')
    digest(cal['sha256']);integer(cal['bytes'],1,4<<20);identity(cal['identity']);identity(cal['parent_identity']);same(cal['mode'],'0600');same(cal['parent_mode'],'0700');same(worker.get('calibration_sha256'),cal['sha256'])
    private=worker.get('private_output');fields(private,('path','identity','mode'));absolute(private['path']);identity(private['identity']);same(private['mode'],'0700');same(grant.get('private_output'),private['path'])
    private_root=PurePosixPath(private['path']);calibration_root=PurePosixPath(cal['path']).parent
    if private_root==calibration_root or private_root in calibration_root.parents or calibration_root in private_root.parents:raise ValueError('Calibration and candidate journal roots must be disjoint')
    fixed=dict(schema='qwen-candidate-recorded-reconstruction-grant-v1',execute=True,candidate_ordinal=1,candidate_manifest_sha256=expected['candidate_manifest'],producer_parent_report_sha256=expected['parent'],producer_worker_report_sha256=expected['worker'],producer_outer_receipt_sha256=expected['producer_outer'],producer_source_manifest_sha256=expected['source_manifest'],producer_source_archive_sha256=expected['source_archive'],producer_grant_sha256=expected['producer_grant'],calibration=cal,private_output=private,protocol_sha256=PROTOCOL_SHA256,model_pins_sha256=expected['model_pins'],model_forward=False,validation_read=False,row0_evaluation=False,selection=False)
    extra=('reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_script_sha256','request_sha256','evidence_output')
    fields(reconstruction,(*fixed,*extra))
    for name,value in fixed.items():same(reconstruction[name],value)
    for name in extra[:-1]:digest(reconstruction[name])
    absolute(reconstruction['evidence_output'])
    return AdmittedAttempt(tuple((n,public[n])for n in NAMES),tuple((n,expected[n])for n in NAMES),canonical(cal),private['path'],tuple(private['identity']),public['candidate_manifest'],public['worker'],passed,*(reconstruction[n]for n in extra[:3]))

def require_admitted(value):
    if type(value)is not AdmittedAttempt:raise PermissionError('Externally authenticated reconstruction grant required')
    other=admit(dict(value.public_blobs),dict(value.expected_pins))
    def typed(v):
        if type(v)in(tuple,list):return(type(v),tuple(typed(x)for x in v))
        if type(v)is dict:return(type(v),tuple(sorted((typed(k),typed(x))for k,x in v.items())))
        return(type(v),v)
    if any(typed(getattr(value,n))!=typed(getattr(other,n))for n in AdmittedAttempt.__dataclass_fields__):raise ValueError('Immutable admitted binding changed')
    return value
