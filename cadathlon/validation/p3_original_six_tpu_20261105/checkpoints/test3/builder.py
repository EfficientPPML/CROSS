import ast
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import stat
import tarfile

BASE=Path('/tmp/p3-original-six-preparation-20261105')
OUT=Path('/tmp/p3-original-six-test3-checkpoint-20261105')
VERIFY=Path('/tmp/cnn-p3-original-six-test3-portable-verifier-20261105.py')
KEY_PARENT=Path('/dev/shm/cross-p3-original-six-key-parent-f383y30c')
KEY_CHILD=KEY_PARENT/'cross-transformer-case-vg_ja2yj'
pins={};origins={}
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path,expected=None,cap=16*1024*1024):
    path=Path(path);before=path.lstat();assert stat.S_ISREG(before.st_mode),str(path)
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        opened=os.fstat(fd);assert (before.st_dev,before.st_ino)==(opened.st_dev,opened.st_ino)
        assert stat.S_ISREG(opened.st_mode) and opened.st_size<=cap
        pieces=[];total=0
        while True:
            block=os.read(fd,min(1024*1024,cap-total+1))
            if not block:break
            pieces.append(block);total+=len(block);assert total<=cap
        raw=b''.join(pieces);after=os.fstat(fd)
        assert (opened.st_size,opened.st_mtime_ns,opened.st_ctime_ns)==(after.st_size,after.st_mtime_ns,after.st_ctime_ns)
        assert len(raw)==opened.st_size
    finally:os.close(fd)
    named=path.lstat();assert (named.st_dev,named.st_ino)==(opened.st_dev,opened.st_ino)
    digest=sha(raw)
    if expected is not None:assert digest==expected,(str(path),digest,expected)
    assert pins.setdefault(str(path),digest)==digest
    return raw
def write(name,raw):
    path=OUT/name;path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('xb') as stream:stream.write(raw)
def put(name,obj):write(name,(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n').encode())
def retain(source,name,expected=None):
    raw=read(source,expected);write(name,raw)
    origins[name]={'path':str(source),'sha256':sha(raw),'bytes':len(raw)}
    return raw
def archive(raw):
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as a:
        members=a.getmembers();names=[m.name for m in members]
        assert len(names)==len(set(names))
        assert all(m.isfile() and not m.issym() and not m.islnk() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in members)
        return {m.name:a.extractfile(m).read() for m in members}

assert not OUT.exists();OUT.mkdir()
began=utc()
prep=json.loads(retain(BASE/'preparation.json','preparation/preparation.json','80e3b7fab9e439df7fe62a42ef98ef69288ff366f5a0b4ad07dcaedc3cd75a65'))
source_map=json.loads(retain(BASE/'sources.json','preparation/sources.json',prep['source_map_sha256']))
aux=json.loads(retain(BASE/'auxiliary.json','preparation/auxiliary.json',prep['auxiliary_map_sha256']))
start=json.loads(retain(BASE/'run-v1/source-start.json','raw/source-start.json'))
raw={n:retain(BASE/'run-v1/test3'/n,'raw/test3/'+n) for n in ['stdout.json','stderr.log','time.txt','process.json','source-after.json']}
process=json.loads(raw['process.json']);result=json.loads(raw['stdout.json']);after=json.loads(raw['source-after.json'])
assert process['result']==result and source_map==start==after and len(start)==128
assert {p.name for p in (BASE/'run-v1/test3').iterdir()}==set(raw)
parent_path=BASE/'run-v1/report.json'
parent_raw=read(parent_path);parent=json.loads(parent_raw)
assert parent['cases']['test3.in']==process and parent['collector_pid']==1598331
assert parent['passed'] is False and parent.get('complete') is not True
assert parent['active_case']!='test3.in'
parent_name='raw/parent-partial-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'.json'
write(parent_name,parent_raw)
origins[parent_name]={'path':str(parent_path),'sha256':sha(parent_raw),'bytes':len(parent_raw),'mutable_origin':True,'snapshot_only':True}
launch=json.loads(retain(BASE/'launch-v1.json','raw/launch-v1.json'))
live=json.loads(retain(BASE/'launch-v1-root-readback.json','raw/launch-v1-root-readback.json'))
actual=archive(retain(BASE/'run-v1/sources.tar.gz','sources/actual-128.tar.gz','a5e8b41586e683bb6eb23d57266195bfcb0206acb94f2f1a4af9fef6636074eb'))
prepared=archive(retain(BASE/'sources.tar.gz','sources/prepared-129.tar.gz',prep['archive_sha256']))
assert len(actual)==128 and set(actual)==set(start)
assert len(prepared)==129 and set(prepared)==set(start)|set(aux)
current=[]
for name,digest in {**start,**aux}.items():
    prefix,relative=name.split('/',1)
    copied=Path(prep['checkout'] if prefix=='repo' else prep['collector'])/relative
    original=Path(prep['source_repo'] if prefix=='repo' else prep['source_collector'])/relative
    b=read(copied,digest);assert b==read(original,digest)==prepared[name]
    if name in actual:assert actual[name]==b
    current.append({'name':name,'prepared_path':str(copied),'original_path':str(original),'sha256':digest})
assert parent['source_hashes_start']==start and parent['source_archive_sha256']==sha(read(BASE/'run-v1/sources.tar.gz'))
assert result['case']=='test3.in' and result['batch']==4
assert result['mode']=='HE-synthetic-smoke' and result['implementation']=='baseline.py'
fixture=json.loads(actual['repo/cadathlon/non_agent_pset/p3_cnn/test/test3.in'])
assert fixture=={'purpose':'synthetic-arithmetic-not-trained-accuracy','model':'lenet','batch':4,'weight_seed':7203,'input_seed':8303,'input_kind':'uniform'}
assert process['returncode']==0 and process['pid']==1606928 and 'error' not in process
assert all(process[k] is True for k in ['passed','process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'])
assert process['owned_public_key_directory']==str(KEY_CHILD)
assert process['owned_public_key_directory_identity']=={'device':24,'inode':4067}

# One bounded, independent absence observation; never wait on or signal the parent.
observed=utc();pids={};pgids={}
for pid in [1606928]:
    try:os.kill(pid,0)
    except ProcessLookupError:pids[str(pid)]='absent_ESRCH'
    else:raise AssertionError(('PID remains present',pid))
    try:Path('/proc/'+str(pid)).lstat()
    except FileNotFoundError:pass
    else:raise AssertionError(('proc PID remains present',pid))
for pgid in [1606928]:
    try:os.killpg(pgid,0)
    except ProcessLookupError:pgids[str(pgid)]='absent_ESRCH'
    else:raise AssertionError(('PGID remains present',pgid))
try:KEY_CHILD.lstat()
except FileNotFoundError:child_absent=True
else:raise AssertionError('completed child key directory remains')
# The live key parent is not statted, traversed, read or modified by this checkpoint.
absence={'observed_utc':observed,'pids':pids,'pgids':pgids,'proc_entries_absent':True,
 'child_key_directory':str(KEY_CHILD),'child_key_directory_absent':child_absent,
 'parent_key_directory':str(KEY_PARENT),'parent_directory_statted':False,
 'parent_directory_removed':False,'parent_directory_contents_read':False,
 'original_session_polled':False,'wait_authority_inherited':False,'live_observation_inherited':False,
 'inner_pid_observed':False,'unknown_inner_pid_absence_claim':False,
 'pgid_binding':'Recorded wrapper PID and original run_case subprocess.Popen(start_new_session=True); no separate test3 live observation.',
 'method':'signal 0 existence checks only for completed wrapper PID and source-bound PGID; lstat only for that proc name and exact completed child directory'}
put('process-absence.json',absence)

source_defs={}
for name,wanted in {
 'repo/cadathlon/non_agent_pset/p3_cnn/src/runner.py':['run','main'],
 'repo/cadathlon/common/support.py':['metrics','require_ciphertexts','require_secure_execution'],
 'repo/jaxite_word/he_params.py':['openfhe_logq_bits','modulus_product_is_secure','qp_is_secure'],
 'collector/check_p3_original_six.py':['child_runner','validate_result','evaluate'],
 'collector/process_helpers.py':['run_case'],
 'repo/cadathlon/non_agent_pset/p3_cnn/src/reference.py':['make_case'],
 'repo/cadathlon/non_agent_pset/p3_cnn/src/baseline.py':['build_mapping'],
}.items():
    tree=ast.parse(actual[name].decode())
    for fn in wanted:
        matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==fn];assert len(matches)==1
        node=matches[0]
        source_defs[name+':'+fn]={'line':node.lineno,'end_line':node.end_lineno,'AST_sha256':sha(ast.dump(node,include_attributes=False).encode())}
ring=result['execution_rings'];assert len(ring)==1;ring=ring[0]
assert (ring['degree'],ring['num_q'],ring['num_p'],ring['security_bits'],ring['security_model'])==(32768,16,4,128,'classical')
product=math.prod(ring['q_towers']+ring['p_towers']);assert product.bit_length()==605<=881
put('source-audit.json',{'source_functions':source_defs,'current_sources':current,'actual_source_count':128,'prepared_archive_count':129,'auxiliary_doc_sentinel':'repo/jaxite_word/API_REFERENCE.md','per_logit_gate':'abs(output - expected) <= 0.02 + 0.01 * abs(expected), every logit','top1_gate':0.99,'numeric_scope':'Recorded original runner pass/metrics bound to exact sources; actual decrypted output logits are not retained, so no independent numerical recomputation is claimed.','security_integer_check':{'N':32768,'num_q':16,'num_p':4,'product_decimal':str(product),'product_bit_length':605,'classical_128_ceiling':881,'inclusive_pass':True},'original_security_scope':'Pinned local runtime-context/ciphertext/noise/distribution checks, not a sandbox or independent lattice-estimator run.'})
summary={'case':'test3.in','completed_cases_admitted':1,'scheduled_cases':6,'model':'lenet','batch':4,'case_ordinal':3,'profile':'original','implementation':'baseline.py','case_passed':True,'sweep_complete':False,'sweep_passed_claim':False,'parent_snapshot':parent_name,'parent_snapshot_active_case':parent['active_case'],'max_abs_error':result['max_abs_error'],'nrmse':result['nrmse'],'top1_agreement':result['top1_agreement'],'runner_total_s':result['total_s'],'external_process_elapsed_s':process['process_elapsed_s'],'setup_s':result['setup_s'],'cold_request_s':result['end_to_end_s'][0],'server_s':result['server_s'][0],'warmups':0,'repeats':1,'trained_accuracy_evaluated':False,'max_error_interpretation':'Exact source-bound reported metrics, no normalization or causal attribution.','captured_constants_warning':'Original stderr records 12.36GB of constants captured during lowering; no causal or performance claim.','hugepage_diagnostic':'Original stderr records a file-backed THP alignment diagnostic; preserved without causal attribution, case exit0/PASS.','security':'Actual reported N32768/Q16/P4 classical128; exact product bit_length 605 <= pinned ceiling 881.','cleanup':'Original collector records exit0/reaped/child key removal/handle close; independent late recorded wrapper PID/source-bound PGID and child-directory absence. Parent and its key parent remain owned by root.','excluded_claims':['final six-case sweep','ranked scoring','trained accuracy','compact five-minute schedule','independent test3 live/inner-PID observation or direct wait by packager','new test execution']}
put('summary.json',summary)
put('capture.json',{'started_utc':began,'finished_utc':utc(),'parent_partial_snapshot':parent_name,'parent_snapshot_sha256':sha(parent_raw),'parent_report_was_live':True,'parent_terminal_claim':False,'authoritative_case_wait_owner':'Original collector run_case; packager has no direct wait authority.','original_root_session':81235,'original_root_live_receipt':'raw/launch-v1-root-readback.json','active_case_numerical_or_key_body_read':False,'key_body_read':False,'candidate_import_or_execution':False,'current_prepared_original_and_archived_sources_exact':True,'read_scope':'Exact5 completedcase files, public source/fixture archives and maps, source-fixed originals, root launch/live metadata, timestamped partial parent report and bounded completed-child absence.'})
put('tooling-lineage.json',{'parent_checkpoint':'/tmp/p3-original-six-test2-checkpoint-20261105',
 'parent_builder_sha256':'fc6ecadee43ee2bc2a5b2aae2fa53f3f9e9d688a7c240d6f14c06f00ffd57e16',
 'parent_verifier_sha256':'b28d031c55c1967b8059598cf5cea3b18917a577a67f404117d82ac592326ac5',
 'adaptation':'Exact test3 values/fixture; wrapper-only absence; parent untouched; historical test1 live receipt is not test3 live evidence.',
 'source_execution':False,'numerical_rerun':False})
retain(Path(__file__),'builder.py')
retain(VERIFY,'verify.py')
put('origins.json',origins)
readme='''# P3 original test3 checkpoint — third of six

The original LeNet, batch-4 synthetic encrypted TPU case passed its pinned runner gates. This is a public checkpoint for test3 only (one case admitted here, not a three-case aggregate); the original six-case parent was still running when its report was captured.

| Recorded result | Value |
|---|---:|
| Maximum absolute error | 9.141108009425025e-13 |
| NRMSE | 3.2013200584170407e-11 |
| Top-1 agreement | 1.0 |
| Runner total, including setup and release | 635.2648103709798 s |
| External process elapsed | 651.4186651210184 s |

The source-bound runner checks every logit against `0.02 + 0.01 * abs(reference)` and top-1 agreement >= 0.99. Raw decrypted logits are not archived, so this is verification of recorded, source-bound evidence rather than independent recomputation. The reported actual ring is N=32768, Q16/P4, classical 128; the exact modulus product has bit length 605, within the pinned inclusive 881-bit ceiling.

All five completed-case files are retained exactly. The run archive has 128 measured files; the preparation archive has those same files plus the separately hashed API_REFERENCE sentinel. Sources match the start/after maps and current prepared/original files at capture. The parent report is an exact timestamped partial snapshot, not a terminal sweep report.

The original collector records exit 0, reaping, child key-directory removal and handle closure. Independent late checks found the recorded wrapper PID, its source-bound process group, and the exact completed child directory absent. No test3 inner PID was recorded or independently observed. The retained initial launch/live receipt describes the parent and test1, not test3 live identity. Wrapper group ownership is derived from the exact original collector's `start_new_session=True` call. The live parent and key-parent metadata were not newly statted. The live parent and root-owned key parent were not stopped, removed or traversed. The packager did not inherit direct-wait or live-observation authority.

Original stderr includes a file-backed hugepage alignment diagnostic and a 12.36GB captured-constants warning. The case nevertheless records exit 0 and passed gates. It is preserved without a causal or performance interpretation. This checkpoint is not final sweep success, ranked scoring, trained-model accuracy, or proof of the separate compact five-minute schedule.

Run `python -I -S verify.py` for portable standard-library verification. Optional `--current` additionally hashes only the declared public source and stable evidence origins; it never polls processes or reads the live parent report. The verifier imports no candidate or numerical backend and extracts no archive files.
'''
write('README.md',readme.encode())
# Stable completed evidence and source bytes, excluding the intentionally live parent.
for path,digest in pins.items():
    if path!=str(parent_path):assert sha(read(path))==digest
manifest={str(p.relative_to(OUT)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file()}
put('artifact_hashes.json',manifest)
print(json.dumps({'root':str(OUT),'files':len(manifest)+1,'indexed':len(manifest),'index_sha256':sha((OUT/'artifact_hashes.json').read_bytes()),'summary_sha256':sha((OUT/'summary.json').read_bytes()),'observed_utc':observed},sort_keys=True))
