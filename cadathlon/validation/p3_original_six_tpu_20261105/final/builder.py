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
OUT=Path('/tmp/p3-original-six-final-public-20261105')
VERIFY=Path('/tmp/cnn-p3-original-six-final-verifier-20261105.py')
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

assert not OUT.exists();OUT.mkdir();began=utc()
admit=json.loads(retain(BASE/'root-terminal-admission-20261105.json','root/terminal-admission.json','8724aaa62574c9346affd06797517569d018ee6a8a89332a29484e6d04f3ddd9'))
terminal=json.loads(retain(BASE/'root-original-terminal-receipt-20261105.json','root/terminal-receipt.json',admit['terminal_receipt_sha256']))
pending=json.loads(retain(BASE/'root-terminal-cleanup-pending-20261105.json','root/cleanup-pending.json',admit['cleanup_pending_receipt_sha256']))
assert terminal['original_session']==81235 and terminal['exit_code']==0 and terminal['terminal'] is True and terminal['tool_chunk']=='84c601'
assert admit['passed'] is True and admit['complete'] is True
prep=json.loads(retain(BASE/'preparation.json','preparation/preparation.json','80e3b7fab9e439df7fe62a42ef98ef69288ff366f5a0b4ad07dcaedc3cd75a65'))
sources=json.loads(retain(BASE/'sources.json','preparation/sources.json',prep['source_map_sha256']))
aux=json.loads(retain(BASE/'auxiliary.json','preparation/auxiliary.json',prep['auxiliary_map_sha256']))
for n in ['readiness-after-tpu12-20261105.json','prior-collector-report.json','prior-collector-junit.xml']:
    retain(BASE/n,'preparation/'+n)
launch=json.loads(retain(BASE/'launch-v1.json','root/launch.json','f219750cc03acb90d05283d2076f53cb9623df62e35f0ef7b9e5021434b3d5d5'))
retain(BASE/'launch-v1-root-readback.json','root/launch-live-readback.json','1f5bca1333ac7797bd94fd07982f6bb56f59b84c37fc28cb023bf850e245c0ad')
retain(Path(launch['launcher_source']),'root/launcher-source.py.txt',launch['launcher_sha256'])
retain(Path(launch['initial_launcher_source']),'root/initial-launcher-source.py.txt',launch['initial_launcher_sha256'])
retain(Path(admit['verifier_path']),'root/terminal-reader-source.py.txt',admit['verifier_sha256'])
report=json.loads(retain(BASE/'run-v1/report.json','raw/report.json',admit['report_sha256']))
start=json.loads(retain(BASE/'run-v1/source-start.json','raw/source-start.json'))
end=json.loads(retain(BASE/'run-v1/source-end.json','raw/source-end.json'))
assert report['complete'] is True and report['passed'] is True and 'active_case' not in report
assert sources==start==end==report['source_hashes_start']==report['source_hashes_end'] and len(sources)==128
actual=archive(retain(BASE/'run-v1/sources.tar.gz','sources/actual-128.tar.gz',admit['source_archive_sha256']))
prepared=archive(retain(BASE/'sources.tar.gz','sources/prepared-129.tar.gz',prep['archive_sha256']))
assert set(actual)==set(sources) and set(prepared)==set(sources)|set(aux) and len(prepared)==129
current=[]
for name,digest in (sources|aux).items():
    prefix,relative=name.split('/',1);assert prefix in ['repo','collector']
    copied=Path(prep['checkout'] if prefix=='repo' else prep['collector'])/relative
    original=Path(prep['source_repo'] if prefix=='repo' else prep['source_collector'])/relative
    b=read(copied,digest);assert b==read(original,digest)==prepared[name]
    if name in actual:assert actual[name]==b
    current.append({'name':name,'prepared_path':str(copied),'original_path':str(original),'sha256':digest})
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
    tree=ast.parse(actual[name])
    for f in wanted:
        node,=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==f]
        source_defs[name+':'+f]={'line':node.lineno,'end_line':node.end_lineno,'AST_sha256':sha(ast.dump(node,include_attributes=False).encode())}
rows=[];warnings=[]
for i in range(1,7):
    case='test'+str(i)+'.in';folder=BASE/'run-v1'/('test'+str(i))
    names=['process.json','stdout.json','stderr.log','time.txt','source-after.json']
    assert {p.name for p in folder.iterdir()}==set(names)
    b={n:retain(folder/n,'raw/test'+str(i)+'/'+n,admit['case_file_hashes']['test'+str(i)+'/'+n]) for n in names}
    process=json.loads(b['process.json']);result=json.loads(b['stdout.json'])
    assert process==report['cases'][case] and process['result']==result and json.loads(b['source-after.json'])==sources
    assert process['returncode']==0 and all(process[n] is True for n in ['passed','source_files_unchanged','process_reaped','public_key_store_removed','owned_directory_handle_closed'])
    row=admit['cases'][i-1];assert row['case']==case and row['runner_total_s']==result['total_s'] and row['wrapper_pid']==process['pid']
    rows.append(row)
    text=b['stderr.log'].decode()
    warnings.append({'case':case,'captured_constants_warning':'A large amount of constants were captured' in text,
                     'hugepage_alignment_diagnostic':'File offset incorrectly aligned for file-backed THP' in text,
                     'stderr_sha256':sha(b['stderr.log']),'interpretation':'Raw diagnostics preserved without causal or performance attribution.'})
assert len(rows)==6 and len(admit['case_file_hashes'])==30
put('source-audit.json',{'source_functions':source_defs,'current_sources':current,'actual_source_count':128,
 'prepared_archive_count':129,'auxiliary_doc_sentinel':'repo/jaxite_word/API_REFERENCE.md',
 'per_logit_gate':'abs(output - expected) <= 0.02 + 0.01 * abs(expected), every logit','top1_gate':0.99,
 'security':'Exact reported integer product of Q16/P4 has bit_length605 <= pinned inclusive classical128/N32768 ceiling881; original runtime gates/source-bound evidence, not new estimator proof.',
 'no_independent_decrypted_output_recomputation':True})
put('summary.json',{'schema':'original-p3-six-final-public-summary-v1','complete':True,'passed_cases':6,'failed_cases':0,
 'cases':rows,'parent_elapsed_s':report['elapsed_s'],'original_root_session':81235,'root_terminal_tool_chunk':'84c601',
 'profile':'original baseline; fresh process, zero warmups, one cold repeat per public synthetic fixture',
 'independent_decrypted_logit_recomputation':False,'new_numerical_execution':False,'trained_accuracy_evaluated':False,
 'diagnostics':warnings,'cleanup_authority':'Original collector case waits/cleanup; root original terminal wait and identity-bound empty key-parent removal. Packager inherits neither authority.',
 'excluded_claims':['ranked scoring or trained-model accuracy','compact five-minute schedule','GNU RSS as peak HBM or aggregate memory','new inner-PID/live observations or waits','runtime promotion']})
put('capture.json',{'started_utc':began,'finished_utc':utc(),'scope':'All30 completed public case files and finalreport/source/provenance/rootterminal receipts only',
 'packager_polled_original_session':False,'packager_direct_wait_authority':False,'key_body_read':False,
 'new_process_or_key_path_observation':False,'candidate_import_or_execution':False,'source_current_archive_exact':True,
 'root_terminal_admission_sha256':sha(read(BASE/'root-terminal-admission-20261105.json')),
 'previous_partial_checkpoints_unchanged':True})
retain(Path(__file__),'builder.py');retain(VERIFY,'verify.py')
put('origins.json',origins)
table='\n'.join('| '+r['case']+' | '+('LeNet' if r['model']=='lenet' else 'AlexNet-tiny')+' | '+str(r['batch'])+' | '+format(r['max_abs_error'],'.8g')+' | '+format(r['nrmse'],'.8g')+' | '+str(r['top1_agreement'])+' | '+format(r['runner_total_s'],'.6f')+' |' for r in rows)
readme='''# Original P3 six-case TPU validation — all six passed

The six original public synthetic fixtures passed on the pinned baseline source: LeNet and AlexNet-tiny, each at batches 1, 2 and 4. Root's original session81235 returned exit0 at tool chunk84c601; the final parent report records3369.480570556014s. This capsule archives that completed run and does not launch or rerun anything.

| Case | Model | Batch | Recorded max abs error | Recorded NRMSE | Top-1 agreement | Runner total (s) |
|---|---|---:|---:|---:|---:|---:|
'''+table+'''

Exact metrics and setup, request, server, external wall-time and GNU RSS values are in `summary.json` and the unmodified case files. Each fresh process used zero warmups and one cold request. Runner total includes mapping release; parent elapsed includes the sequential sweep. These measurements do not establish the separate compact five-minute schedule or ranked performance. GNU time RSS is not a peak-HBM or aggregate-memory measurement.

The original source gates every logit with `abs(output-reference) <= 0.02 + 0.01*abs(reference)` and requires top-1 agreement at least0.99. Decrypted output arrays were not retained: verification is of recorded, source-bound evidence, not independent recomputation. All cases report actual N32768/Q16/P4 classical128; the exact public QP product has bit length605, within the pinned inclusive881-bit ceiling. This is synthetic arithmetic validation, not trained-model accuracy or a new security-estimator result.

All30 completed-case files are retained exactly, with final report and source-start/after/end joins. The actual archive contains128 measured source/fixture files. The preparation archive contains those same bytes plus one separately hashed API_REFERENCE sentinel (129total). The historical preparation and15 fake-process collector tests retain their original statuses; those counts are not added to the six actual cases. No dynamic-constants experiment replaced the original baseline.

The original collector records each exit0/reaping, child key-directory removal and handle closure. Root separately records original terminal authority, completed wrapper/source-bound group absence, original key-parent identity and emptiness before removal, then final removal admission. Later-case inner PIDs are not inferred. The initial live receipt covers its original parent/test1 observation only. The packager neither polls the original P3 session nor restats/removes key paths; portable verification only joins these public receipts. The archived root cleanup script is provenance text and must not be executed by the portable verifier.

Raw stderr is preserved, including captured-constants warnings and any hugepage alignment diagnostics, without causal attribution. Earlier test1–3 partial checkpoints remain unchanged and remain partial historical snapshots.

Run `python -I -S -B verify.py` for portable stdlib-only evidence verification. `--current` additionally hashes only the declared public source/evidence origins. Neither mode imports a candidate/backend, reruns numerical work, polls processes, reads key/CT bodies, extracts archives, or invokes the archived root cleanup program. No repository edit, promotion or grant is part of this capsule.
'''
write('README.md',readme.encode())
for path,digest in pins.items():assert sha(read(path))==digest
manifest={str(p.relative_to(OUT)):{'sha256':sha(p.read_bytes()),'bytes':p.stat().st_size} for p in sorted(OUT.rglob('*')) if p.is_file()}
put('artifact_hashes.json',manifest)
print(json.dumps({'root':str(OUT),'files':len(manifest)+1,'indexed':len(manifest),'index_sha256':sha((OUT/'artifact_hashes.json').read_bytes()),'summary_sha256':sha((OUT/'summary.json').read_bytes()),'README_sha256':sha((OUT/'README.md').read_bytes())},sort_keys=True))

