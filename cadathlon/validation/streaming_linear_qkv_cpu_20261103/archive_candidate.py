"""Archive the reusable-module component proof without client secrets/payloads."""
from pathlib import Path
import difflib,hashlib,io,json,os,shutil,subprocess,tarfile
LAB=Path(__file__).resolve().parent;ROOT=Path('/home/jianming_gatech/CROSS')
OUT=ROOT/'cadathlon/validation/streaming_linear_qkv_cpu_20261103'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
r=json.loads((LAB/'run-v1/report.json').read_text())
assert r['passed'] and r['source_files_unchanged'] and r['owned_component_directory_removed'] and r['all_processes_reaped'] and r['owned_directory_handle_closed']
pids=[r['wrapper_pid'],*[s['pid'] for s in r['stages'].values()]]
for pid in pids:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError(f'owned process {pid} still present')
assert not Path(r['owned_component_directory']).exists()
assert sha(LAB/'run-v1/sources.tar.gz')==r['source_archive_sha256']
with tarfile.open(LAB/'run-v1/sources.tar.gz','r:gz') as stream:
 archived={m.name:stream.extractfile(m).read() for m in stream.getmembers()}
expected={}
for name,digest in r['source_hashes_start'].items():
 p=Path(name);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==digest
 member='repository/'+str(p.relative_to(LAB/'checkout')) if p.is_relative_to(LAB/'checkout') else 'prototype/'+p.name
 assert member not in expected;expected[member]=data
assert archived==expected
metadata=json.loads((LAB/'metadata_tests.json').read_text());assert metadata['passed'] and metadata['source_files_unchanged']
portable=json.loads((LAB/'promotion_tests.json').read_text());assert portable['passed'] and portable['source_files_unchanged']
OUT.mkdir(exist_ok=False);shutil.copytree(LAB/'run-v1',OUT/'measured');shutil.copyfile(LAB/'run-v1.log',OUT/'measured/collector.log')
for name in ('metadata_tests.log','metadata_tests.json','metadata_tests.xml','source_origin.json','prior_transport_sources.json','run_metadata.py','ideal-tests.log','ideal-tests-initial.py.txt','ideal-tests-initial.log','promotion_tests.log','promotion_tests.json','promotion_tests.xml','run_promotion_tests.py'):
 shutil.copyfile(LAB/name,OUT/name)
shutil.copytree(LAB/'initial_metadata_failure',OUT/'initial_metadata_failure')
texts=OUT/'measured_prototype';texts.mkdir()
for p in sorted((LAB/'prototype').glob('*.py')):shutil.copyfile(p,texts/(p.name+'.txt'))
for p in sorted(LAB.glob('test_*.py')):shutil.copyfile(p,texts/(p.name+'.txt'))
patch=[];changes={};files={
 'cadathlon/common/encrypted_streaming_linear.py':LAB/'checkout/cadathlon/common/encrypted_streaming_linear.py',
 'cadathlon/common/package.py':LAB/'checkout/cadathlon/common/package.py',
 'cadathlon/tests/test_streaming_linear.py':LAB/'promotion/cadathlon/tests/test_streaming_linear.py',
 'cadathlon/tests/test_streaming_linear_contracts.py':LAB/'promotion/cadathlon/tests/test_streaming_linear_contracts.py',
}
follow=OUT/'reusable_candidate';follow.mkdir()
for rel,p in files.items():
 target=ROOT/rel;before=target.read_bytes() if target.exists() else b'';after=p.read_bytes()
 changes[rel]={'before_sha256':hashlib.sha256(before).hexdigest() if target.exists() else None,'after_sha256':hashlib.sha256(after).hexdigest()}
 patch.extend(difflib.unified_diff(before.decode().splitlines(True),after.decode().splitlines(True),fromfile='a/'+rel if target.exists() else '/dev/null',tofile='b/'+rel))
 copy=follow/(rel+'.txt');copy.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,copy)
patch_path=OUT/'streaming_linear.patch';patch_path.write_text(''.join(patch));subprocess.run(['git','apply','--check',str(patch_path)],cwd=ROOT,check=True)
client=r['stages']['client-decrypt']['result'];server=r['stages']['keyless-server']['result'];public=json.loads((LAB/'run-v1/public_manifest.json').read_text())
summary={'scope':'Reusable distinct-row streaming-linear module plus genuine pinned BERT layer0 QKV three-process CPU component; not a full layer/model, ranked stage, MRPC accuracy or calibration.',
 'passed':True,'source_count':len(expected),'source_files_unchanged':True,'source_archive_sha256':r['source_archive_sha256'],'processes_absent':pids,'owned_files_removed':True,
 'checks':client['checks'],'worst_abs_error':client['worst_abs_error'],'fixed_gate':1e-6,'all_slots_checked':client['all_slots_checked'],
 'ring':public['parameters'],'actual_log2_qp':r['stages']['client-encrypt']['result']['actual_log2_qp'],'packing_contract':public['packing'],'streaming_linear_source_sha256':public['streaming_linear_source_sha256'],
 'native_preprocessing_s':r['preprocessing']['elapsed_s'],'preprocessing_process_s':r['stages']['client-preprocess']['process_elapsed_s'],'cryptographic_subprocess_total_s':r['cryptographic_subprocess_total_s'],'collector_total_s':r['elapsed_s'],'server_process_s':r['stages']['keyless-server']['process_elapsed_s'],'per_projection_server_s':{k:v['server_s'] for k,v in server['checks'].items()},
 'keyless_server_guards':{k:server[k] for k in ('server_secret_absent','client_codecs_and_keygen_forbidden','client_file_access_attempts')},'metadata_tests':{'count':90,'elapsed_s':metadata['elapsed_s'],'source_count':len(metadata['source_hashes_start']),'source_files_unchanged':True},'portable_candidate_tests':portable,'patch_sha256':sha(patch_path),'changed_files':changes,
 'failure_scope':'Both preserved initial failures were test setup errors: missing non-Python Makefile in the inherited source snapshot; missing checkout root in the first ideal-test import path. Neither attempted encryption.',
 'timing_scope':'Observed CPU development timings, not calibrated performance or a controlled comparison. No other work was intentionally scheduled on affinity168–171 during this encryption run; shared-host load was uncontrolled.',
 'full_bert_evaluated':False,'full_qwen_evaluated':False,'mrpc_accuracy_evaluated':False,'scoring_calibrated':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');shutil.copyfile(Path(__file__),OUT/'archive_candidate.py')
print(json.dumps({k:summary[k] for k in ('passed','source_count','worst_abs_error','all_slots_checked','collector_total_s','patch_sha256')},indent=2))
