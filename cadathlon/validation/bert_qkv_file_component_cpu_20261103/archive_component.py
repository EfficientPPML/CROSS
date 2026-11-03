"""Preserve the exact completed QKV proof and a distinct cleanup-only follow-up."""
from pathlib import Path
import difflib,hashlib,io,json,os,shutil,subprocess,tarfile
LAB=Path(__file__).resolve().parent;CANDIDATE=Path('/tmp/bert-qkv-file-component-cleanup-20261103');ROOT=Path('/home/jianming_gatech/CROSS')
OUT=ROOT/'cadathlon/validation/bert_qkv_file_component_cpu_20261103';OUT.mkdir(exist_ok=False)
r=json.loads((LAB/'run-v1/report.json').read_text());assert r['passed'] and r['source_files_unchanged'] and r['owned_component_directory_removed'] and r['all_processes_reaped']
pids=[r['wrapper_pid'],*[s['pid'] for s in r['stages'].values()]]
for pid in pids:
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError(f'owned process{pid} still present')
assert not Path(r['owned_component_directory']).exists()
assert hashlib.sha256((LAB/'run-v1/sources.tar.gz').read_bytes()).hexdigest()==r['source_archive_sha256']
with tarfile.open(LAB/'run-v1/sources.tar.gz','r:gz') as stream:
 archived={m.name:stream.extractfile(m).read() for m in stream.getmembers()}
expected={}
for name,sha in r['source_hashes_start'].items():
 p=Path(name);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
 member='repository/'+str(p.relative_to(LAB/'checkout')) if p.is_relative_to(LAB/'checkout') else 'prototype/'+p.name
 expected[member]=data
assert archived==expected
shutil.copytree(LAB/'run-v1',OUT/'measured');shutil.copyfile(LAB/'run-v1.log',OUT/'measured/collector.log')
for name in ('metadata_tests.log','metadata_tests.json','metadata_tests.xml','source_origin.json','prior_transport_sources.json','run_metadata.py'):
 shutil.copyfile(LAB/name,OUT/name)
shutil.copytree(LAB/'initial_metadata_failure',OUT/'initial_metadata_failure')
# Review copies do not masquerade as a newly measured runtime.
texts=OUT/'measured_prototype';texts.mkdir()
for p in sorted((LAB/'prototype').glob('*.py')):shutil.copyfile(p,texts/(p.name+'.txt'))
follow=OUT/'cleanup_followup';follow.mkdir();patch=[];changed={}
for rel in ('prototype/collect.py','prototype/test_component.py'):
 old=(LAB/rel).read_bytes();new=(CANDIDATE/rel).read_bytes();assert old!=new
 changed[rel]={'before_sha256':hashlib.sha256(old).hexdigest(),'after_sha256':hashlib.sha256(new).hexdigest()}
 patch.extend(difflib.unified_diff(old.decode().splitlines(True),new.decode().splitlines(True),fromfile='a/'+rel,tofile='b/'+rel))
 shutil.copyfile(CANDIDATE/rel,follow/(Path(rel).name+'.txt'))
patch_path=follow/'collector_cleanup.patch';patch_path.write_text(''.join(patch));subprocess.run(['git','apply','--check',str(patch_path)],cwd=LAB,check=True)
cr=json.loads((CANDIDATE/'metadata_tests.json').read_text());assert cr['passed'] and cr['source_files_unchanged']
with tarfile.open(follow/'sources.tar.gz','w:gz') as stream:
 for name,sha in cr['source_hashes_start'].items():
  p=Path(name);data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==sha
  member='repository/'+str(p.relative_to(CANDIDATE/'checkout')) if p.is_relative_to(CANDIDATE/'checkout') else 'validation/'+p.name
  info=tarfile.TarInfo(member);info.size=len(data);info.mode=0o644;stream.addfile(info,io.BytesIO(data))
for name in ('metadata_tests.json','metadata_tests.log','metadata_tests.xml','run_metadata.py'):shutil.copyfile(CANDIDATE/name,follow/name)
client=r['stages']['client-decrypt']['result'];server=r['stages']['keyless-server']['result'];prep=r['preprocessing'];public=json.loads((LAB/'run-v1/public_manifest.json').read_text())
summary={'scope':'One genuine CPU full-checkpoint BERT layer0 QKV file-boundary component on two distinct rows; not a complete layer/model/MRPC stage or calibration.','passed':True,'source_count':len(expected),'source_files_unchanged':True,'source_archive_sha256':r['source_archive_sha256'],'processes_absent':pids,'owned_files_removed':True,'checks':client['checks'],'worst_abs_error':client['worst_abs_error'],'fixed_gate':1e-6,'all_slots_checked':client['all_slots_checked'],'ring':public['parameters'],'actual_log2_qp':r['stages']['client-encrypt']['result']['actual_log2_qp'],'native_preprocessing_s':prep['elapsed_s'],'preprocessing_process_s':r['stages']['client-preprocess']['process_elapsed_s'],'cryptographic_subprocess_total_s':r['cryptographic_subprocess_total_s'],'collector_total_s':r['elapsed_s'],'server_process_s':r['stages']['keyless-server']['process_elapsed_s'],'per_projection_server_s':{k:v['server_s'] for k,v in server['checks'].items()},'keyless_server_guards':{k:server[k] for k in ('server_secret_absent','client_codecs_and_keygen_forbidden','client_file_access_attempts')},'cleanup_followup':{'passed_cpu_tests':34,'source_count':len(cr['source_hashes_start']),'source_files_unchanged':True,'patch_sha256':hashlib.sha256(patch_path.read_bytes()).hexdigest(),'changed_files':changed,'encrypted_rerun_claim':False,'independent_security_review':'No additional blocker; unknown active evidence fails closed and directory owner is pinned by fd.'},'known_measured_collector_limit':'Original healthy numerical proof predates the cleanup follow-up for unavailable active-stage evidence and held directory-fd ownership. Its observed cleanup succeeded; it is not relabeled as a follow-up-source run.','timing_scope':'Observed CPU component timings, not calibrated performance. A brief separate metadata-test process shared CPU affinity during the server run; host load was uncontrolled.','full_bert_evaluated':False,'mrpc_accuracy_evaluated':False,'scoring_calibrated':False}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');shutil.copyfile(Path(__file__),OUT/'archive_component.py');print(json.dumps({k:summary[k] for k in ('passed','source_count','worst_abs_error','all_slots_checked','collector_total_s')},indent=2))
