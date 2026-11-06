"""Copy one public failed case; no live-job polling, payload reads or code execution."""
from pathlib import Path
import datetime,hashlib,json,math,shutil,tarfile
LAB=Path('/tmp/cross-optimized-twelve-20261103')
SOURCE=LAB/'evidence/qwen/test5'
ROOTOBS=Path('/tmp/qwen-toy5-negative-root-observation-20261104')
SHARED=Path('/home/jianming_gatech/CROSS/cadathlon/validation/optimized_transformer_tpu_sweep_20261103')
DEST=Path('/tmp/qwen-toy5-negative-evidence-checkpoint-20261104')
ARCHIVE='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
NAMES=('process.json','stdout.json','stderr.log','resources.json','resource_samples.jsonl','security_device.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def main():
    assert not DEST.exists()
    before={n:sha(SOURCE/n)for n in NAMES}
    assert sha(ROOTOBS/'observation.json')=='551d7c4693df06d7dfab612cbd83801e10f85cc6d943bea0f2280112d0afd8ae'
    observation=read(ROOTOBS/'observation.json')
    for n in NAMES:
        if n in observation['files']:assert before[n]==observation['files'][n]==sha(ROOTOBS/n)
    manifest=read(SHARED/'snapshot_manifest.json');expected=manifest['source_hashes']
    assert len(expected)==199 and sha(SHARED/'sources.tar.gz')==ARCHIVE
    assert (SHARED/'snapshot_manifest.json').read_bytes()==(LAB/'evidence/snapshot_manifest.json').read_bytes()
    assert {n:sha(LAB/'checkout'/n)for n in expected}==expected
    with tarfile.open(SHARED/'sources.tar.gz','r:gz')as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==199 and all(m.isfile()for m in members)
        bodies={m.name:tar.extractfile(m).read()for m in members}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}==expected
    process=read(SOURCE/'process.json');numeric=read(SOURCE/'stdout.json');security=read(SOURCE/'security_device.json');resources=read(SOURCE/'resources.json')
    assert process['passed']is False and process['returncode']==1 and process['result']==numeric
    assert numeric['error_type']=='ApproximationErrorTooHigh'and numeric['passed']is False
    assert all(process[n]is True for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
    samples=[json.loads(line)for line in(SOURCE/'resource_samples.jsonl').read_text().splitlines()]
    assert resources==process['resources'] and len(samples)==resources['samples']==28916
    assert max(x['aggregate_rss_bytes']for x in samples)==resources['sampled_peak_process_tree_rss_bytes']==83863691264
    parent_raw=(ROOTOBS/'sweep-in-progress.json').read_bytes();assert hashlib.sha256(parent_raw).hexdigest()==observation['files']['sweep-in-progress.json']
    parent=json.loads(parent_raw);cases=[r for m in parent['models'].values()for r in m['cases'].values()]
    assert len(cases)==10 and sum(x['passed']is True for x in cases)==7 and sum(x['passed']is False for x in cases)==3
    assert parent['models']['qwen']['cases']['test5.in']==process and parent['passed']is False
    case=json.loads(bodies['cadathlon/agent_pset/p4_qwen/test/test5.in']);output=security['events'][-1]['ciphertexts'][0]
    assert before=={n:sha(SOURCE/n)for n in NAMES}
    DEST.mkdir();(DEST/'root-review').mkdir()
    for n in NAMES:shutil.copyfile(SOURCE/n,DEST/n)
    shutil.copyfile(ROOTOBS/'observation.json',DEST/'root-review/observation.json')
    (DEST/'parent_report_at_checkpoint.json').write_bytes(parent_raw)
    shutil.copyfile(__file__,DEST/'archive_checkpoint.py')
    summary={'schema':'optimized-tpu-qwen-test5-failure-checkpoint-v1','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'case_passed':False,'case_returncode':1,'case_pid':2992744,'case_reaping_observation':'Original collector records reap; root independently observed PID/group absent. This archiver did not poll/reap the sweep.','root_observation_sha256':sha(DEST/'root-review/observation.json'),'parent_report_sha256':sha(DEST/'parent_report_at_checkpoint.json'),'parent_tool_session':12657,'parent_tool_reaping_owner':'root','parent_tool_exit_code':None,'completed_cases':10,'passed_cases':7,'failed_cases':3,'required_total_cases':12,'full_sweep_terminal_admitted':False,'current_phase_at_root_observation':'qwen/test6','source_count':199,'source_archive_sha256':ARCHIVE,'source_manifest_sha256':sha(SHARED/'snapshot_manifest.json'),'source_file_sha256':expected,'source_archive_and_current_snapshot_verified':True,'failed_case_files_before':before,'failed_case_files_after':{n:sha(SOURCE/n)for n in NAMES},'case_public':case,'numerical_result':numeric,'logit_metrics_available':False,'runner_timing_metrics_available':False,'external_case_elapsed_s':process['process_elapsed_s'],'resources':resources,'completed_refresh_pairs':19,'input_output_security_gate_return_observed':True,'security_report_passed':False,'security_report_error':security['error'],'output_ciphertext':output,'cleanup':{n:process[n]for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed')},'limitations':['Negative synthetic toy case only, not fullQwen/calibration/ranked or HE acceptance.','No decoded logits/nRMSE/maxabs or successful runner timings available.','Generic completed refresh/layer messages do not localize first numerical fault or prove domain membership.','Sampled RSS may double-count shared pages and miss peaks; TPU memory values are checkpoint metadata.','Shared199 source archive remains in enclosing sweep; no extra copy.','No backend/key/ciphertext/model/private bodies, restart, source edit or live-job ownership change.']}
    (DEST/'summary.json').write_text(json.dumps(summary,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'destination':str(DEST),'copied_case_files':len(NAMES),'source_count':199,'case_passed':False,'checkpoint_cases':10,'sample_count':len(samples)}))
if __name__=='__main__':main()
