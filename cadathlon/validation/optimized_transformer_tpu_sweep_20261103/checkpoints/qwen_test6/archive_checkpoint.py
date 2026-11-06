"""Archive one completed public failure; never poll/reap or execute a candidate."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, math, shutil, tarfile

LAB=Path('/tmp/cross-optimized-twelve-20261103')
SOURCE=LAB/'evidence/qwen/test6'
SHARED=Path('/home/jianming_gatech/CROSS/cadathlon/validation/optimized_transformer_tpu_sweep_20261103')
OBS=Path('/tmp/root-tpu12-q6-progress-20261105')
DEST=Path('/tmp/qwen-toy6-negative-evidence-checkpoint-20261105')
ARCHIVE='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
OBS_SHA='b6425825396b831c9a5d5cf5b209593b74b4d3c9ed7702b0012f5bbd9fe2efc3'
NAMES=('process.json','stdout.json','stderr.log','resources.json','resource_samples.jsonl','security_device.json')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def write(p,value):p.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')

def main():
    assert not DEST.exists()
    assert sha(OBS/'receipt.json')==OBS_SHA
    observation=read(OBS/'receipt.json')
    before={n:sha(SOURCE/n)for n in NAMES}
    for n in NAMES:
        saved=observation['origins'].get('q6-'+n)
        if saved is not None:assert before[n]==saved['sha256']==sha(OBS/('q6-'+n))
    expected=read(SHARED/'snapshot_manifest.json')['source_hashes']
    assert len(expected)==199 and sha(SHARED/'sources.tar.gz')==ARCHIVE
    assert (SHARED/'snapshot_manifest.json').read_bytes()==(LAB/'evidence/snapshot_manifest.json').read_bytes()
    assert {n:sha(LAB/'checkout'/n)for n in expected}==expected
    with tarfile.open(SHARED/'sources.tar.gz','r:gz')as archive:
        members=archive.getmembers()
        assert len(members)==len({m.name for m in members})==199
        assert all(m.isfile()and not PurePosixPath(m.name).is_absolute()and '..'not in PurePosixPath(m.name).parts for m in members)
        bodies={m.name:archive.extractfile(m).read()for m in members}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}==expected
    process=read(SOURCE/'process.json');numeric=read(SOURCE/'stdout.json');security=read(SOURCE/'security_device.json');resources=read(SOURCE/'resources.json')
    assert process['pid']==76681 and process['passed']is False and process['returncode']==1 and process['result']==numeric
    assert numeric['passed']is False and numeric['error_type']=='ApproximationErrorTooHigh'
    assert 'estimated stddev=6.314e+81, limit=3.125e-02' in numeric['error']
    assert all(process[n]is True for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
    assert not Path('/proc/76681').exists()
    try:Path(process['owned_public_key_directory']).lstat()
    except FileNotFoundError:pass
    else:raise AssertionError('completed child key directory still present')
    members=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        except(FileNotFoundError,ProcessLookupError):continue
        if int(fields[2])==76681:members.append(int(p.name))
    assert not members
    samples=[json.loads(line)for line in(SOURCE/'resource_samples.jsonl').read_text().splitlines()]
    assert resources==process['resources'] and len(samples)==resources['samples']==30137
    assert max(x['aggregate_rss_bytes']for x in samples)==resources['sampled_peak_process_tree_rss_bytes']==141938708480
    parent_raw=(OBS/'sweep-report-snapshot.json').read_bytes()
    assert hashlib.sha256(parent_raw).hexdigest()==observation['origins']['sweep-report-snapshot.json']['sha256']
    parent=json.loads(parent_raw);cases=[r for entry in parent['models'].values()for r in entry['cases'].values()]
    assert len(cases)==11 and sum(x['passed']is True for x in cases)==7 and sum(x['passed']is False for x in cases)==4
    assert parent['models']['qwen']['cases']['test6.in']==process and parent['phase']=='bert/test6' and parent['passed']is False
    case=json.loads(bodies['cadathlon/agent_pset/p4_qwen/test/test6.in'])
    output=security['events'][-1]['ciphertexts'][0]
    assert [e['boundary']for e in security['events']]==['input','output']
    assert security['passed']is False and output['moduli_count']==12
    q=security['events'][-1]['actual_rings'][0]['q_towers'][:output['moduli_count']]
    assert math.prod(q)<int(float.fromhex('0x1.fffffffffffffp+1023'))
    assert before=={n:sha(SOURCE/n)for n in NAMES}
    DEST.mkdir();(DEST/'root-review').mkdir()
    for n in NAMES:shutil.copyfile(SOURCE/n,DEST/n)
    shutil.copyfile(OBS/'receipt.json',DEST/'root-review/observation.json')
    (DEST/'parent_report_at_checkpoint.json').write_bytes(parent_raw)
    shutil.copyfile(__file__,DEST/'archive_checkpoint.py')
    cleanup=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=76681,pid_absent=True,matching_process_group_id_members=[],owned_key_directory=process['owned_public_key_directory'],owned_key_directory_absent=True,collector_reaped=True,scope='Read-only current /proc and directory absence, plus original collector record; not whole-sweep terminal admission or escaped-descendant proof.')
    write(DEST/'root-review/cleanup-observation.json',cleanup)
    summary=dict(schema='optimized-tpu-qwen-test6-failure-checkpoint-v1',verified_utc=cleanup['utc'],case_passed=False,case_returncode=1,case_pid=76681,parent_tool_session=12657,parent_tool_reaping_owner='root',parent_tool_exit_code=None,completed_cases=11,passed_cases=7,failed_cases=4,required_total_cases=12,full_sweep_terminal_admitted=False,current_phase_at_root_observation='bert/test6',root_observation_sha256=OBS_SHA,cleanup_observation_sha256=sha(DEST/'root-review/cleanup-observation.json'),parent_report_sha256=sha(DEST/'parent_report_at_checkpoint.json'),source_count=199,source_archive_sha256=ARCHIVE,source_manifest_sha256=sha(SHARED/'snapshot_manifest.json'),source_file_sha256=expected,source_archive_and_current_snapshot_verified=True,failed_case_files_before=before,failed_case_files_after={n:sha(SOURCE/n)for n in NAMES},case_public=case,numerical_result=numeric,logit_metrics_available=False,runner_timing_metrics_available=False,external_case_elapsed_s=process['process_elapsed_s'],resources=resources,completed_refresh_pairs=19,input_output_security_gate_return_observed=True,security_report_passed=False,security_report_error=security['error'],output_ciphertext=output,output_Q_log2=sum(math.log2(v)for v in q),output_Q_within_binary64_finite_range=True,cleanup={n:process[n]for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed')},limitations=['Negative synthetic public fixture only; not full pretrained Qwen, calibration, ranked or HE acceptance.','No decoded logits, error metrics or successful runner timings. External failed-process duration is not a passing benchmark.','Refresh/layer completion and final Q range do not identify the first numerical divergence. Strict decoding remains unchanged.','Sampled process-tree RSS can double-count shared pages and miss transients. Device statistics are checkpoint observations, not continuous peaks.','Shared199 archive is referenced from the enclosing sweep; no duplicate source archive.','No private Qwen calibrator outcomes or key/ciphertext/model bodies; no runtime changes, restart or new actual execution.'])
    write(DEST/'summary.json',summary)
    print(json.dumps(dict(destination=str(DEST),source_count=199,copied_case_files=6,samples=len(samples),case_passed=False,completed_cases=11)))

if __name__=='__main__':main()
