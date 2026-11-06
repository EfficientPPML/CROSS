"""Independent public-only review of BERT5 within the still-running TPU sweep."""
from pathlib import Path
import hashlib,json,math,os,tarfile
P=Path('/tmp/cross-optimized-twelve-20261103')
CASE=P/'evidence/bert/test5'
OUT=Path('/tmp/bert-toy5-terminal-root-review-20261104')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
names=('process.json','stdout.json','stderr.log','security_device.json','resources.json','resource_samples.jsonl')
assert all(not(CASE/n).is_symlink()and(CASE/n).is_file()for n in names)
blobs={n:(CASE/n).read_bytes()for n in names};pins={n:sha(b)for n,b in blobs.items()}
process=json.loads(blobs['process.json']);result=json.loads(blobs['stdout.json']);security=json.loads(blobs['security_device.json'])
assert result=={'case':'test5.in','passed':False,'error':'int too large to convert to float','error_type':'OverflowError','notes':[]}
assert process['result']==result and process['passed']is False and type(process['returncode'])is int and process['returncode']==1
assert all(process[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
assert process['pid']==2610682 and process['timeout_s']is None
assert process['owned_public_key_directory']=='/dev/shm/cross-optimized-twelve-20261103/cross-transformer-case-li25p0uu'
assert process['owned_public_key_directory_identity']==dict(device=24,inode=2467)
assert not Path(process['owned_public_key_directory']).exists()
for f in (os.kill,os.killpg):
    try:f(2610682,0)
    except ProcessLookupError:pass
    else:raise AssertionError('Completed BERT5 process/group remains')
assert security['passed']is False and security['error']=='SystemExit: 1'
assert [e['boundary']for e in security['events']]==['input','output']
qp=[]
for e in security['events']:
    assert len(e['ciphertexts'])==1 and len(e['actual_rings'])==2
    for ring in e['actual_rings']:
        assert ring['degree']==131072 and len(ring['q_towers'])==100 and len(ring['p_towers'])==24
        assert ring['security_bits']==128 and ring['security_model']=='classical'
        bits=sum(math.log2(n)for n in ring['q_towers']+ring['p_towers'])
        assert abs(bits-ring['log2_QP'])<1e-8 and math.ceil(bits)==3491 and bits<=3523
        qp.append(bits)
    assert all(d['platform']=='tpu'for d in e['ciphertexts'][0]['devices'])
output=security['events'][1]['ciphertexts'][0]
assert output['moduli_count']==40 and output['nsd']==1 and output['scale']==6.8994011570896904e16
stderr=blobs['stderr.log'].decode();assert stderr.count('bounded homomorphic refresh starting')==stderr.count('bounded homomorphic refresh complete')==8
assert 'packed BERT layer 2: output ready'in stderr
resources=json.loads(blobs['resources.json']);assert resources==process['resources']
samples=[json.loads(line)for line in blobs['resource_samples.jsonl'].splitlines()]
assert len(samples)==resources['samples']==18957
assert max(x['aggregate_rss_bytes']for x in samples)==resources['sampled_peak_process_tree_rss_bytes']==151212969984
manifest=read(P/'evidence/snapshot_manifest.json');expected=manifest['source_hashes'];assert len(expected)==199
archive=P/'evidence/sources.tar.gz';assert sha(archive.read_bytes())=='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
with tarfile.open(archive)as tar:
    members=tar.getmembers();assert len(members)==len({m.name for m in members})==199
    assert all(m.isfile()and m.name in expected for m in members)
    bodies={m.name:tar.extractfile(m).read()for m in members}
assert {n:sha(b)for n,b in bodies.items()}==expected
assert {n:sha((P/'checkout'/n).read_bytes())for n in expected}==expected
case=json.loads(bodies['cadathlon/agent_pset/p2_bert/test/test5.in']);config=case['config']
assert (config['hidden_size'],config['intermediate_size'],config['num_attention_heads'],config['num_hidden_layers'])==(24,48,4,3)
assert len(case['input_ids'])==2 and all(len(x)==12 for x in case['input_ids'])and case['weight_seed']==9505
parent_raw=(P/'evidence/report.json').read_bytes();parent=json.loads(parent_raw)
assert parent['models']['bert']['cases']['test5.in']==process and parent['passed']is False
cases=[x for model in parent['models'].values()for x in model['cases'].values()]
assert len(cases)==9 and sum(x['passed']is True for x in cases)==7 and sum(x['passed']is False for x in cases)==2
assert parent['phase']=='qwen/test5' and parent['source_archive_sha256']==sha(archive.read_bytes())
assert {n:sha((CASE/n).read_bytes())for n in names}==pins
report=dict(schema='bert-toy5-independent-terminal-root-review-v1',review_passed=True,case_passed=False,
    case_files_sha256=pins,source_count=199,source_archive_sha256=sha(archive.read_bytes()),current_archive_sources_equal=True,
    parent_report_sha256=sha(parent_raw),completed_cases_at_checkpoint=9,prior_passes=7,failed_cases=['bert/test4','bert/test5'],
    original_sweep_still_running=True,active_case='qwen/test5',parent_terminal_or_global_final_source_check_claim=False,
    child_pid=2610682,child_pgid=2610682,live_group_observation_tool_chunk='5556dd',parent_reported_child_reaped=True,
    independently_verified_child_and_group_absent=True,owned_key_path_absent=True,owned_key_identity=process['owned_public_key_directory_identity'],
    error_type='OverflowError',error='int too large to convert to float',refresh_completions=8,
    output_q_limbs=40,output_scale=output['scale'],output_nsd=1,actual_qp_log2=qp[0],actual_qp_ceil_bits=3491,security_ceiling_bits=3523,
    both_recorded_boundaries_tpu=True,security_report_overall_passed=False,decoded_metrics_available=False,
    external_process_elapsed_s=process['process_elapsed_s'],resources=resources,
    fixture=dict(batch=2,sequence_length=12,hidden=24,intermediate=48,heads=4,layers=3,seed=9505),
    scope='Actual encrypted public toy failure; no full pretrained model, ranked accuracy, calibrated score or precise throw-site claim',
    source_diagnosis_limit='The output metadata boundary completed, but no traceback or ciphertext coefficient bodies are retained; exact throw site remains unlocalized.',
    review_script_sha256=sha(Path(__file__).read_bytes()))
OUT.mkdir(mode=0o700)
with (OUT/'review.json').open('x')as f:json.dump(report,f,indent=2);f.write('\n')
with (OUT/'review.py').open('xb')as f:f.write(Path(__file__).read_bytes())
with (OUT/'parent_report_at_review.json').open('xb')as f:f.write(parent_raw)
print(json.dumps(dict(review_passed=True,case_passed=False,review=str(OUT/'review.json'),sha256=sha((OUT/'review.json').read_bytes()),completed_cases=9)))
