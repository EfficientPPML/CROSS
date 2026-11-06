"""Public failed-case checkpoint; does not poll or reap the live TPU sweep."""
from pathlib import Path
import datetime,hashlib,json,math,os,shutil,tarfile

LAB=Path('/tmp/cross-optimized-twelve-20261103');EV=LAB/'evidence';SRC=EV/'bert/test5'
ROOT=Path('/home/jianming_gatech/CROSS/cadathlon/validation/optimized_transformer_tpu_sweep_20261103')
DEST=ROOT/'checkpoints/bert_test5'
ARCHIVE='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
NAMES={'process.json','stdout.json','stderr.log','resource_samples.jsonl','resources.json','security_device.json'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
load=lambda p:json.loads(p.read_bytes())

def absent(pid):
    assert not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:return True
    raise AssertionError('Failed case group remains')

def main():
    assert {p.name for p in SRC.iterdir()}==NAMES
    assert not any(p.is_symlink()for p in SRC.iterdir())
    before={n:sha(SRC/n)for n in sorted(NAMES)}
    manifest=load(EV/'snapshot_manifest.json');expected=manifest['source_hashes'];assert len(expected)==199
    assert {n:sha(LAB/'checkout'/n)for n in expected}==expected
    assert sha(EV/'sources.tar.gz')==sha(ROOT/'sources.tar.gz')==ARCHIVE
    assert (EV/'snapshot_manifest.json').read_bytes()==(ROOT/'snapshot_manifest.json').read_bytes()
    with tarfile.open(EV/'sources.tar.gz','r:gz')as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==199
        assert all(m.isfile()for m in members)
        bodies={m.name:tar.extractfile(m).read()for m in members}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}==expected
    process=load(SRC/'process.json');numeric=load(SRC/'stdout.json');security=load(SRC/'security_device.json')
    assert process['passed']is False and process['returncode']==1 and process['result']==numeric and numeric['passed']is False
    assert numeric['error_type']=='OverflowError'and numeric['error']=='int too large to convert to float'
    assert not any(n in numeric for n in('max_abs_error','nrmse','top1_agreement','server_s','setup_s','end_to_end_s','total_s'))
    assert all(process[n]is True for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
    assert absent(process['pid'])and not Path(process['owned_public_key_directory']).exists()
    assert security['passed']is False and security['error']=='SystemExit: 1'
    assert [e['boundary']for e in security['events']]==['input','output']
    for event in security['events']:
        assert len(event['ciphertexts'])==1 and len(event['actual_rings'])==2
        for ring in event['actual_rings']:
            assert ring['degree']==131072 and ring['security_bits']==128 and ring['security_model']=='classical'
            assert len(ring['q_towers'])==100 and len(ring['p_towers'])==24
            qp=sum(math.log2(q)for q in ring['q_towers']+ring['p_towers'])
            assert abs(qp-ring['log2_QP'])<1e-8 and qp<=3523
        assert all(d['platform']=='tpu'for c in event['ciphertexts']for d in c['devices'])
    output=security['events'][1]['ciphertexts'][0]
    assert output['moduli_count']==40 and output['nsd']==1 and output['scale']==6.8994011570896904e16
    stderr=(SRC/'stderr.log').read_text()
    assert stderr.count('bounded homomorphic refresh starting')==stderr.count('bounded homomorphic refresh complete')==8
    assert all(f'packed BERT layer {i}: output ready'in stderr for i in range(3))
    resources=load(SRC/'resources.json');assert resources==process['resources']
    samples=[json.loads(line)for line in(SRC/'resource_samples.jsonl').read_text().splitlines()]
    assert len(samples)==resources['samples']==18957
    assert max(r['aggregate_rss_bytes']for r in samples)==resources['sampled_peak_process_tree_rss_bytes']==151212969984
    parent_raw=(EV/'report.json').read_bytes();parent=json.loads(parent_raw)
    cases=[r for m in parent['models'].values()for r in m['cases'].values()]
    assert len(cases)==9 and sum(r['passed']is True for r in cases)==7 and sum(r['passed']is False for r in cases)==2
    assert parent['models']['bert']['cases']['test5.in']==process and parent['passed']is False
    parent_stat=Path('/proc/944985/stat').read_text();fields=parent_stat[parent_stat.rfind(')')+2:].split()
    parent_observation=dict(pid=944985,ppid=int(fields[1]),pgid=int(fields[2]),present=True)
    case=json.loads(bodies['cadathlon/agent_pset/p2_bert/test/test5.in']);config=case['config']
    case_public=dict(purpose=case['purpose'],weight_seed=case['weight_seed'],batch_size=len(case['input_ids']),sequence_length=len(case['input_ids'][0]),
        hidden_size=config['hidden_size'],intermediate_size=config['intermediate_size'],num_attention_heads=config['num_attention_heads'],
        num_hidden_layers=config['num_hidden_layers'],num_labels=config['num_labels'],valid_keys=[sum(row)for row in case['attention_mask']])
    assert (case_public['batch_size'],case_public['sequence_length'],case_public['hidden_size'],case_public['intermediate_size'],case_public['num_hidden_layers'])==(2,12,24,48,3)
    assert before=={n:sha(SRC/n)for n in NAMES}
    DEST.mkdir()
    for name in NAMES:shutil.copyfile(SRC/name,DEST/name)
    (DEST/'parent_report_at_checkpoint.json').write_bytes(parent_raw)
    summary=dict(schema='optimized-tpu-bert-test5-failure-checkpoint-v1',verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Ninth completed case of original optimized toy TPU sweep; second failed case. Parent still live and root-owned.',
        case_passed=False,case_returncode=1,case_pid=process['pid'],case_pid_and_group_absent=True,
        case_reaping_observation='Recorded by original validation_sweep parent; archival reviewer did not reap the child.',
        parent_process_observation=parent_observation,parent_tool_session=12657,parent_tool_reaping_owner='root',parent_tool_exit_code=None,
        completed_cases=9,passed_cases=7,failed_cases=2,required_total_cases=12,overall_sweep_passed=False,
        source_count=199,source_archive_and_current_snapshot_verified=True,source_archive_sha256=ARCHIVE,source_file_sha256=expected,
        failed_case_files_before=before,failed_case_files_after={n:sha(SRC/n)for n in NAMES},case_public=case_public,
        numerical_result=numeric,logit_metrics_available=False,runner_timing_metrics_available=False,
        external_case_elapsed_s=process['process_elapsed_s'],resources=resources,completed_refresh_pairs=8,
        input_output_security_gate_return_observed=True,security_report_passed=False,security_report_error=security['error'],output_ciphertext=output,
        cleanup={n:process[n]for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed')},
        limitations=['Synthetic BERT dimensions only; no full pretrained/MRPC/ranked accuracy claim.',
            'OverflowError is the recorded exception; no decoded logits/error/nRMSE/top1 or successful runner timing is available.',
            'No traceback identifies the exception site in saved stderr. Source diagnosis is separate.',
            'Finite scale/remaining levels/security and completed refresh logs do not establish numerical correctness.',
            'Sampled RSS may double count shared pages and miss peaks; device statistics are boundary checkpoints.',
            'No keys or ciphertext bodies were read/copied, and no TPU job, retry, parent reap or source change was performed.'])
    (DEST/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    shutil.copyfile(__file__,DEST/'archive_checkpoint.py')
    print(json.dumps(dict(destination=str(DEST),source_count=199,case_passed=False,completed_cases=9,passed_cases=7,failed_cases=2,
        six_case_files_sha256=before),sort_keys=True))

if __name__=='__main__':main()
