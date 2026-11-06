"""Verify public failed-case evidence; no imports of runtime or payload readers."""
from pathlib import Path,PurePosixPath
import argparse,hashlib,json,math,tarfile
P=Path(__file__).resolve().parent
CURRENT=Path('/tmp/cross-optimized-twelve-20261103')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def verify(source_root,current=False):
    source_root=Path(source_root)
    index=read(P/'artifact_hashes.json')
    assert not any(p.is_symlink()for p in P.rglob('*'))
    assert {str(p.relative_to(P))for p in P.rglob('*')if p.is_file()}==set(index)|{'artifact_hashes.json'}
    for n,h in index.items():assert sha(P/n)==h
    s=read(P/'summary.json');manifest=read(source_root/'snapshot_manifest.json');expected=manifest['source_hashes']
    assert len(expected)==199 and expected==s['source_file_sha256']
    assert sha(source_root/'snapshot_manifest.json')==s['source_manifest_sha256']
    assert sha(source_root/'sources.tar.gz')==s['source_archive_sha256']=='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
    with tarfile.open(source_root/'sources.tar.gz','r:gz')as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==199 and all(m.isfile()for m in members)
        assert all(not PurePosixPath(m.name).is_absolute() and '..'not in PurePosixPath(m.name).parts for m in members)
        bodies={m.name:tar.extractfile(m).read()for m in members}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}==expected
    assert len(s['failed_case_files_before'])==6
    for n,h in s['failed_case_files_before'].items():assert sha(P/n)==h==s['failed_case_files_after'][n]
    process=read(P/'process.json');numeric=read(P/'stdout.json');security=read(P/'security_device.json');parent=read(P/'parent_report_at_checkpoint.json');root=read(P/'root-review/observation.json')
    assert sha(P/'root-review/observation.json')==s['root_observation_sha256']=='551d7c4693df06d7dfab612cbd83801e10f85cc6d943bea0f2280112d0afd8ae'
    assert sha(P/'parent_report_at_checkpoint.json')==root['files']['sweep-in-progress.json']==s['parent_report_sha256']
    assert root['negative_case_observed']is True and root['child_group_absent_now']is True and root['full_sweep_terminal_admitted']is False
    for n in ('process.json','stdout.json','stderr.log','security_device.json','resources.json'):assert root['files'][n]==sha(P/n)
    assert process['passed']is False and type(process['returncode'])is int and process['returncode']==1 and process['result']==numeric and numeric['passed']is False
    assert numeric['error_type']=='ApproximationErrorTooHigh'and 'estimated stddev=4.099e+48, limit=3.125e-02' in numeric['error']
    assert not any(n in numeric for n in('max_abs_error','nrmse','top1_agreement','server_s','setup_s','end_to_end_s','total_s'))
    assert all(process[n]is True for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
    assert process['pid']==root['child_pid']==2992744
    assert security['passed']is False and security['error']=='SystemExit: 1'
    assert [e['boundary']for e in security['events']]==['input','output']
    for event in security['events']:
        assert len(event['ciphertexts'])==1 and len(event['actual_rings'])==2
        for ring in event['actual_rings']:
            q,p=ring['q_towers'],ring['p_towers'];assert ring['degree']==131072 and len(q)==100 and len(p)==24
            assert ring['security_bits']==128 and ring['security_model']=='classical'
            assert abs(sum(math.log2(x)for x in q+p)-ring['log2_QP'])<1e-8 and math.prod(q+p).bit_length()<=3523
            assert math.prod(p)>=max(math.prod(q[i:i+25])for i in range(0,100,25))
        assert event['actual_rings'][1]['dnum']==4
        assert all(d['platform']=='tpu'for c in event['ciphertexts']for d in c['devices'])
    output=security['events'][1]['ciphertexts'][0]
    assert output==s['output_ciphertext'] and output['moduli_count']==12 and output['nsd']==1 and output['scale']==6.89940115708969e16
    assert output['shape']==[1,2,256,512,12] and output['payload_bytes']==12582912
    stderr=(P/'stderr.log').read_text();assert stderr.count('bounded homomorphic refresh starting')==stderr.count('bounded homomorphic refresh complete')==19
    assert all(f'packed Qwen layer {i}: {kind} ready'in stderr for i in range(3)for kind in('attention','MLP'))
    resources=read(P/'resources.json');assert resources==process['resources']==s['resources']
    samples=[json.loads(line)for line in(P/'resource_samples.jsonl').read_text().splitlines()]
    assert len(samples)==resources['samples']==28916
    assert max(x['aggregate_rss_bytes']for x in samples)==resources['sampled_peak_process_tree_rss_bytes']==83863691264
    cases=[r for m in parent['models'].values()for r in m['cases'].values()]
    assert len(cases)==10 and sum(c['passed']is True for c in cases)==7 and sum(c['passed']is False for c in cases)==3
    assert parent['models']['qwen']['cases']['test5.in']==process and parent['passed']is False
    case=json.loads(bodies['cadathlon/agent_pset/p4_qwen/test/test5.in']);assert case==s['case_public']
    assert hashlib.sha256(bodies['cadathlon/agent_pset/p4_qwen/test/test5.in']).hexdigest()==root['files']['test5.in']
    c=case['config'];assert (c['hidden_size'],c['intermediate_size'],c['num_attention_heads'],c['num_key_value_heads'],c['head_dim'],c['num_hidden_layers'],c['vocab_size'])==(16,24,4,2,8,3,32)
    assert case['weight_seed']==9105 and len(case['token_ids'])==2 and all(len(row)==4 for row in case['token_ids'])
    assert s['case_passed']is False and s['full_sweep_terminal_admitted']is False and s['parent_tool_exit_code']is None
    if current:
        assert {n:sha(CURRENT/'checkout'/n)for n in expected}==expected
        assert {n:sha(CURRENT/'evidence/qwen/test5'/n)for n in s['failed_case_files_before']}==s['failed_case_files_before']
    print(json.dumps(dict(archive_verified=True,source_count=199,completed_cases_at_checkpoint=10,prior_passes=7,failed_cases=3,failed_case='qwen/test5',case_passed=False,current_checked=current,full_sweep_terminal_admitted=False,full_model_claim=False,indexed_files=len(index))))
if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-proof-root',type=Path,default=P.parents[1],help='Enclosing sweep directory containing exact snapshot_manifest.json and sources.tar.gz.')
    parser.add_argument('--current',action='store_true')
    args=parser.parse_args();verify(args.source_proof_root,args.current)
