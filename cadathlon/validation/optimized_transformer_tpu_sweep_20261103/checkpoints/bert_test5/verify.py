"""Portable public failed-case check; no backend imports or private body reads."""
from pathlib import Path
import argparse,hashlib,json,math,tarfile
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
CURRENT=Path('/tmp/cross-optimized-twelve-20261103')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())

def verify(current=False):
    index=read(P/'artifact_hashes.json')
    assert not any(p.is_symlink()for p in P.rglob('*'))
    assert {p.relative_to(P).as_posix()for p in P.rglob('*')if p.is_file()}==set(index)|{'artifact_hashes.json'}
    for name,pin in index.items():assert sha(P/name)==pin
    summary=read(P/'summary.json');expected=read(ROOT/'snapshot_manifest.json')['source_hashes']
    assert len(expected)==199 and expected==summary['source_file_sha256']
    assert sha(ROOT/'sources.tar.gz')==summary['source_archive_sha256']=='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
    with tarfile.open(ROOT/'sources.tar.gz','r:gz')as tar:
        members=tar.getmembers();assert len(members)==len({m.name for m in members})==199 and all(m.isfile()for m in members)
        bodies={m.name:tar.extractfile(m).read()for m in members}
    assert {n:hashlib.sha256(b).hexdigest()for n,b in bodies.items()}==expected
    for name,pin in summary['failed_case_files_before'].items():assert sha(P/name)==pin==summary['failed_case_files_after'][name]
    process=read(P/'process.json');numeric=read(P/'stdout.json');security=read(P/'security_device.json');parent=read(P/'parent_report_at_checkpoint.json')
    assert process['passed']is False and process['returncode']==1 and process['result']==numeric and numeric['passed']is False
    assert numeric['error_type']=='OverflowError'and numeric['error']=='int too large to convert to float'
    assert not any(n in numeric for n in('max_abs_error','nrmse','top1_agreement','server_s','setup_s','end_to_end_s','total_s'))
    assert all(process[n]is True for n in('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
    assert security['passed']is False and security['error']=='SystemExit: 1'
    assert [e['boundary']for e in security['events']]==['input','output']
    for event in security['events']:
        assert len(event['ciphertexts'])==1 and len(event['actual_rings'])==2
        for ring in event['actual_rings']:
            assert ring['degree']==131072 and len(ring['q_towers'])==100 and len(ring['p_towers'])==24
            assert ring['security_bits']==128 and ring['security_model']=='classical'
            assert abs(sum(math.log2(q)for q in ring['q_towers']+ring['p_towers'])-ring['log2_QP'])<1e-8 and ring['log2_QP']<=3523
        assert all(d['platform']=='tpu'for c in event['ciphertexts']for d in c['devices'])
    output=security['events'][1]['ciphertexts'][0]
    assert output['moduli_count']==40 and output['nsd']==1 and output['scale']==6.8994011570896904e16
    stderr=(P/'stderr.log').read_text();assert stderr.count('bounded homomorphic refresh starting')==stderr.count('bounded homomorphic refresh complete')==8
    assert all(f'packed BERT layer {i}: output ready'in stderr for i in range(3))
    resources=read(P/'resources.json');assert resources==process['resources']==summary['resources']
    samples=[json.loads(line)for line in(P/'resource_samples.jsonl').read_text().splitlines()]
    assert len(samples)==resources['samples']==18957
    assert max(r['aggregate_rss_bytes']for r in samples)==resources['sampled_peak_process_tree_rss_bytes']==151212969984
    cases=[r for m in parent['models'].values()for r in m['cases'].values()]
    assert len(cases)==9 and sum(c['passed']is True for c in cases)==7 and sum(c['passed']is False for c in cases)==2
    assert parent['models']['bert']['cases']['test5.in']==process and parent['passed']is False
    case=json.loads(bodies['cadathlon/agent_pset/p2_bert/test/test5.in']);c=case['config']
    assert (c['hidden_size'],c['intermediate_size'],c['num_attention_heads'],c['num_hidden_layers'])==(24,48,4,3)
    assert len(case['input_ids'])==2 and all(len(row)==12 for row in case['input_ids'])
    assert [sum(row)for row in case['attention_mask']]==[8,7]
    assert summary['case_passed']is False and summary['overall_sweep_passed']is False and summary['case_pid_and_group_absent']is True
    review=read(P/'root-review/review.json')
    assert review['review_passed']is True and review['case_passed']is False
    assert review['case_files_sha256']==summary['failed_case_files_before']
    assert review['source_archive_sha256']==summary['source_archive_sha256']
    assert review['review_script_sha256']==sha(P/'root-review/review.py')
    assert review['parent_report_sha256']==sha(P/'root-review/parent_report_at_review.json')
    assert review['parent_reported_child_reaped']is True and review['independently_verified_child_and_group_absent']is True
    if current:
        assert {n:sha(CURRENT/'checkout'/n)for n in expected}==expected
        assert {n:sha(CURRENT/'evidence/bert/test5'/n)for n in summary['failed_case_files_before']}==summary['failed_case_files_before']
    print(json.dumps(dict(archive_verified=True,source_count=199,completed_cases=9,prior_passes=7,failed_cases=2,
        failed_case='bert/test5',case_passed=False,current_checked=current,full_model_claim=False)))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--current',action='store_true');verify(p.parse_args().current)
