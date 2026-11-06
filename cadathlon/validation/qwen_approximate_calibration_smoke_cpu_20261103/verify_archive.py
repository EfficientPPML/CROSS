"""Offline verification of the11-context approximate smoke and provenance."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();v=json.loads((P/'verification.json').read_text())
for n,h in v['artifact_sha256'].items():assert sha(P/n)==h,n
for run in ('collector-preflight-v1','collector-preflight-v2','approximate-smoke-v1'):
    r=json.loads((P/run/'report.json').read_text());assert r['passed'] and r['source_hashes_before']==r['source_hashes_after']
    with tarfile.open(P/run/'sources.tar.gz') as t:actual={i.name:hashlib.sha256(t.extractfile(i).read()).hexdigest() for i in t.getmembers()}
    assert actual==r['source_hashes_before'] and sha(P/run/'sources.tar.gz')==r['source_archive_sha256']
    if run!='collector-preflight-v1':
        for n,h in actual.items():
            if not n.startswith('software/'):assert sha(P/n)==h,n
r=json.loads((P/'approximate-smoke-v1/report.json').read_text());m=json.loads((P/'candidate-v1/manifest.json').read_text());rows=json.loads((P/'tokenized_calibration_inputs.json').read_text())
assert sha(P/'candidate-v1/manifest.json')==v['manifest_sha256']==r['candidate_manifest_sha256']
assert r['verified_manifest_input_sha256']==m['source_input_sha256']
assert r['asset_sha256_before']==r['asset_sha256_after']==json.loads((P/'model_pins.json').read_text())['asset_sha256']
assert r['smoke'] and not r['complete'] and all(r['gates'].values()) and r['all_workload_entries_accounted']==11
records=list(r['cases']);assert len(records)==3
for d in r['decode']:
    initial=next(x['token_ids'] for x in rows if x['id']==d['id'] and x['sequence_length']==16)
    assert d['complete_sequence_agreement'] and not d['free_suffix_stopped']
    for kind,tokens in [('teacher',d['native_tokens']),('free',d['approximate_free_tokens'])]:
        part=d[kind+'_records'];assert len(part)==4
        for i,x in enumerate(part):assert x['token_ids']==initial+tokens[:i]
        records+=part
assert len(records)==11
for x in records:
    assert x['approximate_complete'] and x['numeric_pass'] and x['domain_pass'] and x['exact_semantic_pass'] and x['failure'] is None
    assert len(x['operator_metrics'])==421 and len(x['block_metrics'])==28 and x['all_final_logits_checked']==151936
    assert x['native_top1']==x['approximate_top1']
    for met in list(x['operator_metrics'].values())+x['block_metrics']:assert met['passed'] and met['finite'] and met['shape_matches'] and met['nrmse']<=.05
    assert x['full_logit_metric']['passed'] and x['full_logit_metric']['nrmse']<=.10
    assert x['exact_vs_native']['max_abs']<=1e-3 and x['exact_vs_native']['nrmse']<=1e-5
    data=b''.join(int(z).to_bytes(8,'little',signed=True) for z in x['token_ids']);assert hashlib.sha256(data).hexdigest()==x['context_sha256']
assert all(not row['below'] and not row['above'] and not row['nonfinite'] for row in r['propagated_domain_ranges'].values())
assert len(r['hook_invariance_checks'])==4 and all(x['all_logits_bitwise_invariant'] for x in r['hook_invariance_checks'])
print(json.dumps({'passed':True,'scope':'11-context smoke only','runtime_sources':31,'operators_per_context':421,'blocks_per_context':28,'full_logit_count':151936}))
