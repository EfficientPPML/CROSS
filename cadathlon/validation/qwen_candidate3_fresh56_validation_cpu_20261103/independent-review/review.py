"""Public report accounting only; no model/runtime imports or evaluation."""
from pathlib import Path
import hashlib,importlib.metadata,json,math,struct,tarfile,time
ROOT=Path('/tmp/qwen-candidate3-fresh56-validation-20261103')
HERE=Path(__file__).resolve().parent
EXPECTED_REPORT='9925407314aa1545e26fa7fbcaba9de0bb6f1b20f2b131c31dc72a5969b01b48'
EXPECTED_CANDIDATE='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def require(value,label):
    if not value:raise AssertionError(label)
def finite_number(x):return type(x) in (float,int) and math.isfinite(x)
def metric(m,limit,exact=False):
    ok=(type(m)is dict and m.get('finite')is True and m.get('shape_matches')is True
        and finite_number(m.get('max_abs')) and m['max_abs']>=0 and finite_number(m.get('nrmse')) and 0<=m['nrmse']<=limit)
    if exact:ok=ok and m['max_abs']<=1e-3 and m.get('top1_match')is True
    require(m.get('passed')is ok,'stored metric gate differs')
    return ok

def main():
    started=time.monotonic();path=ROOT/'fresh-v1/report.json';require(sha(path)==EXPECTED_REPORT,'report hash')
    r=json.loads(path.read_text());before=r['source_hashes_before'];require(len(before)==76 and before==r['source_hashes_after'],'76 unchanged sources')
    archive=ROOT/'fresh-v1/sources.tar.gz';require(sha(archive)==r['source_archive_sha256']=='ef6e7e2a4b99bc6e3651148638d6fe97d5864e368baeaef3f6d0c04bfdc0746e','source archive hash')
    with tarfile.open(archive) as tar:
        members=tar.getmembers();require(len(members)==76 and all(m.isfile() for m in members),'archive inventory')
        archived={m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in members}
    require(archived==before,'all archive bytes')
    dist=importlib.metadata.distribution('transformers');current={}
    for name in before:
        p=Path(dist.locate_file(name.removeprefix('software/'))) if name.startswith('software/') else ROOT/name
        external={
            'validation_inputs/tokenized_fixtures.json':Path('/tmp/qwen-fresh56-outerdomain-freeze-20261103/freeze-v1/tokenized_fresh_inputs.json'),
            'fresh_freeze/sources.tar.gz':Path('/tmp/qwen-fresh56-outerdomain-freeze-20261103/freeze-v1/sources.tar.gz'),
            'calibration_prerequisite/report.json':Path('/tmp/qwen-candidate3-calibration-20261103/full-v1/report.json'),
            'calibration_prerequisite/runtime_sources.tar.gz':Path('/tmp/qwen-candidate3-calibration-20261103/full-v1/sources.tar.gz')}
        current[name]=sha(external.get(name,p))
    require(current==before,'all current source bytes')
    require(sha(ROOT/'candidate-v3/manifest.json')==r['candidate_manifest_sha256']==EXPECTED_CANDIDATE,'candidate identity')
    require(sha(ROOT/'calibration/protocol.json')==r['protocol_sha256'],'original protocol identity')
    require(sha(ROOT/'selection/protocol.json')==r['selection_protocol_sha256']!=r['protocol_sha256'],'distinct selection protocol')
    require(r['asset_sha256_before']==r['asset_sha256_after'] and len(r['asset_sha256_before'])==7 and r['assets_unchanged']is True,'reported seven asset identities')
    require(r['selection_prerequisite']['passed']is True and r['selection_prerequisite']['manifest_sha256']==EXPECTED_CANDIDATE,'selection gate identity')
    require(r['workload']=='fresh','fresh workload')
    require(sha(ROOT/'validation_protocol.json')==r['validation_protocol_sha256'],'validation protocol')
    require(r['calibration_prerequisite']['verified']is True and r['calibration_prerequisite']['report_sha256']=='1e827000cc0368853f2e8932ec17ee5f318e973e3c80f67652790491fcf7c628','calibration prerequisite')
    require(r['fresh_freeze_prerequisite']['verified']is True and r['fresh_freeze_prerequisite']['tokenized_records']==56 and r['fresh_freeze_prerequisite']['planned_contexts']==112,'fresh freeze prerequisite')
    rows=r['cases'];teachers=[x for d in r['decode'] for x in d['teacher_records']];frees=[x for d in r['decode'] for x in d['free_records']]
    require([len(x) for x in (rows,teachers,frees,r['decode'])]==[48,32,32,8],'complete112 accounting')
    allrows=rows+teachers+frees;require(len({x['id'] for x in allrows})==112,'unique context IDs')
    metrics={'operator':[],'block':[],'logit':[],'exact':[]};matches={}
    for row in allrows:
        require(row['approximate_complete']is True and row['failure']is None,'all trajectories measured')
        require(row['missing_operator_count']==0 and row['missing_block_count']==0 and len(row['operator_metrics'])==421 and len(row['block_metrics'])==28,'all421 operators/28blocks')
        require(row['all_final_logits_checked']==151936,'all logits')
        require(type(row['domain_observations'])is int and row['domain_observations']>0 and row['domain_pass']is True,'domain observations')
        numeric=all(metric(m,.05) for m in row['operator_metrics'].values()) and all(metric(m,.05) for m in row['block_metrics']) and metric(row['full_logit_metric'],.1)
        require(numeric and row['numeric_pass']is True,'numeric gate')
        require(metric(row['exact_vs_native'],1e-5,True) and row['exact_semantic_pass']is True,'exact-native semantics')
        ids=row['token_ids'];require(len(ids)==row['sequence_length'] and all(type(x)is int for x in ids),'public context shape')
        require(hashlib.sha256(b''.join(struct.pack('<q',x) for x in ids)).hexdigest()==row['context_sha256'],'context bytes hash')
        require(all(type(row[k])is int and 0<=row[k]<151936 for k in ('native_top1','approximate_top1')),'top1 types/range')
        metrics['operator'].extend(row['operator_metrics'].values());metrics['block'].extend(row['block_metrics']);metrics['logit'].append(row['full_logit_metric']);metrics['exact'].append(row['exact_vs_native'])
    def summary(records,n):
        return dict(expected_contexts=n,records=len(records),complete_approximate_contexts=sum(x['approximate_complete']is True for x in records),
            top1_matches=sum(x['approximate_top1']==x['native_top1'] for x in records),top1_denominator=n,
            numeric_and_domain_pass=len(records)==n and all(x['numeric_pass'] and x['domain_pass'] and x['exact_semantic_pass'] for x in records),complete=len(records)==n and all(x['approximate_complete'] for x in records))
    summaries={name:summary(records,n) for name,records,n in [('prefill',rows,48),('teacher',teachers,32),('free',frees,32)]}
    for name,value in summaries.items():require(value==r[name+'_summary'],'stored '+name+' summary')
    for n in (8,16,32):require(summary([x for x in rows if x['sequence_length']==n],16)==r['per_suite'][str(n)],'suite accounting')
    complete=0
    for d in r['decode']:
        require(len(d['teacher_records'])==len(d['free_records'])==4 and d['free_suffix_stopped']is False,'four actual steps per branch')
        teacher=d['teacher_records'];free=d['free_records']
        require(teacher[0]['token_ids']==free[0]['token_ids'],'same initial seed')
        for i in range(4):
            require(teacher[i]['native_top1']==d['native_tokens'][i] and free[i]['approximate_top1']==d['approximate_free_tokens'][i],'declared decode choices')
            if i:
                require(teacher[i]['token_ids']==teacher[i-1]['token_ids']+[teacher[i-1]['native_top1']],'teacher prefix')
                require(free[i]['token_ids']==free[i-1]['token_ids']+[free[i-1]['approximate_top1']],'free prefix')
        match=d['native_tokens']==d['approximate_free_tokens'];require(d['complete_sequence_agreement']is match,'sequence gate');complete+=match
    require(complete==r['complete_free_sequence_matches']==8,'sequence aggregate')
    require((r['required_prefill_top1_matches'],r['required_teacher_top1_matches'],r['required_complete_free_sequences'])==(44,29,6),'unchanged fresh ceilings')
    gates=dict(all_numeric_and_domains=True,aggregate_prefill_top1=summaries['prefill']['top1_matches']>=44,teacher_top1=summaries['teacher']['top1_matches']>=29,complete_free_sequences=complete>=6)
    require(gates==r['gates'] and all(gates.values()),'original aggregate gates')
    domains=r['propagated_domain_ranges'];observations=0
    for label,d in domains.items():
        lo,hi=d['domain'];require(all(finite_number(x) for x in (lo,hi,d['minimum'],d['maximum'])),'finite domain range')
        require(lo<=d['minimum']<=d['maximum']<=hi and d['below']==d['above']==d['nonfinite']==0,'propagated domain range '+label)
        require(type(d['count'])is int and d['count']>0,'observed domain count');observations+=d['count']
    require(len(r['hook_invariance_checks'])==4 and all(x['all_logits_bitwise_invariant']is True for x in r['hook_invariance_checks']),'native hook invariance')
    require(r['complete']is True and r['passed']is True and r['smoke']is False and r['errors']==[] and r['all_workload_entries_accounted']==112,'final report state')
    require(r['validation_data_read']is True and r['validation_input_read_attempted']is True and r['validation_input_read_status']=='confirmed' and r['terminal_scope_completed']is True and r['coefficient_selection_performed']is False and r['encrypted_execution']is False and r['no_jax_imported']is True,'scope flags')
    result=dict(passed=True,scope='Independent aggregate/accounting review of public recorded fresh-validation metrics and domain extrema; no rerun of tensor errors, assets, model or HE; authorized fresh token bytes hashed for source identity only.',
        report_sha256=EXPECTED_REPORT,candidate_manifest_sha256=EXPECTED_CANDIDATE,source_count=76,source_current_archive_before_after_identical=True,
        asset_scope='Seven before/after reported hashes compared; asset payloads not reopened.',contexts=112,prefill=summaries['prefill'],teacher=summaries['teacher'],free=summaries['free'],complete_sequences=complete,
        metric_counts={name:len(values) for name,values in metrics.items()},maximum_nrmse={name:max(m['nrmse'] for m in values) for name,values in metrics.items()},maximum_abs={name:max(m['max_abs'] for m in values) for name,values in metrics.items()},
        propagated_domain_labels=len(domains),propagated_scalar_observations=observations,original_gates_recomputed=gates,
        declared_count_and_prefix_checks=True,elapsed_s=time.monotonic()-started,review_script_sha256=sha(__file__))
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
