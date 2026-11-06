"""Fail-closed calibration prerequisite; imports no model or validation input."""
from pathlib import Path
import hashlib,importlib.metadata,importlib.util,json,math,tarfile

def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def require(condition,message):
    if not condition:raise ValueError('Calibration prerequisite failed: '+message)
def verify(report_path,archive_path,root,requirements):
    root=Path(root);require(requirements.get('ready') is True,'terminal calibration requirement is not ready')
    require(sha(report_path)==requirements['full_report_sha256'],'full report hash')
    require(sha(archive_path)==requirements['full_source_archive_sha256'],'full source archive hash')
    r=json.loads(Path(report_path).read_text());expected=requirements['expected_runtime_source_hashes']
    require(len(expected)==55,'expected55-source runtime closure')
    require(r.get('passed') is True and r.get('complete') is True and r.get('smoke') is False,'successful complete full calibration')
    require(r.get('source_files_unchanged') is True and r['source_hashes_before']==r['source_hashes_after']==expected,'source before/after identity')
    require(r['source_archive_sha256']==requirements['full_source_archive_sha256'],'reported source archive hash')
    with tarfile.open(archive_path) as t:
        members=t.getmembers()
        require(len(members)==len(expected),'source archive member count')
        require(len({i.name for i in members})==len(members),'source archive duplicate members')
        require(all(i.isfile() for i in members),'source archive contains non-regular entries')
        archived={i.name:hashlib.sha256(t.extractfile(i).read()).hexdigest() for i in members}
    require(archived==expected,'exact source archive members')
    package=Path(importlib.util.find_spec('transformers').origin).parent
    for name,h in expected.items():
        path=package/name.removeprefix('software/transformers/') if name.startswith('software/transformers/') else root/name
        require(sha(path)==h,'current runtime source '+name)
    require(r['versions']==requirements['expected_versions'],'reported software versions')
    require({n:importlib.metadata.version(n) for n in r['versions']}==r['versions'],'current software versions')
    require(sha(root/'candidate-v3/manifest.json')==requirements['candidate_manifest_sha256']==r['candidate_manifest_sha256'],'candidate identity')
    m=json.loads((root/'candidate-v3/manifest.json').read_text())
    require(r['verified_manifest_input_sha256']=={'selection/'+n:h for n,h in m['source_input_sha256'].items()},'all twelve frozen selection inputs verified')
    for name,h in m['source_input_sha256'].items():require(sha(root/'selection'/name)==h,'candidate input '+name)
    pins=json.loads((root/'calibration/model_pins.json').read_text())
    require(r['assets_unchanged'] and r['asset_sha256_before']==r['asset_sha256_after']==pins['asset_sha256'],'calibration asset identity')
    require(r.get('validation_data_read') is False and r.get('coefficient_selection_performed') is False and r.get('encrypted_execution') is False and r.get('no_jax_imported') is True,'calibration scope')
    require(len(r['cases'])==288 and len(r['decode'])==16 and r['all_workload_entries_accounted']==416,'complete calibration counts')
    inputs=json.loads((root/'calibration/tokenized_calibration_inputs.json').read_text());corpus=json.loads((root/'calibration/corpus.json').read_text())
    for S in (8,16,32):
        group=[x for x in r['cases'] if x['sequence_length']==S]
        require(len(group)==96 and len({x['id'] for x in group})==96,'unique calibration bucket count')
        for row in group:
            ident=row['id'].rsplit('_S',1)[0];expected_tokens=next((x['token_ids'] for x in inputs if x['id']==ident and x['sequence_length']==S),None)
            require(row['token_ids']==expected_tokens,'calibration input identity')
    require([d['id'] for d in r['decode']]==corpus['decode_prompt_ids'],'decode seed inventory')
    teacher=[];free=[]
    for d in r['decode']:
        initial=next(x['token_ids'] for x in inputs if x['id']==d['id'] and x['sequence_length']==16)
        require(len(d['teacher_records'])==len(d['free_records'])==len(d['native_tokens'])==len(d['approximate_free_tokens'])==4,'complete decode history')
        require(not d['free_suffix_stopped'],'no unmeasured suffix in successful calibration')
        for kind,tokens,key in [('teacher',d['native_tokens'],'native_top1'),('free',d['approximate_free_tokens'],'approximate_top1')]:
            for i,row in enumerate(d[kind+'_records']):require(row['token_ids']==initial+tokens[:i] and row[key]==tokens[i],'same-prefix '+kind+' history')
        require(d['complete_sequence_agreement']==(d['native_tokens']==d['approximate_free_tokens']),'free agreement calculation')
        teacher+=d['teacher_records'];free+=d['free_records']
    for row in r['cases']+teacher+free:
        require(all(row.get(k) is True for k in ('approximate_complete','numeric_pass','domain_pass','exact_semantic_pass')) and row.get('failure') is None,'complete numerical/domain context')
        require(len(row['operator_metrics'])==421 and len(row['block_metrics'])==28 and row['all_final_logits_checked']==151936,'complete operator/block/logit comparisons')
        for metric in list(row['operator_metrics'].values())+row['block_metrics']:
            require(metric['passed'] and metric['finite'] and metric['shape_matches'] and 0<=metric['nrmse']<=.05,'operator/block gate')
        metric=row['full_logit_metric'];require(metric['passed'] and metric['finite'] and metric['shape_matches'] and 0<=metric['nrmse']<=.10,'full-logit gate')
        metric=row['exact_vs_native'];require(metric['passed'] and metric['top1_match'] and 0<=metric['nrmse']<=1e-5 and 0<=metric['max_abs']<=1e-3,'exact-native semantic gate')
    require(all(not x['below'] and not x['above'] and not x['nonfinite'] for x in r['propagated_domain_ranges'].values()),'all propagated calibration domains')
    matches=lambda rows:sum(x['approximate_top1']==x['native_top1'] for x in rows)
    require(matches(r['cases'])>=260 and matches(teacher)>=58 and sum(d['complete_sequence_agreement'] for d in r['decode'])>=12,'published aggregate/token gates')
    require(all(r['gates'].values()) and r['required_prefill_top1_matches']==260 and r['required_teacher_top1_matches']==58 and r['required_complete_free_sequences']==12,'declared gates unchanged')
    return {'verified':True,'report_sha256':requirements['full_report_sha256'],'source_archive_sha256':requirements['full_source_archive_sha256'],'runtime_sources':55,'calibration_contexts':416,'manifest_sha256':requirements['candidate_manifest_sha256']}
