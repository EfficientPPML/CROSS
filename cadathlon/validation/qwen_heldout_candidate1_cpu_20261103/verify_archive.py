"""Offline integrity and fixed-gate audit of the frozen held-out attempt."""
from pathlib import Path
import hashlib,json,math,tarfile
P=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def verify():
    v=json.loads((P/'verification.json').read_text())
    for n,h in v['artifact_sha256'].items():assert sha(P/n)==h,n
    r=json.loads((P/'heldout-v1/report.json').read_text());m=json.loads((P/'candidate-v1/manifest.json').read_text());inputs=json.loads((P/'validation_inputs/tokenized_fixtures.json').read_text());req=json.loads((P/'calibration_requirement.json').read_text())
    assert r['source_files_unchanged'] and r['source_hashes_before']==r['source_hashes_after']
    with tarfile.open(P/'heldout-v1/sources.tar.gz') as t:
        members=t.getmembers();assert len(members)==len({i.name for i in members})==38 and all(i.isfile() for i in members)
        actual={i.name:hashlib.sha256(t.extractfile(i).read()).hexdigest() for i in members}
        prior=json.loads(t.extractfile('calibration_prerequisite/report.json').read())
    assert actual['calibration_prerequisite/report.json']==req['full_report_sha256']
    assert actual['calibration_prerequisite/runtime_sources.tar.gz']==req['full_source_archive_sha256']
    assert prior['passed'] and prior['complete'] and prior['all_workload_entries_accounted']==416
    assert prior['source_hashes_before']==prior['source_hashes_after']==req['expected_runtime_source_hashes']
    assert r['calibration_prerequisite']['verified'] and r['calibration_prerequisite']['calibration_contexts']==416
    assert all(actual[n]==h for n,h in req['expected_runtime_source_hashes'].items())
    assert actual==r['source_hashes_before'] and sha(P/'heldout-v1/sources.tar.gz')==r['source_archive_sha256']
    for n,h in actual.items():
        if not n.startswith(('software/','calibration_prerequisite/')):assert sha(P/n)==h,n
    assert sha(P/'candidate-v1/manifest.json')==r['candidate_manifest_sha256']==v['manifest_sha256']
    assert r['verified_manifest_input_sha256']==m['source_input_sha256']
    assert r['asset_sha256_before']==r['asset_sha256_after']==json.loads((P/'model_pins.json').read_text())['asset_sha256'] and r['assets_unchanged']
    assert not r['smoke'] and r['validation_data_read'] and not r['coefficient_selection_performed'] and not r['encrypted_execution'] and r['no_jax_imported']
    assert len(r['cases'])==48 and len(r['decode'])==8 and r['all_workload_entries_accounted']==112
    assert [d['id'] for d in r['decode']]==[x['id'] for x in inputs if x['kind']=='decode']
    teacher=[];free=[]
    for d in r['decode']:
        initial=next(x['token_ids'] for x in inputs if x['id']==d['id'] and x['kind']=='decode')
        assert len(d['teacher_records'])==len(d['free_records'])==4
        for i,x in enumerate(d['teacher_records']):
            assert x['token_ids']==initial+d['native_tokens'][:i] and x['native_top1']==d['native_tokens'][i]
        n=len(d['approximate_free_tokens'])
        for i,x in enumerate(d['free_records']):
            if i<n:
                assert x['token_ids']==initial+d['approximate_free_tokens'][:i] and x['approximate_top1']==d['approximate_free_tokens'][i]
            elif i==n and n<4:
                assert x['token_ids']==initial+d['approximate_free_tokens'] and x['approximate_top1'] is None and not x['approximate_complete']
            else:
                assert x['status']=='unmeasured_suffix_after_approximate_failure' and not x['approximate_complete'] and not x['numeric_pass'] and not x['domain_pass'] and 'token_ids' not in x
        assert d['complete_sequence_agreement']==(n==4 and d['approximate_free_tokens']==d['native_tokens'])
        assert d['free_suffix_stopped']==(n<4)
        teacher+=d['teacher_records'];free+=d['free_records']
    for x in r['cases']:
        ident=x['id'];S=x['sequence_length']
        assert x['token_ids']==next(q['token_ids'] for q in inputs if q['id']==ident and q['sequence_length']==S)
    for S in (8,16,32):
        assert {x['id'] for x in r['cases'] if x['sequence_length']==S}=={x['id'] for x in inputs if x['kind']=='prefill' and x['sequence_length']==S}
    allrows=r['cases']+teacher+free;numeric=True;complete=True
    for x in allrows:
        if 'token_ids' not in x:numeric=False;complete=False;continue
        b=b''.join(int(z).to_bytes(8,'little',signed=True) for z in x['token_ids']);assert hashlib.sha256(b).hexdigest()==x['context_sha256']
        assert x['sequence_length']==len(x['token_ids'])
        for met in list(x['operator_metrics'].values())+x['block_metrics']:
            expected=met['finite'] and met['shape_matches'] and met['nrmse']<=.05
            assert met['passed']==expected
        allops=len(x['operator_metrics'])==421 and len(x['block_metrics'])==28
        assert x['missing_operator_count']==421-len(x['operator_metrics']) and x['missing_block_count']==28-len(x['block_metrics'])
        lm=x['full_logit_metric'];done=lm is not None and allops
        assert x['approximate_complete']==done
        if lm is not None:
            assert x['all_final_logits_checked']==151936
            assert lm['passed']==(lm['finite'] and lm['shape_matches'] and lm['nrmse']<=.10)
        sem=x['exact_vs_native'];assert x['exact_semantic_pass']==(sem['finite'] and sem['shape_matches'] and sem['nrmse']<=1e-5 and sem['max_abs']<=1e-3 and sem['top1_match'])
        expected_numeric=done and all(q['passed'] for q in x['operator_metrics'].values()) and all(q['passed'] for q in x['block_metrics']) and lm['passed']
        assert x['numeric_pass']==expected_numeric
        numeric &= x['numeric_pass'] and x['domain_pass'] and x['exact_semantic_pass'];complete &= done
    matches=lambda rows:sum(x.get('approximate_top1') is not None and x['approximate_top1']==x.get('native_top1') for x in rows)
    expected_gates={'all_numeric_and_domains':bool(numeric),'aggregate_prefill_top1':matches(r['cases'])>=44,'teacher_top1':matches(teacher)>=29,'complete_free_sequences':sum(d['complete_sequence_agreement'] for d in r['decode'])>=6}
    if r['passed']:assert all(not d['below'] and not d['above'] and not d['nonfinite'] for d in r['propagated_domain_ranges'].values())
    assert r['gates']==expected_gates and r['passed']==all(expected_gates.values()) and r['complete']==bool(complete)
    assert r['required_prefill_top1_matches']==44 and r['required_teacher_top1_matches']==29 and r['required_complete_free_sequences']==6
    assert {S:r['per_suite'][S]['expected_contexts'] for S in ('8','16','32')}=={'8':16,'16':16,'32':16}
    for rows,summary,count in [(r['cases'],r['prefill_summary'],48),(teacher,r['teacher_summary'],32),(free,r['free_summary'],32)]:
        assert summary['top1_matches']==matches(rows) and summary['top1_denominator']==count and summary['records']==count
    assert len(r['hook_invariance_checks'])==4 and all(x['all_logits_bitwise_invariant'] for x in r['hook_invariance_checks'])
    print(json.dumps({'integrity_verified':True,'numerical_passed':r['passed'],'workload_entries':112,'runtime_sources':len(actual),'held_out_after_calibration':True,'encrypted_execution':False}))
if __name__=='__main__':verify()
