"""Offline frozen112-context source/accounting/gate verification."""
from pathlib import Path
import hashlib,io,json,math,struct,tarfile
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
for n,h in json.loads((P/'artifact_manifest.json').read_bytes()).items():assert sha((P/n).read_bytes())==h,n
r=json.loads((P/'report.json').read_bytes());s=json.loads((P/'summary.json').read_bytes())
assert sha((P/'report.json').read_bytes())==s['report_sha256']
assert r['source_files_unchanged'] is True and r['assets_unchanged'] is True
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==s['source_count']
assert len(r['asset_sha256_before'])==7 and r['asset_sha256_before']==r['asset_sha256_after']
assert sha((P/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==s['source_archive_sha256']
with tarfile.open(P/'sources.tar.gz') as t:
    members=t.getmembers();assert len(members)==len({m.name for m in members})==s['source_count'] and all(m.isfile() for m in members)
    source={m.name:t.extractfile(m).read() for m in members}
assert {n:sha(b) for n,b in source.items()}==r['source_hashes_before']
req=json.loads(source['validation_requirement.json'])
assert len(req['expected_runtime_source_hashes'])==55
assert all(sha(source[n])==h for n,h in req['expected_runtime_source_hashes'].items())
assert all(sha(source[n])==h for n,h in req['validation_code_sha256'].items())
assert sha(source['candidate-v3/manifest.json'])==r['candidate_manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
assert sha(source['calibration_prerequisite/report.json'])==req['full_report_sha256']=='1e827000cc0368853f2e8932ec17ee5f318e973e3c80f67652790491fcf7c628'
assert sha(source['calibration_prerequisite/runtime_sources.tar.gz'])==req['full_source_archive_sha256']
assert sha(source['validation_inputs/tokenized_fixtures.json'])==req['validation_input_sha256']==r['validation_inputs_sha256']
assert r['calibration_prerequisite']['verified'] is True and r['selection_prerequisite']['passed'] is True
assert r['validation_data_read'] is True and r['validation_input_read_attempted'] is True and r['validation_input_read_status']=='confirmed'
assert not r['coefficient_selection_performed'] and not r['encrypted_execution'] and r['no_jax_imported']
if r['workload']=='published':
    assert s['source_count']==71 and req['validation_input_sha256']=='5caad37526dfda5a379f029ddfc75fb10879b3313d3533ec4cbf60d4192ef359'
if r['workload']=='fresh':
    assert s['source_count']==76 and req['validation_input_sha256']=='b20f28fc66260d8bd0ec704d6138da9a60ba513a53111d789cfd10f431dd80af'
    freeze=json.loads(source['fresh_freeze/report.json'])
    assert sha(source['fresh_freeze/report.json'])==req['fresh_freeze']['report_sha256']
    assert sha(source['fresh_freeze/sources.tar.gz'])==req['fresh_freeze']['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(source['fresh_freeze/sources.tar.gz'])) as t:
        ms=t.getmembers();assert len(ms)==11==len({m.name for m in ms}) and all(m.isfile() for m in ms)
        assert {m.name:sha(t.extractfile(m).read()) for m in ms}==freeze['source_hashes_before']==freeze['source_hashes_after']
inputs=json.loads(source['validation_inputs/tokenized_fixtures.json'])
prefill=[x for x in inputs if x['kind']=='prefill'];seeds=[x for x in inputs if x['kind']=='decode']
assert len(inputs)==56 and len(prefill)==48 and len(seeds)==8
assert [(x['id'],x['token_ids']) for x in r['cases']]==[(x['id'],x['token_ids']) for x in prefill]
assert [d['id'] for d in r['decode']]==[x['id'] for x in seeds]
records=list(r['cases']);teacher=[];free=[]
for d,seed in zip(r['decode'],seeds):
    assert len(d['teacher_records'])==len(d['free_records'])==4
    nt=[];ft=[];stopped=False
    for row in d['teacher_records']:
        assert row['token_ids']==seed['token_ids']+nt;nt.append(row['native_top1'])
    for row in d['free_records']:
        if stopped:
            assert row['status']=='unmeasured_suffix_after_approximate_failure' and row['approximate_top1'] is None
            continue
        assert row['token_ids']==seed['token_ids']+ft
        if row['approximate_top1'] is None:stopped=True
        else:ft.append(row['approximate_top1'])
    assert d['native_tokens']==nt and d['approximate_free_tokens']==ft
    assert d['complete_sequence_agreement']==(len(ft)==4 and ft==nt)
    teacher+=d['teacher_records'];free+=d['free_records']
records+=teacher+free
assert len(records)==r['all_workload_entries_accounted']==112
assert (r['required_prefill_top1_matches'],r['required_teacher_top1_matches'],r['required_complete_free_sequences'])==(44,29,6)
if r['passed']:
    assert r['complete'] and not r['errors'] and all(r['gates'].values())
    assert sum(x['native_top1']==x['approximate_top1'] for x in r['cases'])>=44
    assert sum(x['native_top1']==x['approximate_top1'] for x in teacher)>=29
    assert sum(d['complete_sequence_agreement'] for d in r['decode'])>=6
    for row in records:
        assert all(row[k] is True for k in ('approximate_complete','numeric_pass','domain_pass','exact_semantic_pass'))
        assert row['failure'] is None and len(row['operator_metrics'])==421 and len(row['block_metrics'])==28 and row['all_final_logits_checked']==151936
        assert row['context_sha256']==sha(struct.pack('<'+'q'*len(row['token_ids']),*row['token_ids']))
        for m in [*row['operator_metrics'].values(),*row['block_metrics']]:
            assert m['passed'] and m['finite'] and m['shape_matches'] and 0<=m['nrmse']<=.05
        m=row['full_logit_metric'];assert m['passed'] and m['finite'] and m['shape_matches'] and 0<=m['nrmse']<=.10
        m=row['exact_vs_native'];assert m['passed'] and m['top1_match'] and 0<=m['nrmse']<=1e-5 and 0<=m['max_abs']<=1e-3
    assert all(not x['below'] and not x['above'] and not x['nonfinite'] for x in r['propagated_domain_ranges'].values())
assert s['observed_terminal_exit_code']==(0 if r['passed'] else 1) and s['session_reaped'] is True and s['pid_absent'] is True
print(json.dumps(dict(integrity_passed=True,validation_passed=r['passed'],contexts=112,sources=s['source_count'],manifest=r['candidate_manifest_sha256'])))
