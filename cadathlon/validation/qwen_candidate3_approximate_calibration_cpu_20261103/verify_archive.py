"""Offline exact-source and workload-accounting verification; no model imports."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name, expected in json.loads((P/'artifact_manifest.json').read_bytes()).items():
    assert sha(P/name)==expected, name
r=json.loads((P/'report.json').read_text());s=json.loads((P/'summary.json').read_text())
assert r['source_files_unchanged'] is True and r['assets_unchanged'] is True
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==55
assert r['asset_sha256_before']==r['asset_sha256_after'] and len(r['asset_sha256_before'])==7
assert sha(P/'sources.tar.gz')==r['source_archive_sha256']==s['source_archive_sha256']
with tarfile.open(P/'sources.tar.gz') as archive:
    members=archive.getmembers();assert len(members)==55 and len({m.name for m in members})==55 and all(m.isfile() for m in members)
    source={m.name:archive.extractfile(m).read() for m in members}
assert {n:hashlib.sha256(b).hexdigest() for n,b in source.items()}==r['source_hashes_before']
assert r['candidate_manifest_sha256']=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
assert hashlib.sha256(source['candidate-v3/manifest.json']).hexdigest()==r['candidate_manifest_sha256']
req=json.loads(source['collector_requirement.json'])
for group in ('selection_artifact_sha256','calibration_input_sha256','runtime_dependency_sha256'):
    assert all(hashlib.sha256(source[n]).hexdigest()==h for n,h in req[group].items())
assert all(hashlib.sha256(source['selection/'+n]).hexdigest()==h for n,h in req['selection_input_sha256'].items())
assert len(req['selection_input_sha256'])==12 and r['selection_prerequisite']['passed'] is True
assert req['calibration_input_sha256']['calibration/protocol.json']!=req['selection_input_sha256']['protocol.json']
assert r['no_jax_imported'] and not r['validation_data_read'] and not r['coefficient_selection_performed'] and not r['encrypted_execution'] and not r['smoke']
cases=r['cases'];decodes=r['decode'];assert len(cases)==288 and len(decodes)==16
assert r['all_workload_entries_accounted']==416
rows=json.loads(source['calibration/tokenized_calibration_inputs.json']);assert len(rows)==288
assert [(x['token_ids'],x['sequence_length']) for x in cases]==[(x['token_ids'],x['sequence_length']) for x in rows]
records=list(cases)
for decoded in decodes:
    teacher=decoded['teacher_records'];free=decoded['free_records'];assert len(teacher)==len(free)==4
    initial=next(x['token_ids'] for x in rows if x['id']==decoded['id'] and x['sequence_length']==16)
    nt=[];ft=[];stopped=False
    for i,row in enumerate(teacher):
        assert row['token_ids']==initial+nt and row['group']=='decode_teacher';nt.append(row['native_top1'])
    for i,row in enumerate(free):
        if stopped:assert row['status']=='unmeasured_suffix_after_approximate_failure' and row['approximate_top1'] is None;continue
        assert row['token_ids']==initial+ft and row['group']=='decode_free'
        if row['approximate_top1'] is None:stopped=True
        else:ft.append(row['approximate_top1'])
    assert decoded['native_tokens']==nt and decoded['approximate_free_tokens']==ft
    assert decoded['complete_sequence_agreement']==(len(ft)==4 and ft==nt)
    records.extend(teacher+free)
assert len(records)==416
assert r['required_prefill_top1_matches']==260 and r['required_teacher_top1_matches']==58 and r['required_complete_free_sequences']==12
if r['passed']:
    assert r['complete'] and not r['errors'] and all(r['gates'].values())
    for row in records:
        assert row['approximate_complete'] and row['numeric_pass'] and row['domain_pass'] and row['exact_semantic_pass']
        assert len(row['operator_metrics'])==421 and len(row['block_metrics'])==28 and row['all_final_logits_checked']==151936
        assert all(x['passed'] for x in row['operator_metrics'].values()) and all(x['passed'] for x in row['block_metrics']) and row['full_logit_metric']['passed']
assert s['observed_terminal_exit_code']==(0 if r['passed'] else 1) and s['session_reaped'] is True
print(json.dumps({'integrity_passed':True,'calibration_passed':r['passed'],'records':416,'source_files':55,'model_assets':7}))
