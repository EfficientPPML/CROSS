"""Offline integrity and honest112-context accounting; no model import."""
from pathlib import Path
import hashlib,json,tarfile
P=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
r=json.loads((P/'report.json').read_text());s=json.loads((P/'summary.json').read_text())
assert r['source_files_unchanged'] and r['assets_unchanged'] and not r['errors']
assert r['source_hashes_before']==r['source_hashes_after'] and len(r['source_hashes_before'])==63
assert r['asset_sha256_before']==r['asset_sha256_after'] and len(r['asset_sha256_before'])==7
assert sha((P/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==s['source_archive_sha256']
with tarfile.open(P/'sources.tar.gz') as archive:
 members=archive.getmembers();assert len(members)==63==len({m.name for m in members}) and all(m.isfile() for m in members)
 data={m.name:archive.extractfile(m).read() for m in members}
assert {n:sha(b) for n,b in data.items()}==r['source_hashes_before']
req=json.loads(data['validation_requirement.json']);proto=json.loads(data['validation_protocol.json'])
assert r['candidate_manifest_sha256']==req['candidate_manifest_sha256']=='0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489'
assert sha(data['candidate-v2/manifest.json'])==r['candidate_manifest_sha256']
assert all(sha(data[n])==h for n,h in req['expected_runtime_source_hashes'].items()) and len(req['expected_runtime_source_hashes'])==53
assert all(sha(data[n])==h for n,h in req['validation_code_sha256'].items())
assert r['calibration_prerequisite']['verified'] and r['calibration_prerequisite']['calibration_contexts']==416
assert sha(data['calibration_prerequisite/report.json'])==req['full_report_sha256']=='8bcc7b6e7275914b85245e4ee65481ca87b37fed565dda8bac59fe97563b6641'
assert sha(data['calibration_prerequisite/runtime_sources.tar.gz'])==req['full_source_archive_sha256']=='42cf71780d0f8d84f29791f8e18ed13f133589bb5c66fed9a45790b4d3d1578b'
assert sha(data['validation_inputs/tokenized_fixtures.json'])==r['validation_inputs_sha256']==req['validation_input_sha256']==proto['input_sha256']
assert sha(data['validation_protocol.json'])==req['validation_protocol_sha256']
assert r['workload']==proto['workload']==s['workload']
inputs=json.loads(data['validation_inputs/tokenized_fixtures.json']);pref=[x for x in inputs if x['kind']=='prefill'];seeds=[x for x in inputs if x['kind']=='decode']
assert len(inputs)==56 and len(pref)==48 and len(seeds)==8 and len(r['cases'])==48 and len(r['decode'])==8
assert [(x['id'],x['token_ids']) for x in r['cases']]==[(x['id'],x['token_ids']) for x in pref]
assert [x['id'] for x in r['decode']]==[x['id'] for x in seeds]
records=list(r['cases'])
for decoded,seed in zip(r['decode'],seeds):
 teacher=decoded['teacher_records'];free=decoded['free_records'];assert len(teacher)==len(free)==4
 nt=[];ft=[];stopped=False
 for row in teacher:
  assert row['token_ids']==seed['token_ids']+nt;nt.append(row['native_top1'])
 for row in free:
  if stopped:assert row['status']=='unmeasured_suffix_after_approximate_failure' and row['approximate_top1'] is None;continue
  assert row['token_ids']==seed['token_ids']+ft
  if row['approximate_top1'] is None:stopped=True
  else:ft.append(row['approximate_top1'])
 assert decoded['native_tokens']==nt and decoded['approximate_free_tokens']==ft
 assert decoded['complete_sequence_agreement']==(len(ft)==4 and ft==nt)
 records.extend(teacher+free)
assert len(records)==r['all_workload_entries_accounted']==112
complete=[x for x in records if x.get('approximate_complete')];stops=[x for x in records if x.get('failure')];suffix=[x for x in records if x.get('status')=='unmeasured_suffix_after_approximate_failure']
assert len(complete)==s['complete_contexts'] and len(stops)==s['domain_or_nonfinite_stops'] and len(suffix)==s['unmeasured_suffixes']
for row in complete:
 assert len(row['operator_metrics'])==421 and len(row['block_metrics'])==28 and row['all_final_logits_checked']==151936
assert r['required_prefill_top1_matches']==44 and r['required_teacher_top1_matches']==29 and r['required_complete_free_sequences']==6
if r['passed']:
 assert len(complete)==112 and r['complete'] and all(r['gates'].values())
 assert all(x['numeric_pass'] and x['domain_pass'] and x['exact_semantic_pass'] for x in complete)
assert r['validation_data_read'] and not r['coefficient_selection_performed'] and not r['encrypted_execution'] and r['no_jax_imported']
assert s['session_reaped'] and s['observed_terminal_exit_code']==(0 if r['passed'] else 1)
print(json.dumps({'integrity_passed':True,'validation_passed':r['passed'],'workload':r['workload'],'complete':len(complete),'domain_stops':len(stops),'unmeasured_suffixes':len(suffix),'source_files':63}))
