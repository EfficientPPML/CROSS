from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
h=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args()
indexed=load(ROOT/'files.sha256.json');files={f.relative_to(ROOT).as_posix():f for f in ROOT.rglob('*')if f.is_file()and f.name!='files.sha256.json'and '__pycache__'not in f.parts};assert set(files)==set(indexed)and all(h(f.read_bytes())==indexed[n]for n,f in files.items())
d=ROOT/'measured';r=load(d/'report.json');w=load(d/'worker/report.json');term=load(d/'terminal_observation.json');expected=load(d/'source_manifest.json');review=load(d/'root-review/review.json');peer=load(ROOT/'peer-review/security.json');grant=load(ROOT/'authorization/grant.json');launch=load(ROOT/'authorization/launch_review.json')
REPORT='0c9f0521465f0fee5085c1c86e46cef25ada3b5216cda9098ea8d6b459d6516c';TAR='eac29c7d1d456d1c3ebc4042b59789eb54f0233b128169af7036b17612f1f5ec';MAP='99f60b4b4401ca246364127984af69df739f1fd0c4afa96e5e962558d44cde54'
assert h((d/'report.json').read_bytes())==REPORT and h((d/'sources.tar.gz').read_bytes())==TAR and h((d/'source_manifest.json').read_bytes())==MAP
with tarfile.open(d/'sources.tar.gz')as t:
 ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==33 and sum(m.size for m in ms)<=16<<20
 data={}
 for m in ms:
  n=Path(m.name);assert m.isfile()and not n.is_absolute()and '..'not in n.parts and m.size<=4<<20
  assert n.suffix in('.py','.json','.md','.txt','.xml')or m.name in('.pytest_cache/.gitignore','.pytest_cache/CACHEDIR.TAG','.pytest_cache/v/cache/nodeids')
  b=t.extractfile(m).read();assert len(b)==m.size;data[m.name]=b
assert {n:h(b)for n,b in data.items()}==expected==r['source_hashes_start']==r['source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']
assert r['worker']==w and r['worker_report_sha256']==h((d/'worker/report.json').read_bytes())and r['source_archive']['sha256']==TAR
for row in(r,w):
 assert all(row[n]is True for n in('passed','complete','cleanup_completed','signal_restoration_completed','source_files_unchanged','terminal_after_body_cleanup_and_signal_restoration_attempts','acceptance_requires_independent_exit0_reap_and_exact_report_hash'))
 assert row['source_manifest_sha256']==MAP and not row['audit_denials']and row['model_forward']is False and row['coefficient_selection']is False
assert r['actual_tokenizer_stage']is True and w['generated_decode_contexts']is False and w['external_inputs_never_removed']and w['private_outputs_retained']
assert r['child_returncode']==0 and r['child_reaped']and r['child_group_absent']and r['child_pid']==w['pid']
assert term['outer_tool_chunk']=='000207'and term['persistent_session']is None and term['outer_returncode']==term['child_returncode']==0 and all(term[n]is True for n in('outer_completed','child_reaped','collector_absent','child_group_absent'))
assert term['report_sha256']==REPORT and term['source_archive_sha256']==TAR and term==review['outer']
assert grant['scope']=='tokenizer-only'and grant['execute']is True and grant['model_forward']is False and grant['coefficient_selection']is False
assert h((ROOT/'authorization/grant.json').read_bytes())==w['grant_sha256']==review['grant_sha256']==peer['public_grant_sha256']
assert h((ROOT/'authorization/launch_review.json').read_bytes())==grant['independent_review_sha256']and grant['source_manifest_sha256']==MAP and grant['protocol_sha256']==h(data['public/protocol.json'])==w['protocol_sha256']
assert json.loads(data['execution_authorization.json'])=={'execute':False}
pr=load(ROOT/'preflight/report.json');pw=load(ROOT/'preflight/worker/report.json');assert h((ROOT/'preflight/report.json').read_bytes())==grant['preflight_report_sha256']==launch['preflight_report_sha256']=='e8f3e1ab64e2f1bc17273ee7cbac74b5c4c7cfc42fe3ee547b4112207f0fed31'
assert pr['passed']and pr['worker']==pw and pr['source_hashes_start']==pr['source_hashes_end']==expected and pr['worker_report_sha256']==h((ROOT/'preflight/worker/report.json').read_bytes())
cases=list(ET.parse(ROOT/'preflight/worker/tests.xml').iter('testcase'));assert len(cases)==91 and not any(c.findall(n)for c in cases for n in('failure','error','skipped'))and len(pw['installed_import_denials'])==6
external=json.loads(data['external_inputs.json']);assert set(external)==set(w['input_sha256'])and all(external[k]['sha256']==v for k,v in w['input_sha256'].items())
assert len(w['tokenizer_asset_sha256'])==4 and w['original_calibration_records_verified']==288
ov=w['overlap'];assert ov['passed']and not ov['known_full_context_intersections']and ov['future_decode_tokens_generated']is False and ov['row0_token0_excluded']is True
assert all(ov[n]==0 for n in('known_new_context_vs_old_decode_family_prefixes','new_decode_seed_vs_excluded_context_prefixes','new_vs_old_decode_family_prefixes'))
for name,count,unique,per_length in(('calibration',3072,2792,96),('validation',1024,1021,32)):
 rows=ov[name+'_prefills'];assert set(rows)==set(map(str,range(1,33)))and all(type(v['entries'])is int and v['entries']==per_length and type(v['unique'])is int and 0<v['unique']<=per_length for v in rows.values())
 assert sum(v['entries']for v in rows.values())==count==ov['frozen_prefill_entries'][name]==review[name+'_prefills']and sum(v['unique']for v in rows.values())==unique==review[name+'_unique_prefills']
assert ov['frozen_calibration_seed_entries']==review['calibration_decode_seeds']==16 and ov['frozen_validation_seed_entries']==review['validation_decode_seeds']==8
assert ov['calibration_workload_entries']==3072+64+64==3200 and ov['validation_workload_entries']==1024+32+32==1088
assert review['counts_by_length']==ov and [w['new_natural_token_min'],w['new_natural_token_max']]==review['new_natural_token_range']==[60,63]
assert review['private_outputs']==w['private_outputs']and all(v['mode']=='0600'and v['bytes']>0 and len(v['sha256'])==64 for v in w['private_outputs'].values())
assert all(review[n]is True for n in('passed','all_five_original_inputs_and_four_tokenizer_assets_reauthenticated','original288_mapping_and16_seeds_exact','private_directory0700_files0600','source_current_execution_start_end_archive_equal'))
assert review['report_sha256']==peer['report_sha256']==REPORT and review['source_archive_sha256']==peer['source_archive_sha256']==TAR and review['sources']==peer['source_count']==33 and peer['passed']
assert review['known_token_context_intersections']==review['append_only_decode_family_prefix_intersections']==0 and review['generated_decode_contexts']is False and review['model_forward']is False and review['coefficient_selection']is False
if a.current:assert {n:h((a.current/n).read_bytes())for n in expected}==expected
print(json.dumps(dict(passed=True,files=len(files),sources=33,calibration_prefills=3072,validation_prefills=1024,seed_counts=[16,8],private_bodies_read_by_verifier=False,current=a.current is not None,model_forward=False)))
