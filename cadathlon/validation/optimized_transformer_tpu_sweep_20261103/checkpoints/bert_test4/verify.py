"""Portable failed-case/source verification only; no backend imports or execution."""
from pathlib import Path
import hashlib,json,math,tarfile
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def verify():
 for name,h in read(P/'artifact_hashes.json').items():assert sha(P/name)==h,name
 summary=read(P/'summary.json');manifest=read(ROOT/'snapshot_manifest.json');expected=manifest['source_hashes']
 assert len(expected)==199 and expected==summary['source_file_sha256']
 assert sha(ROOT/'sources.tar.gz')==summary['source_archive_sha256']=='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
 with tarfile.open(ROOT/'sources.tar.gz')as t:
  ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==199 and all(m.isfile() for m in ms)
  body={m.name:t.extractfile(m).read() for m in ms}
 assert {n:hashlib.sha256(b).hexdigest() for n,b in body.items()}==expected
 for name,h in summary['failed_case_files_before'].items():assert sha(P/name)==h==summary['failed_case_files_after'][name]
 process=read(P/'process.json');numeric=read(P/'stdout.json');security=read(P/'security_device.json');parent=read(P/'parent_report_at_checkpoint.json')
 assert process['passed']is False and process['returncode']==1 and process['result']==numeric and numeric['passed']is False
 assert numeric['error_type']=='ApproximationErrorTooHigh' and 'stddev=inf' in numeric['error'] and '3.125e-02' in numeric['error']
 assert not any(n in numeric for n in ('max_abs_error','nrmse','top1_agreement','server_s','setup_s','end_to_end_s','total_s'))
 assert all(process[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed','source_files_unchanged'))
 cases=[c for model in parent['models'].values() for c in model['cases'].values()]
 assert len(cases)==8 and sum(c['passed']is True for c in cases)==7 and sum(c['passed']is False for c in cases)==1
 assert parent['models']['bert']['cases']['test4.in']==process and parent['passed']is False
 assert security['passed']is False and security['error']=='SystemExit: 1'
 assert [e['boundary'] for e in security['events']]==['input','output']
 for e in security['events']:
  assert len(e['ciphertexts'])==1 and len(e['actual_rings'])==2
  for ring in e['actual_rings']:
   assert ring['degree']==131072 and len(ring['q_towers'])==100 and len(ring['p_towers'])==24
   assert ring['security_bits']==128 and ring['security_model']=='classical'
   assert abs(sum(math.log2(q) for q in ring['q_towers']+ring['p_towers'])-ring['log2_QP'])<1e-8 and ring['log2_QP']<=3523
  assert all(d['platform']=='tpu' for d in e['ciphertexts'][0]['devices'])
 output=security['events'][1]['ciphertexts'][0]
 assert output['moduli_count']==24 and output['nsd']==1 and output['scale']==6.8994011570896904e16
 stderr=(P/'stderr.log').read_text();assert stderr.count('bounded homomorphic refresh starting')==5 and stderr.count('bounded homomorphic refresh complete')==5
 assert 'packed BERT layer 1: output ready' in stderr and 'overflow encountered in square' in stderr
 resources=read(P/'resources.json');assert resources==process['resources']==summary['resources']
 samples=[json.loads(line) for line in (P/'resource_samples.jsonl').read_text().splitlines()]
 assert len(samples)==resources['samples']==13251 and max(r['aggregate_rss_bytes'] for r in samples)==resources['sampled_peak_process_tree_rss_bytes']==135146196992
 case=json.loads(body['cadathlon/agent_pset/p2_bert/test/test4.in']);c=case['config']
 assert (c['hidden_size'],c['intermediate_size'],c['num_attention_heads'],c['num_hidden_layers'])==(24,48,4,2)
 assert len(case['input_ids'])==2 and all(len(x)==8 for x in case['input_ids'])
 assert summary['case_passed']is False and summary['overall_sweep_passed']is False and summary['case_pid_and_group_absent']is True
 print(json.dumps(dict(archive_verified=True,source_count=199,completed_cases=8,prior_passes=7,failed_case='bert/test4',case_passed=False,full_model_claim=False)))
if __name__=='__main__':verify()
