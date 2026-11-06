"""Portable public metadata/source proof verifier; no tensor, token, backend reads."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((P/n).read_bytes())
def members(folder,report):
 path=P/folder/'sources.tar.gz';assert sha(path)==report['source_archive_sha256']
 with tarfile.open(path)as t:
  ms=t.getmembers();assert len(ms)==25 and len({m.name for m in ms})==25 and all(m.isfile() and m.size<16<<20 for m in ms)
  data={m.name:t.extractfile(m).read() for m in ms}
 assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==report['source_hashes_before']==report['source_hashes_after']==report['execution_source_hashes_before']==report['execution_source_hashes_after']
 return data

def verify():
 for name,pin in read('artifact_hashes.json').items():assert sha(P/name)==pin,name
 reports=[read(f'{n}/report.json') for n in ('preflight','actual')]
 for r in reports:
  assert all(r[n]is True for n in ('passed','complete','cleanup_completed','source_files_unchanged','execution_files_unchanged','terminal_scope_completed','termination_scope_completed'))
  assert len(r['children'])==1
  assert all(r['children'][0][n]is True for n in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
  assert r['children'][0]['direct_returncode']==0
 pre,actual=reports;source=members('actual',actual);assert source==members('preflight',pre)
 assert pre['tests']==dict(tests=59,errors=0,failures=0,skipped=0) and pre['zero_audited_body_reads']is True
 xml=ET.parse(P/'preflight/tests.xml').getroot().findall('testsuite');assert sum(int(s.get('tests')) for s in xml)==59
 assert all(sum(int(s.get(k,0)) for s in xml)==0 for k in ('errors','failures','skipped'))
 audit=read('preflight/audit.json');assert audit==dict(denied_attempts=[],backend_modules=[],passed=True,scope='ordinary-Python audit; no OS isolation claim')
 copied=json.loads(source['copied_sources.json'])
 assert all(hashlib.sha256(source[n]).hexdigest()==h for n,h in copied['sources'].items())
 assert hashlib.sha256(source['inputs/candidate_manifest.json']).hexdigest()=='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
 w=read('actual/worker/report.json');assert sha(P/'actual/worker/report.json')==actual['worker']['report_sha256']
 assert w['source_hashes_before']==w['source_hashes_after']==actual['source_hashes_before']
 assert all(w[n]is True for n in ('passed','complete','cleanup_completed','source_files_unchanged','assets_unchanged','provenance_unchanged','terminal_scope_completed','termination_scope_completed','historical_boundary_reproduced_exactly'))
 assert all(w[n]is False for n in ('HE_executed','native_framework_executed','final_normalization_evaluated','vocabulary_evaluated','candidate_changed','original_S1_gate_passed'))
 assert w['asset_sha256_before']==w['asset_sha256_after'] and not w['backend_modules']
 m=json.loads(source['public_metadata.json']);a=m['original_union'];assert a['native_count']==a['exact_count']==416
 native=m['original_native_range'];exact=m['original_exact_range']
 assert a['minimum']==min(native['minimum'],exact['minimum']) and a['maximum']==max(native['maximum'],exact['maximum'])
 domain=[.8*a['minimum'],1.2*a['maximum']];assert domain==m['candidate2_final_rule']['domain']
 enlarged=[max(1e-6,domain[0]/4),domain[1]*4];assert enlarged==m['candidate3_final_rule']['domain']==[29.88948209636508,2923.83603515625]
 assert m['calibration_workload']['buckets']==[8,16,32]
 assert m['calibration_workload']['decode_seed_length']==16 and m['calibration_workload']['decode_steps']==4
 assert w['embedding_sha256']==m['historical_S1']['embedding_sha256'] and w['fixture']==m['historical_S1']['fixture']
 paired=w['paired'];assert len(paired['blocks'])==28 and len(paired['variance_sites'])==113
 assert paired['approximate_final']==m['historical_S1']['failure_detail']['record']
 ap=paired['approximate_final']['maximum'];ex=paired['exact_function_final']['maximum']
 assert ap==3391.698360993511 and ex==3391.7152988079342 and ex>ap>enlarged[1]
 assert paired['final_variance_difference']==ap-ex and paired['final_variance_relative_difference']==(ap-ex)/ex
 assert paired['approximate_prefix_domain_records']==307 and paired['exact_function_domain_diagnostics']==140
 assert paired['approximate_prior_domains_passed']is True and paired['exact_function_domain_diagnostic_failures']=={}
 for i,row in enumerate(paired['blocks']):
  assert row['layer']==i and all(type(row[k])is float and 0<=row[k]<float('inf') for k in ('max_abs_error','rms_error','exact_rms'))
 assert read('terminal_readback.json')['authoritative_reap']==dict(session=38855,exit_code=0,reaped=True)
 print(json.dumps(dict(passed=True,sources=25,tests=59,paired_blocks=28,variance_sites=113,original_S1_gate_passed=False,scope='public proof readback only')))
if __name__=='__main__':verify()
