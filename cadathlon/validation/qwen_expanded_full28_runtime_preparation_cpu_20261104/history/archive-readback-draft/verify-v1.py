"""Portable public evidence readback; never imports measured runtime or test code."""
from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def read(n):return json.loads((P/n).read_bytes())
def archive(path,expected):
 with tarfile.open(path)as t:
  rows=t.getmembers();assert len(rows)==len(expected)==len({r.name for r in rows})
  assert all(r.isfile()for r in rows)
  assert {r.name:hashlib.sha256(t.extractfile(r).read()).hexdigest()for r in rows}==expected

def verify(current=False):
 pins=read('proof/sources.json');assert len(pins)==199
 assert sha(P/'proof/sources.json')=='c9b9dfd19f6514d6c43fd39378d1ea54a188b3b9d57085985dfc4ce8ae0dfe83'
 assert sha(P/'proof/sources.tar.gz')=='99de0e64165e26b9bd51352a1cbcd982187fdb3d093643f2a47afe3ddae26caa'
 archive(P/'proof/sources.tar.gz',pins)
 assert {str(p.relative_to(P/'source')):sha(p)for p in (P/'source').rglob('*')if p.is_file()}==pins
 r=read('proof/worker/report.json');o=read('proof/owner-terminal.json')
 assert all(r[k]is True for k in ('passed','synthetic_tests_started','synthetic_component_tests_completed','actions_completed','outer_receipt_required'))
 assert all(r[k]is False for k in ('actual_execution_authorized','scientific_calibration_verified','learned_wholegraph_executed','encrypted_execution'))
 assert r['denied_attempts']==[]and r['test_exit_code']==0 and r['cpu_affinity']==[112,113,114,115]
 assert all(r[k]==pins for k in ('execution_sources_before','original_sources_before','execution_sources_after','original_sources_after'))
 assert r['source_manifest_sha256']==sha(P/'proof/sources.json')
 assert all(pins[n]==h for n,h in r['fresh_derivation'].items())and len(r['fresh_derivation'])==5
 assert r['tests']==dict(tests=180,failures=0,errors=0,skipped=0)
 cases=list(ET.parse(P/'proof/worker/tests.xml').getroot().iter('testcase'))
 assert len(cases)==180 and all(not any(c.find(k)is not None for k in ('error','failure','skipped'))for c in cases)
 assert sum(c.get('name','').startswith(('test_callback_wrapper_replacement_rejected_before_invocation','test_clean_owner_control','test_genuine_callback_primary_and_retained_payload_control'))for c in cases)==4
 assert type(o['session_id'])is int and o['exit_code']==0 and o['reaped']is True and o['recorded_group_absent']is True
 assert o['worker_report_sha256']==sha(P/'proof/worker/report.json')
 assert all(row['absent']is True for row in o['observed_processes'])
 assert 'SOFTMAX_REVIEW_SOURCE'in o['outer_environment_unset']
 for row in r['loaded_local_modules'].values():assert pins[row['path']]==row['sha256']
 plan=read('proof/worker/public_plan.json');assert sha(P/'proof/worker/public_plan.json')==r['public_plan_sha256']
 assert plan['candidate_sha256']=='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
 assert len(plan['layers'])==28 and len(plan['workloads'])==32 and plan['resources']['rotation_keys']==166
 final=plan['final_norm'];assert final['degree']==256 and final['input_level']==25 and final['output_level']==3 and final['raw_input_bound']==8192 and final['inverse_refresh_count']==0
 vocab=plan['vocabulary'];assert vocab['output_level']==2 and vocab['output_count']==len(vocab['blocks'])==38 and vocab['all_logits']==151936
 assert all(b['input_level']==3 and b['output_level']==2 for b in vocab['blocks'])
 assert plan['execution_authorized']is False and plan['scientific_calibration_verified']is False and plan['first_pass_freeze_sha256']is None
 old=read('history/startup198/sources.json');archive(P/'history/startup198/sources.tar.gz',old);assert len(old)==198
 oldr=read('history/startup198/report.json');oldo=read('history/startup198/owner-terminal.json')
 assert oldr['passed']is False and oldr['synthetic_component_tests_completed']is False and oldo['exit_code']==70 and oldo['tests_executed']==0
 failed=read('history/audit199/sources.json');archive(P/'history/audit199/sources.tar.gz',failed);assert len(failed)==199
 failedr=read('history/audit199/worker/report.json');assert failedr['passed']is False and len(failedr['denied_attempts'])==12 and failedr['tests']==dict(tests=180,failures=0,errors=0,skipped=0)
 delta=read('source/final_origin.json');assert sorted(n for n in failed if failed[n]!=pins.get(n))==sorted(delta['changed_parent_files']);assert set(pins)==set(failed)
 prior=delta['prior_closure'];assert sorted(n for n in old if old[n]!=failed.get(n))==sorted(prior['changed_parent_files']);assert sorted(set(failed)-set(old))==prior['added_files']
 cache=r['import_cache'];assert cache['dont_write_bytecode']is True and cache['empty_at_terminal']is True and len(cache['removed_absent_stdlib_zip'])==1
 smoke=read('history/cache-startup-smoke/worker/report.json');assert smoke['passed']is True and smoke['denied_attempts']==[]and smoke['tests']==dict(tests=1,failures=0,errors=0,skipped=0)
 predecessor=read('source/predecessor87.json');assert len(predecessor['sources'])==87
 assert all(pins['before87/'+n]==h for n,h in predecessor['sources'].items())
 for label,count,passed,failed in [('runtime-v1',171,101,70),('runtime-v2',79,75,4)]:
  t=read('history/'+label+'/terminal.json');assert t['exit_code']==1 and t['reaped']is True and t['tests']==count and t['passed']==passed and t['failed']==failed
 for label,expected in [('negative-v3',(4,2)),('successor-v1',(4,0))]:
  rows=list(ET.parse(P/'history/softmax-wrapper'/label/'tests.xml').getroot().iter('testcase'))
  assert len(rows)==expected[0]and sum(c.find('failure')is not None for c in rows)==expected[1]
 siteold=read('history/site-old/report.json');sitefixed=read('history/site-fixed/report.json')
 assert sum(c['passed']is False for c in siteold['cases'])==1 and sitefixed['passed']is True and len(sitefixed['cases'])==4
 for label in ('site-old','site-fixed'):
  sr=read('history/'+label+'/report.json');assert sr['source_before']==sr['source_after']
  assert all(sha(P/'history'/label/n)==h for n,h in sr['source_after'].items())
 if current:
  for root in (Path(r['original_root']),Path(r['execution_root'])):
   assert {str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()}==pins
  assert all(sha(Path(predecessor['root'])/n)==h for n,h in predecessor['sources'].items())
  assert all(sha(Path(delta['parent_root'])/n)==h for n,h in failed.items())
  assert all(sha(Path(prior['parent_root'])/n)==h for n,h in old.items())
 index=read('artifact_hashes.json');observed={str(p.relative_to(P)):sha(p)for p in P.rglob('*')if p.is_file()and p.name!='artifact_hashes.json'};assert observed==index
 print(json.dumps(dict(passed=True,source_count=199,tests=180,current_checked=current,scope='Public/synthetic preparation and recorded evidence; no model, HE or calibration claim.')))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--current',action='store_true');a=p.parse_args();verify(a.current)
