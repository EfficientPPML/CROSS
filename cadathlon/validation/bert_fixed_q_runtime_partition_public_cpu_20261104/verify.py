"""Offline exact-source and public shape-record verifier; no numerical imports."""
from pathlib import Path
import argparse,hashlib,json,math,tarfile,xml.etree.ElementTree as ET

def sha(data):return hashlib.sha256(data).hexdigest()
def load(path):return json.loads(path.read_bytes())
def archive(path):
 with tarfile.open(path,'r:gz')as t:
  rows=t.getmembers();assert all(m.isfile()and not m.name.startswith('/')and '..'not in Path(m.name).parts for m in rows)
  result={m.name:t.extractfile(m).read()for m in rows};assert len(result)==len(rows);return result
def verify(root,current=None):
 root=Path(root);expected=load(root/'artifact_sha256.json')
 assert {str(p.relative_to(root)):sha(p.read_bytes())for p in root.rglob('*')if p.is_file()and p.name!='artifact_sha256.json'}==expected
 r=load(root/'measured/report.json');w=load(root/'proof/report.json')
 assert r['passed']is True and w['passed']is True
 for directory,report,count in [('measured',r,43),('proof',w,44)]:
  assert report['source_files_unchanged']is True and report['source_archive_verified']is True
  content=archive(root/directory/'sources.tar.gz');hashes={n:sha(b)for n,b in content.items()}
  assert len(content)==count and hashes==report['source_hashes_before']==report['source_hashes_after']
  assert sha((root/directory/'sources.tar.gz').read_bytes())==report['source_archive_sha256']
  if directory=='measured':source=content
  else:
   assert all(content[n]==v for n,v in source.items())
   assert sha(content['external/check_fixed_q_runtime_partition_20261104.py'])==sha((root/'driver.py').read_bytes())
 assert w['audit_report_sha256']==sha((root/'measured/report.json').read_bytes())
 assert [x['name']for x in w['phases']]==['tests','public-audit']
 assert all(x['returncode']==0 and x['waited']is True for x in w['phases'])
 assert r['forbidden_calls']==[]and r['guards_restored']is True and r['denied_entrypoints']==32 and r['CPU_backend_initialized']is True
 origin=json.loads(source['source_origins.json']);assert len(origin)==36
 assert all(sha(source[n])==v['sha256']for n,v in origin.items())
 for record in r['actual_methods'].values():assert sha(source['core/'+record['source'].split('/core/',1)[1]])==record['sha256']
 original=json.loads(source['inputs/profile.json'])['profile'];q=original['q'];prior=json.loads(source['inputs/public_profile_audit.json'])
 assert len(q)==100 and sha(source['inputs/public_profile_audit.json'])=='5224f8922abba50ef28bcdb6c844268f2dbda123cc6ccc49487a79d09012c7eb'
 assert [(x['dnum'],x['aux_bits'])for x in r['profiles']]==[(8,30),(6,30),(6,31)]
 negative_count=0
 for profile,estimated in zip(r['profiles'],(15,19,18)):
  d=profile['dnum'];p=profile['P'];old=next(x for x in prior['profiles']if (x['dnum'],x['aux_bits'])==(d,profile['aux_bits']))
  assert p==old['P'];registration=profile['registration'];fields=registration['fields']
  assert fields['q_towers']==q and fields['p_towers']==p and fields['register_word_size']==profile['aux_bits']+1
  assert registration['accepted']is False and registration['actual_public_security']is True
  assert registration['estimated_logical_P_count']==estimated
  assert registration['rejection']=={'type':'ValueError','message':f'OpenFHE HYBRID profile requires exactly {estimated} P towers, got {len(p)}'}
  assert len(profile['levels'])==50
  for level,record in enumerate(profile['levels']):
   n=2*(level+1);alpha=(100+d-1)//d;active=(n+alpha-1)//alpha;qp=n+len(p)
   assert record['level']==level and record['passed']is True and record['keygen_sizeQ_frozen']is True
   assert record['full_kernel_control_generated']is False and record['compiled_HE_executed']is False
   assert record['outputs']==[{'shape':[active,256,512,qp],'dtype':'uint32'}]*2+[{'shape':[131072],'dtype':'int32'}]
   dispatch=record['dispatch_only'];assert dispatch['keygen_sizeQ']==100 and dispatch['active_digits']==active
   plan=record['independent_fixed_partition_plan'];parts=[list(range(start,min(start+alpha,n)))for start in range(0,n,alpha)]
   assert plan['partition_indices']==parts and plan['P_covers_fixed_partitions']is True
   assert math.prod(p)>=max(math.prod(q[i]for i in part)for part in parts)
   assert plan['raw_key_shape']==[d,100+len(p),131072]and plan['formatted_full_key_shape']==[d,131072,100+len(p)]
   transposes=[x for x in record['equations']if x['primitive']=='transpose'];assert len(transposes)==2
   assert all(x['permutation']==[0,2,1]and x['outputs'][0]['shape']==plan['formatted_full_key_shape']for x in transposes)
   slices=[x for x in record['equations']if x['primitive']=='slice'and len(x['inputs'][0]['shape'])==3]
   assert len(slices)==4 and {(x['start_indices'][-1],x['limit_indices'][-1])for x in slices}=={(0,n),(100,100+len(p))}
   assert set(record['negative_shapes'])=={'too_few_digits','wrong_tail','unpaired'}
   assert all(v['rejected']is True for v in record['negative_shapes'].values());negative_count+=len(record['negative_shapes'])
 tree=ET.parse(root/'proof/tests.xml');suites=tree.findall('.//testsuite')
 assert sum(int(s.attrib['tests'])for s in suites)==21
 assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('failures','errors','skipped'))
 terminal=load(root/'terminal.json');assert terminal['session_id']==83053 and terminal['observed_exit_code']==0 and terminal['reaped']is True
 if current:
  assert all(sha((Path(current)/n).read_bytes())==h for n,h in r['source_hashes_before'].items())
  assert sha(Path('/tmp/check_fixed_q_runtime_partition_20261104.py').read_bytes())==w['source_hashes_before']['external/check_fixed_q_runtime_partition_20261104.py']
 return {'passed':True,'sources':43,'proof_sources':44,'tests':21,'registered_accepted':0,'profiles':3,'abstract_levels':150,'malformed_shape_rejections':negative_count,'current_checked':current is not None}
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('--current');args=parser.parse_args()
 print(json.dumps(verify(args.archive,args.current),sort_keys=True))
