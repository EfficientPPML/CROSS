"""Portable exact-source/public arithmetic verifier; no JAX or HE imports."""
import argparse,hashlib,json,math,tarfile,xml.etree.ElementTree as ET
from pathlib import Path

def digest(data):return hashlib.sha256(data).hexdigest()
def read(path):return json.loads(path.read_bytes())
def members(path):
 with tarfile.open(path,'r:gz')as tar:
  entries=tar.getmembers();assert all(m.isfile()and not m.name.startswith('/')and '..'not in Path(m.name).parts for m in entries)
  result={m.name:tar.extractfile(m).read()for m in entries}
  assert len(result)==len(entries)
  return result
def verify(root,current=None):
 root=Path(root);manifest=read(root/'artifact_sha256.json')
 actual={str(p.relative_to(root)):digest(p.read_bytes())for p in root.rglob('*')if p.is_file()and p.name!='artifact_sha256.json'}
 assert actual==manifest,'Archive artifact inventory differs'
 r=read(root/'measured/report.json');w=read(root/'proof/report.json')
 assert r['passed']is True and w['passed']is True
 assert len(r['source_hashes_before'])==39 and len(w['source_hashes_before'])==40
 for base,report in [('measured',r),('proof',w)]:
  assert report['source_files_unchanged']is True and report['source_archive_verified']is True
  body=members(root/base/'sources.tar.gz')
  assert digest((root/base/'sources.tar.gz').read_bytes())==report['source_archive_sha256']
  hashes={n:digest(b)for n,b in body.items()}
  assert hashes==report['source_hashes_before']==report['source_hashes_after']
  if base=='measured':source=body
  else:
   assert all(body[n]==b for n,b in source.items())
   assert digest(body['external/check_fixed_q_dnum_public_20261103.py'])==digest((root/'driver.py').read_bytes())
 assert w['audit_report_sha256']==digest((root/'measured/report.json').read_bytes())
 assert [p['name']for p in w['phases']]==['tests','public-audit']
 assert all(p['returncode']==0 and p['waited']is True for p in w['phases'])
 assert r['forbidden_calls']==[]and r['guards_restored']is True and r['denied_entrypoint_count']==28
 assert r['cpu_backend_initialized_after_import']is True
 origins=json.loads(source['source_origins.json'])
 assert len(origins)==36 and all(digest(source[n])==v['sha256']for n,v in origins.items())
 for item in r['loaded_local_modules'].values():assert digest(source[item['path']])==item['sha256']
 for item in r['actual_entrypoint_sources'].values():
  name='core/'+item['path'].split('/core/',1)[1];assert digest(source[name])==item['sha256']
 profile=json.loads(source['inputs/profile.json'])['profile'];q=profile['q'];N=profile['degree']
 assert q==r['baseline']['q']and len(q)==100 and N==131072
 assert r['baseline']['default_generated_P_matches']is True and r['baseline']['default_aux_bits']==30
 inventory=json.loads(source['inputs/bert118_inventory.json']);assert inventory['passed']is True and len(set(inventory['required_keys']))==118
 assert len(r['profiles'])==12
 assert [(x['dnum'],x['aux_bits'])for x in r['profiles']]==[(d,a)for d in (5,6,7,8)for a in (29,30,31)]
 secure=0
 for row in r['profiles']:
  d=row['dnum'];p=row['P'];assert row['p_count']==len(p)
  assert all(1<v<2**31 and v%(2*N)==1 for v in (*q,*p))and len(set((*q,*p)))==len(q)+len(p)
  assert len(row['prefixes'])==100 and row['cd2_levels_checked']==50
  for n,record in enumerate(row['prefixes'],1):
   alpha=(n+d-1)//d;maximum=max(math.prod(q[i:min(i+alpha,n)])for i in range(0,n,alpha));product=math.prod(q[:n])*math.prod(p)
   assert record['q_count']==n and record['cd2_level']==(n//2-1 if n%2==0 else None)
   assert record['QP_bitlength']==product.bit_length()and record['security_ceiling']==3523
   assert record['secure_128_classical']==(product.bit_length()<=3523)
   assert record['kernel_valid']is True and record['P_coverage']is True and record['NTT_congruence']is True
   assert math.prod(p)>=maximum and record['max_partition_bitlength']==maximum.bit_length()
   assert record['P_product_bitlength']==math.prod(p).bit_length()
  assert row['full']==row['prefixes'][-1]
  assert row['public_valid_full_profile']==row['full']['secure_128_classical']
  secure+=row['public_valid_full_profile']
  for label,count in [('bert118',118),('qwen166',166)]:
   resource=row['resources'][label];pair=2*d*N*(100+len(p))*4
   assert resource['rotation_count']==count and resource['rotation_pair_raw_bytes']==pair
   assert resource['rotation_keys_raw_bytes']==count*pair and resource['rotation_keys_raw_GiB']==count*pair/2**30
   assert resource['public_plus_relinearization_plus_rotation_raw_bytes']==(count+1)*pair+2*N*100*4
   assert resource['one_ciphertext_raw_bytes']==104857600
  for level in (0,25,49):
   v=row['resource_at_cd2_levels'][str(level)];pair=2*d*N*(2*(level+1)+len(p))*4
   assert v['rotation_pair_raw_bytes']==pair and v['one_ciphertext_raw_bytes']==2*N*2*(level+1)*4
 assert secure==9
 currentrow=next(x for x in r['profiles']if (x['dnum'],x['aux_bits'])==(8,30))
 assert currentrow['P']==profile['p']==r['baseline']['P']
 tree=ET.parse(root/'proof/tests.xml');suites=tree.findall('.//testsuite')
 assert sum(int(s.attrib['tests'])for s in suites)==8
 assert all(int(s.attrib.get(k,0))==0 for s in suites for k in ('errors','failures','skipped'))
 terminal=read(root/'terminal.json');assert terminal['session_id']==82594 and terminal['observed_exit_code']==0 and terminal['reaped']is True
 if current:
  current=Path(current)
  assert all(digest((current/n).read_bytes())==h for n,h in r['source_hashes_before'].items())
  assert digest(Path('/tmp/check_fixed_q_dnum_public_20261103.py').read_bytes())==w['source_hashes_before']['external/check_fixed_q_dnum_public_20261103.py']
 return {'passed':True,'candidate_sources':39,'proof_sources':40,'tests':8,'profiles':12,'prefix_checks':1200,'secure_full_profiles':secure,'current_checked':current is not None}

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('archive');parser.add_argument('--current');args=parser.parse_args()
 print(json.dumps(verify(args.archive,args.current),sort_keys=True))
