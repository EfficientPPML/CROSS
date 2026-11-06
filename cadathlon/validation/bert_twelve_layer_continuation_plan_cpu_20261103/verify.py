"""Offline independent recurrence/source checks; no backend, weights or inference."""
from pathlib import Path
import hashlib,json,math,tarfile
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def main():
 r=json.loads((HERE/'report.json').read_text());before=r['source_hashes_before']
 assert r['passed'] and r['source_files_unchanged'] and before==r['source_hashes_after'] and len(before)==117
 assert sha((HERE/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
 with tarfile.open(HERE/'sources.tar.gz') as t:
  rows=t.getmembers();assert len(rows)==len(before)==len({m.name for m in rows})
  assert {m.name:sha(t.extractfile(m).read()) for m in rows}==before
  m=json.loads(t.extractfile('candidate_manifest.json').read())
  profile=json.loads(t.extractfile('p60_profile_and_refresh.json').read())['profile']
  cert=json.loads(t.extractfile('gelu_refresh_certificates.json').read())
  assert sha(t.extractfile('candidate_manifest.json').read())==r['manifest_sha256']
 for name,rule in m['norms'].items():
  d=rule['polynomial']['degree'];depth=(d-1).bit_length()*(1 if rule['method']=='power' else 2)+7
  assert r['normalizations'][name]['row_norm_depth']==depth
 count={k:0 for k in range(1,129)}
 assert len(r['schedules'])==1536 and len({(v['K'],v['layer']) for v in r['schedules']})==1536
 for s in r['schedules']:
  i,k=s['layer'],s['K'];lv=s['levels'];round_depth=m['normalization_reciprocal_by_public_K'][str(k)]['iterations']+5
  assert lv['attention_input']==(49 if i==0 else 25)
  assert lv['QKV']==lv['attention_input']-1 and lv['QK']==lv['QKV']-3
  assert lv['normalized_t']==lv['QK']-3
  level=lv['normalized_t']-19 if i==0 else 25-19;assert level==lv['initial_probabilities']
  events=(0 if i==0 else 2)
  for row in s['rounds']:
   assert row['input_level']==level and row['refresh_before']==(level<round_depth+2)
   if row['refresh_before']:assert level>=2;level=25;events+=1
   level-=round_depth;assert level==row['output_level'] and level>=2
  assert len(s['rounds'])==m['softmax'][str(i)]['squarings']
  assert level<3+1+r['normalizations'][f'{i}.attention']['row_norm_depth']+2
  assert lv['final_probabilities_after_refresh_mask']==24 and lv['PV']==21 and lv['attention_dense']==20
  assert lv['attention_norm']==20-r['normalizations'][f'{i}.attention']['row_norm_depth']>=2
  assert lv['GELU']==24-(2*(m['gelu'][str(i)]['degree']-1).bit_length()+2)>=2
  assert lv['MLP_norm']==24-r['normalizations'][f'{i}.mlp']['row_norm_depth']>=2
  events+=3
  assert events==s['refresh_calls_per_query_tile'] and events*((k+15)//16)==s['refresh_calls_all_real_tiles']
  count[k]+=events*((k+15)//16)
 assert count[128]==r['K128_refresh_calls']==1064 and count[1]==r['K1_refresh_calls']==98
 assert min(s['levels']['MLP_norm'] for s in r['schedules'])==3
 assert r['gelu_refresh_bounds']==[cert['sites'][str(i)]['refresh_bound'] for i in range(12)]
 p=r['pooler'];level=12-1-6;events=0
 for row in p['rounds']:
  assert row['input_level']==level and row['refresh_before']==(level<13)
  if level<13:level=25;events+=1
  level-=11;assert row['output_level']==level>=2
 assert events==3 and level-1==p['classifier_output_level']==2 and r['K128_with_pooler_refresh_calls']==1067
 q,pv=profile['q'],profile['p'];assert len(q)==100 and len(pv)==14 and math.prod(q+pv).bit_length()<=3523
 assert math.prod(pv)>=max(math.prod(q[i:i+13]) for i in range(0,100,13))
 assert set(r['union_keys'])==set(r['graph_keys'])|set(r['refresh_keys']) and len(r['union_keys'])==130
 assert r['raw_union_rotation_bytes']==len(r['union_keys'])*r['raw_rotation_pair_bytes']
 assert not r['encrypted_execution'] and not r['model_forward'] and not r['native_bound_transfer_to_approximate_or_CKKS_proved']
 print(json.dumps({'passed':True,'exact_sources':len(before),'layer_K_schedules':1536,'encoder_K128_refreshes':1064,'with_pooler':1067,'minimum_MLP_level':3}))
if __name__=='__main__':main()
