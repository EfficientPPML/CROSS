"""Independent NumPy semantic/cost review; no ciphertext or accelerator claim."""
import hashlib, importlib.util, json, os, time
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SOURCE=Path('/tmp/factored-bert-scores-20261103/score_layout.py')
COEFFICIENTS=Path('/tmp/streaming-bert-scores/run2/manifest.json')
def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()
before={str(p):digest(p) for p in (Path(__file__),SOURCE,COEFFICIENTS)}
spec=importlib.util.spec_from_file_location('reviewed_score_layout',SOURCE)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
started=time.perf_counter();rng=np.random.default_rng(461)
slots=65536;positions=np.arange(slots);keys=set();counts=[]
seen=np.zeros((16,128,12),np.int64)
hostile=rng.normal(size=slots)
for tile in range(8):
 for offset in range(16):
  mapper=m.ScoreMap(tile,offset)
  expected=np.zeros(slots);inverse=np.zeros(slots)
  for q in range(16):
   key=16*tile+(q+offset)%16
   for head in range(12):
    src=q*4096+head*64;dst=q*4096+key*16+head
    expected[dst]=hostile[src];inverse[src]=hostile[dst]
    seen[q,key,head]+=1
  actual=mapper.apply(hostile);back=mapper.inverse(hostile)
  assert np.array_equal(actual,expected)
  assert np.array_equal(back,inverse)
  assert not np.any(actual.reshape(16,4096)[:,2048:])
  counts.append(dict(mapper.counts));keys.update(mapper.rotations)
assert np.array_equal(seen,np.ones_like(seen))

def rotate(x,k):return np.roll(x,-k)
anchor=(positions%2048)<16
def reduce_keys(x):
 # Only chunk anchors are interpreted; hostile neighboring chunks may occur.
 out=x.copy()
 for bit in range(7):out=out+rotate(out,16*(1<<bit))
 return out*anchor
def broadcast(x):
 out=x.copy()
 for bit in range(7):out=out+rotate(out,-16*(1<<bit))
 return out
raw=rng.normal(size=slots)
expected_sum=raw.reshape(32,128,16).sum(axis=1)
actual_sum=reduce_keys(raw)
assert np.allclose(actual_sum.reshape(32,128,16)[:,0,:],expected_sum,rtol=0,atol=2e-14)
assert np.array_equal(broadcast(actual_sum),np.repeat(actual_sum.reshape(32,128,16)[:,:1,:],128,axis=1).reshape(-1))

coef=json.loads(COEFFICIENTS.read_text())['power_coefficients_in_t_2x']
cases=[]
for sequence in (1,17,128):
 masks=[np.ones(sequence,dtype=bool),np.arange(sequence)%3==0,np.arange(sequence)==sequence-1]
 for mask in masks:
  for qt in range((sequence+15)//16):
   active=m.public_mask(mask,qt,sequence).reshape(16,2,128,16)
   expected=np.zeros_like(active)
   for q in range(min(16,sequence-16*qt)):
    for k in range(sequence):
     if mask[k]:expected[q,0,k,:12]=True
   assert np.array_equal(active,expected)
   # Large hostile padding must not influence valid or padded-query results.
   score=rng.normal(size=(16,2,128,16))*.4+np.arange(16)[:,None,None,None]*.03
   score[~active]=1e4
   flattened=score.reshape(-1);masked=active.reshape(-1)
   engine=m.FixedSoftmax(coef)
   got=engine.evaluate(flattened,masked,int(mask.sum())).reshape(score.shape)
   exact=np.zeros_like(score)
   for q in range(min(16,sequence-16*qt)):
    for h in range(12):
     index=np.flatnonzero(mask);s=score[q,0,index,h];ex=np.exp(s-s.max())
     exact[q,0,index,h]=ex/ex.sum()
   error=float(np.max(np.abs(got-exact)))
   assert error<1e-8 and not engine.failures(),(sequence,qt,error,engine.failures())
   assert np.count_nonzero(got[~active])==0
   # A concrete rotation schedule may broadcast the reduced denominator
   # BEFORE reciprocal, fill inactive slots with one, then multiply masked
   # numerator. This avoids an uncounted reciprocal-output anchor mask.
   positive=np.exp(np.where(masked,flattened,0))*masked
   denominator=broadcast(reduce_keys(positive))
   live_group=np.repeat(active.reshape(32,128,16).any(axis=1,keepdims=True),128,axis=1).reshape(-1)
   safe=denominator+(~live_group)
   scheduled=positive/safe
   assert np.allclose(scheduled.reshape(score.shape),exact,rtol=0,atol=2e-16)
   cases.append({'sequence':sequence,'query_tile':qt,'keys':int(mask.sum()),'max_error':error})

inverse_engine=m.FixedSoftmax(coef)
recip,depth=inverse_engine.inverse(np.array([1/128,.5/128,1.]),(.5/128,2.),13,'review')
assert np.max(np.abs(recip*np.array([1/128,.5/128,1.])-1))<1e-12
after={str(p):digest(p) for p in (Path(__file__),SOURCE,COEFFICIENTS)}
report={'scope':__doc__,'passed':before==after,'source_before':before,'source_after':after,
 'all_128_forward_inverse_maps_exact':True,'all_query_key_head_routes_cover_once':True,
 'segmented_reduction_and_broadcast_rotation_schedule':True,'mask_softmax_cases':cases,
 'mapping_distinct_rotation_indices':sorted(keys),'mapping_distinct_rotation_count':len(keys),
 'mapping_rotation_calls_per_query_tile':sum(c['rotations'] for c in counts),
 'mapping_public_multiplies_per_query_tile':sum(c['public_multiplications'] for c in counts),
 'per_map_forward_plus_inverse_depth':sorted(set(c['public_multiplication_levels'] for c in counts)),
 'segmented_softmax_schedule':{'reductions_per_score_ciphertext':7,
   'rotations_per_reduction':14,'additions_per_reduction':14,'anchor_multiplies_per_reduction':1,
   'rotations_per_score_ciphertext':98,'rotations_per_128_query_example':784,
   'new_rotation_indices_beyond_map_union':sorted(({(sign*16*(1<<b))%slots for sign in (-1,1) for b in range(7)})-keys),
   'mean_input_mask_separate_from_anchor_mask':True,
   'abstract_softmax_depth_without_new_fusion':227,
   'dummy_fill':'Add1 only on entirely inactive2048-slot/head groups after denominator broadcast; masked keys in livegroups retain their common denominator.'},
 'normalization_reciprocals_at_public_example_inputs':recip.tolist(),'reciprocal_depth':depth,
 'elapsed_s':time.perf_counter()-started,'numpy_version':np.__version__,
 'limitations':['NumPy public routing and arithmetic only; no encrypted map or refresh.',
 'The full 65536-slot period has no duplicate aliases; period128 refresh evidence does not transfer.',
 'FixedSoftmax depth226 excludes a complete implemented ciphertext schedule, CKKS error, refresh and capacity validation.',
 'Reciprocal values can reach128 for a uniform128-key distribution and256 at its declared lower-domain boundary. Bound16 is insufficient for those intermediates.',
 'Key and queued-ciphertext counts do not establish total HBM fit.']}
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
(HERE/'reviewed_score_layout.py').write_bytes(SOURCE.read_bytes())
(HERE/'coefficients_manifest.json').write_bytes(COEFFICIENTS.read_bytes())
print(json.dumps({k:v for k,v in report.items() if k not in ('source_before','source_after','mask_softmax_cases','mapping_distinct_rotation_indices')},indent=2))
raise SystemExit(0 if report['passed'] else 1)
