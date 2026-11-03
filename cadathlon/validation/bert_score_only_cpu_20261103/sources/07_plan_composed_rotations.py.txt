"""Public key inventory and exact plaintext rotation-composition parity."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from score_layout import ScoreMap,SLOTS
HERE=Path(__file__).resolve().parent;started=time.perf_counter()
paths=[Path(__file__),HERE/'score_layout.py',Path('/tmp/streaming-bert-review/inventory.json')]
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
inventory=json.loads(paths[-1].read_text())
cat=inventory['rotation_categories']
existing=set(cat['linear'])|set(cat['head_reduce_broadcast'])|set(cat['row_reduce_broadcast'])|set(cat['direct_token_align_broadcast'])
binary={sign*(1<<bit)%SLOTS for sign in (-1,1) for bit in range(16)}
assert binary<=existing and len(binary)==31

def signed_binary(shift):
 n=int(shift)%SLOTS
 if n>SLOTS//2:n-=SLOTS
 sign=1 if n>=0 else -1;n=abs(n);bit=0;out=[]
 while n:
  if n&1:
   digit=2-(n%4) if n!=1 else 1
   out.append(sign*digit*(1<<bit));n-=digit
  n//=2;bit+=1
 assert sum(out)%SLOTS==shift%SLOTS
 assert all(s%SLOTS in binary for s in out)
 return tuple(out)

def policy(shift):
 shift%=SLOTS
 return (() if shift==0 else ((shift,) if shift in existing else signed_binary(shift)))

def rotate(value,shift):
 out=value
 for component in policy(shift):out=np.roll(out,-component)
 return out

maps=[ScoreMap(t,o) for t in range(8) for o in range(16)]
rows=[]
for baby_step in (16,32,64,128,256,512,1024):
 direct=set();composed=set();requested_count=executed_count=pure_signed_count=0
 for m in maps:
  for displacements in (m.indices,[(-d)%SLOTS for d in m.indices]):
   requested=({d%baby_step for d in displacements}|{d//baby_step*baby_step for d in displacements})-{0}
   direct|=requested;requested_count+=len(requested)
   for shift in requested:
    sequence=policy(shift);executed_count+=len(sequence);pure_signed_count+=len(signed_binary(shift));composed|={v%SLOTS for v in sequence}
 rows.append({'baby_step':baby_step,'one_key_per_logical_shift_union':len(direct),
  'requested_rotations_per_query_tile_forward_and_scatter':requested_count,
  'executed_rotations_per_query_tile_hybrid_existing_else_signed_binary':executed_count,
  'executed_rotations_per_example_S128_hybrid':executed_count*8,
  'executed_rotations_per_query_tile_pure_signed_binary':pure_signed_count,
  'used_existing_key_count':len(composed),'additional_keys_beyond_existing_graph':len(composed-existing),
  'additional_multiplicative_depth':0})
# Every possible logical rotation has the same composed slot permutation.
for shift in range(SLOTS):assert sum(signed_binary(shift))%SLOTS==shift
rng=np.random.default_rng(4622);value=rng.normal(size=SLOTS);checks=[]
for tile,offset in ((0,0),(0,15),(7,0),(7,15)):
 m=ScoreMap(tile,offset)
 for inverse in (False,True):
  source,target=(m.target,m.source) if inverse else (m.source,m.target)
  displacement=(source-target)%SLOTS;actual=np.zeros(SLOTS)
  for shift in sorted(set(map(int,displacement))):
   where=target[displacement==shift];actual[where]+=rotate(value,shift)[where]
  expected=m.inverse(value) if inverse else m.apply(value)
  np.testing.assert_array_equal(actual,expected)
  checks.append({'tile':tile,'offset':offset,'inverse':inverse,'exact':True})
report={'scope':__doc__,'passed':True,'key_policy':'Use one existing full-graph key when present; otherwise compose signed non-adjacent power-of-two rotations. Fixed public policy, no private branch.',
 'existing_full_graph_key_count':len(existing),'signed_power_basis_key_count':31,'additional_basis_keys_required':0,
 'plans':rows,'literal_composed_diagonal_mask_checks':checks,'all65536_rotation_displacements_algebraically_verified':True,
 'ciphertext_error_scope':'Exact NumPy permutation parity only. Every added key switch contributes real CKKS error/workspace/runtime; none is measured or bounded here.',
 'attention_mask_depth':'Composition uses rotations only and adds no multiplicative levels. Compaction/scatter each still consume one public-linear/rescale level under the assumed BSGS circuit.',
 'normalized_softmax_reduction_rotations':sorted({sign*16*(1<<bit)%SLOTS for sign in (-1,1) for bit in range(7)}),
 'limitations':['Counts concern only the128forward+128inverse maps for each16query tile, excluding QK/head reductions, V multiplication, softmax reductions, dense layers, and refresh.',
 'No keys generated or ciphertext operations executed. Existing graph key count190 is itself a planned public inventory, not a proven full model.',
 'A500logical-key direct BSGS variant is not required:31signed-power keys already occur in the graph, at substantially higher execution rotation counts.',
 'Future factored head/row/key-position maps can trade additional public-mask levels for fewer rotations; not implemented here.'],
 'source_hashes_before':before,'source_hashes_after':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'seconds':time.perf_counter()-started}
report['source_files_unchanged']=report['source_hashes_before']==report['source_hashes_after']
report['passed']=report['passed'] and report['source_files_unchanged']
(HERE/'plan_composed_rotations.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='normalized_softmax_reduction_rotations'},indent=2))
