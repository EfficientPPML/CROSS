from pathlib import Path
import hashlib,json,time
import numpy as np
from score_layout import *
started=time.perf_counter();source=Path(__file__).with_name('score_layout.py');before=hashlib.sha256(source.read_bytes()).hexdigest()
checks=[];alltargets=np.zeros(SLOTS,int);unique=np.arange(SLOTS,dtype=float)+.25
union=set();inverse_union=set();direct=set()
for tile in range(8):
 for offset in range(16):
  m=ScoreMap(tile,offset);assert len(set(m.source))==len(set(m.target))==192
  alltargets[m.target]+=1
  actual=m.apply(unique);restored=m.inverse(actual)
  expected=np.zeros(SLOTS);expected[m.source]=unique[m.source]
  np.testing.assert_array_equal(restored,expected)
  for r in range(16):
   j=16*tile+(r+offset)%16
   np.testing.assert_array_equal(actual.reshape(2,16,128,16)[0,r,j,:12],unique[r*4096+np.arange(12)*64])
  union|=m.bsgs_rotations;direct|=set(m.indices)
  ds={(-d)%SLOTS for d in m.indices}
  inverse_union|=({d%m.n1 for d in ds}|{d//m.n1*m.n1 for d in ds})-{0}
  if (tile,offset) in ((0,0),(0,15),(7,0),(7,15)):
   np.testing.assert_array_equal(actual,m.apply_diagonals(unique))
   np.testing.assert_array_equal(m.inverse(unique),m.apply_diagonals(unique,inverse=True))
assert np.all(alltargets.reshape(2,16,128,16)[0,:,:,:12]==1)
assert np.all(alltargets.reshape(2,16,128,16)[0,:,:,12:]==0) and not np.any(alltargets[PERIOD:])
checks.append('All128 maps: exact independent row/head/key addressing, one-to-one active score coverage and inverse projection; four boundary maps also execute literal rotation/mask sums.')
for sequence in (17,128):
 for kind in ('all','last','interleaved'):
  mask={'all':np.ones(sequence,bool),'last':np.arange(sequence)==sequence-1,'interleaved':np.arange(sequence)%3==1}[kind]
  for qt in range((sequence+15)//16):
   public=public_mask(mask,qt,sequence).reshape(2,16,128,16)
   assert np.array_equal(public[0],public[1]) and not np.any(public[:,:,:,12:])
   for r in range(min(16,sequence-qt*16)):
    np.testing.assert_array_equal(public[0,r,:sequence,0],mask)
   assert not np.any(public[:,min(16,sequence-qt*16):]) and not np.any(public[:,:,sequence:])
checks.append('S17/S128 allvalid, lastkey and interleaved masks preserve padded queries, mask only absent keys/head padding and public tail rows.')
report={'passed':before==hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Exact plaintext public map/mask semantics only; no encrypted or numerical softmax execution.',
'checks':checks,'map_count':128,'direct_diagonal_rotation_union':len(direct-{0}),
'planned_bsgs_forward_rotation_union':len(union),'planned_bsgs_inverse_rotation_union':len(inverse_union),
'planned_bsgs_combined_rotation_union':len(union|inverse_union),'bsgs_combined_rotation_indices':sorted(union|inverse_union),
'period':PERIOD,'aliases':SLOTS//PERIOD,'seconds':time.perf_counter()-started,
'source_sha256':before,'source_unchanged':before==hashlib.sha256(source.read_bytes()).hexdigest(),
'checker_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='bsgs_combined_rotation_indices'},indent=2))
