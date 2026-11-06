"""Exact public mask/rotation network checks; no ciphertext arithmetic."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from score_layout import ScoreMap,SLOTS,public_mask

paths=[Path(__file__),Path(__file__).with_name('score_layout.py')]
before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
started=time.perf_counter();rng=np.random.default_rng(5331)
value=rng.integers(-65536,65536,SLOTS).astype(np.float64)/1024
rows=[];keys=set();total_rotations=0
for tile in range(8):
 for offset in range(16):
  mapping=ScoreMap(tile,offset)
  forward=mapping.apply(value);backward=mapping.inverse(value)
  assert np.array_equal(forward,mapping.oracle(value))
  assert np.array_equal(backward,mapping.oracle(value,inverse=True))
  assert np.count_nonzero(forward)<=192 and np.count_nonzero(backward)<=192
  expected_rotations=16+2*int(tile!=0)
  assert mapping.counts['rotations']==expected_rotations
  assert mapping.counts['public_multiplication_levels']==18
  total_rotations+=mapping.counts['rotations'];keys.update(mapping.rotations)
  rows.append({'key_tile':tile,'offset':offset,'forward_inverse_equal_direct':True,
               'rotations':expected_rotations,'public_multiplies':mapping.counts['public_multiplications']})
for sequence in (1,17,128):
 mask=np.zeros(sequence,dtype=bool);mask[-1]=True
 for qt in range((sequence+15)//16):
  shaped=public_mask(mask,qt,sequence).reshape(16,2,128,16)
  valid_rows=min(16,sequence-qt*16)
  assert not shaped[:,1].any() and not shaped[...,12:].any()
  assert int(shaped.sum())==valid_rows*12
  assert np.all(shaped[:valid_rows,0,sequence-1,:12])

# Every public shift can use the already-planned signed power-of-two keys.
def naf_shifts(number):
 number=number%SLOTS
 if number>SLOTS//2:number-=SLOTS
 sign=1 if number>=0 else -1;number=abs(number);bit=0;parts=[]
 while number:
  if number&1:
   digit=2-(number%4) if number>1 else 1
   parts.append((sign*digit*(1<<bit))%SLOTS);number-=digit
  number//=2;bit+=1
 return parts
decomposed={shift:naf_shifts(shift) for shift in sorted(keys)}
for shift,parts in decomposed.items():
 rotated=value
 for part in parts:rotated=np.roll(rotated,-part)
 assert np.array_equal(rotated,np.roll(value,-shift))
composed_per_query=0
for tile in range(8):
 for offset in range(16):
  requested=[63*(1<<b) for b in range(4)]+[-16*(1<<b) for b in range(4)]+[-256*tile]
  requested += [-x for x in reversed(requested)]
  composed_per_query+=sum(len(naf_shifts(x)) for x in requested if x)
after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
report={'scope':__doc__,'passed':before==after,'cases':rows,
 'all128_forward_and_inverse_maps_equal_direct':True,
 'period':65536,'alias_count':1,'score_shape':[16,2,128,16],
 'upper_chunk_per_query':'inactive safe dummy denominator1; no secret data is discarded',
 'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
 'elapsed_s':time.perf_counter()-started,
 'planning':{'full128_sequence_query_tiles':8,
  'forward_plus_inverse_requested_rotations_per_example':total_rotations*8,
  'forward_plus_inverse_power_composed_rotations_per_example':composed_per_query*8,
  'direct_key_count':len(keys),'direct_key_indices':sorted(keys),
  'power_key_count':len(set(part for parts in decomposed.values() for part in parts)),
  'forward_mapping_depth':9,'inverse_mapping_depth':9,
  'optional_forward_depth_if_anchor_mask_folds_into_existing_head_reduction':8},
 'limitations':['Only public routing semantics and operation counts; no encrypted key-switch noise, refresh or TPU performance.',
 'Additional routing levels can trigger more refreshes; output period65536 has no alias averaging.',
 'The retained normalized-squaring circuit still needs226 abstract levels and can form reciprocals above the validated bound16.']}
Path(__file__).with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ('cases','source_hashes_before','source_hashes_after')},indent=2))
raise SystemExit(0 if report['passed'] else 1)
