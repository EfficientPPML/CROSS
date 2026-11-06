"""Public-only diagnostic plan, no launchable HE worker or key material."""
from pathlib import Path
import ast,hashlib,json,math,sys
import numpy as np
import public_graph as graph
from dense_reference import references
P=Path(__file__).resolve().parent

def source_gate():
 p=json.loads((P/'predecessor.json').read_bytes());actual={n:hashlib.sha256((P/'predecessor'/n).read_bytes()).hexdigest() for n in p['source_hashes']}
 if len(actual)!=199 or actual!=p['source_hashes']:raise ValueError('original199 predecessor bytes differ')
 return actual

def derive():
 source_gate()
 if np.__version__!='2.5.4':raise ValueError('original interpolation NumPy2.5.4 required')
 r=graph.run();refs,refresh=references(r['case']['config'],r['weights'],r['embeddings'],r['mask'],r['approximations']);phase=0;checkpoints=[];worst=0.
 for row,value in zip(r['trace'],r['values'],strict=True):
  if row['kind']=='phase' and row['label'].startswith('bounded'):
   expected=refresh[phase//2];phase+=1
  else:expected=refs[(row['kind'],row['label'])]
  error=float(np.max(np.abs(value-expected)))
  if not np.isfinite(value).all() or not np.isfinite(expected).all() or not math.isfinite(error) or error>1e-10:raise ValueError('public all-slot independent reference mismatch')
  worst=max(worst,error);checkpoints.append(dict(row,independent_array_sha256=hashlib.sha256(expected.tobytes()).hexdigest(),public_reference_error=error))
 ns=dict(math=math,Sequence=graph.Sequence,Tuple=graph.Tuple)
 graph.selected(graph.ROOT/'jaxite_word/bootstrapping.py',['_select_layers','_reduce_rotation','_bootstrap_rotation_indices'],ns)
 bootstrap=sorted(set(ns['_bootstrap_rotation_indices'](131072,8192,(4,4)))|{262143});keys=sorted(set(r['graph_plan']['rotations'])|set(bootstrap))
 historical=json.loads((P/'historical_public_plan.json').read_bytes())['case'];ring=r['ring'];q,p=ring['q_towers'],ring['p_towers'];dnum=4
 if keys!=historical['rotations'] or q!=historical['q_towers'] or p!=historical['p_towers']:raise ValueError('actual199 key/profile differs from historical plan')
 if r['graph_plan']['depth']!=93 or len(r['refreshes'])!=5 or checkpoints[-1]['level']!=11:raise ValueError('fixed circuit or five-refresh schedule changed')
 partition=(len(q)+dnum-1)//dnum;partitions=[q[i:i+partition] for i in range(0,len(q),partition)]
 if math.prod(p)<max(math.prod(x)for x in partitions) or math.prod(q+p).bit_length()>3523:raise ValueError('complete fixed profile/P partition check fails')
 pair=2*len(partitions)*(len(q)+len(p))*131072*4
 ct_bytes=sum(row['ciphertext_payload_bytes'] for row in checkpoints)
 coefficients={n:dict(domain=list(a.domain),degree=len(a.coefficients)-1,chebyshev_coefficients=list(a.coefficients),power_coefficients=np.polynomial.chebyshev.cheb2poly(a.coefficients).tolist(),strategy='power',input_power_basis=a.input_power_basis) for n,a in r['approximations'].items()}
 return dict(schema='bert-toy4-checkpoint-plan-v1',scope=__doc__,original199_source_count=199,no_HE_or_TPU_execution=True,
   case_sha256=hashlib.sha256((graph.ROOT/'cadathlon/agent_pset/p2_bert/test/test4.in').read_bytes()).hexdigest(),case=r['case'],
   profile=dict(degree=131072,slots=65536,composite_degree=2,degree_layout=[256,512],chain_depth=49,scale_bits=56,dnum=4,q_towers=q,p_towers=p,scale=r['scale'],log2_QP=sum(math.log2(x)for x in q+p),security_ceiling=3523,P_covers_actual_partitions=True),
   packing=dict(batch=2,sequence=8,feature_width=64,period=8192,replicas=8),graph_depth=93,
   coefficients=coefficients,checkpoints=checkpoints,checkpoint_count=len(checkpoints),refresh_events=r['refreshes'],refresh_bounds=[x['public_bound']for x in r['refreshes']],
   ideal_counts=r['counts'],polynomial_domains=r['domains'],ideal_final_toy_gate_passed=r['original_toy_clear_gate_passed'],ideal_max_logit_error=r['clear_max_logit_error'],
   independent_all_slot_reference_max_error=worst,all_slot_reference_comparisons=len(checkpoints)*65536,
   graph_rotation_indices=r['graph_plan']['rotations'],bootstrap_rotation_indices=bootstrap,required_rotation_indices=keys,
   storage=dict(rotation_keys=len(keys),one_rotation_pair_bytes=pair,raw_rotation_bytes=len(keys)*pair,relinearization_pair_bytes=pair,
      client_public_key_bytes=2*131072*100*4,secret_i64_bytes=131072*8,
      checkpoint_raw_payload_bytes=ct_bytes,checkpoint_npy_header_reserve_bytes=len(checkpoints)*8192,
      checkpoint_disk_budget_bytes=ct_bytes+len(checkpoints)*8192+(1<<30),
      raw_public_store_plus_header_and_reserve_bytes=(len(keys)+1)*(pair+8192)+(64<<20),
      maximum_single_checkpoint_payload_bytes=max(row['ciphertext_payload_bytes']for row in checkpoints),
      transport_extra_host_payloads=1,extra_device_retained_checkpoint_payloads=1,
      not_a_peak_claim='Raw payload/header reserves exclude JAX compiler caches, parameter tables, live graph temporaries, file-cache duplication and runtime/library memory.'),
   gates=dict(original_decoder_stddev_limit=.03125,original_final_absolute=.02,original_final_relative=.01,original_binary_prediction_exact=True,
       diagnostic_checkpoint_absolute=1e-6,diagnostic_gate_is_additional_and_does_not_replace_original=True,clear_reference_self_check=1e-10),
   scale_scope='These are nominal metadata recurrences for identity refresh returning canonical scale. Future actual worker must record each actual refresh result and replay downstream ordered float operations without relabeling; no canonical-return assumption is an HE proof.',
   deferred_implementation=['Pinned separate-process client setup/encryption, keyless server instrumentation/opaque save, private client readback',
      'Exact actual-profile security/P checks before keygen; dual keygen aliases/seven codecs forbidden in server',
      'Authenticated96-envelope transport with bounded headers/residues/context/sequence and owned-directory guards',
      'Actual rounded polynomial first-term metadata replay from refreshed scales; captured-source mutation guards',
      'All server work/checkpoint persistence/source verification before any private decode; preserve every decode failure independently',
      'Dedicated owner group/direct wait/cleanup quarantine; resources preflight on available accelerator lease'])

if __name__=='__main__':
 import argparse
 parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();r=derive()
 with args.output.open('x')as f:json.dump(r,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps({k:r[k]for k in ('checkpoint_count','refresh_bounds','independent_all_slot_reference_max_error','ideal_final_toy_gate_passed')}))
