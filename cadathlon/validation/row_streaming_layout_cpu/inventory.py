"""Public row-packed BERT storage/key planning; no encryption or activations."""
from pathlib import Path
import hashlib,json,math,os,sys
ROOT=Path('/home/jianming_gatech/CROSS');OUT=Path(__file__).resolve().parent
os.environ['JAX_PLATFORMS']='cpu'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
from jaxite_word.bsgs import compute_bsgs_params
from encrypted_refresh import RefreshProfile,refresh_plan
profile=RefreshProfile(degree=131072,depth=49,dnum=4)
parameters=profile.parameters();q=parameters['q_towers'];p=parameters['p_towers']
slots=65536;features=4096;rows=slots//features;sequence=128;tiles=math.ceil(sequence/rows)
matrices={};linear=set()
for name,r,c in [('qkv_output_pooler',768,768),('mlp_up',3072,768),('mlp_down',768,3072),('classifier',2,768)]:
 active=tuple(d%slots for d in range(-(r-1),c));n1,n2=compute_bsgs_params(slots,len(active),2.)
 keys=({d%n1 for d in active}|{d//n1*n1 for d in active})-{0};linear.update(keys)
 matrices[name]=dict(rows=r,columns=c,active_diagonals=len(active),baby_step=n1,giant_factor=n2,rotation_count=len(keys),rotations=sorted(keys))
head={sign*(1<<bit)%slots for sign in (-1,1) for bit in range(6)}
row={sign*(1<<bit)%slots for sign in (-1,1) for bit in range(12)}
binary_token={sign*features*(1<<bit)%slots for sign in (-1,1) for bit in range(4)}
direct_token={features*index for index in range(1,16)}
refresh=refresh_plan(parameters['degree'],slots,(4,4),max_level=49);boot=set(refresh['rotations'])
alpha=math.ceil(len(q)/4);parts=math.ceil(len(q)/alpha)
key_bytes=2*parts*(len(q)+len(p))*parameters['degree']*4;ct_bytes=2*parameters['degree']*len(q)*4
report=dict(scope=__doc__,degree=parameters['degree'],slots=slots,features=features,tokens=sequence,rows_per_tile=rows,tiles=tiles,
 profile=dict(q_count=len(q),p_count=len(p),dnum=4,actual_log2_qp=sum(math.log2(m) for m in q+p),security_ceiling=3523,refresh_period=slots,normalized_refresh_output_level=refresh['output_level']),
 memory=dict(ciphertext_at_max_level_bytes=ct_bytes,eight_tile_activation_set_bytes=tiles*ct_bytes,
   conservative_x_q_k_v_payload_bytes=4*tiles*ct_bytes,one_public_key_pair_bytes=key_bytes,
   note='Payload accounting only, batch dimension1; excludes transform/key/control/plaintext/workspace overhead. Actual X/Q/K/V levels can reduce payload below this max-level upper bound.'),
 dense_linear_matrices=matrices,
 rotation_categories=dict(linear=sorted(linear),head_reduce_broadcast=sorted(head),row_reduce_broadcast=sorted(row),binary_token_align_broadcast=sorted(binary_token),direct_token_align_broadcast=sorted(direct_token),full_slot_bootstrap=sorted(boot)),
 alternatives={},limitations=[
  'Full-width dense matrix inventory is a public upper bound; assumes all supported diagonals nonzero and the current BSGS heuristic.',
  'Binary token alignment needs up to4 sequential rotations; direct token alignment uses15 keys and one alignment rotation.',
  'Period65536 is not covered by the earlier period128 refresh numerical proof; it has no redundant slot averaging.',
  'No polynomial domains, CKKS numerical error, model gates, actual HBM or runtime are validated here.'])
for name,token in [('binary_token',binary_token),('direct_token',direct_token)]:
 graph=linear|head|row|token;union=graph|boot
 report['alternatives'][name]=dict(graph_rotation_count=len(graph),bootstrap_rotation_count=len(boot),combined_rotation_count=len(union),
   graph_raw_public_key_bytes=len(graph)*key_bytes,combined_raw_public_key_bytes=len(union)*key_bytes,combined_rotations=sorted(union))
source_paths=[Path(__file__),ROOT/'jaxite_word/bsgs.py',ROOT/'jaxite_word/bootstrapping.py',ROOT/'cadathlon/common/encrypted_refresh.py',ROOT/'jaxite_word/he_params.py',ROOT/'jaxite_word/util.py',ROOT/'jaxite_word/composite_prime_gen.py']
report['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths}
(OUT/'inventory.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(profile=report['profile'],memory=report['memory'],alternatives={name:{k:v for k,v in row.items() if k!='combined_rotations'} for name,row in report['alternatives'].items()}),indent=2))
