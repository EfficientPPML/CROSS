"""Pinned seven-case approximate score-only attention proof; plaintext only."""
import argparse,hashlib,importlib.metadata,json,math,os
from pathlib import Path
import resource,sys,time
os.environ['JAX_PLATFORMS']='cpu'
import numpy as np
ROOT=Path(os.environ.get('CROSS_ROOT','/home/jianming_gatech/CROSS')).resolve()
sys.path[:0]=[str(ROOT/'cadathlon/common'),str(Path(__file__).resolve().parent),'/tmp/streaming-bert-layout']
from support import load_module
from encrypted_polynomial import nonlinear_approximation
from row_attention import RowPackedAttention
from score_layout import ScoreMap, FixedSoftmax, duplicate_period, public_mask, exact_probabilities

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def candidate(q,k,v,mask,coefficients):
 sequence=len(q);tiles=(sequence+15)//16
 row=RowPackedAttention();out=np.zeros_like(q);reports=[]
 maps={(t,o):ScoreMap(t,o) for t in range(tiles) for o in range(16)}
 for qt in range(tiles):
  query=row.pack_rows(q,qt);packed_scores=np.zeros(65536)
  for kt in range(tiles):
   key=row.pack_rows(k,kt)
   for offset in range(16):
    score=row.head_reduce(query*row.rotate_rows(key,offset),1/8)
    packed_scores+=maps[kt,offset].apply(score)
  packed_scores=duplicate_period(packed_scores)
  active=public_mask(mask,qt,sequence)
  engine=FixedSoftmax(coefficients,score_domain=(-16.,16.),squarings=5,iterations=13)
  probabilities=engine.evaluate(packed_scores,active,int(np.sum(mask)))
  exact=exact_probabilities(packed_scores,active)
  probability_error=float(np.max(np.abs(probabilities-exact)))
  numerator=np.zeros(65536)
  for kt in range(tiles):
   value=row.pack_rows(v,kt)
   for offset in range(16):
    anchors=maps[kt,offset].inverse(probabilities)
    numerator+=row.head_broadcast(anchors)*row.rotate_rows(value,offset)
  count=min(16,sequence-qt*16)
  out[qt*16:qt*16+count]=numerator.reshape(16,4096)[:count,:768]
  reports.append({'query_tile':qt,'probability_max_abs_error':probability_error,
    'fixed_domain_failures':engine.failures(),'historical_narrow_domain_failures':{k:v for k,v in engine.failures(historical=True).items() if k.startswith('historical_')},
    'observations':engine.domains,'abstract_softmax_depth_from_compacted_scores':engine.depth,
    'arithmetic_counts_per_score_ciphertext':dict(engine.counts)})
 return out,reports

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();args.output.mkdir(exist_ok=False)
 audit=load_module(ROOT/'cadathlon/validation/check_pretrained_bert_domains.py')
 reference=load_module(ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py')
 assets={name:digest(args.snapshot/name) for name in audit.EXPECTED_ASSETS};assert assets==audit.EXPECTED_ASSETS
 sources=[Path(__file__),Path(__file__).with_name('score_layout.py'),Path('/tmp/streaming-bert-layout/row_attention.py'),
 ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py',ROOT/'cadathlon/validation/check_pretrained_bert_domains.py',ROOT/'cadathlon/common/encrypted_polynomial.py']
 before={str(p):digest(p) for p in sources};started=time.perf_counter()
 # Public analytic function constants, selected before reading model activations.
 approximation=nonlinear_approximation('exp',(-.5,.5),8)
 coefficients=np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
 grid=np.linspace(-.5,.5,20001);exp_error=float(np.max(np.abs(np.polynomial.polynomial.polyval(2*grid,coefficients)-np.exp(grid))))
 rho=(2.-.5/128)/(2.+.5/128);newton_relative_bound=rho**(2**13)
 assert exp_error<1e-10 and newton_relative_bound<1e-12
 manifest={'centered_score_domain':[-16.,16.],'domain_scope':'Fixed conditional domain selected from the preceding public full-row audit, not a guaranteed bound on unseen inputs.',
 'historical_domain_preserved_for_diagnostics':[-9.,12.],'exp_domain':[-.5,.5],'exp_degree':8,'power_coefficients_in_t_2x':coefficients.tolist(),
 'exp_grid_max_abs_error':exp_error,'exp_grid_gate':1e-10,'normalized_squarings':5,
 'initial_reciprocal':{'domain':[.5,2.],'iterations':6},
 'round_reciprocal':{'domain_formula':'[0.5/public_valid_count,2]','iterations':13,'max_keys':128,'worst_case_exact_arithmetic_relative_residual_bound':newton_relative_bound},
 'selection':'Only declared function domains and analytic Newton convergence decide constants/iterations. No labels/logits or per-value stopping/clipping.',
 'fixed_gates':{'probability_max_abs':1e-8,'context_and_block_max_abs':1e-6,'probability_sum_abs':1e-10,'domain_violations':0}}
 (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 import torch
 from transformers import AutoTokenizer,BertForSequenceClassification
 torch.set_num_threads(4)
 model,loading=BertForSequenceClassification.from_pretrained(args.snapshot,local_files_only=True,use_safetensors=True,torch_dtype=torch.float64,attn_implementation='eager',output_loading_info=True)
 assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
 assert set(loading['unexpected_keys'])<={'bert.embeddings.position_ids'}
 model.eval();config=model.config.to_dict();weights={name:value.detach().numpy() for name,value in model.state_dict().items()}
 tokenizer=AutoTokenizer.from_pretrained(args.snapshot,local_files_only=True)
 public=list(audit.EXAMPLES)+[('The committee reviews the proposal and discusses the evidence. '*20,'Members discuss the proposal before reaching a decision. '*20)]
 tokens=tokenizer([a for a,b in public],[b for a,b in public],max_length=128,truncation=True,padding='max_length',return_tensors='pt')
 embeddings=reference.embed(config,weights,tokens['input_ids'].numpy(),tokens['token_type_ids'].numpy());masks=tokens['attention_mask'].numpy().astype(bool)
 cases=[(f'public_pair_{i}',embeddings[i:i+1],masks[i:i+1]) for i in range(4)]
 boundary=np.zeros((1,128),bool);boundary[0,[0,15,16,31,32,63,64,95,111,112,127]]=True
 last=np.zeros((1,128),bool);last[0,-1]=True
 cases.extend([('public_boundary_mask',embeddings[:1],boundary),('public_last_key_only',embeddings[:1],last),('public_partial_tile17',embeddings[:1,:17],masks[:1,:17])])
 reports=[];prefix='bert.encoder.layer.0.'
 for name,x,mask in cases:
  print(json.dumps({'starting':name,'sequence':x.shape[1],'valid_keys':int(mask.sum())}),flush=True)
  q,k,v=[reference.linear(x,weights,prefix+'attention.self.'+kind)[0] for kind in ('query','key','value')]
  context,tile_reports=candidate(q,k,v,mask[0],coefficients);context=context[None]
  post=reference.layer_norm(x+reference.linear(context,weights,prefix+'attention.output.dense'),weights,prefix+'attention.output.LayerNorm',config['layer_norm_eps'])
  intermediate=reference.gelu(reference.linear(post,weights,prefix+'intermediate.dense'))
  actual=reference.layer_norm(post+reference.linear(intermediate,weights,prefix+'output.dense'),weights,prefix+'output.LayerNorm',config['layer_norm_eps'])
  expected=reference.encoder_block(config,weights,x,mask,0)
  with torch.no_grad():
   native_x=torch.from_numpy(x.copy());native_mask=torch.from_numpy(mask.copy())
   additive=torch.zeros(native_mask[:,None,None,:].shape,dtype=torch.float64).masked_fill(~native_mask[:,None,None,:],torch.finfo(torch.float64).min)
   native=model.bert.encoder.layer[0](native_x,attention_mask=additive)[0].numpy()
   native_context=model.bert.encoder.layer[0].attention.self(native_x,attention_mask=additive)[0].numpy()
  errors={'probability_vs_exact':max(row['probability_max_abs_error'] for row in tile_reports),
   'context_vs_transformers':float(np.max(np.abs(context-native_context))),
   'full_block_vs_numpy':float(np.max(np.abs(actual-expected))),
   'full_block_vs_transformers':float(np.max(np.abs(actual-native)))}
  domain_count=sum(len(row['fixed_domain_failures']) for row in tile_reports)
  passed=bool(np.isfinite(actual).all() and errors['probability_vs_exact']<1e-8 and all(value<1e-6 for key,value in errors.items() if key!='probability_vs_exact') and domain_count==0)
  reports.append({'case':name,'sequence':int(x.shape[1]),'valid_keys':int(mask.sum()),'passed':passed,'errors':errors,
   'domain_failure_records':domain_count,'tiles':tile_reports,'padded_query_output_max_abs':float(np.max(np.abs(actual[~mask]))) if np.any(~mask) else None})
  print(json.dumps({'completed':name,'passed':passed,'errors':errors,'domain_failure_records':domain_count}),flush=True)
 after={str(p):digest(p) for p in sources}
 report={'scope':__doc__,'passed':all(row['passed'] for row in reports) and before==after,'model_revision':args.snapshot.name,'asset_sha256':assets,
 'manifest':manifest,'cases':reports,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
 'elapsed_s':time.perf_counter()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'versions':{name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','safetensors')},
 'limitations':['Only attention softmax uses the proposed polynomial/Newton circuit. QKV projections and remaining block dense/GELU/LayerNorm are exact plaintext oracles.',
 'Score compaction/scatter use sparse linear maps; literal rotation equivalence is separately tested, but no ciphertext map executes here.',
 'Abstract depth ignores CKKS rounding, scale alignment, refresh errors and implementation fusion. No secure arithmetic or full-model gate is established.',
 'Period32768 has two aliases; it is not covered by the period128 refresh proof.',
 'Three public pairs plus adversarial public masks and one long pair are not MRPC accuracy or certified unseen-input domains.']}
 (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 return 0 if report['passed'] and before==after else 1
if __name__=='__main__':raise SystemExit(main())
