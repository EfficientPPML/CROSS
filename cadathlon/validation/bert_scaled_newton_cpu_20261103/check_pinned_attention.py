"""Pinned seven-case fixed-s4 Newton comparison; plaintext planning only."""
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
from scaled_softmax import ScaledNewtonSoftmax

def digest(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()

def candidate(q,k,v,mask,coefficients):
 sequence=len(q);tiles=(sequence+15)//16
 row=RowPackedAttention();outputs={name:np.zeros_like(q) for name in ('baseline','scaled')};reports=[]
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
  engines={'baseline':FixedSoftmax(coefficients,score_domain=(-16.,16.),squarings=5,iterations=13),
           'scaled':ScaledNewtonSoftmax(coefficients,score_domain=(-16.,16.),squarings=5,iterations=13)}
  probabilities={name:engine.evaluate(packed_scores,active,int(np.sum(mask))) for name,engine in engines.items()}
  exact=exact_probabilities(packed_scores,active)
  errors={name:float(np.max(np.abs(probability-exact))) for name,probability in probabilities.items()}
  comparison=float(np.max(np.abs(probabilities['scaled']-probabilities['baseline'])))
  numerators={name:np.zeros(65536) for name in engines}
  for kt in range(tiles):
   value=row.pack_rows(v,kt)
   for offset in range(16):
    moved_value=row.rotate_rows(value,offset)
    for name,probability in probabilities.items():
     anchors=maps[kt,offset].inverse(probability)
     numerators[name]+=row.head_broadcast(anchors)*moved_value
  count=min(16,sequence-qt*16)
  for name,numerator in numerators.items():
   outputs[name][qt*16:qt*16+count]=numerator.reshape(16,4096)[:count,:768]
  engine=engines['scaled'];baseline=engines['baseline']
  reports.append({'query_tile':qt,'probability_max_abs_error':errors['scaled'],
    'baseline_probability_max_abs_error':errors['baseline'],'probability_vs_unchanged':comparison,
    'fixed_domain_failures':engine.failures(),'baseline_domain_failures':baseline.failures(),
    'newton_bound_failures':engine.bound_failures(),
    'historical_narrow_domain_failures':{k:v for k,v in engine.failures(historical=True).items() if k.startswith('historical_')},
    'observations':engine.domains,'newton_intermediate_bounds':engine.intermediate_bounds,
    'public_newton_coefficients':engine.public_round_coefficients,
    'abstract_softmax_depth_with_distinct_mean_anchor_mask':engine.depth,
    'unchanged_baseline_abstract_depth_missing_mean_anchor_mask':baseline.depth,
    'arithmetic_counts_per_score_ciphertext':dict(engine.counts),
    'unchanged_baseline_arithmetic_counts':dict(baseline.counts)})
 return outputs,reports

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--snapshot',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
 args=parser.parse_args();args.output.mkdir(exist_ok=False)
 audit=load_module(ROOT/'cadathlon/validation/check_pretrained_bert_domains.py')
 reference=load_module(ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py')
 assets={name:digest(args.snapshot/name) for name in audit.EXPECTED_ASSETS};assert assets==audit.EXPECTED_ASSETS
 sources=[Path(__file__),Path(__file__).with_name('score_layout.py'),Path(__file__).with_name('scaled_softmax.py'),Path('/tmp/streaming-bert-layout/row_attention.py'),
 ROOT/'cadathlon/agent_pset/p2_bert/src/reference.py',ROOT/'cadathlon/validation/check_pretrained_bert_domains.py',ROOT/'cadathlon/common/encrypted_polynomial.py']
 before={str(p):digest(p) for p in sources};started=time.perf_counter()
 source_directory=args.output/'sources';source_directory.mkdir()
 for i,path in enumerate(sources):(source_directory/f'{i:02d}_{path.name}.txt').write_bytes(path.read_bytes())
 # Public analytic function constants, selected before reading model activations.
 approximation=nonlinear_approximation('exp',(-.5,.5),8)
 coefficients=np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
 grid=np.linspace(-.5,.5,20001);exp_error=float(np.max(np.abs(np.polynomial.polynomial.polyval(2*grid,coefficients)-np.exp(grid))))
 rho=(2.-.5/128)/(2.+.5/128);newton_relative_bound=rho**(2**13)
 assert exp_error<1e-10 and newton_relative_bound<1e-12
 manifest={'normalized_newton':{'s':4.,'C':'public_valid_count/4','target':'z=4/(K*x)','first_mean_exp_reciprocal':'unchanged','round_iterations':13,'declared_z_upper':8.,'declared_xz_upper':'4/K','public_product_upper':1.,'correction_interval':[1.,2.],'final_unscaled_product_upper':'4/K','binary64_range_diagnostic_slack':'64*machine_epsilon*max(1,abs(interval endpoints)); no CKKS interpretation'},'centered_score_domain':[-16.,16.],'domain_scope':'Fixed conditional domain selected from the preceding public full-row audit, not a guaranteed bound on unseen inputs.',
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
  contexts,tile_reports=candidate(q,k,v,mask[0],coefficients);context=contexts['scaled'][None];baseline_context=contexts['baseline'][None]
  post=reference.layer_norm(x+reference.linear(context,weights,prefix+'attention.output.dense'),weights,prefix+'attention.output.LayerNorm',config['layer_norm_eps'])
  intermediate=reference.gelu(reference.linear(post,weights,prefix+'intermediate.dense'))
  actual=reference.layer_norm(post+reference.linear(intermediate,weights,prefix+'output.dense'),weights,prefix+'output.LayerNorm',config['layer_norm_eps'])
  baseline_post=reference.layer_norm(x+reference.linear(baseline_context,weights,prefix+'attention.output.dense'),weights,prefix+'attention.output.LayerNorm',config['layer_norm_eps'])
  baseline_intermediate=reference.gelu(reference.linear(baseline_post,weights,prefix+'intermediate.dense'))
  baseline_actual=reference.layer_norm(baseline_post+reference.linear(baseline_intermediate,weights,prefix+'output.dense'),weights,prefix+'output.LayerNorm',config['layer_norm_eps'])
  expected=reference.encoder_block(config,weights,x,mask,0)
  with torch.no_grad():
   native_x=torch.from_numpy(x.copy());native_mask=torch.from_numpy(mask.copy())
   additive=torch.zeros(native_mask[:,None,None,:].shape,dtype=torch.float64).masked_fill(~native_mask[:,None,None,:],torch.finfo(torch.float64).min)
   native=model.bert.encoder.layer[0](native_x,attention_mask=additive)[0].numpy()
   native_context=model.bert.encoder.layer[0].attention.self(native_x,attention_mask=additive)[0].numpy()
  errors={'probability_vs_unchanged':max(row['probability_vs_unchanged'] for row in tile_reports),
   'context_vs_unchanged':float(np.max(np.abs(context-baseline_context))),
   'full_block_vs_unchanged':float(np.max(np.abs(actual-baseline_actual))),
   'probability_vs_exact':max(row['probability_max_abs_error'] for row in tile_reports),
   'context_vs_transformers':float(np.max(np.abs(context-native_context))),
   'full_block_vs_numpy':float(np.max(np.abs(actual-expected))),
   'full_block_vs_transformers':float(np.max(np.abs(actual-native)))}
  domain_count=sum(len(row['fixed_domain_failures']) for row in tile_reports)
  bound_count=sum(len(row['newton_bound_failures']) for row in tile_reports)
  baseline_pass=all(not row['baseline_domain_failures'] and row['baseline_probability_max_abs_error']<1e-8 for row in tile_reports)
  passed=bool(np.isfinite(actual).all() and errors['probability_vs_exact']<1e-8 and all(value<1e-6 for key,value in errors.items() if key!='probability_vs_exact') and domain_count==0 and bound_count==0 and baseline_pass and errors['probability_vs_unchanged']<1e-8)
  reports.append({'case':name,'sequence':int(x.shape[1]),'valid_keys':int(mask.sum()),'passed':passed,'errors':errors,
   'domain_failure_records':domain_count,'newton_bound_failure_records':bound_count,'unchanged_baseline_passed':baseline_pass,'tiles':tile_reports,'padded_query_output_max_abs':float(np.max(np.abs(actual[~mask]))) if np.any(~mask) else None})
  print(json.dumps({'completed':name,'passed':passed,'errors':errors,'domain_failure_records':domain_count}),flush=True)
 after={str(p):digest(p) for p in sources}
 report={'scope':__doc__,'passed':all(row['passed'] for row in reports) and before==after,'model_revision':args.snapshot.name,'asset_sha256':assets,
 'manifest':manifest,'cases':reports,'source_hashes_before':before,'source_hashes_after':after,'source_files_unchanged':before==after,
 'elapsed_s':time.perf_counter()-started,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 'versions':{name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','safetensors')},
 'limitations':['Only attention softmax uses the proposed polynomial/Newton circuit. QKV projections and remaining block dense/GELU/LayerNorm are exact plaintext oracles.',
 'Score compaction/scatter execute factored public-mask and rotation networks in NumPy; no ciphertext map executes here.',
 'Abstract depth ignores CKKS rounding, scale alignment, refresh errors and implementation fusion. No secure arithmetic or full-model gate is established.',
 'The exact normalized-Newton bounds rely on correlated positive values. Binary64 diagnostics do not prove CKKS/refresh stability or unseen-input domains.',
 'The abstract depth is232=227 reviewed-unfused baseline+five final public scales; no actual HE schedule or precision claim.',
 'The score layout has period65536 and no alias averaging; it is not covered by the period128 refresh proof.',
 'Three public pairs plus adversarial public masks and one long pair are not MRPC accuracy or certified unseen-input domains.']}
 (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 return 0 if report['passed'] and before==after else 1
if __name__=='__main__':raise SystemExit(main())
