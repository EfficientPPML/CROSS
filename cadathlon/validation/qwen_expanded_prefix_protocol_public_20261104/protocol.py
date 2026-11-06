"""Public design only: no tokenizer, calibration body, model or selection run."""
from pathlib import Path
from fractions import Fraction
import hashlib,json,math
P=Path(__file__).resolve().parent

def digest(raw):return hashlib.sha256(raw).hexdigest()
def pinned(name):
 pins=json.loads((P/'input_pins.json').read_bytes());raw=(P/'inputs'/name).read_bytes()
 if digest(raw)!=pins[name]['sha256']:raise ValueError('public design input hash differs')
 return json.loads(raw)

def workload():
 rows=[]
 for prompt in range(96):
  for length in range(1,33):rows.append(dict(kind='prefill',prompt_ordinal=prompt,length=length,original416_member=length in(8,16,32)))
 for kind in ('teacher','free'):
  for seed in range(16):
   for step in range(4):rows.append(dict(kind=kind,seed_ordinal=seed,step=step,length=16+step,original416_member=True))
 return rows

def sites():
 norm=[f'{i}.{site}.inverse_sqrt' for i in range(28)for site in ('input','post','q','k')]+['final.inverse_sqrt']
 return dict(normalization=norm,silu=[f'{i}.silu' for i in range(28)],centered_softmax=[str(i)for i in range(28)])

def uniform_domain(minimum,maximum,epsilon):
 # This function is a policy demonstration for synthetic tests, not a selector.
 if any(type(x)is not float or not math.isfinite(x)for x in (minimum,maximum,epsilon))or not 0<epsilon<=minimum<=maximum:raise ValueError('finite positive variance observations required')
 lo=max(epsilon,(.8*minimum)/4);hi=(1.2*maximum)*4
 if not math.isfinite(hi)or hi<=lo:raise ValueError('unsupported domain')
 return [lo,hi]

def required_matches(n):
 if type(n)is not int or n<=0:raise ValueError('positive exact count required')
 return (9*n+9)//10

def context_accounting(calibration,validation,other_excluded=()):
 # Synthetic/public tokenizer-free seam only; actual token bodies are not read
 # by this proposal. A future reviewed tokenizer collector supplies the arrays.
 def sequences(rows):
  out=[]
  for row in rows:
   if type(row)not in(list,tuple)or not row or any(type(x)is not int or x<0 for x in row):raise ValueError('exact nonnegative integer token sequences required')
   out.append(tuple(row))
  return out
 a,b,c=map(sequences,(calibration,validation,other_excluded))
 def counts(rows):return {str(n):dict(entries=sum(len(x)==n for x in rows),unique=len({x for x in rows if len(x)==n}))for n in sorted({len(x)for x in rows})}
 excluded=set(a)|set(c)|{(0,)};overlap=set(b)&excluded
 return dict(calibration=counts(a),validation=counts(b),calibration_entries=len(a),validation_entries=len(b),
  overlap_by_length={str(n):sum(len(x)==n for x in overlap)for n in sorted({len(x)for x in overlap})},
  disjoint=not overlap,denominators_preserved=True)

def derive():
 old=pinned('original_protocol.json');selection=pinned('candidate3_selection_protocol.json');base=pinned('candidate3_manifest.json');requirements=pinned('original_calibration_requirement.json');validation=pinned('untouched_validation_design.json')
 inventory=sites();assert set(inventory['normalization']+inventory['silu'])==set(base['nonlinear']) and len(base['softmax'])==28
 w=workload();assert len(w)==3200 and sum(x['original416_member']for x in w)==416
 assert selection['degree_ladder']==[16,32,64,128,256,512]
 gates=dict(old['gates']);gates['prefill_top1_aggregation']='Aggregate all declared prefixes; report each length separately, no new per-length acceptance gate.'
 gates['teacher_forced_required_matches']='ceil(0.9*decisions);64 calibration requires58;32 new validation requires29'
 return dict(schema='qwen-expanded-prefix-protocol-proposal-v1',status='public_design_only_root_review_required',
  execution_authorized=False,tokenizer_or_model_or_selection_executed=False,
  distinct_from_original_three_candidate_protocol=True,old_candidate_manifest_sha256= digest((P/'inputs/candidate3_manifest.json').read_bytes()),
  informed_by_prior_failure='Known frozen row0 final-RMS failure motivated prefix coverage. Its3391.698 variance and every old validation outcome are excluded from new domain/coefficient selection. This is not an untouched redesign.',
  model_revision=old['model_revision'],calibration=dict(prompt_source='Exactly the original32topics x3framings=96 prompts; original text bytes remain externally pinned and unopened here.',
   input_sha256=requirements['calibration_input_sha256'],prefill_prefixes=list(range(1,33)),prefill_contexts=3072,teacher_contexts=64,free_contexts=64,contexts=3200,
   original416_subset=dict(prefill_lengths=[8,16,32],prefill_contexts=288,teacher_contexts=64,free_contexts=64,total=416),
   construction='From each original S32 token prefix take its exact firstL tokens for L1..32; require exact equality with original S8/S16 entries and original seed mapping before any model reads. No new tokenization of calibration text or choice by activation ranges.',
   decode='Same16 original seeds, original native teacher prefixes and actual approximate free prefixes; evaluate native/exact/approximate on each actual shared prefix; preserve stopped suffixes and denominator3200.',
   workload=w,entry_scope='3200 entries, not3200 unique sequences. Preserve every declared entry and denominator even when prefixes repeat. Tokenizer freeze reports within-calibration and within-validation entry/unique counts by length and never deduplicates.'),
  observation=dict(selection_sites=inventory,selection_site_count=169,normalization_count=113,silu_count=28,centered_softmax_count=28,
   full_metrics=dict(operators_per_context=421,blocks_per_context=28,last_position_logits=151936),
   propagation='Observe every declared nonlinear input, each causalK1..32, every initial/round denominator, and each rational-SiLU tanh base/denominator. Stop before any out-of-domain evaluation; never clip or skip domain checks. Native/exact ranges stay distinct from approximate ranges.'),
  selection=dict(scope='Only all113 normalization rules may change under one policy. All28 SiLU and28 softmax rules, exp8, reciprocal domains/counts and their acceptance gates remain byte-identical to candidate3.',
   positive_domain='Expanded-calibration union vmin/vmax from native float32 and independent exact float64, all113 sites: base=[max(epsilon,0.8*vmin),1.2*vmax], then[max(epsilon,base_lo/4),base_hi*4]. Same binary64 ordering for every site; no exceptions, label/error fitting, row0 or validation extrema.',
   failed_candidate_revision='At most3 new-protocol candidates. After a complete failed approximate calibration, union only observed valid-prefix calibration nonlinear inputs with the original native/exact ranges, then reapply the same whole-site rule/ladder. A rejected input may be recorded as an observation without evaluating its invalid operator. Preserve all stopped suffixes; output errors cannot fit coefficients.',
   degree_ladder=selection['degree_ladder'],strategy='balanced Chebyshev only',coefficient_algorithm=selection['coefficient_algorithm'],uniform_relative_target=1e-6,
   uniform_certificate=selection['uniform_certificate'],binary64_diagnostic=selection['binary64_diagnostic'],
   fallback='Smallest static Newton count <=64 with exact rounded initial-affine residual certificate and same1e-6 relative bound, plus fixed2049 linear/geometric binary64 grids. This must be regenerated from each new public interval; do not reuse old counts for new endpoints. Unsupported certificate/overflow/instability fails the whole candidate.',
   protected_rule_checks='Observe all protected SiLU/softmax domains in all3200 trajectories and independently recompute conditional public SiLU radii from new norm relative certificates and pinned learned gate/gamma bounds. Any protected-domain/radius failure marks candidate unsupported; no automatic protected-rule retuning.',
   preserve_all_attempts=True,maximum_candidates=3,freeze='Freeze all113 rules/full candidate/provenance before any approximate calibration. Freeze the first fully calibration-passing candidate before any new validation or row0 measurement.'),
  acceptance=dict(unchanged_numeric_gates=gates,
   calibration=dict(prefill_required_matches=required_matches(3072),prefill_total=3072,teacher_required_matches=58,teacher_total=64,complete_free_required=12,complete_free_total=16,complete_contexts=3200),
   untouched_validation=dict(prefill_required_matches=required_matches(1024),prefill_total=1024,teacher_required_matches=29,teacher_total=32,complete_free_required=6,complete_free_total=8,complete_contexts=1088),
   failure='Complete coverage and all numeric/semantic/domain gates are mandatory. Count every failed or stopped context in its declared denominator; no partial pass.'),
  untouched_validation=dict(design_sha256=digest((P/'inputs/untouched_validation_design.json').read_bytes()),design=validation,
   measured=False,tokenized=False,token_disjointness_verified=False,
   prerequisite='Freeze exact new tokenized bodies and hashes BEFORE expanded calibration results/selection. Authorize a tokenizer-only check separately. Require every planned new validation token sequence to differ from the EXPANDED3072 calibration prefix inventory at allL1..32, original teacher/free decode contexts, old calibration/validation/fresh token contexts, and explicit row0 tokenID0 S1. Report exact-sequence intersections and entry/unique counts by length; comparison is by full token sequence, not text or prompt identity; if this fails, preserve the design and freeze a separately reviewed revision before any new model run. Do not silently replace prompts.',
   no_feedback='New validation and row0 are opened only after final candidate freeze. Their failures cannot feed another selection using this protocol. Old published/fresh112 are optional regression only, never newly untouched validation.'),
  row0_regression=dict(source='Exact existing checkpoint embedding row0/tokenID0 S1, original position0, same source binding. Generic first-S8 calibration provenance does not establish this row0 fixture.',excluded_from_selection=True,when='After first complete calibration-passing candidate is frozen; preserve outcome under unchanged operator/logit/domain gates.',
   original_failure_remains=True,no_targeted_domain_widening=True),
  process=['Root reviews public design and scope. No input/body or model reads authorized by this artifact.',
   'Separate tokenizer-only construction freezes all original prefix mappings and new untouched validation token bodies/disjointness before any expanded results.',
   'Source-freeze native/exact observer and full3200 workload; run only after separate grant; record original416 subset separately.',
   'Perform uniform public selection/certification under this new protocol; freeze complete manifest before approximate3200.',
   'After full calibration pass, evaluate frozen candidate on new1088 validation and row0 regression without selecting a new candidate from their results.'],
  costs=dict(expanded_calibration_entries=3200,original_entries=416,entry_multiplier_exact=[100,13],maximum_polynomial_attempts_per_candidate=113*6,
   caution='Sequence work is not proportional to entry count. High-degree exact certificates, binary64 stability, inverse-only refresh placement and eventual HE cost must be rederived; prior93 reserve gaps and12931S32 refresh plan do not transfer automatically.'),
  source_scope='Only public protocol/manifests/certificate/arithmetic source and a newly authored validation design; no original calibration/old validation/new token bodies, checkpoint/model/HE access.')

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();report=derive()
 with a.output.open('x')as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
 print(json.dumps(dict(calibration=3200,original_subset=416,proposed_validation=1088,execution_authorized=False)))
