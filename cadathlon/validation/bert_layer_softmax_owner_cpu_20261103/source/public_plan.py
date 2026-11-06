"""Public twelve-layer softmax schedules; no CKKS context, keys, or model forward."""
from pathlib import Path
import hashlib,json,math,sys
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'checkout/cadathlon/common'))
from encrypted_bert_split_initial import LayerInitialContract,prefix_scale_plan,tail_scale_plan
from encrypted_bert_initial_softmax import initial_softmax_plan
from encrypted_bert_normalization_round import NormalizationRoundContract,normalization_round_plan
from encrypted_score_reduction import ScoreReductionLayout
from encrypted_scale_planning import weighted_sum_scale,drop_divisor
from encrypted_polynomial import EncryptedPolynomialEvaluator,PolynomialApproximation

def build():
    raw=(HERE/'candidate_manifest.json').read_bytes();manifest=json.loads(raw)
    profile=json.loads((HERE/'p60_profile_and_refresh.json').read_text());p=profile['profile'];q=p['q'];scale=p['canonical_scale']
    layers={}
    for layer in range(12):
        contract=LayerInitialContract(raw,layer);norm=manifest['norms'][str(layer)+'.attention'];poly=norm['polynomial']
        strategy={'power':'power','balanced_chebyshev':'chebyshev'}[norm['method']]
        norm_depth=5+EncryptedPolynomialEvaluator.polynomial_depth(SimpleNamespace(strategy=strategy),
            PolynomialApproximation('inverse_sqrt',tuple(poly['domain']),tuple(poly['coefficients'])))
        cases={}
        for K in range(1,129):
            rule=NormalizationRoundContract(raw,K);variants={}
            for name,start in (('canonical',scale),('one_ulp_below',math.nextafter(scale,-math.inf)),('one_ulp_above',math.nextafter(scale,math.inf))):
                if layer==0:
                    initial=initial_softmax_plan(q[:92],input_level=45,canonical_scale=scale,input_scale=start)
                    prefix=None;level=23
                else:
                    prefix=prefix_scale_plan(q[:44],input_level=21,canonical_scale=scale,input_scale=start)
                    initial=tail_scale_plan(q[:52],input_level=25,canonical_scale=scale,input_scale=scale);level=6
                actual=initial['scales']['probabilities'];rounds=[];refreshes=int(layer!=0)
                for index in range(contract.squarings):
                    before=level;refresh=level<rule.updates+7
                    if refresh:level=25;actual=scale;refreshes+=1
                    plan=normalization_round_plan(q[:2*(level+1)],input_level=level,canonical_scale=scale,contract=rule,input_scale=actual)
                    level=plan['levels']['probabilities'];actual=plan['scales']['probabilities']
                    rounds.append(dict(index=index,preceding_level=before,refreshed=refresh,input_level=plan['input_level'],
                        output_level=level,input_scale=plan['input_scale'],output_scale=actual))
                assert level<norm_depth+6
                final=weighted_sum_scale([scale],scale,drop_divisor(q,25))['output_scale']
                variants[name]=dict(input_level=45 if layer==0 else 21,input_scale=start,
                    initial_prefix_levels=None if prefix is None else prefix['levels'],
                    initial_output_level=23 if layer==0 else 6,initial_output_scale=initial['scales']['probabilities'],
                    rounds=rounds,final_refresh_input_level=level,final_refresh_output_level=25,
                    final_mask_output_level=24,final_output_scale=final,explicit_B2_refresh_count=refreshes+1)
            s=contract.squarings;n=rule.updates
            cases[str(K)]=dict(updates_per_round=n,denominator_domain=rule.domain,
                counts_excluding_refresh=dict(rotations=28+14*s,ciphertext_products=18+2*n*s,
                    plaintext_terms=17+3*s,composite_rescales=27+(2*n+3)*s),scale_variants=variants)
        layers[str(layer)]=dict(centered_domain=contract.centered_domain,normalized_divisor=contract.normalized_divisor,
            normalized_domain=contract.normalized_domain,selected_squarings=contract.squarings,
            attention_norm_depth=norm_depth,continuation_after_final_mask=norm_depth+6,cases=cases)
    graph=set(ScoreReductionLayout.rotation_indices());bootstrap=set(profile['refresh_plan']['rotations']);union=graph|bootstrap
    pair_bytes=8*2*131072*(100+14)*4
    return dict(passed=True,scope=__doc__,manifest_sha256=hashlib.sha256(raw).hexdigest(),layers=layers,
        public_cases=1536,actual_scale_variants_per_case=3,full_slot_ideal_cases=396,
        graph_keys=sorted(graph),bootstrap_keys=sorted(bootstrap),union_keys=sorted(union),
        raw_rotation_pair_bytes=pair_bytes,raw_union_key_bytes=len(union)*pair_bytes,
        encrypted_execution=False,actual_refresh=False,model_forward=False,selected_coefficients_changed=False,
        limitations=['Public centered-score and mapped-t domains remain caller assumptions.',
            'B2 refreshed probabilities and normalized-t require distinct full-slot range/error evidence.',
            'Repeated refresh and CKKS noise do not inherit the ideal probability correlation or denominator bounds.',
            'Invalid refreshed slots may be nonzero; original masks and mandatory final real PT mask are retained.',
            'No complete attention, file-stage, twelve-layer HE, latency, or memory readiness follows.'])
