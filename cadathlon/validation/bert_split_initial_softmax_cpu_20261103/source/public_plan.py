"""Public split-softmax level/scale inventory, with no context or keys."""
from pathlib import Path
import hashlib,json,math,sys
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE/'checkout/cadathlon/common'))
from encrypted_bert_split_initial import LayerInitialContract,prefix_scale_plan,tail_scale_plan
from encrypted_score_reduction import ScoreReductionLayout
def build():
    raw=(HERE/'candidate_manifest.json').read_bytes()
    profile=json.loads((HERE/'p60_profile_and_refresh.json').read_text());p=profile['profile']
    q=p['q'];scale=p['canonical_scale'];layers={}
    for layer in range(12):
        contract=LayerInitialContract(raw,layer)
        variations={}
        for name,actual in (('canonical',scale),('one_ulp_below',math.nextafter(scale,-math.inf)),
                            ('one_ulp_above',math.nextafter(scale,math.inf))):
            variations[name]=dict(prefix=prefix_scale_plan(q[:44],input_level=21,canonical_scale=scale,input_scale=actual),
                tail=tail_scale_plan(q[:52],input_level=25,canonical_scale=scale,input_scale=actual))
        layers[str(layer)]=dict(centered_domain=contract.centered_domain,squarings=contract.squarings,
            normalized_divisor=contract.normalized_divisor,normalized_domain=contract.normalized_domain,
            base_exp_coefficient_sha256=hashlib.sha256(__import__('numpy').asarray(contract.exponential_coefficients,dtype='<f8').tobytes()).hexdigest(),
            initial_domain=contract.initial_domain,initial_seed=contract.initial_seed,initial_updates=contract.initial_updates,
            scales=variations)
    graph=set(ScoreReductionLayout.rotation_indices());bootstrap=set(profile['refresh_plan']['rotations']);union=graph|bootstrap
    pair_bytes=8*2*131072*(100+14)*4
    return dict(passed=True,scope=__doc__,manifest_sha256=hashlib.sha256(raw).hexdigest(),layers=layers,
        public_K_values=list(range(1,129)),all1536_layer_K_metadata_schedules_covered=True,
        full_slot_arithmetic_cases=384,distinct_divisor_layers=[0,4,1],all12_endpoint_sites_checked=True,
        prefix_levels=dict(input=21,mean=19,normalized_t=18),refresh=dict(public_bound=2.,output_level=25),
        tail_levels=dict(input=25,base_exp=21,mean_exp=19,inverse=8,product=7,probabilities=6),
        counts_excluding_refresh=dict(logical_rotations=28,ciphertext_products=18,plaintext_terms=16,
            composite_rescales=26,explicit_refresh_calls=1),graph_keys=sorted(graph),bootstrap_keys=sorted(bootstrap),union_keys=sorted(union),
        raw_rotation_pair_bytes=pair_bytes,raw_union_key_bytes=len(union)*pair_bytes,
        encrypted_execution=False,actual_refresh=False,model_forward=False,selected_coefficients_changed=False,
        tail_recenters=False,extra_pre_power_mask=False,
        limitations=['Valid centered scores and mapped t domains are public caller obligations.',
            'Every refreshed slot uses the B2 range/error contract; padding is not assumed to remain zero.',
            'All coefficient and bias masks retain their original primitive order; no extra pre-power PT/rescale is hidden.',
            'The six static inverse updates retain their original first-affine encoding target; actual scales are replayed.',
            'This component omits the selected5/6/7 later squaring rounds, attention and model/file-stage work.',
            'No CKKS error, refresh accuracy, full-model readiness, latency or peak-memory inference follows.'])
