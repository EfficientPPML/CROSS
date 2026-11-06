"""Fixed public Qwen toy5 diagnostic; caller must authenticate sources/grant first.

This module imports no NumPy/backend until run(). It never reads retained model,
key, ciphertext, calibration or private token data. No arithmetic is run at import.
"""
import hashlib
import json
import math
from pathlib import Path


def canonical(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def finite_json(value):
    if type(value)is float and not math.isfinite(value):
        return {'nonfinite':('nan' if math.isnan(value)else 'positive_infinity' if value>0 else 'negative_infinity')}
    if type(value)is dict:return {k:finite_json(v)for k,v in value.items()}
    if type(value)in(list,tuple):return [finite_json(v)for v in value]
    return value


def run(root):
    import numpy as np
    from literal_graph import build,selected
    from dense_reference import references
    root=Path(root)
    case=json.loads((root/'frozen/cadathlon/agent_pset/p4_qwen/test/test5.in').read_bytes())
    expected=np.asarray(json.loads((root/'frozen/cadathlon/agent_pset/p4_qwen/test/test5.out').read_bytes())['logits'],dtype=np.float64)
    profile=json.loads((root/'profile.json').read_bytes())
    config=case['config']
    if (case['weight_seed']!=9105 or np.shape(case['token_ids'])!=(2,4)
        or tuple(config[k] for k in ('hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','num_hidden_layers','vocab_size'))!=(16,24,4,2,8,3,32)):
        raise ValueError('fixed public toy5 only')
    ns=build(np,root)
    selected(root/'frozen/cadathlon/agent_pset/p4_qwen/src/reference.py',
             ['rmsnorm','rope','forward','toy_weights','make_case'],ns)
    selected(root/'frozen/cadathlon/common/support.py',['metrics'],ns)
    selected(root/'frozen/cadathlon/agent_pset/p4_qwen/src/runner.py',['check_output'],ns)
    weights,embeddings,native=ns['make_case'](case)
    native_fixture=bool(np.allclose(native,expected,atol=1e-10,rtol=1e-10))
    if not native_fixture:raise ValueError('original native reference fixture drift')
    rules=ns['public_toy_approximations']('qwen')
    # Overflow remains a recorded diagnostic; never clip values or replace rules.
    # Both paths stop at their first nonfinite stage and retain preceding records.
    with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
        dense=references(np,config,weights,embeddings,rules)
        layout=ns['PackedPairLayout'](2,4,ns['packed_feature_width']('qwen',config),65536)
        arithmetic=ns['Arithmetic'](profile,dense['records'])
        graph=ns['Graph'](layout,arithmetic)
        value=ns['Value'](layout.pack_client_embeddings(embeddings)[0],49,profile['scale'])
        complete=False;error_type=None;logits=None;result=None
        try:
            arithmetic.record('input','original_embeddings',value)
            result=graph.evaluate(value,'qwen',config,weights,rules)
            arithmetic.record('output','logits',result)
            logits=layout.unpack_client_features([result.slots],32)[:,-1,:]
            complete=True
        except (FloatingPointError,OverflowError) as error:
            error_type=type(error).__name__
    trace=arithmetic.trace
    compared=[row for row in trace if row['independent_dense_max_error']is not None]
    unpaired=sorted(set(dense['records'])-arithmetic.compared)
    maximum=max((row['independent_dense_max_error'] for row in compared),default=0.)
    periodicity=max((row['periodicity_max_error'] for row in trace),default=0.)
    # Fixed comparison diagnostics only; this is not an encrypted acceptance gate.
    L=config['num_hidden_layers']
    expected_counts={'linear':7*L+1,'normalize':2*L+1,'normalize_heads':2*L,'rope':2*L,'attention':L,'polynomial':7*L+1}
    expected_records=2+2*sum(expected_counts.values())+2*L
    count_agreement=arithmetic.counter==dense['counts']==expected_counts
    dense_agreement=bool(complete and dense['complete'] and not unpaired and maximum<=1e-10 and count_agreement and len(compared)==expected_records)
    phase=[row for row in trace if row['kind']=='phase' and row['label'].startswith('packed Qwen layer ')]
    checkpoints=[row for row in trace if row['kind'] in ('input','output','refresh_before','refresh_after')
                 or row in phase or (row['kind']=='normalize_after' and row['label']==str(2*config['num_hidden_layers']))]
    R=len(arithmetic.refresh_events)
    recurrence={'derived_refresh_count':R,'checkpoint_count':len(checkpoints),
                'complete_graph_expected_formula':'1 input + 6 layer phase ends + 2*R refresh boundaries + 1 final RMS + 1 output = 9+2R',
                'formula_holds':bool(complete and len(phase)==6 and len(checkpoints)==9+2*R),
                'observed_TPU_refresh_count':19,'matches_observed_count':bool(complete and R==19),
                'observed_count_used_to_construct_schedule':False}
    # Sparse1024/four-level transform path has no _collapsed_hoisted_plan call.
    if layout.period!=1024 or profile['level_budget']!=[4,4]:raise ValueError('fixed bootstrap planner branch')
    bootstrap=set(ns['_bootstrap_rotation_indices'](131072,layout.period,profile['level_budget']))
    fold={layout.period*(1<<i) for i in range(int(math.log2(layout.slots//layout.period)))}
    rotations=set(arithmetic.rotations)|bootstrap|fold|{2*131072-1}
    metrics=None;original_numeric_gate=None
    if complete:
        metrics=ns['metrics'](logits,expected)
        try:ns['check_output'](logits,expected,metrics);original_numeric_gate=True
        except ValueError:original_numeric_gate=False
    coeffs={name:{'domain':list(rule.domain),'chebyshev_coefficients':list(rule.coefficients),
                  'power_coefficients':(np.polynomial.chebyshev.cheb2poly(rule.coefficients)/(rule.domain[1]**np.arange(len(rule.coefficients))) if rule.input_power_basis else np.polynomial.chebyshev.cheb2poly(rule.coefficients)).tolist(),
                  'input_power_basis':rule.input_power_basis,'polynomial_depth':arithmetic.polynomial_depth(rule)}
            for name,rule in rules.items()}
    weight_hashes={name:{'shape':list(array.shape),'dtype':str(array.dtype),'sha256':hashlib.sha256(array.tobytes()).hexdigest()} for name,array in sorted(weights.items())}
    return finite_json({'schema':'qwen-toy5-public-polynomial-replay-v1','public_fixed_case_only':True,
            'complete':complete,'error_type':error_type,'dense_complete':dense['complete'],
            'dense_error_type':dense['error_type'],'literal_first_nonfinite':arithmetic.first_nonfinite,
            'dense_first_nonfinite':dense['first_nonfinite'],
            'diagnostic_complete':bool(complete and dense['complete']),'native_fixture_matches':native_fixture,
            'native_fixture_metrics':ns['metrics'](native,expected),
            'literal_matches_independent_dense_at_1e_10':dense_agreement,
            'maximum_semantic_absolute_error':maximum,'maximum_periodicity_absolute_error':periodicity,
            'compared_semantic_records':len(compared),'unpaired_dense_records':unpaired,
            'literal_domain_checks':arithmetic.domains,'dense_domain_checks':dense['domains'],
            'all_literal_domains_satisfied':all(row['passed'] for row in arithmetic.domains),
            'refresh_magnitude_checks':arithmetic.refresh_ranges,
            'all_refresh_magnitudes_satisfied':all(row['passed'] for row in arithmetic.refresh_ranges),
            'schedule_is_conditional_on_public_domains':True,'schedule':recurrence,
            'refresh_events':arithmetic.refresh_events,'trace':trace,'minimal_checkpoints':checkpoints,
            'primitive_counts':arithmetic.counts,'semantic_counts':arithmetic.counter,'dense_semantic_counts':dense['counts'],
            'derived_expected_semantic_counts':expected_counts,'derived_expected_semantic_records':expected_records,'semantic_count_agreement':count_agreement,
            'rotations':{'graph':sorted(arithmetic.rotations),'bootstrap':sorted(bootstrap),'fold':sorted(fold),
                         'conjugation':262143,'union':sorted(rotations),'union_count':len(rotations),
                         'complete_graph_inventory':complete,'not_actual_keygen_or_capacity_proof':True},
            'final_metadata':None if result is None else {'level':result.level,'scale':result._ckks_scale,'nsd':result._ckks_nsd},
            'fixed_original_numeric_metrics':metrics,'fixed_original_numeric_gate':original_numeric_gate,
            'numeric_gate_scope':'clear original-polynomial diagnostic only; never an encrypted acceptance result',
            'coefficients':coeffs,'public_weight_hashes':weight_hashes,
            'public_embedding_sha256':hashlib.sha256(embeddings.tobytes()).hexdigest(),
            'source_case_sha256':hashlib.sha256((root/'frozen/cadathlon/agent_pset/p4_qwen/test/test5.in').read_bytes()).hexdigest(),
            'packed_period':layout.period,'physical_slots':65536,'replicas':64,
            'derived_refresh_capacity':arithmetic.refresh_capacity,
            'actual_model':False,'backend_context':False,'key_generation':False,'HE':False,
            'refresh_model':'identity on plaintext; nominal scale and derived capacity only',
            'limitations':['No encoding, modular reduction, RNS, key switching, bootstrap approximation or CKKS noise.',
                           'Domain failure invalidates bound-based refresh justification even if continuation is finite.',
                           'Dense dot products and literal BSGS have distinct binary64 summation order.',
                           'No first encrypted corruption site follows from this replay or from completed progress messages.',
                           'No change to original coefficients, domains, decoder, acceptance thresholds or TPU queue.']})
