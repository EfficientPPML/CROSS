"""Canonical-input public scale hypothesis; no ciphertext, key, encoding or accuracy evidence."""
from pathlib import Path
import hashlib,json,math
from tracked_reciprocal import tracked_scale_plan,_forward
from encrypted_bert_normalization_round import NormalizationRoundContract,normalization_round_plan

HERE=Path(__file__).resolve().parent
FAILED_SHA='f97f839d23a2d91e258f8937a92bc8173ebd1bc77483db803a100531c8985b45'
AUDIT_SHA='2b2457e50da73b9ccf87984c089afcb99586fde863a48e98b89a1b7ccf872a17'
MANIFEST_SHA='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'

def public_inputs():
    raw=(HERE/'failed_report.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=FAILED_SHA:raise ValueError('fixed failed evidence differs')
    failed=json.loads(raw)
    if failed['passed']is not False or failed['complete']is not True or failed['fixed_absolute_error_gate']!=1e-6:
        raise ValueError('original failure and gate required')
    pins=json.loads((HERE/'predecessor_pins.json').read_bytes())
    for name,h in pins.items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest()!=h:raise ValueError('predecessor source differs')
        source=name if name.endswith('.json') else 'checkout/cadathlon/common/'+name
        if name!='failed_report.json' and failed['source_hashes_before'].get(source)!=h:raise ValueError('unbound predecessor source')
    audit=(HERE/'actual_profile_audit.json').read_bytes();manifest=(HERE/'candidate_manifest.json').read_bytes()
    if hashlib.sha256(audit).hexdigest()!=AUDIT_SHA or hashlib.sha256(manifest).hexdigest()!=MANIFEST_SHA:
        raise ValueError('fixed public profile/manifest required')
    ring=next(r for r in json.loads(audit)['profiles']if r['name']=='depth49_p60_d8')
    q,p=ring['q'],ring['p'];N=ring['degree']
    if (N,len(q),len(p),ring['dnum'])!=(131072,100,14,8):raise ValueError('common ring differs')
    if len(set(q+p))!=114 or any(type(v)is not int or v%(2*N)!=1 for v in q+p):raise ValueError('ordered actual moduli differ')
    if math.prod(q+p).bit_length()>3523:raise ValueError('actual public security ceiling exceeded')
    alpha=math.ceil(len(q)/8)
    if any(math.prod(q[i:i+alpha])>math.prod(p)for i in range(0,len(q),alpha)):raise ValueError('P coverage failed')
    return ring,manifest,failed

def plan(ring,manifest,K,gain):
    if type(gain)is not int or gain not in (1,2,4,8):raise ValueError('reviewed power-of-two mask scale gain required')
    contract=NormalizationRoundContract(manifest,K)
    q=ring['q'];delta=ring['canonical_scale'];L=25
    def divisor(level):return q[2*level]*q[2*level+1]
    def affine(source,target,level):
        encode=target*divisor(level)/source
        return dict(input_scale=source,target_scale=target,plaintext_scale=encode,
                    output_scale=source*encode/divisor(level),input_level=level,output_level=level-1)
    mask=affine(delta,gain*delta,L)
    numerator=mask['output_scale']*mask['output_scale']/divisor(L-1)
    # Seven rotations sum at numerator scale; after this real anchor affine,
    # seven rotations broadcast at the high denominator scale.
    anchors=affine(numerator,(gain*gain)*delta,L-2)
    denominator_level=L-3;inverse_level=denominator_level-contract.updates-1
    # Match the original target-selection order at gain1, then jointly solve
    # y and its separate residual branch. No wrapper metadata is reassigned.
    nominal_numerator=(gain*delta)*(gain*delta)/divisor(L-1)
    target=delta*divisor(inverse_level)/nominal_numerator
    nominal=tracked_scale_plan(q[:2*(denominator_level+1)],input_scale=(gain*gain)*delta,
        input_level=denominator_level,updates=contract.updates,final_scale=target)
    inverse=_forward(q[:2*(denominator_level+1)],input_scale=anchors['output_scale'],
        input_level=denominator_level,updates=contract.updates,first_output_scale=nominal['first_output_scale'])
    output=numerator*inverse['rows'][-1]['scale']/divisor(inverse_level)
    scales=[mask['output_scale'],numerator,anchors['output_scale'],output]
    scales.extend(v for row in inverse['rows']for k,v in row.items()if k in ('scale','residual_scale'))
    if not all(math.isfinite(v)and v>0 for v in scales):raise ValueError('finite positive metadata required')
    if not math.isclose(output,delta,rel_tol=1e-12):raise ValueError('next-round canonical tolerance failed')
    return dict(K=K,gain=gain,updates=contract.updates,mask=mask,numerator_scale=numerator,
        anchors=anchors,inverse=inverse,inverse_level=inverse_level,output_level=inverse_level-1,
        output_scale=output,output_relative_scale_error=output/delta-1,
        minimum_y_scale_over_canonical=min(r['scale']/delta for r in inverse['rows']),
        minimum_residual_scale_over_canonical=min(r['residual_scale']/delta for r in inverse['rows']if 'residual_scale'in r),
        maximum_residual_scale_over_canonical=max(r['residual_scale']/delta for r in inverse['rows']if 'residual_scale'in r),
        counts=dict(ciphertext_products=2*contract.updates,plaintext_products=3,rotations=14,
                    composite_rescales=2*contract.updates+3,critical_levels=contract.updates+5),
        real_polynomial_unchanged=True,encoding_schedule_changed=gain!=1,metadata_relabeling=False)

def conditional_sensitivity(failed,gain,unreduced_fraction):
    """Illustrative noise allocation, explicitly NOT a bound on the new circuit."""
    name='K128_rows16';d=failed['checks'][name+':denominator'];derived=failed['derived_checks'][name]
    E=d['max_abs_error'];m=d['decoded_minimum'];e=failed['checks'][name+':e13']
    internal=(derived['invariant13']['max_abs_error']+max(abs(e['decoded_minimum']),abs(e['decoded_maximum'])))/m
    assumed_E=E*(unreduced_fraction+(1-unreduced_fraction)/(gain*gain))
    return dict(gain=gain,assumed_unreduced_error_fraction=unreduced_fraction,
        assumed_denominator_error=assumed_E,
        illustrative_inverse_error=assumed_E/(m*(m-assumed_E))+internal*(gain*gain),
        assumption='Remaining denominator noise falls by gain squared; inverse internal term grows by gain squared. Neither assumption is established by HE evidence.')

def derive():
    ring,manifest,failed=public_inputs();rows=[plan(ring,manifest,K,g)for K in range(1,129)for g in (1,2,4,8)]
    return dict(passed=True,scope=__doc__,schedules=rows,security=dict(actual_QP_bit_length=math.prod(ring['q']+ring['p']).bit_length(),classical128_ceiling=3523,P_coverage=True),
        sensitivity=[conditional_sensitivity(failed,g,rho)for g in (1,2,4,8)for rho in (0.,.25,.5,1.)],
        limitations=['No encoded plaintext, ciphertext, backend, key generation, or performance measurement.',
            'Input scale is fixed to the canonical value; actual adjacent-scale refresh or previous-output metadata needs a separate literal replay.',
            'Higher scales cannot recover information already lost in the input ciphertext.',
            'The jointly solved y/residual scales preserve the real polynomial but change encoded bytes and CKKS error.',
            'Backend encoding resolution, coefficient headroom, callback/lifecycle guards and all fixed numerical gates need a separate implementation and genuine proof.',
            'Sensitivity entries are conditional illustrations, not noise bounds or a repaired-run prediction.'])

if __name__=='__main__':print(json.dumps(derive(),indent=2,allow_nan=False))
