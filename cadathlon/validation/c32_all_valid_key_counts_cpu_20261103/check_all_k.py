"""Public conditional C32 Newton certificates; no CKKS, refresh, or softmax execution."""
from pathlib import Path
from fractions import Fraction as F
from decimal import Decimal, localcontext, ROUND_CEILING
from collections import Counter
import hashlib,json,math,tarfile,time
import numpy as np
from public_schedule import PublicSchedule

HERE=Path(__file__).resolve().parent
C=32
GATE=F(1,10**12)


def upper_decimal_power(value,squarings):
    assert value>=0
    with localcontext() as ctx:
        ctx.prec=100;ctx.rounding=ROUND_CEILING
        out=Decimal(value.numerator)/Decimal(value.denominator)
        for _ in range(squarings):out=out*out
    return out


def outward(value):
    result=float(value)
    if F.from_float(result)<value:result=float(np.nextafter(result,np.inf))
    return result


def lower_outward(value):
    result=float(value)
    if F.from_float(result)>value:result=float(np.nextafter(result,-np.inf))
    return result


def validate_public_dummy(value,a,b):
    # This backend-free check must run before any ciphertext method/key setup.
    if isinstance(value,(bool,np.bool_)) or not isinstance(value,(int,float,np.integer,np.floating)):
        raise ValueError('public dummy must be a positive finite real scalar')
    value=float(value)
    if not (math.isfinite(a) and math.isfinite(b) and 0<a<b):
        raise ValueError('declared positive finite domain required')
    if not math.isfinite(value) or not a<=value<=b:
        raise ValueError('public dummy outside its declared domain')
    return value


def first_residual_bounds(a,b,weight,bias):
    # Actual rounded public affine z1=w*d+b is treated as an exact real
    # polynomial. Its residual1-C*d*z1 is convex becausew<0.
    def r(x):return 1-C*x*(weight*x+bias)
    points=[a,b]
    vertex=-bias/(2*weight)
    if a<=vertex<=b:points.append(vertex)
    values=list(map(r,points))
    return min(values),max(values)


def run():
    started=time.perf_counter()
    paths=[HERE/name for name in ('check_all_k.py','public_schedule.py','baseline_manifest.json','reviewed_integer_newton.py')]
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    before={p.name:digest(p) for p in paths}
    baseline=json.loads((HERE/'baseline_manifest.json').read_text())
    schedule=PublicSchedule(baseline['power_coefficients_in_t_2x'])
    rows=[];hist=Counter();unsafe_dummy=[];rounded_bound_cases=[];dummy_negative_checks=0
    for k in range(1,129):
        for stage in range(5):
            contract=schedule.stage(k,stage)
            a,b=contract['a'],contract['b'];af,bf=F.from_float(a),F.from_float(b)
            n=contract['iterations'];hist[n]+=1
            assert n>=2
            # Do not round K/4 or reuse the historical K128 seed. C32 is fixed;
            # s=K/32 is descriptive, not an encoded/restored runtimeconstant.
            seed=2./(C*(a+b))
            weight=float((-C*seed)*seed);bias=float(2.*seed)
            wf,betaf=F.from_float(weight),F.from_float(bias)
            rlo,rhi=first_residual_bounds(af,bf,wf,betaf)
            first_abs=max(abs(rlo),abs(rhi))
            assert first_abs<1
            exact_rounded_residual=upper_decimal_power(first_abs,n-1)
            ideal_residual=upper_decimal_power((bf-af)/(bf+af),n)
            assert exact_rounded_residual<=Decimal('1e-12')
            assert ideal_residual<=Decimal('1e-12')
            # After at least two updates, r_n=r_1^(2**(n-1)) lies in[0,gate].
            # Therefore0<z_n<=1/(C*d),1-gate<=C*d*z_n<=1.
            true_inverse_bound=1/(C*af)
            assert true_inverse_bound<=8
            if true_inverse_bound>F(k,16):rounded_bound_cases.append([k,stage])
            initial_z_low=wf*bf+betaf
            initial_z_high=wf*af+betaf
            assert 0<initial_z_low<=initial_z_high<=true_inverse_bound
            dummy=contract['inactive_dummy']
            assert validate_public_dummy(dummy,a,b)==dummy
            assert F.from_float(a)<=F.from_float(dummy)<=F.from_float(b)
            for invalid in (True,False,0.,-1.,float('nan'),float('inf'),
                            float(np.nextafter(a,-np.inf)),float(np.nextafter(b,np.inf))):
                try:validate_public_dummy(invalid,a,b)
                except ValueError:dummy_negative_checks+=1
                else:raise AssertionError('invalid public dummy accepted')
            assert validate_public_dummy(a,a,b)==a and validate_public_dummy(b,a,b)==b
            if not a<=1.<=b:unsafe_dummy.append([k,stage])
            points=np.unique(np.r_[np.linspace(a,b,1025),np.geomspace(a,b,1025),
                a,b,np.nextafter(a,b),np.nextafter(b,a),dummy])
            z=weight*points+bias
            min_z=float(z.min());max_z=float(z.max());max_xz=0.;max_cxz=0.
            min_correction=2.;max_correction=0.
            for _ in range(1,n):
                product=points*z
                correction=2.-C*product
                max_xz=max(max_xz,float(product.max()));max_cxz=max(max_cxz,float((C*product).max()))
                min_correction=min(min_correction,float(correction.min()))
                max_correction=max(max_correction,float(correction.max()))
                z=z*correction
                min_z=min(min_z,float(z.min()));max_z=max(max_z,float(z.max()))
            expected=1./(C*points)
            residual=float(np.max(np.abs(1.-C*points*z)))
            abs_error=float(np.max(np.abs(z-expected)))
            inverse_error=float(np.max(np.abs(C*z-1./points)))
            # This diagnostic tolerance is for public binary64 simulation only.
            # The exact real coefficient certificate above has no slack.
            float_slack=128*np.finfo(np.float64).eps
            fp_pass=(np.isfinite(z).all() and min_z>=-float_slack and max_z<=outward(true_inverse_bound)+float_slack
                and residual<=1e-12+float_slack and max_xz<=1/C+float_slack
                and max_cxz<=1+float_slack and min_correction>=1-float_slack and max_correction<=2+float_slack)
            assert fp_pass,(k,stage,residual,max_z)
            rows.append({'K':k,'stage':stage,'a':a,'b':b,'declared_updates':n,
                'C':C,'equivalent_s':k/C,'initial_estimate_binary64':seed,
                'first_affine_weight_binary64':weight,'first_affine_bias_binary64':bias,
                'inactive_dummy':dummy,'midpoint_exact_domain_membership_checked':True,
                'dummy_one_within_domain':bool(a<=1<=b),
                'first_update_residual_lower_outward':lower_outward(rlo),
                'first_update_residual_upper_outward':outward(rhi),
                'first_following_correction_lower_outward':lower_outward(1+rlo),
                'first_following_correction_upper_outward':outward(1+rhi),
                'first_update_d_times_z_upper_outward':outward((1-rlo)/C),
                'first_update_Cd_times_z_upper_outward':outward(1-rlo),
                'ideal_relative_residual_upper':str(ideal_residual),
                'actual_rounded_first_affine_relative_residual_upper':str(exact_rounded_residual),
                'true_z_upper_outward':outward(true_inverse_bound),
                'K_over_16_is_exact_bound_for_rounded_domain':bool(true_inverse_bound<=F(k,16)),
                'ciphertext_path_levels_if_existing_integer_helper_used':2*n-1,
                'sample_count':len(points),'sampled_relative_residual':residual,
                'sampled_z_max_abs_error':abs_error,'sampled_restored_reciprocal_max_abs_error':inverse_error,
                'sampled_all_updates_max_z':max_z,'sampled_max_d_times_z':max_xz,
                'sampled_max_Cd_times_z':max_cxz,'sampled_correction_range':[min_correction,max_correction],
                'binary64_diagnostic_slack':float_slack,'passed':bool(fp_pass)})
    # Distinguish the old variableC=K/4 from this new fixedC32 normalization.
    # Rounding K/4 to an integer would target a different inverse.
    k=15;d=F(1,k);old_c=F(k,4);wrong_c=F(round(float(old_c)))
    rounded_k_example={'K':k,'old_C_exact':str(old_c),'rounded_old_C':str(wrong_c),
        'wrong_mass_if_old_restoration_retained':str(old_c/wrong_c),
        'new_fixed_C':C,'new_equivalent_s':float(F(k,C))}
    assert old_c/wrong_c!=1
    after={p.name:digest(p) for p in paths}
    report={'scope':__doc__,'passed':all(x['passed'] for x in rows) and before==after,
        'source_sha256_before':before,'source_sha256_after':after,'source_files_unchanged':before==after,
        'fixed_C':C,'contracts_checked':len(rows),'public_K':list(range(1,129)),
        'stages_checked':[0,1,2,3,4],'declared_update_histogram':dict(sorted(hist.items())),
        'global_exact_z_bound_all_updates':8,
        'exact_d_times_z_bound_from_update2':1/C,
        'exact_Cd_times_z_bound_from_update2':1,
        'exact_correction_interval_for_updates3_onward':[1,2],
        'first_update_d_times_z_global_upper':max(x['first_update_d_times_z_upper_outward'] for x in rows),
        'first_update_Cd_times_z_global_upper':max(x['first_update_Cd_times_z_upper_outward'] for x in rows),
        'first_following_correction_global_lower':min(x['first_following_correction_lower_outward'] for x in rows),
        'first_following_correction_global_upper':max(x['first_following_correction_upper_outward'] for x in rows),
        'first_update_negative_residual_contracts':sum(x['first_update_residual_lower_outward']<0 for x in rows),
        'midpoint_membership_contracts_checked':len(rows),
        'invalid_public_dummy_checks':dummy_negative_checks,
        'generic_dummy_acceptance_plan':'Validate public finite positive scalar and closed declared-domain membership before any ciphertext/key method. For mixed public row domains, validate each row/stage dummy first, then place it at the corresponding inactive anchors as weighted_sum bias; no data-dependent selection.',
        'bound_scope':'For exact real arithmetic after update2; first affine uses exact interpretations of actually rounded public coefficients. Binary64 observations below are separate diagnostics.',
        'whole_graph_conditional_assumptions':[
            'Baseexp positive polynomial, centeredscore domain[-16,16] and scaledexp domain[-.5,.5].',
            'Initial meanexp inverse remains the historical sixupdate C1 circuit on[.5,2].',
            'Before each squaring, common positive normalizers preserve the public ratio bound and mass lies in[1-1e-12,1].',
            'The denominator equals the sum of the same squared probabilities; no independent adversarial numerator/denominator claim.',
            'Public K is the exact number of valid keys; padded queries remain independent, invalidkey terms are masked.',
            'Each inactive denominator must use its declared stage dummy;1 is outside some tightened domains.',
            'No CKKS quantization/noise/refresh error, actual fullslot layout execution or composed model is certified.'],
        'composition_identity':'p_i_squared * z *32 = p_i_squared/d * (1-r_n); exact mass1-r_n and common ratio preserve existing schedule assumptions.',
        'bounded_restoration_order':'Multiply p_i_squared*z first (<=1/32 under their correlated sum relation), then restore32. Materializing32*z alone can reach256 and exceeds bound16.',
        'stages_where_dummy1_is_outside_domain':unsafe_dummy,
        'outward_a_rounding_requires_bound_1_over_32a_instead_of_exact_K_over_16':rounded_bound_cases,
        'do_not_round_K_over_4_counterexample':rounded_k_example,
        'K128_updates':[row['declared_updates'] for row in rows if row['K']==128],
        'K15_updates':[row['declared_updates'] for row in rows if row['K']==15],
        'worst_sampled_relative_residual':max(x['sampled_relative_residual'] for x in rows),
        'worst_sampled_z_max_abs_error':max(x['sampled_z_max_abs_error'] for x in rows),
        'worst_sampled_restored_reciprocal_max_abs_error':max(x['sampled_restored_reciprocal_max_abs_error'] for x in rows),
        'sample_points_total':sum(x['sample_count'] for x in rows),'rows':rows,
        'elapsed_s':time.perf_counter()-started}
    (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    with tarfile.open(HERE/'sources.tar.gz','w:gz') as tar:
        for p in paths:tar.add(p,arcname=p.name)
    with tarfile.open(HERE/'sources.tar.gz') as tar:
        assert {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest() for m in tar.getmembers()}==before
    summary={name:report[name] for name in ('passed','contracts_checked','K128_updates','K15_updates','sample_points_total','worst_sampled_relative_residual','worst_sampled_z_max_abs_error','source_files_unchanged','elapsed_s')}
    summary['dummy1_unsafe_contracts']=len(unsafe_dummy)
    summary['per_K_exact_bound_rounding_caveats']=len(rounded_bound_cases)
    print(json.dumps(summary))
    return 0 if report['passed'] else 1

if __name__=='__main__':raise SystemExit(run())
