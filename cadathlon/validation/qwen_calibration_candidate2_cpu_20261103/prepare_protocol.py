"""Freeze all actual public multipliers and input/source identities before selection."""
from pathlib import Path
from fractions import Fraction as F
import copy,datetime,hashlib,json,math,shutil,struct
import numeric
from domain_rule import silu_domain
from select_candidate2 import canonical,fixed_parts
P=Path(__file__).resolve().parent
V=Path('/home/jianming_gatech/CROSS/cadathlon/validation')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    if (P/'protocol.json').exists():raise ValueError('refuse to overwrite frozen protocol')
    sources={'base_manifest.json':V/'qwen_calibration_candidate1_cpu_20261103/candidate-v1/manifest.json',
      'public_gate_bounds.json':V/'qwen_public_silu_bounds_cpu_20261103/public-v1/report.json',
      'uniform_polynomial_certificates.json':Path('/tmp/qwen-rms-polynomial-certificate-20261103/report.json'),
      'uniform_certificate_sources.tar.gz':Path('/tmp/qwen-rms-polynomial-certificate-20261103/sources.tar.gz'),
      'selection_proposal.json':V/'qwen_candidate2_selection_design_20261103/proposal.json',
      'fresh_input_freeze.json':V/'qwen_public_silu_bounds_cpu_20261103/fresh_input_freeze.json',
      'fresh_tokenized_inputs.json':V/'qwen_public_silu_bounds_cpu_20261103/public-v1/tokenized_fresh_inputs.json',
      'calibration_inputs.json':V/'qwen_approximate_calibration_cpu_20261103/tokenized_calibration_inputs.json',
      'calibration_corpus.json':V/'qwen_approximate_calibration_cpu_20261103/corpus.json',
      'model_pins.json':V/'qwen_approximate_calibration_cpu_20261103/model_pins.json'}
    for name,path in sources.items():shutil.copyfile(path,P/'inputs'/name)
    assert sha(P/'inputs/base_manifest.json')=='d16cfe0eaf4079b1e782790dcff277ed6901aa7ea1376ee7877b40b54dde93a2'
    assert sha(P/'inputs/public_gate_bounds.json')=='45bcf78ea0e2a45fe455c4de504269f5bbe1392f2ada91598db9327689cfdc7f'
    assert sha(P/'inputs/uniform_certificate_sources.tar.gz')=='2545fb88ed41d2e3b592cba68dc7a9f267ddfe9a7b39a8fbc240d7a414cd4f41'
    assert sha(P/'inputs/fresh_tokenized_inputs.json')=='af93ae2a580bc4ff51d1463e8ab50ab054e295d0e4932340ae81d18185696988'
    base=json.loads((P/'inputs/base_manifest.json').read_text());bounds=json.loads((P/'inputs/public_gate_bounds.json').read_text());cert=json.loads((P/'inputs/uniform_polynomial_certificates.json').read_text())
    assert bounds['passed'] and bounds['rows_computed']==86016 and cert['passed']
    actual={}
    for layer in range(28):
        label=f'{layer}.post.inverse_sqrt';rule=base['nonlinear'][label]
        if rule['method']=='newton':
            check=numeric.rsqrt_contract(rule['domain'])
            assert all(rule[k]==v for k,v in check.items())
            delta=F(rule['final_relative_error_bound']);origin='unchanged exact rounded-first-affine ideal Newton certificate'
        else:
            check=cert['results'][label];exact=check['exact_relative_bound']
            assert check['certified'] and check['positive_polynomial_midpoint'] and check['domain']==rule['domain'] and check['strategy']==rule['strategy']
            packed=struct.pack('<'+'d'*len(rule['evaluation_coefficients']),*rule['evaluation_coefficients'])
            assert hashlib.sha256(packed).hexdigest()==check['evaluation_coefficient_f64le_sha256']
            delta=F(int(exact['numerator_hex'],16),int(exact['denominator_hex'],16));origin='independently reviewed exact Chebyshev H certificate, rational delta/(2-delta)'
        assert 0<=delta<F(1,10**6)
        layer_bounds=bounds['layers'][str(layer)];native=float(layer_bounds['maximum_row_upper'])
        assert native==max(r['native_exact_real_upper'] for r in layer_bounds['rows'])
        domain=silu_domain(native,delta)
        displayed=float(delta)
        if F(displayed)<delta:displayed=math.nextafter(displayed,math.inf)
        actual[str(layer)]={'site':label,'certified':True,'unchanged_norm_rule_sha256':canonical(rule),'certificate_origin':origin,
          'delta_fraction':{'numerator_hex':hex(delta.numerator),'denominator_hex':hex(delta.denominator)},'relative_upper_binary64':displayed,
          'native_gate_upper':native,'public_silu_domain':domain['domain'],'domain_computation':domain,
          'scope':'Exact-real norm/gate relation conditional on unchanged variance domain; excludes subsequent floating/CKKS/refresh error.'}
    (P/'norm_multipliers.json').write_text(json.dumps(actual,indent=2)+'\n')
    norms,soft=fixed_parts(base);assert len(norms)==113
    proposal=json.loads((P/'inputs/selection_proposal.json').read_text())
    protocol=copy.deepcopy(proposal)
    protocol.update(schema='qwen-candidate2-selection-protocol-v1',status='frozen_before_analytic_selection',
       all28_actual_uniform_certificates=True,unchanged_norms_sha256=canonical(norms),unchanged_softmax_sha256=canonical(soft),
       degrees_in_order=proposal['analytic_selection']['degrees_in_order'],fallback_base_degrees=proposal['analytic_selection']['fallback']['base_degrees_in_order'],
       arithmetic_sha256=sha(P/'arithmetic.py'),numeric_sha256=sha(P/'numeric.py'),created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
       final_public_silu_domains={k:v['public_silu_domain'] for k,v in actual.items()},
       selection_authorized=True,model_execution_authorized=False)
    protocol['viability_assessment']['provisional_radius_if_all_polynomial_delta_le_1e-6']=[min(x['public_silu_domain'][1] for x in actual.values()),max(x['public_silu_domain'][1] for x in actual.values())]
    protocol['public_domain_rule']['delta_sources']['8polynomial']='Final independent v2 certificates are accepted and bound in inputs/uniform_polynomial_certificates.json; no placeholders remain.'
    paths=sorted(P.glob('*.py'))+[P/'norm_multipliers.json']+sorted((P/'inputs').iterdir())
    protocol['source_input_sha256']={str(path.relative_to(P)):sha(path) for path in paths}
    (P/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (P/'protocol-freeze.json').write_text(json.dumps({'passed':True,'protocol_sha256':sha(P/'protocol.json'),'actual_certificates':28,'norm_rules_unchanged':113,'all_softmax_unchanged':True,'model_evaluated':False,'coefficient_selection_performed':False,'source_input_count':len(paths)},indent=2)+'\n')
    print(json.dumps({'passed':True,'protocol_sha256':sha(P/'protocol.json'),'actual_certificates':28,'sources':len(paths)}))

if __name__=='__main__':main()
