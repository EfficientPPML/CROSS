"""Public aggregate conditioning diagnosis; no keys, slot arrays or backend."""
from pathlib import Path
import hashlib,json,math
P=Path(__file__).resolve().parent
def derive():
 raw=(P/'failed_report.json').read_bytes()
 assert hashlib.sha256(raw).hexdigest()=='f97f839d23a2d91e258f8937a92bc8173ebd1bc77483db803a100531c8985b45'
 r=json.loads(raw);assert r['passed']is False and r['complete']is True and r['fixed_absolute_error_gate']==1e-6
 assert hashlib.sha256((P/'measured_fixtures.py').read_bytes()).hexdigest()==r['source_hashes_before']['fixtures.py']
 results={}
 for name,derived in r['derived_checks'].items():
  last=max(int(k[-2:]) for k in derived if k.startswith('invariant'))
  denominator=r['checks'][name+':denominator'];e=r['checks'][name+f':e{last:02d}']
  E=denominator['max_abs_error'];m=denominator['decoded_minimum'];I=derived[f'invariant{last:02d}']['max_abs_error']
  residual=max(abs(e['decoded_minimum']),abs(e['decoded_maximum']))
  assert all(math.isfinite(v) for v in (E,m,I,residual)) and 0<E<m
  # If published rounded aggregates were exact bounds: d_tilde>=m,
  # |d_tilde-d|<=E and |1-d_tilde*y|<=I+|e_tilde| imply this sum.
  denominator_term=E/(m*(m-E));internal_term=(I+residual)/m
  observed=derived['final_true_inverse']['max_abs_error']
  assert observed<=denominator_term+internal_term
  results[name]=dict(updates=last,denominator_error=E,decoded_denominator_minimum=m,
   reciprocal_perturbation_estimate=denominator_term,internal_residual_estimate=internal_term,
   combined_estimate=denominator_term+internal_term,observed_true_inverse_error=observed,
   observed_over_denominator_estimate=observed/denominator_term,
   denominator_error_sufficient_for_reciprocal_term_below_1e_minus_6=1e-6*m*m/(1+1e-6*m),
   unchanged_run_passed=False)
 return dict(scope='Conditional aggregate sensitivity from rounded published metrics; not a directed-rounding interval certificate or slotwise causal reconstruction.',cases=results,
  limits=['No decoded arrays were retained/read; maxima may occur at different slots.',
   'The recorded invariant uses binary64 subtraction/multiplication; its rounding and publication rounding are not separately bounded here.',
   'The sufficient denominator threshold excludes internal residual/HE perturbations and is not an execution acceptance gate.',
   'This does not establish a repaired circuit, more precise profile, successful rerun or full-model accuracy.'])
if __name__=='__main__':print(json.dumps(derive(),indent=2,allow_nan=False))
