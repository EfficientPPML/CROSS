"""All-K public analytic interval checks plus binary64 grid observations."""
from pathlib import Path
import hashlib,json,time
import numpy as np
from public_schedule import PublicSchedule
from scaled_softmax import ScaledNewtonSoftmax

HERE=Path(__file__).resolve().parent
COEFFICIENTS=HERE/'baseline_manifest.json'
FILES=[Path(__file__),HERE/'public_schedule.py',HERE/'scaled_softmax.py',HERE/'score_layout.py',COEFFICIENTS]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in FILES}
before=hashes();started=time.perf_counter()
coefficients=json.loads(COEFFICIENTS.read_text())['power_coefficients_in_t_2x']
schedule=PublicSchedule(coefficients)
rows=[];worst=0.;range_failures=0
for k in range(1,129):
 for stage in range(5):
  plan=schedule.stage(k,stage);a,b=plan['a'],plan['b'];n=plan['iterations']
  x=np.unique(np.r_[np.geomspace(a,b,129),np.linspace(a,b,129),plan['inactive_dummy']])
  engine=ScaledNewtonSoftmax(coefficients)
  z,_,c=engine.scaled_inverse(x,(a,b),n,'domain',k)
  residual=float(np.max(np.abs(1-c*x*z)))
  assert residual<=1e-12 and not engine.bound_failures(),(k,stage,residual,engine.bound_failures())
  r=plan['ideal_ratio_upper']
  vectors=[np.ones(k),np.r_[r,np.ones(k-1)],np.r_[1.,np.full(k-1,r)],np.linspace(1.,r,k),np.geomspace(1.,r,k)]
  denominator_cases=[]
  for mass in (1.-1e-12,1.):
   for vector in vectors:
    probability=mass*vector/vector.sum()
    square=probability*probability;denominator=square.sum()
    assert a<=denominator<=b,(k,stage,denominator,a,b)
    checked=ScaledNewtonSoftmax(coefficients)
    zz,_,cc=checked.scaled_inverse(np.array([denominator]),(a,b),n,'structured',k)
    out=cc*(square*zz[0]);out_mass=float(out.sum())
    assert np.min(out)>=0 and 1.-1e-12-64*np.finfo(float).eps<=out_mass<=1.+64*np.finfo(float).eps
    assert not checked.bound_failures()
    denominator_cases.append({'input_mass':mass,'denominator':float(denominator),'output_mass':out_mass})
  worst=max(worst,residual)
  rows.append({'K':k,'stage':stage,'iterations':n,'domain':[a,b],'points':len(x),
               'worst_grid_relative_residual':residual,'structured_cases':denominator_cases})
grid=np.linspace(-.5,.5,20001)
grid_error=float(np.max(np.abs(np.polynomial.polynomial.polyval(2*grid,coefficients)-np.exp(grid))))
assert grid_error<1e-10 and grid_error<=schedule.certificate['whole_interval_error_E_float_upper']
first=schedule.stage(128,0);c=128/4;z0=2/(c*(first['a']+first['b']))
bad_first=1.*(-c*z0*z0)+2*z0
assert bad_first<0 # Public counterexample to retaining inactive dummy1.
after=hashes()
report={'scope':__doc__,'passed':before==after,'source_hashes_before':before,'source_hashes_after':after,
 'source_files_unchanged':before==after,'schedule':schedule.report(),'checks':rows,
 'worst_observed_grid_relative_residual':worst,'exp_grid_error':grid_error,
 'inactive_dummy1_counterexample':{'K':128,'stage':0,'domain':[first['a'],first['b']],
    'initial_z':z0,'first_Newton_output_for_x1':bad_first,'replacement_public_midpoint':first['inactive_dummy']},
 'elapsed_s':time.perf_counter()-started,'numpy':np.__version__,
 'limitations':['Conditional ideal positive-arithmetic ratio/mass proof; actual binary64 observations are separate.',
 'No CKKS, refresh, interval proof of floating arithmetic, unseen-input guarantee or performance claim.']}
(HERE/'public_contracts.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'K_stage_contracts':len(rows),'worst_relative_residual':worst,
 'K128_iterations':[p['iterations'] for p in schedule.rows[128]],'analytic_E':schedule.certificate['whole_interval_error_E_float_upper'],
 'inactive_dummy1_first_output':bad_first,'elapsed_s':report['elapsed_s']}))
raise SystemExit(0 if report['passed'] else 1)
