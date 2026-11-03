"""Declared-domain binary64 grid and illustrative additive-error sensitivity.

The additive step error is an artificial scalar perturbation, not a CKKS model.
No encryption, refresh, model fitting, clipping, or adaptive iterations occur.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from scaled_softmax import ScaledNewtonSoftmax

HERE=Path(__file__).resolve().parent
files=[Path(__file__),HERE/'scaled_softmax.py',HERE/'score_layout.py']
before={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
started=time.perf_counter();residual=0.;worst_bound_excess=0.;rows=[]
for k in range(1,129):
 x=np.unique(np.r_[np.geomspace(.5/k,2.,257),np.linspace(.5/k,2.,257)])
 # Coefficients are unused by scaled_inverse; no approximation is selected here.
 engine=ScaledNewtonSoftmax(np.zeros(9))
 z,depth,c=engine.scaled_inverse(x,(.5/k,2.),13,'declared_domain',k)
 assert not engine.bound_failures(),(k,engine.bound_failures())
 maximum=float(np.max(np.abs(c*x*z-1)))
 assert maximum<1e-12
 residual=max(residual,maximum)
 rows.append({'K':k,'points':len(x),'relative_residual':maximum,
              'observed_max_z':float(z.max()),'declared_max_z':8.})

sensitivity=[]
for k in (1,15,16,17,64,128):
 x=np.unique(np.r_[np.geomspace(.5/k,2.,257),np.linspace(.5/k,2.,257)])
 for scale in (1.,4.):
  c=k/scale;initial=2./(c*(.5/k+2.))
  for magnitude in (0.,1e-12,1e-10,1e-8):
   for sign in (-1,1):
    eta=sign*magnitude
    z=x*(-c*initial*initial)+2*initial+eta
    for _ in range(1,13):z=z*(2.-c*(x*z))+eta
    sensitivity.append({'K':k,'s':scale,'C':c,'step_eta':eta,
      'max_abs_relative_residual':float(np.max(np.abs(1-c*x*z))),
      'minimum_z':float(z.min()),'maximum_z':float(z.max()),
      'largest_C_times_x':float(np.max(c*x))})

after={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
report={'scope':__doc__,'passed':before==after,'source_hashes_before':before,
 'source_hashes_after':after,'source_files_unchanged':before==after,
 'grid_K1_to128':rows,'worst_relative_residual':residual,
 'illustrative_sensitivity':sensitivity,
 'algebra':{'definition':'a=.5/K,b=2,C=K/s,z0=2/[C(a+b)],e_n=1-C*x*z_n',
   'exact_recurrence':'e_(n+1)=e_n^2',
   'initial_error_bound':'|e_0| <= (b-a)/(b+a) < 1',
   'after_first_step':'0<z_n<=1/(Cx)<=2s;0<x*z_n<=s/K;0<C*x*z_n<=1;1<=correction<2',
   'final_probability':'C*p_i^2*z <= p_i^2/x <=1 when x=sum_j p_j^2 is the same shared denominator',
   'step_perturbation':'If z_next=z*(2-C*x*z)+eta, e_next=e^2-C*x*eta',
   'amplification_scope':'For fixed absolute eta and s4,C*x<=64 versus<=256 for s1. Actual CKKS eta is not specified or held equal by this experiment.',
   'roundoff_scope':'Binary64 diagnostics permit64*epsilon*max(1,interval endpoint magnitude), with strict excursions separately recorded in the model-run reports.'},
 'elapsed_s':time.perf_counter()-started,'numpy':np.__version__}
(HERE/'public_algebra.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':report['passed'],'worst_relative_residual':residual,
 'elapsed_s':report['elapsed_s'],'sensitivity_cases':len(sensitivity)}))
raise SystemExit(0 if report['passed'] else 1)
