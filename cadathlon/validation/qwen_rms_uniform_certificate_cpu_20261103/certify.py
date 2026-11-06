"""Exact-real uniform Qwen RMS polynomial certificates; no model/JAX/HE."""
from pathlib import Path
from fractions import Fraction as F
from datetime import datetime,timezone
import hashlib,io,json,math,os,struct,sys,tarfile,time
HERE=Path(__file__).resolve().parent
MANIFEST=Path(os.environ.get('QWEN_FROZEN_MANIFEST','/tmp/qwen-calibration-candidate1-20261103/candidate-v1/manifest.json'))
EVALUATOR=Path(os.environ.get('QWEN_FROZEN_ARITHMETIC','/tmp/qwen-calibration-candidate1-20261103/arithmetic.py'))
MANIFEST_HASH='d16cfe0eaf4079b1e782790dcff277ed6901aa7ea1376ee7877b40b54dde93a2'
EVALUATOR_HASH='9689b843b4ec2d0a0ef724dbdca7a516e9a4f23da83f83af5571da13380e9a6f'
SITES=[f'{i}.post.inverse_sqrt' for i in (0,1,2,23,24,25,26,27)]

def sha(data):return hashlib.sha256(data).hexdigest()
def up(value):
 f=float(value)
 return math.nextafter(f,math.inf) if F(f)<value else f

def down(value):
 f=float(value)
 return math.nextafter(f,-math.inf) if F(f)>value else f

def dyadic(value):
 assert value.denominator&(value.denominator-1)==0
 return dict(numerator_hex=hex(value.numerator),denominator_power_of_two=value.denominator.bit_length()-1,upper_binary64=up(value))

def add(a,b,weight=F(1)):
 out=list(a)+[F(0)]*max(0,len(b)-len(a))
 for i,x in enumerate(b):out[i]+=weight*x
 return out

def linear(a,alpha,beta):
 out=[F(0)]*(len(a)+1)
 for n,c in enumerate(a):
  out[n]+=beta*c
  if n==0:out[1]+=alpha*c
  else:out[n-1]+=alpha*c/2;out[n+1]+=alpha*c/2
 return out

def pullback(coefficients,strategy,alpha,beta):
 out=[coefficients[0]];prev=[F(1)];current=[beta,alpha]
 if len(coefficients)>1:out=add(out,current,coefficients[1])
 for n in range(2,len(coefficients)):
  following=linear(current,alpha,beta)
  if strategy=='chebyshev':following=add([2*x for x in following],prev,F(-1))
  elif strategy!='power':raise ValueError('unknown frozen basis')
  out=add(out,following,coefficients[n]);prev,current=current,following
 return out

def square(coefficients):
 # All frozen constants and pullbacks are dyadic. A common integer denominator
 # avoids costly Fraction gcd inside the exact Chebyshev convolution.
 D=max(c.denominator for c in coefficients)
 assert D&(D-1)==0
 integers=[c.numerator*(D//c.denominator) for c in coefficients]
 out=[0]*(2*len(coefficients)-1)
 for i,a in enumerate(integers):
  for j in range(i,len(integers)):
   value=a*integers[j]*(1 if i==j else 2)
   out[i+j]+=value;out[j-i]+=value
 return [F(x,2*D*D) for x in out]

def evaluate(coefficients,x,strategy='chebyshev'):
 if strategy=='power':
  result=F(0)
  for c in reversed(coefficients):result=result*x+c
  return result
 before,current=F(1),x;result=coefficients[0]
 if len(coefficients)>1:result+=coefficients[1]*current
 for c in coefficients[2:]:before,current=current,2*x*current-before;result+=c*current
 return result

def exact_certificate(rule):
 lo,hi=map(F,rule['domain']);assert 0<lo<hi
 coefficients=list(map(F,rule['evaluation_coefficients']));strategy=rule['strategy']
 # These are the exact binary64 public constants from the frozen evaluator.
 low,high=rule['domain'];A=F(2/(high-low));B=F(-(high+low)/(high-low))
 midpoint=(lo+hi)/2;half=(hi-lo)/2;alpha=A*half;beta=A*midpoint+B
 pulled=pullback(coefficients,strategy,alpha,beta)
 h=linear(square(pulled),half,midpoint);h[0]-=1
 delta=sum(map(abs,h),F(0));positive_midpoint=evaluate(pulled,F(0))>0
 probes=[]
 for u in (F(-1),F(0),F(1),F(1,3)):
  v=midpoint+half*u;original=evaluate(coefficients,A*v+B,strategy)
  assert evaluate(pulled,u)==original
  assert evaluate(h,u)==v*original*original-1
  probes.append(str(u))
 certified=bool(delta<1 and positive_midpoint)
 relative=delta/(2-delta) if certified else None
 return dict(domain=rule['domain'],strategy=strategy,degree=rule['degree'],
  evaluation_coefficient_f64le_sha256=sha(struct.pack('<'+str(len(coefficients))+'d',*rule['evaluation_coefficients'])),
  actual_affine=dict(weight_float_hex=float(A).hex(),bias_float_hex=float(B).hex(),alpha=dyadic(alpha),beta=dyadic(beta)),
  exact_H_coefficient_L1=dyadic(delta),positive_domain=True,positive_polynomial_midpoint=positive_midpoint,
  H_degree=len(h)-1,delta_below_one=delta<1,certified=certified,
  crude_relative_error_upper=up(delta) if certified else None,
  uniform_relative_error_upper=up(relative) if certified else None,
  exact_relative_bound=(dict(numerator_hex=hex(relative.numerator),denominator_hex=hex(relative.denominator)) if certified else None),
  relative_error_derivation='|H|<=delta<1 and positive midpoint imply p>0 by continuity; |sqrt(v)*p-1|=|H|/(sqrt(1+H)+1)<=delta/(2-delta), since sqrt(1-delta)>=1-delta.',
  exact_pullback_and_square_identity_probes=probes,
  normalized_multiplier_bounds=[down(1-relative),up(1+relative)] if certified else None,
  max_exact_coefficient_denominator_bits=max(c.denominator.bit_length() for c in h))

def main():
 start=time.monotonic();paths=[Path(__file__),HERE/'test_certify.py',MANIFEST,EVALUATOR];before={str(p):sha(p.read_bytes()) for p in paths}
 assert before[str(MANIFEST)]==MANIFEST_HASH and before[str(EVALUATOR)]==EVALUATOR_HASH
 manifest=json.loads(MANIFEST.read_text());results={}
 for name in SITES:
  then=time.monotonic();results[name]=exact_certificate(manifest['nonlinear'][name]);results[name]['elapsed_s']=time.monotonic()-then
  print(json.dumps(dict(site=name,certified=results[name]['certified'],uniform_relative_error_upper=results[name]['uniform_relative_error_upper'],elapsed_s=results[name]['elapsed_s'])),flush=True)
 report=dict(scope=__doc__,passed=all(r['certified'] for r in results.values()),results=results,
  manifest_sha256=MANIFEST_HASH,evaluator_sha256=EVALUATOR_HASH,
  proof='Exact binary-rational Chebyshev pullback and product coefficients; global coefficient L1 is a uniform interval bound, not sampled error.',
  limitations=['Fixed exact-real polynomial functions only, including actual stored evaluation coefficient and public affine rounding.',
   'Subsequent binary64 intermediate arithmetic rounding and CKKS/refresh/quantization errors are not included.',
   'Positive v must remain inside the declared frozen domain; no arbitrary-input, learned-weight, model, prompt, validation or HE readiness claim.',
   'Power coefficients are the frozen converted/pruned evaluation coefficients, not an unrounded replacement by Chebyshev interpolation values.'],
  source_hashes_before=before,elapsed_s=time.monotonic()-start,cpu_affinity=sorted(os.sched_getaffinity(0)),jax_imported='jax'in sys.modules)
 assert not report['jax_imported']
 with tarfile.open(HERE/'sources.tar.gz','w:gz') as t:
  for p in paths:
   data=p.read_bytes();i=tarfile.TarInfo(str(p).lstrip('/'));i.size=len(data);t.addfile(i,io.BytesIO(data))
 after={str(p):sha(p.read_bytes()) for p in paths};assert before==after
 with tarfile.open(HERE/'sources.tar.gz') as t:assert {'/'+m.name:sha(t.extractfile(m).read()) for m in t.getmembers()}==before
 report.update(source_hashes_after=after,source_files_unchanged=True,source_archive_verified=True,source_archive_sha256=sha((HERE/'sources.tar.gz').read_bytes()))
 (HERE/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
 return 0 if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
