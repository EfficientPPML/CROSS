"""Hypothetical periodic-error diagnostic for public deep fixtures; no HE/error-bound claim."""
import sys,json,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path('/home/jianming_gatech/CROSS')
sys.path[:0]=[str(ROOT/'cadathlon/common'),str(ROOT/'cadathlon/tests'),str(ROOT)]
import numpy as np
from support import load_module,metrics
import test_bounded_transformer_plan as planning
from encrypted_bounded_arithmetic import BoundedArithmetic
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_packed_transformer import PackedPairLayout,packed_feature_width,plan_packed_transformer
from encrypted_transformer import public_toy_approximations
Value=planning.Value
files=[Path(__file__),*sorted((ROOT/'cadathlon/common').glob('*.py')),ROOT/'cadathlon/tests/test_bounded_transformer_plan.py',ROOT/'cadathlon/tests/test_packed_transformer.py']
for problem in ('p2_bert','p4_qwen'):
 folder=ROOT/'cadathlon/agent_pset'/problem
 files += [folder/'src/reference.py',*sorted((folder/'test').glob('test*.in')),*sorted((folder/'test').glob('test*.out'))]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
start=hashes()
class Arithmetic(planning.Arithmetic):
 def __init__(self,slots,amplitude,mode,seed):
  super().__init__(slots,True)
  self.observations=[]
  random=np.random.default_rng(seed)
  def refresh(value,*,public_bound):
   original=value.slots.copy()
   if mode=='positive':error=np.full_like(original,amplitude)
   elif mode=='negative':error=np.full_like(original,-amplitude)
   elif mode=='alternating':error=amplitude*np.where(np.arange(original.size)%2,1.,-1.)
   else:error=amplitude*random.choice([-1.,1.],size=original.shape)
   self.observations.append({'input_min':float(original.min()),'input_max':float(original.max()),'input_level':value.level,'public_bound':public_bound,'provenance':getattr(value,'_public_bound_provenance',None)})
   return Value(original+error,25)
  self.refresher=SimpleNamespace(input_level=2,output_level=25,refresh=refresh)
 def declare_bound(self,value,bound,reason='public circuit contract'):
  return BoundedArithmetic.declare_bound(self,value,bound,reason)
rows=[]
for model,problem in [('bert','p2_bert'),('qwen','p4_qwen')]:
 folder=ROOT/'cadathlon/agent_pset'/problem
 reference=load_module(folder/'src/reference.py')
 for number in range(1,7):
  case=json.loads((folder/f'test/test{number}.in').read_text())
  args=reference.make_case(case);weights,embeddings=args[:2];mask=args[2] if model=='bert' else None
  golden=np.asarray(json.loads((folder/f'test/test{number}.out').read_text())['logits'])
  b,s,h=embeddings.shape;f=packed_feature_width(model,case['config'])
  period=(1<<(b-1).bit_length())*(1<<(s-1).bit_length())**2*f
  layout=PackedPairLayout(b,s,f,max(period,2*f))
  plan=plan_packed_transformer(model,case['config'],weights,batch=b,sequence=s,slots=layout.slots,attention_mask=mask)
  if plan['depth']<=52:continue
  for amplitude in (0.,1e-6,1e-5,1e-4,1e-3):
   for mode in ('positive','negative','alternating','rademacher'):
    for seed in ((1,2,3) if mode=='rademacher' else (1,)):
     graph=object.__new__(BoundedPackedTransformer)
     graph.layout,graph.features=layout,f
     graph.arithmetic=Arithmetic(layout.slots,amplitude,mode,seed);graph.ctx=graph.arithmetic.ctx
     row={'model':model,'case':number,'period':period,'amplitude':amplitude,'mode':mode,'seed':seed}
     try:
      result=graph.evaluate(Value(layout.pack_client_embeddings(embeddings)[0],49),model,case['config'],weights,public_toy_approximations(model),mask)
      actual=layout.unpack_client_features([result.slots],output_features=golden.shape[-1])[:,0 if model=='bert' else -1]
      error=metrics(actual,golden)
      passed=(bool(np.all(np.abs(actual-golden)<=.02+.01*np.abs(golden))) and bool(np.array_equal(actual[:,1]>=actual[:,0],golden[:,1]>=golden[:,0]))) if model=='bert' else error['max_abs_error']<=.2 and error['nrmse']<=.1
      row.update(errors=error,gate_passed=passed,final_level=result.level,domains=graph.arithmetic.domains)
     except AssertionError as error:row.update(gate_passed=False,domain_failure=str(error))
     row.update(refreshes=graph.arithmetic.observations,events=graph.arithmetic.refresh_events)
     rows.append(row)
  subset=[r for r in rows if r['model']==model and r['case']==number]
  print(model,number,json.dumps({str(a):{'passed':sum(r['gate_passed'] for r in subset if r['amplitude']==a),'total':sum(r['amplitude']==a for r in subset),'domain_failures':sum('domain_failure' in r for r in subset if r['amplitude']==a),'max_error':max(r.get('errors',{}).get('max_abs_error',0.) for r in subset if r['amplitude']==a)} for a in (0.,1e-6,1e-5,1e-4,1e-3)}),flush=True)
end=hashes()
report={'scope':__doc__,'error_model':'Exactly periodic real additive errors independently injected at each planned refresh output; excludes ordinary CKKS error and does not bound actual refresh noise. Ideal public metadata is retained for schedule selection but is not asserted as a bound on noisy intermediates. Polynomial domain checks and original fixture gates remain active.','source_hashes_start':start,'source_hashes_end':end,'source_files_unchanged':start==end,'rows':rows}
Path('/tmp/public_transformer_refresh_sensitivity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('source unchanged',start==end,flush=True)
