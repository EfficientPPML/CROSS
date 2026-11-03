"""Public parameter and conditional ideal schedule comparison; no HE/keygen/performance claim."""
import sys,json,math,hashlib,time,os,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ.get('CROSS_REPO','/home/jianming_gatech/CROSS')).resolve()
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=Path('/tmp/compare_qwen_refresh_profiles.json'))
args=parser.parse_args()
sys.path[:0]=[str(ROOT/'cadathlon/common'),str(ROOT/'cadathlon/tests'),str(ROOT)]
from encrypted_refresh import refresh_plan
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_packed_transformer import PackedPairLayout,packed_feature_width,plan_packed_transformer
from encrypted_transformer import public_toy_approximations
from support import load_module,metrics
import test_bounded_transformer_plan as planning
from jaxite_word import he_params,util
from jaxite_word.composite_prime_gen import composite_prime_gen
import numpy as np
files=[Path(__file__),*sorted((ROOT/'cadathlon/common').glob('*.py')),*sorted((ROOT/'jaxite_word').glob('*.py')),ROOT/'cadathlon/tests/test_bounded_transformer_plan.py',ROOT/'cadathlon/tests/test_packed_transformer.py']
folder=ROOT/'cadathlon/agent_pset/p4_qwen'
files += [folder/'src/reference.py',folder/'test/test2.in',folder/'test/test2.out']
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
start=hashes();started=time.perf_counter()
case=json.loads((folder/'test/test2.in').read_text());oracle=load_module(folder/'src/reference.py')
weights,embeddings=oracle.make_case(case)[:2];b,s,h=embeddings.shape;f=packed_feature_width('qwen',case['config']);period=(1<<(b-1).bit_length())*(1<<(s-1).bit_length())**2*f
layout=PackedPairLayout(b,s,f,max(period,2*f));golden=np.asarray(json.loads((folder/'test/test2.out').read_text())['logits'])
base_plan=plan_packed_transformer('qwen',case['config'],weights,batch=b,sequence=s,slots=65536)
rows=[];schedules=[]
for depth in range(49,55):
 refresh=refresh_plan(131072,period,(4,4),max_level=depth)
 graph=object.__new__(BoundedPackedTransformer);graph.layout=layout;graph.features=f
 arithmetic=planning.Arithmetic(layout.slots,True)
 def ideal_refresh(value,*,public_bound):
  assert np.max(np.abs(value.slots))<=public_bound+1e-9
  return planning.Value(value.slots,refresh['output_level'])
 arithmetic.refresher=SimpleNamespace(input_level=2,output_level=refresh['output_level'],refresh=ideal_refresh)
 graph.arithmetic=arithmetic;graph.ctx=arithmetic.ctx
 result=graph.evaluate(planning.Value(layout.pack_client_embeddings(embeddings)[0],depth),'qwen',case['config'],weights,public_toy_approximations('qwen'))
 actual=layout.unpack_client_features([result.slots],output_features=golden.shape[-1])[:,-1]
 error=metrics(actual,golden)
 schedule=dict(depth=depth,output_level=refresh['output_level'],final_level=result.level,refresh_count=len(arithmetic.refresh_events),events=arithmetic.refresh_events,ideal_errors=error,gate_passed=error['max_abs_error']<=.2 and error['nrmse']<=.1)
 schedules.append(schedule)
 rotations=set(base_plan['rotations'])|set(refresh['rotations']);rotation_count=len(rotations)
 q=composite_prime_gen(2,2*(depth+1),61,56,262144,31)
 for dnum in (4,5,6,8):
  row={'depth':depth,'dnum':dnum,'rotation_count':rotation_count,'refresh_count':len(arithmetic.refresh_events)}
  try:
   p=util.generate_p_towers(q,dnum,131072,aux_bits=30)
   he_params.assert_p_coverage(q,p,dnum)
   qp_bits=sum(math.log2(x) for x in q+p)
   secure=he_params.qp_is_secure(q,p,131072,he_params.SecurityTarget(128,he_params.CostModel.CLASSICAL))
   words=2*dnum*131072*(len(q)+len(p))
   row.update(q=q,p=p,actual_qp_bits=qp_bits,secure_128_classical=secure,raw_rotation_key_bytes=4*words,all_rotation_key_bytes=rotation_count*4*words,partition_width=(len(q)+dnum-1)//dnum)
  except Exception as e:row.update(error_type=type(e).__name__,error=str(e))
  rows.append(row)
 print('depth',depth,'refreshes',schedule['refresh_count'],'finallevel',result.level,[(r['dnum'],r.get('secure_128_classical'),round(r.get('all_rotation_key_bytes',0)/2**30,2)) for r in rows if r['depth']==depth],flush=True)
end=hashes();report={'scope':__doc__,'fixed_model':'Qwen public test2','scale_bits':56,'first_bits':61,'degree':131072,'period':period,'budgets':[4,4],'no_device_or_key_generation':True,'schedules':schedules,'profiles':rows,'source_hashes_start':start,'source_hashes_end':end,'source_files_unchanged':start==end,'elapsed_s':time.perf_counter()-started}
args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('unchanged',start==end,flush=True)
