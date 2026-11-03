"""Public-ring and conditional ideal graph planning for twelve fixtures; no HE/noise/performance claim."""
import sys,json,math,hashlib,time,os,argparse
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(os.environ.get('CROSS_REPO','/home/jianming_gatech/CROSS')).resolve()
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=Path('/tmp/compare_public_refresh_profiles.json'));args=parser.parse_args()
sys.path[:0]=[str(ROOT/'cadathlon/common'),str(ROOT/'cadathlon/tests'),str(ROOT)]
from encrypted_refresh import refresh_plan
from encrypted_reference import reference_ring_profile
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_packed_transformer import PackedPairLayout,packed_feature_width,plan_packed_transformer
from encrypted_transformer import public_toy_approximations
from support import load_module,metrics
import test_bounded_transformer_plan as planning
from jaxite_word import he_params,util
from jaxite_word.composite_prime_gen import composite_prime_gen
import numpy as np
files=[Path(__file__),*sorted((ROOT/'cadathlon/common').glob('*.py')),*sorted((ROOT/'jaxite_word').glob('*.py')),ROOT/'cadathlon/tests/test_bounded_transformer_plan.py',ROOT/'cadathlon/tests/test_packed_transformer.py']
for problem in ('p2_bert','p4_qwen'):
 folder=ROOT/'cadathlon/agent_pset'/problem
 files += [folder/'src/reference.py',*sorted((folder/'test').glob('test*.in')),*sorted((folder/'test').glob('test*.out'))]
def hashes():return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
start=hashes();started=time.perf_counter();rings={};rows=[]
for depth,dnum in ((49,4),(50,5),(38,2)):
 q=composite_prime_gen(2,2*(depth+1),61,56,262144,31);p=util.generate_p_towers(q,dnum,131072,aux_bits=30);he_params.assert_p_coverage(q,p,dnum)
 secure=he_params.qp_is_secure(q,p,131072,he_params.SecurityTarget(128,he_params.CostModel.CLASSICAL));assert secure
 rings[f'{depth}/{dnum}']={'depth':depth,'dnum':dnum,'q':q,'p':p,'actual_qp_bits':sum(math.log2(x) for x in q+p),'secure_128_classical':secure,'raw_rotation_key_bytes':2*dnum*131072*(len(q)+len(p))*4,'partition_width':(len(q)+dnum-1)//dnum}
for model,problem in [('bert','p2_bert'),('qwen','p4_qwen')]:
 folder=ROOT/'cadathlon/agent_pset'/problem;oracle=load_module(folder/'src/reference.py')
 for number in range(1,7):
  case=json.loads((folder/f'test/test{number}.in').read_text());made=oracle.make_case(case)
  weights,embeddings=made[:2];mask=made[2] if model=='bert' else None
  b,s,h=embeddings.shape;f=packed_feature_width(model,case['config']);period=(1<<(b-1).bit_length())*(1<<(s-1).bit_length())**2*f
  layout=PackedPairLayout(b,s,f,max(period,2*f));golden=np.asarray(json.loads((folder/f'test/test{number}.out').read_text())['logits'])
  base=plan_packed_transformer(model,case['config'],weights,batch=b,sequence=s,slots=65536,attention_mask=mask)
  deep=base['depth']>52
  for depth,dnum in (((49,4),(50,5)) if deep else (reference_ring_profile(base['depth']),)):
   graph=object.__new__(BoundedPackedTransformer);graph.layout=layout;graph.features=f
   arithmetic=planning.Arithmetic(layout.slots,deep);observations=[]
   contract=refresh_plan(131072,period,(4,4),max_level=depth) if deep else None
   if deep:
    def ideal_refresh(value,*,public_bound):
     assert np.max(np.abs(value.slots))<=public_bound+1e-9
     observations.append({'fixture_input_min':float(np.min(value.slots)),'fixture_input_max':float(np.max(value.slots)),'input_level':value.level,'public_bound':public_bound,'provenance':getattr(value,'_public_bound_provenance',None)})
     return planning.Value(value.slots,contract['output_level'])
    arithmetic.refresher=SimpleNamespace(input_level=2,output_level=contract['output_level'],refresh=ideal_refresh)
   graph.arithmetic=arithmetic;graph.ctx=arithmetic.ctx
   result=graph.evaluate(planning.Value(layout.pack_client_embeddings(embeddings)[0],depth),model,case['config'],weights,public_toy_approximations(model),mask)
   actual=layout.unpack_client_features([result.slots],output_features=golden.shape[-1])[:,0 if model=='bert' else -1]
   error=metrics(actual,golden)
   gate=(bool(np.all(np.abs(actual-golden)<=.02+.01*np.abs(golden))) and bool(np.array_equal(actual[:,1]>=actual[:,0],golden[:,1]>=golden[:,0]))) if model=='bert' else error['max_abs_error']<=.2 and error['nrmse']<=.1
   rotations=set(base['rotations'])|(set(contract['rotations']) if deep else set());ring=rings[f'{depth}/{dnum}']
   row={'model':model,'case':number,'deep_profile_applies':deep,'circuit_depth':base['depth'],'depth':depth,'dnum':dnum,'period':period,'output_capacity':contract['output_level'] if deep else None,'final_level':result.level,'refresh_count':len(arithmetic.refresh_events),'refresh_events':arithmetic.refresh_events,'fixture_refresh_observations':observations,'rotation_count':len(rotations),'rotations':sorted(rotations),'all_rotation_key_bytes':len(rotations)*ring['raw_rotation_key_bytes'],'ideal_errors':error,'gate_passed':gate}
   rows.append(row)
  print(model,number,[(r['depth'],r['dnum'],r['refresh_count'],r['final_level'],round(r['all_rotation_key_bytes']/2**30,2)) for r in rows if r['model']==model and r['case']==number],flush=True)
end=hashes();report={'scope':__doc__,'comparison':'For nine deep fixtures compare existing49/d4 versus prospective50/d5. Three shallow fixtures retain their existing leveled profile; no automatic bounded refresh is introduced. Exact generated Q/P and ideal schedule events are retained. Public conditional domains are checked but CKKS/refresh noise is excluded.','no_device_or_key_generation':True,'degree':131072,'scale_bits':56,'first_bits':61,'deep_budgets':[4,4],'rings':rings,'rows':rows,'all_ideal_gates_passed':all(r['gate_passed'] for r in rows),'source_hashes_start':start,'source_hashes_end':end,'source_files_unchanged':start==end,'elapsed_s':time.perf_counter()-started,'nonmonotonic_counterexample':'Separate exact Qwen2 sweep: depth52 gives2 ideal refreshes, depth53 gives3. Both have actual secure generated Q/P at dnum6 or8. More initial levels do not imply fewer refreshes in this greedy circuit schedule.'}
args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('unchanged',start==end,'ideal gates',report['all_ideal_gates_passed'],flush=True)
