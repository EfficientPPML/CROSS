"""Public CPU planning and interface checks only; never launches crypto or TPU."""
import hashlib,importlib.util,json,math,os,subprocess,sys
from pathlib import Path
if os.environ.get('JAX_PLATFORMS')!='cpu':raise RuntimeError('CPU-only planning required')
ROOT=Path('/tmp/runtime-followup-integration-20261103/checkout')
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import numpy as np
from encrypted_reference import reference_ring_profile
from encrypted_packed_transformer import plan_packed_transformer,PackedPairLayout
from encrypted_refresh import refresh_plan
from jaxite_word import he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from support import load_module
spec=importlib.util.spec_from_file_location('suite_planning',ROOT/'cadathlon/validation/check_transformer_suite.py')
suite=importlib.util.module_from_spec(spec);spec.loader.exec_module(suite)

def forbidden(*args,**kwargs):raise AssertionError('public planning must not generate keys')
for module in {key_gen,sys.modules.get('key_gen')}-{None}:
    for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):setattr(module,name,forbidden)

before={};cases={};interfaces={}
for model,problem_name in suite.PROBLEMS.items():
    problem=ROOT/'cadathlon/agent_pset'/problem_name
    implementation=problem/f'src/{model}_optimized_reference.py'
    before[model]=suite.source_manifest(problem,implementation)
    oracle=load_module(problem/'src/reference.py')
    for number in range(1,7):
        case=json.loads((problem/f'test/test{number}.in').read_text())
        ids=case['input_ids' if model=='bert' else 'token_ids'];batch,sequence=len(ids),len(ids[0])
        weights=oracle.toy_weights(case['config'],case['weight_seed'])
        graph=plan_packed_transformer(model,case['config'],weights,batch=batch,sequence=sequence,
            attention_mask=np.ones((batch,sequence),bool))
        degree=131072;slots=degree//2
        layout=PackedPairLayout(batch,sequence,graph['features'],slots)
        depth,dnum=reference_ring_profile(graph['depth'])
        q=composite_prime_gen(2,2*(depth+1),61,56,2*degree,31)
        p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
        target=he_params.SecurityTarget(128,he_params.CostModel.CLASSICAL)
        assert he_params.qp_is_secure(q,p,degree,target)
        he_params.assert_p_coverage(q,p,dnum)
        graph_rotations=set(graph['rotations']);rotations=set(graph_rotations);refresh=None
        if graph['depth']>52:
            budget=min(4,max(2,(layout.period-1).bit_length()))
            refresh=refresh_plan(degree,layout.period,(budget,budget),max_level=depth)
            rotations.update(refresh['rotations'])
        alpha=(len(q)+dnum-1)//dnum;parts=(len(q)+alpha-1)//alpha
        pair_bytes=2*parts*(len(q)+len(p))*degree*4
        required=len(rotations)*(pair_bytes+8192)+64*2**20
        cases[f'{model}/test{number}.in']={
            'batch':batch,'sequence':sequence,'layers':case['config']['num_hidden_layers'],
            'hidden_size':case['config']['hidden_size'],'features':graph['features'],
            'circuit_depth':graph['depth'],'layout_period':layout.period,'degree':degree,
            'chain_depth':depth,'dnum':dnum,'q_towers':q,'p_towers':p,
            'actual_log2_QP':math.log2(math.prod(q+p)),
            'security_ceiling':he_params.security_ceiling(degree,target),
            'graph_rotation_count':len(graph_rotations),'union_rotation_count':len(rotations),
            'rotations':sorted(rotations),'one_raw_key_pair_bytes':pair_bytes,
            'required_key_store_bytes_including_reserve':required,
            'required_key_store_GiB':required/2**30,
            'refresh':refresh,'expected_requests_warm1_repeat3':4}
    command=[sys.executable,str(problem/'src/runner.py'),'--preflight','--implementation',str(implementation)]
    completed=subprocess.run(command,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CROSS_ROOT=str(ROOT)),capture_output=True,text=True)
    interfaces[model]={'command':command,'returncode':completed.returncode,
        'stdout':completed.stdout,'stderr':completed.stderr}
    result=json.loads(completed.stdout)
    assert result['encryption_validated'] is False
    assert completed.returncode==(1 if model=='bert' else 0)
after={model:suite.source_manifest(ROOT/'cadathlon/agent_pset'/name,
       ROOT/'cadathlon/agent_pset'/name/f'src/{model}_optimized_reference.py')
       for model,name in suite.PROBLEMS.items()}
assert before==after
report={'planning_passed':True,'scope':'Public CPU-only no-key planning; no device-fit, HE or TPU success claim.',
    'source_hashes_start':before,'source_hashes_end':after,'source_files_unchanged':True,
    'case_count':len(cases),'cases':cases,'interfaces':interfaces,
    'max_key_store_bytes_including_reserve':max(c['required_key_store_bytes_including_reserve'] for c in cases.values())}
(HERE/'preflight_plan.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'planning_passed':True,'source_files_unchanged':True,'case_count':len(cases),
    'interfaces':{m:v['returncode'] for m,v in interfaces.items()},
    'max_key_store_GiB':report['max_key_store_bytes_including_reserve']/2**30,
    'cases':[{k:v for k,v in dict(case=name,**row).items() if k in ('case','circuit_depth','layout_period','chain_depth','dnum','union_rotation_count','required_key_store_GiB')}
             for name,row in cases.items()]},indent=2))
