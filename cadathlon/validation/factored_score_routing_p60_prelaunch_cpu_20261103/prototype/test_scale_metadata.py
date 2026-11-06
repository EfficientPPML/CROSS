"""Execute unchanged production method ASTs on payload stubs; metadata only, no HE."""
from pathlib import Path
import ast,copy,json,math,sys,types
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;COMMON=HERE.parent/'checkout/cadathlon/common'
sys.path.insert(0,str(COMMON))
import proof
AUDIT=json.loads((HERE.parent/'profile_audit.json').read_text())
PROFILE=next(p for p in AUDIT['profiles'] if p['name']=='depth49_p60_d8')
Q=PROFILE['q'];DELTA=PROFILE['canonical_scale']

class Payload:
    dtype=np.dtype('uint32')
    def __init__(self,data=0.):self.data=np.asarray(data)
    def devices(self):return {'fake-cpu'}
    def astype(self,dtype):return self

class Value:
    modulus_dtype=np.uint32
    def __init__(self,scale,level,*,data=0.):
        self._ckks_scale=scale;self._ckks_nsd=1;self.level=level;self.moduli=Q[:2*(level+1)]
        self.polynomial=Payload(data);self.ntt_ctx=None;self.num_moduli=len(self.moduli)
        self.shape=(1,2,1,1,self.num_moduli)
    def _clone_with_payload(self,payload,**changes):
        result=copy.copy(self);result.polynomial=payload
        for key,value in changes.items():setattr(result,key,value)
        result.num_moduli=len(result.moduli);result.level=result.num_moduli//2-1;result.shape=(1,2,1,1,result.num_moduli)
        return result

def method(path,class_name,name):
    root=ast.parse(path.read_text());cls=next(n for n in root.body if isinstance(n,ast.ClassDef) and n.name==class_name)
    fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name)
    namespace={'np':np,'math':math,'Polynomial':Value,'finite_field':types.SimpleNamespace(check_binary_ct_operands=lambda *a,**k:None)}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path),'exec'),namespace)
    return namespace[name]

WEIGHTED=method(COMMON/'encrypted_polynomial.py','EncryptedPolynomialEvaluator','weighted_sum')
PTCT=method(COMMON/'encrypted_primitives.py','CompiledPlaintextMultiply','multiply')
RESCALE=method(COMMON/'encrypted_primitives.py','CompiledRescale','rescale')
ADD=method(HERE.parent/'checkout/jaxite_word/headd.py','_HEAddKernel','add')

@pytest.fixture
def executor(monkeypatch):
    checks=[]
    monkeypatch.setitem(sys.modules,'finite_field',types.SimpleNamespace(check_ct_operand=lambda *a,**k:checks.append(a)))
    monkeypatch.setitem(sys.modules,'support',types.SimpleNamespace(require_secure_execution=lambda *a:None))
    monkeypatch.setitem(sys.modules,'jax',types.SimpleNamespace(block_until_ready=lambda value:value))
    cache=types.SimpleNamespace(batch=1,degree_layout=(1,1),ff_context_cls=object,
        q_moduli_at_level=lambda level:Q[:2*(level+1)],num_q_at_level=lambda level:2*(level+1),
        level_for_num_q=lambda count:count//2-1,get_sliced_ntt_q=lambda level:object())
    ctx=types.SimpleNamespace(_param_cache=cache)
    ptct=types.SimpleNamespace(ctx=ctx)
    ptct.multiply_encoded=lambda value,plain,level:value._clone_with_payload(Payload(value.polynomial.data*plain.data))
    ptct.multiply=types.MethodType(PTCT,ptct)
    rescaler=types.SimpleNamespace(ctx=ctx)
    def rescale(value,source,target):
        signature=('rescale',source,target,tuple(value.shape),'fake-cpu')
        rescaler.binding=(signature,lambda payload:Payload(payload.data))
        return RESCALE(rescaler,value,source,target)
    rescaler.rescale=rescale
    class AddTable:
        def __getitem__(self,level):
            kernel=types.SimpleNamespace(ff_context_cls=object,num_moduli=2*(level+1),moduli=Q[:2*(level+1)],
                _add_unchecked=lambda left,right:Payload(left.data+right.data))
            return types.SimpleNamespace(add=types.MethodType(ADD,kernel))
    ctx.he_add=AddTable()
    evaluator=types.SimpleNamespace(scale=DELTA,ctx=ctx,_plaintext_multiply=ptct,_rescale=rescaler,
        _public_slots=lambda value:np.asarray(value),_ensure_capacity=lambda values,levels:values,
        _level=lambda value:value.level,_prepare=lambda level:None,_lower=lambda value,level:value,
        _encode_public=lambda weight,level,scale:Value(scale,level,data=weight),_finish=lambda value:value)
    return lambda scales,level,weights:WEIGHTED(evaluator,[Value(s,level,data=2.) for s in scales],weights),checks


@pytest.mark.parametrize('level',(1,5,16,17,24,38,49))
@pytest.mark.parametrize('factor',(1.,.73,1.17))
def test_exact_scale_plan_runs_actual_production_metadata(executor,level,factor):
    evaluate,_=executor;source=DELTA*factor;divisor=math.prod(Q[2*level:2*level+2])
    actual=evaluate([source],level,[.125])
    assert actual._ckks_scale==proof.weighted_scale(source,DELTA,divisor)
    assert actual.level==level-1 and actual._ckks_nsd==1


def test_one_ulp_regression_and_real_noncanonical_product_recurrence(executor):
    evaluate,_=executor;actual=evaluate([DELTA],38,[1.])
    assert actual._ckks_scale==math.nextafter(DELTA,0.) and actual._ckks_scale!=DELTA
    public=json.loads((HERE.parent/'public_plan.json').read_text())['plan']
    product=(DELTA*DELTA)/math.prod(Q[12:14])
    assert public['noncanonical_actual_scale']==product and not math.isclose(product,DELTA,rel_tol=1e-12)
    for case,current in (('canonical',DELTA),('noncanonical',product)):
        stages=[]
        for level in (5,4,3,2):
            current=evaluate([current],level,[.125])._ckks_scale;stages.append(current)
        assert public['exact_stage_scales'][case]==[stages[0],stages[1],stages[1],stages[1],stages[2],stages[3]]
