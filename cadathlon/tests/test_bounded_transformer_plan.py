"""Conditional ideal-value planning only; no encrypted/refresh-error claim.

Run with JAX_PLATFORMS=cpu and CROSS_REPO pointing to the reviewed checkout.
The companion draft modules are imported from this script directory.
"""
import sys,json,math,os,hashlib
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'cadathlon/common')]
import numpy as np
from encrypted_packed_transformer import PackedTransformer,PackedPairLayout,packed_feature_width,plan_packed_transformer
from encrypted_bounded_arithmetic import BoundedArithmetic
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_transformer import public_toy_approximations
from support import load_module,metrics
test=load_module(ROOT/'cadathlon/tests/test_packed_transformer.py')
Value=test._Value

class Primitives(test.PlanningArithmetic):
    @staticmethod
    def clone(source,slots,level):
        result=Value(slots,level)
        result.__dict__.update(source.__dict__)
        result.slots,result.level=slots,level
        return result
    def _lower(self,value,level):
        assert level<=value.level
        return value if level==value.level else self.clone(value,value.slots,level)
    def add(self,left,right,subtract=False):
        return self.clone(left,left.slots-right.slots if subtract else left.slots+right.slots,min(left.level,right.level))
    def weighted_sum(self, values, weights, bias=0.):
        values=self._ensure_capacity(tuple(values),1)
        return self.clone(values[0],sum(value.slots*weight for value,weight in zip(values,weights))+bias,min(x.level for x in values)-1)
    def multiply(self,left,right):
        left,right=self._ensure_capacity((left,right),1)
        return self.clone(left,left.slots*right.slots,min(left.level,right.level)-1)
    def _encode_public(self,slots,level,scale):
        value=super()._encode_public(slots,level,scale)
        return self.declare_bound(value,float(np.max(np.abs(slots))),'public bias constant')

class Arithmetic(BoundedArithmetic,Primitives):
    def __init__(self,slots,refresh):
        Primitives.__init__(self,slots)
        self.refresh_events=[]
        if refresh:
            def ideal_refresh(value,*,public_bound):
                assert np.max(np.abs(value.slots))<=public_bound+1e-9
                return Value(value.slots,25)
            self.refresher=SimpleNamespace(input_level=2,output_level=25,refresh=ideal_refresh)
        else:self.refresher=None
    def rotate(self,value,shift):
        if not shift % self.ctx.num_slots:return value
        self.rotations.add(shift % self.ctx.num_slots)
        result=self.clone(value,np.roll(value.slots,-shift),value.level)
        return self.declare_bound(self.forget_relations(result),self.public_bound(value),'public permutation')
    def declare_bound(self,value,bound,reason='public circuit contract'):
        assert float(np.max(np.abs(value.slots)))<=bound+1e-8*max(1,bound),(reason,bound,float(np.max(np.abs(value.slots))))
        return BoundedArithmetic.declare_bound(self,value,bound,reason)

import pytest

@pytest.mark.parametrize('model,number',[(model,n) for model in ('bert','qwen') for n in range(1,7)])
def test_complete_bounded_graph_preserves_ideal_contracts_and_fixture_gate(model,number):
    problem='p2_bert' if model=='bert' else 'p4_qwen'
    folder=ROOT/'cadathlon/agent_pset'/problem
    oracle=load_module(folder/'src/reference.py')
    case=json.loads((folder/f'test/test{number}.in').read_text());args=oracle.make_case(case)
    weights,embeddings=args[:2];mask=args[2] if model=='bert' else None
    b,s,h=embeddings.shape;f=packed_feature_width(model,case['config'])
    period=(1<<(b-1).bit_length())*(1<<(s-1).bit_length())**2*f
    layout=PackedPairLayout(b,s,f,max(period,2*f))
    plan=plan_packed_transformer(model,case['config'],weights,batch=b,sequence=s,slots=layout.slots,attention_mask=mask)
    graph=object.__new__(BoundedPackedTransformer);graph.layout=layout;graph.features=f
    graph.arithmetic=Arithmetic(layout.slots,plan['depth']>52);graph.ctx=graph.arithmetic.ctx
    initial=49 if plan['depth']>52 else plan['depth']
    result=graph.evaluate(Value(layout.pack_client_embeddings(embeddings)[0],initial),model,case['config'],weights,public_toy_approximations(model),mask)
    golden=np.asarray(json.loads((folder/f'test/test{number}.out').read_text())['logits'])
    actual=layout.unpack_client_features([result.slots],output_features=golden.shape[-1])[:,0 if model=='bert' else -1]
    errors=metrics(actual,golden)
    if model=='bert':
        assert np.all(np.abs(actual-golden)<=.02+.01*np.abs(golden))
        assert np.array_equal(actual[:,1]>=actual[:,0],golden[:,1]>=golden[:,0])
    else:
        assert errors['nrmse']<=.10 and errors['max_abs_error']<=.20
    # Bound checks run at every declaration and every fresh-wrapper refresh.
    assert result.level>=0
    assert max((event['public_bound'] for event in graph.arithmetic.refresh_events),default=0)<=16
