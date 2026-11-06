"""Explicit identity refresh and primitive doubles; no cryptographic execution."""
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence,Tuple
import ast,copy,json,math,sys
import pytest
from test_initial_ideal import ideal_primitives
from ideal_primitives import Value,DELTA,Q,install_matrix_class
import encrypted_refresh
import encrypted_scaled_newton as newton
import encrypted_bert_rational_tanh as tanh
HERE=Path(__file__).resolve().parent
class IdentityRefresh:
    input_level=2;meta_precision=10;public_bound=128.
    def __init__(self,arithmetic):
        self.ctx=arithmetic.ctx;self.evaluator=arithmetic
        self.engine=SimpleNamespace(ctx=copy.copy(self.ctx),_cache=self.ctx._param_cache,_num_slots=65536,_degree=131072,_m=262144,
            _gap=1,_sparse=False,_composite_degree=2,_level_budget=[4,4],_dc_subtract=False)
        self.plan=encrypted_refresh.refresh_plan(131072,65536,(4,4),max_level=49)
        def forbidden(*args,**kwargs):raise AssertionError('crypto helper called by identity refresh')
        self.compiled_primitives=tuple(SimpleNamespace(ctx=self.ctx,**{name:forbidden}) for name in ('rotate','relinearize','rescale','multiply_encoded','multiply'))
        self.calls=[];self.inject=None
        cache=self.ctx._param_cache;cache.raw_rotation_keys={k:{'a':object(),'b':object()} for k in self.plan['rotations']}
        cache.coef_maps={k:object() for k in self.plan['rotations']};cache.per_level_rotation_keys=False
    @property
    def output_level(self):return self.plan['output_level']
    def refresh(self,value,*,public_bound):
        assert public_bound==2. and value.level>=2
        self.calls.append((value.level,value._ckks_scale,public_bound))
        result=Value(value.data,25,math.nextafter(DELTA,math.inf if len(self.calls)%2 else -math.inf))
        if self.inject is not None:self.inject(value,result)
        return result
@pytest.fixture(autouse=True)
def primitives(ideal_primitives,monkeypatch):
    install_matrix_class(monkeypatch)
    tree=ast.parse((HERE/'checkout/jaxite_word/bootstrapping.py').read_text());names={'_select_layers','_reduce_rotation','_collapsed_hoisted_plan','_bootstrap_rotation_indices'}
    nodes=[x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name in names]
    original=next(x for x in tree.body if isinstance(x,ast.ClassDef) and x.name=='Bootstrap')
    methods=[x for x in original.body if isinstance(x,ast.FunctionDef) and x.name in ('compute_bootstrap_nq','required_rotation_indices')]
    ns=dict(math=math,Sequence=Sequence,Tuple=Tuple);stub=ast.ClassDef(name='Bootstrap',bases=[],keywords=[],body=methods,decorator_list=[],type_params=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes+[stub],type_ignores=[])),'exact_public_bootstrap_plan','exec'),ns)
    monkeypatch.setitem(sys.modules,'jaxite_word.bootstrapping',SimpleNamespace(Bootstrap=ns['Bootstrap']))
    monkeypatch.setattr(encrypted_refresh,'BoundedSecureRefresh',IdentityRefresh)
    def affine(arithmetic,value,weight,bias,*,output_scale):
        arithmetic.calls.append('initial_affine');L=value.level;divisor=Q[2*L]*Q[2*L+1]
        encoded=output_scale*divisor/value._ckks_scale
        return Value(value.data*weight+bias,L-1,value._ckks_scale*encoded/divisor)
    def integer(arithmetic,value,multiplier,*,bias=0.):
        arithmetic.calls.append('integer_affine');return Value(value.data*multiplier+bias,value.level,value._ckks_scale)
    monkeypatch.setattr(newton,'affine_at_scale',affine);monkeypatch.setattr(newton,'integer_affine',integer);monkeypatch.setattr(tanh,'integer_affine',integer)
