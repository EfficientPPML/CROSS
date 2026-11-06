"""Exact public planner plus identity-refresh doubles, no HE backend."""
from pathlib import Path
from types import SimpleNamespace,ModuleType
from typing import Sequence,Tuple
import ast,math,sys
from ideal_primitives import Value,DELTA
P=Path(__file__).resolve().parent
def public_plan():
    tree=ast.parse((P/'reference_sources/bootstrapping.py').read_text())
    names={'_select_layers','_reduce_rotation','_collapsed_hoisted_plan','_bootstrap_rotation_indices'}
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Bootstrap')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('compute_bootstrap_nq','required_rotation_indices')]
    stub=ast.ClassDef(name='Bootstrap',bases=[],keywords=[],body=methods,decorator_list=[],type_params=[])
    ns=dict(math=math,Sequence=Sequence,Tuple=Tuple)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes+[stub],type_ignores=[])),'copied_public_bootstrap','exec'),ns)
    m=ModuleType('jaxite_word.bootstrapping');m.Bootstrap=ns['Bootstrap'];sys.modules[m.__name__]=m
    refresh=ast.parse((P/'reference_sources/encrypted_refresh.py').read_text())
    fn=next(n for n in refresh.body if isinstance(n,ast.FunctionDef) and n.name=='refresh_plan')
    exec(compile(ast.Module(body=[fn],type_ignores=[]),'copied_public_refresh_plan','exec'),ns)
    return ns['refresh_plan']
REFRESH_PLAN=public_plan()
class IdentityRefresh:
    input_level=2;output_level=25;meta_precision=10;public_bound=128.
    def __init__(self,a):
        self.ctx=a.ctx;self.plan=REFRESH_PLAN(131072,65536,(4,4),max_level=49);self.calls=[];self.inject=None
        self.evaluator=SimpleNamespace(ctx=a.ctx,scale=DELTA,strategy='power',progress=None,refresher=None)
        self.engine=SimpleNamespace(ctx=a.ctx,_cache=a.ctx._param_cache,_degree=131072,_m=262144,_num_slots=65536,_gap=1,_sparse=False,_composite_degree=2,_level_budget=[4,4],_dc_subtract=False)
        def deny(*args,**kwargs):raise AssertionError('real compiled helper forbidden in ideal refresh')
        self.compiled_primitives=tuple(SimpleNamespace(ctx=a.ctx,**{name:deny}) for name in ('rotate','relinearize','rescale','multiply_encoded','multiply'))
    def refresh(self,value,*,public_bound):
        assert public_bound in (1.,2.,4.,8.,16.,32.,64.) and value.level>=2
        self.calls.append((value.level,public_bound));out=Value(value.polynomial,25,math.nextafter(DELTA,math.inf if len(self.calls)%2 else -math.inf))
        if self.inject is not None:self.inject(value,out)
        return out
m=ModuleType('encrypted_refresh');m.refresh_plan=REFRESH_PLAN;m.BoundedSecureRefresh=IdentityRefresh;sys.modules[m.__name__]=m
