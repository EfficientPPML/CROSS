"""Full-plan identity refresh double; never an encrypted refresh implementation."""
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence,Tuple
import ast,math,sys
from test_softmax_tile import IdentityRefresh as BaseRefresh

HERE=Path(__file__).resolve().parent

class IdentityRefresh(BaseRefresh):
    def __init__(self,arithmetic):
        super().__init__(arithmetic)
        from encrypted_refresh import refresh_plan
        self.plan=refresh_plan(131072,65536,(4,4),max_level=49)
        def forbidden(*args,**kwargs):
            raise AssertionError('compiled crypto helper invoked in identity-refresh ideal test')
        self.compiled_primitives=tuple(SimpleNamespace(ctx=arithmetic.ctx,**{method:forbidden})
            for method in ('rotate','relinearize','rescale','multiply_encoded','multiply'))

def install_public_bootstrap_plan(monkeypatch):
    """Run unchanged public planner methods without importing a crypto backend."""
    tree=ast.parse((HERE/'checkout/jaxite_word/bootstrapping.py').read_text())
    names={'_select_layers','_reduce_rotation','_collapsed_hoisted_plan','_bootstrap_rotation_indices'}
    nodes=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    original=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=='Bootstrap')
    methods=[node for node in original.body if isinstance(node,ast.FunctionDef)
             and node.name in ('compute_bootstrap_nq','required_rotation_indices')]
    namespace=dict(math=math,Sequence=Sequence,Tuple=Tuple)
    stub=ast.ClassDef(name='Bootstrap',bases=[],keywords=[],body=methods,decorator_list=[],type_params=[])
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes+[stub],type_ignores=[])),
                 'exact_public_bootstrap_plan','exec'),namespace)
    monkeypatch.setitem(sys.modules,'jaxite_word.bootstrapping',SimpleNamespace(Bootstrap=namespace['Bootstrap']))
