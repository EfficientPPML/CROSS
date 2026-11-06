"""Slot/metadata test doubles; never an encrypted implementation or refresh proof."""
from pathlib import Path
from types import SimpleNamespace
from numbers import Integral,Real
from typing import Optional,Tuple
import ast,math,sys
import numpy as np
from test_initial_ideal import Arithmetic as BaseArithmetic,Value as BaseValue,Q,DELTA
from encrypted_streaming_linear import RowTileLayout
from encrypted_chained_streaming_linear import ChainedStreamingLinear
from encrypted_polynomial import EncryptedPolynomialEvaluator
HERE=Path(__file__).resolve().parent

class Value(BaseValue):
    degree_layout=(256,512)
    def __init__(self,data,level=49,scale=DELTA):
        super().__init__(data,level,scale)
        # A public payload-shape placeholder exercises the existing queue
        # planner. No limb array or ciphertext is represented by this object.
        self.polynomial=SimpleNamespace(shape=(1,2,256,512,2*(level+1)),dtype='uint32')

class Arithmetic(BaseArithmetic):
    strategy='power'
    polynomial_depth=EncryptedPolynomialEvaluator.polynomial_depth
    evaluate=EncryptedPolynomialEvaluator.evaluate
    def __init__(self):
        super().__init__();self.public_multiplies=0;self.ctx.dnum=8;self.ctx.degree_layout=(256,512);self.ctx.max_level=49
        self.ctx.q_towers=list(Q);self.ctx.p_towers=list(self.ctx._param_cache.p_towers)
        cache=self.ctx._param_cache;cache.degree=131072;cache.dnum=8;cache.degree_layout=(256,512);cache.composite_degree=2
        cache.num_q_at_level=lambda level:2*(level+1)
        self.ctx.he_add=AddAccessor(self)
    def _ensure_capacity(self,values,required):
        if any(v.level<required for v in values):raise ValueError('ideal static capacity exhausted')
        return tuple(values)
    def _prepare(self,level):pass
    def _finish(self,value):return value
    def _public_slots(self,value):return np.broadcast_to(np.asarray(value,dtype=np.float64),(65536,))
    def _encode_public(self,slots,level,scale):return Value(slots,level,scale)
    def rotate(self,value,shift):
        if shift%65536==0:return value
        if shift%65536 not in self.ctx._param_cache.raw_rotation_keys and shift%65536:raise ValueError('missing public test rotation')
        self.calls.append('rotate');return Value(np.roll(value.data,-shift),value.level,value._ckks_scale)
    def add(self,left,right,*,subtract=False):
        if not math.isclose(left._ckks_scale,right._ckks_scale,rel_tol=1e-12):raise ValueError('actual additive metadata mismatch')
        self.calls.append('add')
        return Value(left.data-right.data if subtract else left.data+right.data,min(left.level,right.level),left._ckks_scale)
    def weighted_sum(self,values,weights,*,bias=0.):
        values,weights=tuple(values),tuple(weights);self.calls.append('weighted_sum');self.public_multiplies+=len(values)
        level=min(v.level for v in values);divisor=Q[2*level]*Q[2*level+1]
        before=self.scale*divisor;scales=[float(v._ckks_scale)*float(before/v._ckks_scale) for v in values]
        if not all(math.isclose(scales[0],s,rel_tol=1e-12) for s in scales):raise ValueError('term scales differ')
        out=np.array(values[0].data*weights[0],dtype=np.float64,copy=True)
        for v,w in zip(values[1:],weights[1:]):out+=v.data*w
        out+=bias
        return Value(out,level-1,scales[0]/divisor)
    def multiply(self,left,right):
        self.calls.append('multiply');level=min(left.level,right.level)
        return Value(left.data*right.data,level-1,left._ckks_scale*right._ckks_scale/(Q[2*level]*Q[2*level+1]))

class AddAccessor:
    def __init__(self,arithmetic):self.arithmetic=arithmetic
    def __getitem__(self,level):return SimpleNamespace(add_plain=self.arithmetic.add)

class IdealLinear(ChainedStreamingLinear):
    def __init__(self,arithmetic):
        self.arithmetic=arithmetic;self.ctx=arithmetic.ctx;self.layout=RowTileLayout(65536)
        # All linear_tile validation, diagonal construction, baby rotations,
        # Horner giant accumulation and bias code remains the actual module.

def install_matrix_class(monkeypatch):
    """Execute unchanged public matrix-planner class without importing JAX."""
    common=HERE/'checkout/cadathlon/common'
    core=HERE/'checkout/jaxite_word/bsgs.py'
    fn=next(x for x in ast.parse(core.read_text()).body if isinstance(x,ast.FunctionDef) and x.name=='compute_bsgs_params')
    cls=next(x for x in ast.parse((common/'encrypted_packed_qwen.py').read_text()).body if isinstance(x,ast.ClassDef) and x.name=='MatrixDiagonals')
    ns=dict(Integral=Integral,Real=Real,Optional=Optional,Tuple=Tuple,math=math,np=np)
    exec(compile(ast.Module(body=[fn,cls],type_ignores=[]),'exact_public_matrix_planner','exec'),ns)
    module=sys.modules['encrypted_packed_qwen'];monkeypatch.setattr(module,'MatrixDiagonals',ns['MatrixDiagonals'],raising=False)

