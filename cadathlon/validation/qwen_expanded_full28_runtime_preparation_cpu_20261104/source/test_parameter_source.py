"""Public callback/authentication seams with synthetic tiny arrays only."""
import hashlib
import numpy as np
import pytest
from full_synthetic import contract
from encrypted_qwen_full_model import QwenParameterSource

def source():
    calls=[]
    def loader(name):calls.append(name);return np.zeros((1,),dtype=np.float64)
    c=contract((0,));return QwenParameterSource(c,matrix_loader=loader,gamma_loader=loader),calls

@pytest.mark.parametrize('kind',['matrix','gamma'])
def test_bad_shape_rejects_after_one_requested_load(kind):
    p,calls=source();name='model.layers.0.self_attn.q_proj.weight' if kind=='matrix' else 'final.inverse_sqrt'
    with pytest.raises(ValueError,match='shape'):getattr(p,kind)(name)
    assert calls==[name]

@pytest.mark.parametrize('kind,name',[('matrix','unlisted'),('matrix',False),('gamma','unlisted')])
def test_unknown_public_tensor_rejected_before_callback(kind,name):
    p,calls=source()
    with pytest.raises((ValueError,KeyError,TypeError)):getattr(p,kind)(name)
    assert not calls

@pytest.mark.parametrize('name',['matrix','gamma','check','_freeze','matrix_loader','gamma_loader'])
def test_bound_loader_accessors(name):
    p,calls=source();setattr(p,name,lambda *a,**k:None)
    with pytest.raises(ValueError):QwenParameterSource.check(p)
    assert not calls

def test_freeze_exact_bytes_immutable():
    a=np.array([[1.,-2.],[3.,4.]],dtype=np.float64);pin=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()
    result=QwenParameterSource._freeze(a,(2,2),pin);a[:]=0
    assert np.array_equal(result,[[1,-2],[3,4]]) and not result.flags.writeable
    with pytest.raises(ValueError):result.setflags(write=True)

@pytest.mark.parametrize('kind',['dtype','shape','nan','inf','digest','subclass'])
def test_freeze_invalid_public_arrays(kind):
    a=np.ones((2,2),dtype=np.float64);pin=hashlib.sha256(a.astype('<f8').tobytes()).hexdigest()
    if kind=='dtype':a=a.astype(np.float32)
    elif kind=='shape':a=a.ravel()
    elif kind=='nan':a[0,0]=float('nan')
    elif kind=='inf':a[0,0]=float('inf')
    elif kind=='digest':a[0,0]=2.
    elif kind=='subclass':a=a.view(type('Subclass',(np.ndarray,),{}))
    with pytest.raises(ValueError):QwenParameterSource._freeze(a,(2,2),pin)

def test_callback_cannot_mutate_binding():
    p,calls=source()
    def changed(name):p.matrix_loader=lambda n:None;return np.zeros((1,))
    p=QwenParameterSource(p.contract,matrix_loader=changed,gamma_loader=lambda n:None)
    with pytest.raises(ValueError,match='source changed'):p.matrix('model.layers.0.self_attn.q_proj.weight')
