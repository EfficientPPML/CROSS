import hashlib
import numpy as np
import pytest
from prepare_gamma_fixtures import validate_gamma,no_modules

def test_exact_gamma_bytes_and_independent_copy():
    x=np.arange(128,dtype=np.float64);meta=dict(group_width=128,gamma_f64le_sha256=hashlib.sha256(x.astype('<f8').tobytes()).hexdigest())
    out=validate_gamma('synthetic',x,meta);x[0]=99
    assert out[0]==0
@pytest.mark.parametrize('value',[np.zeros(127),np.zeros(128,dtype=np.float32),np.full(128,np.nan),np.ones(128)])
def test_wrong_shape_dtype_nonfinite_hash(value):
    meta=dict(group_width=128,gamma_f64le_sha256=hashlib.sha256(np.zeros(128,dtype='<f8').tobytes()).hexdigest())
    with pytest.raises(ValueError):validate_gamma('synthetic',value,meta)
def test_module_construction_guard_restores():
    import torch
    original=torch.nn.Module.__init__
    with no_modules():
        with pytest.raises(RuntimeError):torch.nn.Linear(2,2)
        assert torch.tensor([1.,2.],dtype=torch.float64).numpy().dtype==np.dtype('float64')
    assert torch.nn.Module.__init__ is original
