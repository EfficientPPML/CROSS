import json
import numpy as np
import pytest
from full_metrics import check_logits

def fixture():
    expected=np.linspace(-1.,1.,151936,dtype=np.float64);blocks=[]
    for start in range(0,151936,4096):
        x=np.zeros(65536);piece=expected[start:start+4096];x[:len(piece)]=piece;blocks.append(x)
    return tuple(blocks),expected

def test_all38_blocks_and_inactive_padding():
    x,y=fixture();r=check_logits(x,y);assert r['passed'] and r['physical_slots_checked']==2490368 and r['max_abs_error']==0. and r['predicted_token']==151935
    x[-1][384]=2e-8;r=check_logits(x,y);assert not r['passed'] and r['max_abs_error']==2e-8
    json.dumps(r,allow_nan=False)

@pytest.mark.parametrize('kind',['nan','inf','overflow','shape','dtype','reference_nan','count'])
def test_failed_extremes_always_json_safe(kind):
    x,y=fixture()
    if kind=='nan':x[0][0]=float('nan')
    elif kind=='inf':x[-1][-1]=float('inf')
    elif kind=='overflow':x[0][0]=np.finfo(float).max;y[0]=-np.finfo(float).max
    elif kind=='shape':x=(x[0][:-1],*x[1:])
    elif kind=='dtype':x=(x[0].astype(np.complex128),*x[1:])
    elif kind=='reference_nan':y[-1]=float('nan')
    elif kind=='count':x=x[:-1]
    r=check_logits(x,y);assert not r['passed'] and r['max_abs_error']is None
    json.dumps(r,allow_nan=False)
