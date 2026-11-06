"""Independent full-slot Chebyshev/rational reference and owner guard checks."""
from pathlib import Path
from collections import Counter
import copy,json,math,os
import numpy as np
import pytest
from ideal_refresh import primitives,ideal_primitives,IdentityRefresh
from ideal_primitives import Arithmetic,Value,Q,DELTA
from encrypted_bert_rational_tanh import BertTanhContract,BertRationalTanh,tanh_scale_plan
HERE=Path(__file__).resolve().parent;RAW=(HERE/'candidate_manifest.json').read_bytes()
def setup(progress=None):
    a=Arithmetic();r=IdentityRefresh(a);return BertRationalTanh(a,r,BertTanhContract(RAW),progress=progress),a,r

def oracle(x):
    m=json.loads(RAW);y=np.polynomial.chebyshev.chebval(x/64,m['tanh']['base_polynomial']['coefficients']);out={'base.output':y.copy()};seed=m['tanh_reciprocal']['seed']
    for i in range(1,7):
        d=1+y*y;assert np.all((d>=1)&(d<=2.1));inv=d*(-seed*seed)+2*seed
        for _ in range(4):inv=inv*(2-d*inv)
        y=2*y*inv;out[f'round{i}.output']=y.copy()
    return y,out
@pytest.mark.parametrize('level',(8,11,25,49))
@pytest.mark.parametrize('scale',(math.nextafter(DELTA,-math.inf),DELTA,math.nextafter(DELTA,math.inf)))
def test_all_slots_and_noncanonical_refreshed_scales(level,scale):
    owner,a,r=setup();x=np.linspace(-34,34,65536);x[32768]=0.;source=Value(x,level,scale)
    actual,captured=owner.evaluate_tile(source,capture_checkpoints=True);expected,checkpoints=oracle(x)
    error=float(np.max(np.abs(actual.data-expected)));assert error<1e-11
    for name,y in checkpoints.items():np.testing.assert_allclose(captured[name].data,y,rtol=0,atol=1e-11)
    for name,v in captured.items():
        stage=next(s for s in owner.last_plan['stages'] if s['name']==name)
        assert v.level==stage['level'] and v._ckks_scale==stage['scale']
        assert not any(n.startswith('_public_') for n in vars(v))
    assert actual.level>=2 and len(r.calls)==len(owner.last_plan['refresh_events'])
    counts=Counter(a.calls);assert counts['multiply']==68 and counts['weighted_sum']==2 and counts['initial_affine']==6 and counts['integer_affine']==36
    np.testing.assert_array_equal(source.data,x);assert not owner._active
    dest=os.environ.get('TANH_CASE_REPORT_DIR')
    if dest:
        p=Path(dest);p.mkdir(exist_ok=True)
        (p/f'{level}-{scale.hex()}.json').write_text(json.dumps(dict(level=level,scale=scale,slots=65536,max_abs=error,gate=1e-11,counts=dict(counts),refreshes=len(r.calls),plan=owner.last_plan),indent=2)+'\n')
@pytest.mark.parametrize('level',range(8,50))
def test_every_supported_level_plan(level):
    plan=tanh_scale_plan(Q,input_level=level,input_scale=DELTA,canonical_scale=DELTA,contract=BertTanhContract(RAW))
    L=level-6
    for row in plan['rounds']:
        if L<13:L=25;assert row['refresh_before']
        else:assert not row['refresh_before']
        L-=11;assert L==row['output_level']>=2
    assert plan['output_level']==L
@pytest.mark.parametrize('event',('base.map','base.output','round1.inverse1','round3.output','round6.output'))
@pytest.mark.parametrize('kind',('source','result','captured','contract','context'))
def test_callback_mutations_fail_immediately(event,kind):
    source=Value(np.zeros(65536),11);saved=[];at=[]
    def progress(name,result,counts):
        if name=='base.map':saved.append(result)
        if name!=event:return
        at.append(len(a.calls))
        if kind=='source':source._ckks_nsd=True
        elif kind=='result':result._ckks_nsd=True
        elif kind=='captured':saved[0]._ckks_nsd=True
        elif kind=='contract':object.__setattr__(owner.contract,'seed',.5)
        else:a.ctx.secret_key=object()
    owner,a,r=setup(progress)
    with pytest.raises((ValueError,TypeError)):owner.evaluate_tile(source,capture_checkpoints=True)
    assert at==[len(a.calls)] and not owner._active and owner.last_plan is None
@pytest.mark.parametrize('kind',('strategy','layout','contract','binding','low_level','capture'))
def test_entry_rejections(kind):
    owner,a,r=setup();source=Value(np.zeros(65536),11);capture=False
    if kind=='strategy':a.strategy='chebyshev'
    elif kind=='layout':source.degree_layout=(512,256)
    elif kind=='contract':object.__setattr__(owner.contract,'seed',.5)
    elif kind=='binding':owner.binding=copy.copy(owner.binding)
    elif kind=='low_level':source=Value(np.zeros(65536),7)
    else:capture=1
    with pytest.raises((ValueError,TypeError)):owner.evaluate_tile(source,capture_checkpoints=capture)
    assert not a.calls and not r.calls and not owner._active

def test_cancel_reentry_annotations():
    source=Value(np.zeros(65536),11);seen=[]
    def progress(name,result,counts):
        seen.append(result)
        for v in seen:v._public_abs_bound=0.
        with pytest.raises(ValueError,match='reentered'):owner.evaluate_tile(source)
    owner,a,r=setup(progress);result,checkpoints=owner.evaluate_tile(source,capture_checkpoints=True)
    assert all(not any(n.startswith('_public_') for n in vars(v)) for v in checkpoints.values())
    error=KeyboardInterrupt('primary')
    def cancel(*args):raise error
    owner,a,r=setup(cancel)
    with pytest.raises(KeyboardInterrupt) as exc:owner.evaluate_tile(source)
    assert exc.value is error and not owner._active
