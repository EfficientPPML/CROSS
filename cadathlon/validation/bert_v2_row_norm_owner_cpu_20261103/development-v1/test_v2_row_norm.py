"""Independent all-slot ideal/metadata/ownership tests; no encrypted execution."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,copy,math
import numpy as np
import pytest
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE/'checkout/cadathlon/common')]
import encrypted_bert_v2_row_norm as norm
from encrypted_polynomial import EncryptedPolynomialEvaluator
P=next(x for x in json.loads((HERE/'actual_profile_audit.json').read_text())['profiles'] if x['name']=='depth49_p60_d8')
Q=P['q'];DELTA=P['canonical_scale'];RAW=(HERE/'candidate_manifest.json').read_bytes();SLOTS=65536

class Value:
    batch=1;num_elements=2;degree=131072;degree_layout=(256,512)
    def __init__(self,data,level=20,scale=DELTA):
        self.data=np.asarray(data,dtype=np.float64).copy();self.polynomial=self.data;self.level=level;self._ckks_scale=scale;self._ckks_nsd=1
        self.moduli=tuple(Q[:2*(level+1)]);self.num_moduli=len(self.moduli);self.ntt_ctx=None;self._public_row_l2_bound='input remains untouched'

class Arithmetic:
    scale=DELTA;strategy='power';refresher=None;progress=None
    def __init__(self):
        self.ctx=SimpleNamespace(secret_key=None,parameters={},degree=131072,num_slots=SLOTS,composite_degree=2,dnum=8,degree_layout=(256,512),
            _param_cache=SimpleNamespace(secret_key=None,batch=1,composite_degree=2,dnum=8,degree_layout=(256,512),q_towers=Q.copy(),p_towers=P['p'].copy(),q_moduli_at_level=lambda L:Q[:2*(L+1)]))
        self.calls=[]
    def _level(self,v):return v.level
    def _ensure_capacity(self,values,required):return tuple(values)
    def _public_slots(self,x):return np.full(SLOTS,x) if np.ndim(x)==0 else np.asarray(x)
    def rotate(self,v,shift):self.calls.append('rotate');return Value(np.roll(v.data,-shift),v.level,v._ckks_scale)
    def add(self,a,b,*,subtract=False):
        self.calls.append('subtract' if subtract else 'add')
        assert math.isclose(a._ckks_scale,b._ckks_scale,rel_tol=1e-12)
        return Value(a.data-b.data if subtract else a.data+b.data,min(a.level,b.level),a._ckks_scale)
    def weighted_sum(self,values,weights,*,bias=0.):
        self.calls.append('weighted');L=min(v.level for v in values);D=Q[2*L]*Q[2*L+1]
        target=self.scale*D;term_scales=[float(v._ckks_scale)*float(target/v._ckks_scale) for v in values]
        return Value(sum(v.data*w for v,w in zip(values,weights))+bias,L-1,term_scales[0]/D)
    def multiply(self,a,b):
        self.calls.append('multiply');L=min(a.level,b.level)
        return Value(a.data*b.data,L-1,a._ckks_scale*b._ckks_scale/(Q[2*L]*Q[2*L+1]))
    polynomial_depth=EncryptedPolynomialEvaluator.polynomial_depth
    evaluate=EncryptedPolynomialEvaluator.evaluate

@pytest.fixture(autouse=True)
def backend_free(monkeypatch):
    monkeypatch.setitem(sys.modules,'encrypted_packed_qwen',SimpleNamespace(CompiledArithmetic=Arithmetic))
    monkeypatch.setitem(sys.modules,'jaxite_word',SimpleNamespace(Polynomial=Value))
    monkeypatch.setitem(sys.modules,'support',SimpleNamespace(require_secure_execution=lambda *args:None))

def weights(site):
    prefix='' if site=='0.attention' else 'mlp_'
    return tuple(np.load(HERE/(prefix+n+'.npy'),allow_pickle=False) for n in ('gamma','beta'))

def setup(site='0.attention',progress=None):
    ops=Arithmetic();gamma,beta=weights(site)
    return ops,norm.V2RowLayerNorm(ops,RAW,gamma,beta,site=site,progress=progress)

def fixture(contract,valid):
    rng=np.random.default_rng(3100+valid);data=rng.choice([-100.,100.],(16,4096))
    rows=rng.normal(size=(valid,768));rows-=rows.mean(axis=-1,keepdims=True);rows/=np.sqrt(np.mean(rows*rows,axis=-1,keepdims=True))
    lo,hi=contract.domain;variances=np.linspace(lo+(hi-lo)*.05,hi-(hi-lo)*.05,valid)
    data[:valid,:768]=rows*np.sqrt(variances[:,None]-contract.epsilon)+np.arange(valid)[:,None]*3.1
    return data.ravel()

def independent(data,contract,gamma,beta,valid):
    x=data.reshape(16,4096)[:valid,:768];mean=x.mean(axis=-1);centered=x-mean[:,None];variance=np.mean(centered*centered,axis=-1)+contract.epsilon
    assert np.all((variance>=contract.domain[0])&(variance<=contract.domain[1]))
    means=np.zeros((16,4096));means[:valid,:768]=mean[:,None]
    variances=np.full((16,4096),sum(contract.domain)/2);variances[:valid]=variance[:,None]
    lo,hi=contract.domain;inverse=np.polynomial.chebyshev.chebval((2*variances-lo-hi)/(hi-lo),contract.coefficients)
    output=np.zeros((16,4096));output[:valid,:768]=centered*inverse[:valid,:768]*gamma+beta
    return dict(mean=means.ravel(),variance=variances.ravel(),inverse=inverse.ravel(),output=output.ravel())

@pytest.mark.parametrize('site',('0.attention','0.mlp'))
@pytest.mark.parametrize('valid',range(1,17))
@pytest.mark.parametrize('scale',(DELTA,math.nextafter(DELTA,math.inf)))
def test_all_rows_slots_padding_and_actual_scales(site,valid,scale):
    ops,owner=setup(site);data=fixture(owner.contract,valid);source=Value(data,20,scale);saved=copy.deepcopy(vars(source))
    positions=tuple(range(1,2*valid+1,2));result=owner.normalize_tile(source,valid_rows=valid,original_positions=positions,capture_checkpoints=True)
    expected=independent(data,owner.contract,*weights(site),valid)
    for name,v in {**result.checkpoints,'output':result.output}.items():
        np.testing.assert_allclose(v.data,expected[name],rtol=0,atol=3e-12)
        assert v.level==result.plan['levels'][name] and v._ckks_scale==result.plan['scales'][name]
        assert not any(k.startswith('_public_') for k in vars(v))
    assert result.output.level==9 and result.original_positions==positions
    assert np.all(result.output.data.reshape(16,4096)[:valid,768:]==0) and np.all(result.output.data.reshape(16,4096)[valid:]==0)
    assert ops.calls.count('rotate')==42 and ops.calls.count('weighted')==5
    np.testing.assert_array_equal(source.data,saved['data']);assert source._public_row_l2_bound==saved['_public_row_l2_bound']
    assert not owner._active and 'jax' not in sys.modules

@pytest.mark.parametrize('site',('0.attention','0.mlp'))
def test_exact_old_graph_composition_uses_same_public_primitives(site):
    ops,owner=setup(site);data=fixture(owner.contract,7)
    a=owner.normalize_tile(Value(data),valid_rows=7,original_positions=tuple(range(7)),capture_checkpoints=True)
    b=owner._core.normalize_tile(Value(data),owner.gamma,owner.beta,owner.approximation,epsilon=1e-12,valid_rows=7,capture=True)
    np.testing.assert_array_equal(a.output.data,b.output.data)
    assert a.output._ckks_scale==b.output._ckks_scale
    for name in a.checkpoints:np.testing.assert_array_equal(a.checkpoints[name].data,b.checkpoints[name].data)
