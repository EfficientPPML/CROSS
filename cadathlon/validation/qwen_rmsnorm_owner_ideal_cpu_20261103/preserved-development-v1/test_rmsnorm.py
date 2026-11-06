from pathlib import Path
import hashlib,json,math,sys
import numpy as np
import pytest
from ideal_primitives import Arithmetic,Value,Q,DELTA
from ideal_refresh import IdentityRefresh,REFRESH_PLAN
from encrypted_qwen_rmsnorm import QwenRMSContract,QwenRMSNorm,rms_scale_plan
P=Path(__file__).resolve().parent
RAW=(P/'inputs/candidate_manifest.json').read_bytes();META=(P/'site_metadata.json').read_bytes();CONFIG=(P/'inputs/config.json').read_bytes()
SITES=tuple(json.loads(META));RULES=json.loads(RAW)['nonlinear']

@pytest.fixture(scope='module')
def gammas():
    report=json.loads((P/'gamma-preparation-v1/report.json').read_bytes());path=P/'client_gamma_fixtures.npz'
    assert report['passed'] is True and hashlib.sha256(path.read_bytes()).hexdigest()==report['fixture_npz_sha256']
    with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)==set(SITES)
        result={n:archive[n] for n in archive.files}
    for n,v in result.items():assert hashlib.sha256(v.astype('<f8').tobytes()).hexdigest()==json.loads(META)[n]['gamma_f64le_sha256']
    return result
def setup(site,gammas,positions=(0,2,5),callback=None):
    c=QwenRMSContract(RAW,META,site);a=Arithmetic();r=IdentityRefresh(a) if c.events else None
    o=QwenRMSNorm(a,c,CONFIG,gamma=gammas[site],positions=positions,refresher=r,progress=callback)
    return o,a,r
def fixture(o,tile=0):
    c=o.contract;rows=o.layout.tile_rows(tile);rng=np.random.default_rng(1234)
    data=rng.uniform(-200,200,(16,4096));target=np.geomspace(c.domain[0]*1.05,c.domain[1]*.95,rows*c.groups).reshape(rows,c.groups)
    for r in range(rows):
        for g in range(c.groups):
            x=rng.normal(size=c.width);x*=math.sqrt((target[r,g]-c.epsilon)/np.mean(x*x));data[r,g*c.width:(g+1)*c.width]=x
    return Value(data.ravel(),25,math.nextafter(DELTA,math.inf))
def reference(o,value,tile=0):
    c=o.contract;rows=o.layout.tile_rows(tile);raw=value.polynomial.reshape(16,4096)
    variance=np.full((16,4096),c.domain[0]+(c.domain[1]-c.domain[0])/2)
    for row in range(rows):
        for group in range(c.groups):
            block=raw[row,group*c.width:(group+1)*c.width];variance[row,group*c.width:(group+1)*c.width]=np.mean(block*block)+c.epsilon
    assert variance.min()>=c.domain[0] and variance.max()<=c.domain[1]
    if c.method=='polynomial':
        t=variance*(2/(c.domain[1]-c.domain[0]))-(c.domain[1]+c.domain[0])/(c.domain[1]-c.domain[0])
        inverse=np.polynomial.chebyshev.chebval(t,c.coefficients)
    else:
        inverse=variance*c.first_weight+c.first_bias
        for _ in range(1,c.iterations):inverse=inverse*(1.5-.5*(variance*(inverse*inverse)))
    output=np.zeros((16,4096))
    for group in range(c.groups):
        sl=slice(group*c.width,(group+1)*c.width);output[:rows,sl]=(raw[:rows,sl]*inverse[:rows,sl])*o.gamma
    return variance.ravel(),inverse.ravel(),output.ravel()

@pytest.mark.parametrize('site',SITES)
def test_all113_actual_circuits_with_real_gamma_and_fullslot_oracle(site,gammas):
    index=SITES.index(site);positions=tuple(2*i for i in range((1,7,16,19)[index%4]));o,a,r=setup(site,gammas,positions);tile=1 if len(positions)==19 else 0
    x=fixture(o,tile);before=x.polynomial.copy();var,inv,expected=reference(o,x,tile)
    result=o.evaluate_tile(x,tile=tile,capture_checkpoints=True)
    # Distinct reduction orders and Clenshaw versus balanced Chebyshev are
    # compared at this fixed ideal tolerance; it is not an HE precision gate.
    np.testing.assert_allclose(result.checkpoints['variance'].polynomial,var,rtol=3e-15,atol=1e-12)
    np.testing.assert_allclose(result.checkpoints['inverse'].polynomial,inv,rtol=3e-12,atol=3e-11)
    np.testing.assert_allclose(result.output.polynomial,expected,rtol=3e-12,atol=1e-9)
    np.testing.assert_array_equal(x.polynomial,before)
    assert result.output.level==o.contract.output_level and result.original_positions==o.layout.tile_positions(tile)
    bits=o.contract.width.bit_length()-1;c=o.contract
    expected_ct=(c.degree+1) if c.method=='polynomial' else 3*c.iterations-1
    expected_ws=(c.degree+3) if c.method=='polynomial' else c.iterations+2
    assert result.plan['counts']==dict(rotations=2*bits,additions=2*bits,weighted_sums=expected_ws,ciphertext_products=expected_ct,refreshes=len(c.events))
    assert (r.calls if r else [])==[(e[2],e[1]) for e in c.events]
    inactive=np.ones((16,4096),dtype=bool);inactive[:o.layout.tile_rows(tile),:c.groups*c.width]=False
    assert np.count_nonzero(result.output.polynomial.reshape(16,4096)[inactive])==0
    assert all(not any(n.startswith('_public_') for n in vars(v)) for v in (*result.checkpoints.values(),result.output))

@pytest.mark.parametrize('site',SITES)
def test_all113_public_exact_scale_plans(site):
    c=QwenRMSContract(RAW,META,site);scales=tuple(math.nextafter(DELTA,math.inf if i%2 else -math.inf) for i in range(len(c.events)))
    p=rms_scale_plan(Q,c,input_scale=math.nextafter(DELTA,math.inf),refresh_scales=scales)
    assert p['output_level']==c.output_level and min(s['level'] for s in p['stages'])>=2

def test_actual_public_refresh_plan_matches_fullslot_fixture():
    p=REFRESH_PLAN(131072,65536,(4,4),max_level=49)
    audit=json.loads((P/'inputs/profile_audit.json').read_bytes())
    assert set(p['rotations'])==set(audit['bootstrap_rotation_indices']) and p['output_level']==25 and p['period']==65536

def test_no_backend_or_model_imported_in_ideal_owner_suite():
    assert not any(n in sys.modules for n in ('jax','jaxlib','torch','transformers','safetensors'))
