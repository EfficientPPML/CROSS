"""Secure raw-tensor and unchanged Bootstrap dispatch parity; CPU only."""
import copy
import gc
import math
import sys
import weakref
from pathlib import Path
import pytest

REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO))
sys.path.insert(0,str(REPO/'cadathlon/common'))
import jax
import jax.numpy as jnp
import numpy as np
from jaxite_word import CKKSContext,he_params,key_gen,util
from jaxite_word.composite_prime_gen import composite_prime_gen
from jaxite_word.bootstrapping import Bootstrap
from encrypted_primitives import CompiledTensorOnly
from encrypted_refresh import _install_compiled_primitives,_Accessor,_Operation
from encrypted_polynomial import EncryptedPolynomialEvaluator

jax.config.update('jax_enable_x64',True)
RECORDS=[]

@pytest.fixture(scope='module')
def ring():
    degree,depth,dnum=8192,2,3
    q=composite_prime_gen(2,2*(depth+1),45,40,2*degree,31)
    p=util.generate_p_towers(q,dnum,degree,aux_bits=30)
    assert he_params.qp_is_secure(q,p,degree,he_params.SecurityTarget(128,'classical'))
    he_params.assert_p_coverage(q,p,dnum)
    keys=key_gen.gen_pke_pair(q,[],degree)
    evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
    params=dict(degree=degree,num_slots=degree//2,scaling_factor=float(1<<40),
        output_scale=float(1<<40),q_towers=q,p_towers=p,composite_degree=2,
        degree_layout=(64,128),p=40,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
    client=CKKSContext(dict(params,**keys))
    RECORDS.append(dict(ring_degree=degree,depth=depth,dnum=dnum,q_towers=q,p_towers=p,
                        actual_log2_qp=math.log2(math.prod(q+p))))
    return client,params,evaluation,dnum

def server_for(ring,batch):
    client,params,evaluation,dnum=ring
    server=CKKSContext(dict(params,evaluation_key=[
        jnp.asarray(evaluation[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
    server.program_initialization([],dnum=dnum,r=64,c=128,batch=batch)
    assert server.secret_key is None and server._param_cache.secret_key is None
    return server

def assert_same(actual,expected):
    np.testing.assert_array_equal(actual.polynomial,expected.polynomial)
    assert actual.num_elements==expected.num_elements
    assert actual.moduli==expected.moduli
    assert actual.precision==expected.precision==32
    assert actual.degree_layout==expected.degree_layout
    assert getattr(actual,'_ckks_scale',None)==getattr(expected,'_ckks_scale',None)
    assert actual._ckks_nsd==expected._ckks_nsd
    actual.validate()

def bootstrap_shell(context):
    engine=object.__new__(Bootstrap)
    engine.ctx=copy.copy(context)
    engine._cache=context._param_cache
    engine._composite_degree=2
    engine._low_mem=False
    return engine

@pytest.mark.parametrize('batch',[1,2])
def test_secure_raw_tensor_and_bootstrap_parity(ring,batch,monkeypatch):
    client=ring[0];server=server_for(ring,batch)
    evaluator=EncryptedPolynomialEvaluator(server)
    tensor=CompiledTensorOnly(server)
    x=np.stack([np.linspace(-.45,.45,client.num_slots)*(i+1)/batch for i in range(batch)])
    a=client.encrypt_slots_batch(x,scale=float(1<<40))
    b=client.encrypt_slots_batch(-.7*x,scale=float(1<<39))
    # Genuine PTCT products yield nsd=2, rather than relabeling nsd=1 inputs.
    pt=server.encode_at_level(np.full(client.num_slots,.5),2,scale=float(1<<40))
    a2=server.ptct_mul[2].mul(a,pt)
    b2=server.ptct_mul[2].mul(b,pt)
    snapshots=[(v.polynomial,v._ckks_scale,getattr(v,'_ckks_nsd',1),v.moduli[:]) for v in (a,b,a2,b2)]
    facade_type=type(server.he_mul[1]);original_init=facade_type.__init__
    constructed=[]
    def track_init(self,*args,**kwargs):
        constructed.append(1)
        return original_init(self,*args,**kwargs)
    monkeypatch.setattr(facade_type,'__init__',track_init)
    cases=[(a,b),(a2,b),(a2,b2),
           (evaluator._lower(a,1),evaluator._lower(b,1))]
    for left,right in cases:
        level=server._param_cache.level_for_num_q(left.num_moduli)
        expected=server.he_mul[level-1].hemul_no_relin(left,right)
        count=len(constructed)
        actual=tensor.multiply(left,right)
        assert len(constructed)==count, 'tensor-only helper constructed a key-switch facade'
        assert actual.num_elements==3 and actual.num_moduli==left.num_moduli
        assert_same(actual,expected)
        assert_same(tensor.multiply(left,right),expected)
    assert tensor.compiler.compile_count==2
    tensor.clear();jax.clear_caches()
    assert_same(tensor.multiply(*cases[-1]),server.he_mul[0].hemul_no_relin(*cases[-1]))
    assert tensor.compiler.compile_count==2
    for source,snapshot in zip((a,b,a2,b2),snapshots):
        assert source.polynomial is snapshot[0]
        assert (source._ckks_scale,getattr(source,'_ckks_nsd',1),source.moduli)==snapshot[1:]
    # Missing scale must remain untracked, as the core no-rescale facade does.
    untracked=copy.copy(a);untracked.__dict__.pop('_ckks_scale')
    assert_same(tensor.multiply(untracked,b),server.he_mul[1].hemul_no_relin(untracked,b))
    # Exercise the actual Bootstrap method, including its nsd=2 pre-rescale,
    # level alignment, relinearization, canonical validation and scale updates.
    eager=bootstrap_shell(server);compiled=bootstrap_shell(server)
    helpers=_install_compiled_primitives(compiled,server)
    bootstrap_cases=[(2,a,b,x*(-.7*x)),
        (1,a2,b,(.5*x)*(-.7*x)),(1,a2,b2,(.5*x)*(-.35*x)),
        (1,a,evaluator._lower(b,1),x*(-.7*x))]
    for level,left,right,truth in bootstrap_cases:
        expected=eager._he_op_mul(level,left,right)
        actual=compiled._he_op_mul(level,left,right)
        assert_same(actual,expected)
        assert actual.num_elements==2 and actual._ckks_nsd==2
        assert server._param_cache.level_for_num_q(actual.num_moduli)==level
        decoded=np.asarray(client.decrypt_slots_batch(actual))
        error=float(np.max(np.abs(decoded-truth)))
        assert error<2e-6,error
        # With active compiled bindings, accessor bookkeeping must construct
        # no fresh multiply facade even when the eager instance cache is empty.
        server.he_mul._instances.clear();count=len(constructed)
        repeated=compiled._he_op_mul(level,left,right)
        assert_same(repeated,expected)
        assert len(constructed)==count
        RECORDS.append(dict(batch=batch,level=level,input_nsds=[getattr(left,'_ckks_nsd',1),getattr(right,'_ckks_nsd',1)],
            raw_payload_exact=True,bootstrap_payload_exact=True,metadata_exact=True,max_abs_error=error))
    # Invalid canonical contracts fail before ciphertext execution.
    bad_moduli=copy.copy(a);bad_moduli.moduli=list(reversed(a.moduli))
    bad_layout=copy.copy(a);bad_layout.degree_layout=(128,64)
    bad_scale=copy.copy(a);bad_scale._ckks_scale=float('inf')
    bad_nsd=copy.copy(a);bad_nsd._ckks_nsd=0
    invalid=[(a,evaluator._lower(b,1)),(a,bad_moduli),(bad_layout,b),(bad_scale,b),(bad_nsd,b),
             (evaluator._lower(a,0),evaluator._lower(b,0))]
    if batch==2:
        invalid.append((a.batch_slice(0),b))
    for left,right in invalid:
        with pytest.raises((ValueError,TypeError)):
            tensor.multiply(left,right)
    for level in (-1,server.max_level,True):
        with pytest.raises(ValueError):
            compiled.ctx.he_mul[level]
    for parameter,bad in [('sigma',0.),('secret_distribution','gaussian')]:
        original=server.parameters.get(parameter)
        server.parameters[parameter]=bad
        with pytest.raises(ValueError):
            tensor.multiply(a,b)
        if original is None:server.parameters.pop(parameter)
        else:server.parameters[parameter]=original
    # Controls are owned by bindings/cache, not persistent executable objects.
    refs=[weakref.ref(v) for v in tensor.binding[1].controls]
    assert refs
    compiler=tensor.compiler
    for helper in helpers:helper.clear()
    tensor.clear()
    from jaxite_word import ckks_ctx,polynomial
    import ntt_mm
    ckks_ctx.evict_context_caches(server)
    del helpers,compiled,eager,tensor,server,evaluator,a,b,a2,b2,pt,actual,expected,repeated
    del cases,bootstrap_cases,snapshots,invalid,left,right,source,snapshot,untracked,bad_moduli,bad_layout,bad_scale,bad_nsd
    jax.clear_caches();polynomial._POLYNOMIAL_REGISTRY.clear();ntt_mm._barrett_ntt_context_cache.clear();gc.collect()
    assert compiler._executables
    assert all(ref() is None for ref in refs)


def test_accessor_does_not_construct_unused_eager_facade():
    calls=[]
    class Original:
        def __getitem__(self,key):
            calls.append(key)
            return type('Facade',(),{'original_only':'preserved'})()
    accessor=_Accessor(Original(),lambda key,original:_Operation(original,run=lambda:('compiled',key)))
    operation=accessor[7]
    assert operation.run()==('compiled',7) and calls==[]
    assert operation.original_only=='preserved' and calls==[7]
