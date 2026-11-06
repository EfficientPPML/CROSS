"""Secure shallow CKKS metadata regression; no bootstrap or full model claim."""
import os
import sys,json,math,time,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'cadathlon/common')]
import jax,numpy as np
from jaxite_word import CKKSContext,key_gen,util,he_params
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_refresh import ClientRotationKeyStore
from encrypted_bounded_arithmetic import BoundedArithmetic


def test_secure_arithmetic_invalidates_only_unsupported_specialized_contracts():
    jax.config.update('jax_enable_x64',True)
    start=time.monotonic();N=8192;depth=1;dnum=2
    report=dict(checks={})
    q=composite_prime_gen(2,2*(depth+1),61,56,2*N,31)
    p=util.generate_p_towers(q,dnum,N,aux_bits=30)
    assert he_params.qp_is_secure(q,p,N,he_params.SecurityTarget(128,'classical'))
    he_params.assert_p_coverage(q,p,dnum)
    report['ring']=dict(degree=N,depth=depth,dnum=dnum,q_count=len(q),p_count=len(p),actual_log2_qp=sum(map(math.log2,q+p)),security_bits=128)
    k=key_gen.gen_pke_pair(q,[],N)
    ek=key_gen.gen_evaluation_key(k['secret_key'],q=q,P=p,dnum=dnum)
    store=ClientRotationKeyStore.generate(k['secret_key'],q,p,degree=N,dnum=dnum,rotations=[1])
    S=float(q[-2])*q[-1]
    params=dict(degree=N,num_slots=N//2,scaling_factor=S,output_scale=S,q_towers=q,p_towers=p,composite_degree=2,degree_layout=(64,128),p=56,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
    client=CKKSContext(dict(params,**k))
    ctx=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(ek[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
    ctx.program_initialization([1],dnum=dnum,r=64,c=128,batch=1,pregenerated_rotation_keys=store.keys)
    client._polynomial_ntt_ctx=ctx._param_cache.ntt_q_max
    assert ctx.secret_key is None and ctx._param_cache.secret_key is None and ctx.parameters.get('secret_key') is None
    def forbidden(*a,**kw):raise AssertionError('server encrypt/decrypt forbidden')
    ctx.encrypt=ctx.decrypt=ctx.encrypt_slots=ctx.decrypt_slots=forbidden
    ops=BoundedArithmetic(ctx,low_memory=True)
    x=np.tile([.1,.35,-.2,.3],N//8)
    a=ops.declare_bound(client.encrypt_slots(x),.35)
    relations=dict(_public_row_l2_bound=.35,_public_head_l2_bounds=(.35,),_public_normalization_affine=((1.,),(0.,),.35))
    a.__dict__.update(relations)
    clone=a._clone_with_payload(a.polynomial)
    assert all(getattr(clone,n)==v for n,v in relations.items())
    del clone
    report['actual_clone_copies_relations']=True

    def check(name,result,expected,level,retains):
        result.polynomial.block_until_ready()
        observed=client.decrypt_slots(result)
        error=float(np.max(np.abs(observed-expected)))
        for key,value in relations.items():
            assert (getattr(result,key,None)==value) if retains else not hasattr(result,key), (name,key)
            assert getattr(a,key)==value
        assert ops._level(result)==level and getattr(result,'_ckks_nsd',1)==1
        assert math.isclose(result._ckks_scale,S,rel_tol=1e-12)
        assert error<1e-6,(name,error)
        report['checks'][name]=dict(max_abs_error=error,output_level=level,specialized_contracts_preserved=retains)

    check('addition_invalidates',ops.add(a,a),2*x,1,False)
    check('multiplication_invalidates',ops.multiply(a,a),x*x,0,False)
    check('weighted_sum_invalidates',ops.weighted_sum([a],[2.],bias=.1),2*x+.1,0,False)
    check('feature_rotation_invalidates',ops.rotate(a,1),np.roll(x,-1),1,False)
    assert ops.rotate(a,0) is a
    check('identity_preserves',ops.rotate(a,N//2),x,1,True)
    check('level_reduction_preserves',ops._lower(a,0),x,0,True)
    assert len(report['checks'])==6
    from jaxite_word import ckks_ctx
    ckks_ctx.evict_context_caches(client)
    ckks_ctx.evict_context_caches(ctx)
    store.close()
