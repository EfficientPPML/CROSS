"""Actual public JAX residues with synthetic arrays; no CKKSContext or secret."""
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,os,sys,time,gc,weakref
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE/'base'),str(HERE/'checkout/jaxite_word')]

def main(output,quick=False):
    if os.environ.get('JAX_PLATFORMS')!='cpu':raise ValueError('explicit CPU required')
    import numpy as np
    import jax,jax.numpy as jnp
    import herot,finite_field,util
    # The raw public fixture supplies shape/metadata wrappers, NOT CKKSContext.
    # This mock security boundary is explicit; no ring-security/HE claim follows.
    sys.modules['encrypted_polynomial']=SimpleNamespace(EncryptedPolynomialEvaluator=lambda ctx:None)
    def require(owner,value):
        ctx=getattr(owner,'ctx',owner)
        if any(x is not None for x in (ctx.secret_key,ctx._param_cache.secret_key,ctx.parameters.get('secret_key'))):raise ValueError('secret in public fixture')
    sys.modules['support']=SimpleNamespace(require_secure_execution=require)
    from grouped_rotation import GroupedCompiledRotation
    from prepared_rotation import AuthenticatedRotationLease
    from encrypted_rotation import CompiledRotation
    output=Path(output);output.mkdir(exist_ok=False);started=time.monotonic()
    profile=json.loads((HERE/'p60_profile_and_refresh.json').read_bytes())['profile']
    report=dict(passed=False,scope='Actual public JAX group completion on canonical residues and synthetic keys, with explicit mocked context/security metadata. No CKKSContext, keygen, encryption, decryption or secure-HE proof.',fixtures=[],cpu_affinity=sorted(os.sched_getaffinity(0)))
    def write(): (output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    write()
    cases=[(16,14)] if quick else [(n,q) for n in (16,256) for q in (12,14,26,28,52,100)]
    for N,Q in cases:
        t=time.monotonic();r=4 if N==16 else 16;c=N//r;slots=N//2
        q=profile['q'][:Q];p=profile['p'];fullq=profile['q'];rng=np.random.default_rng(9173+N+Q)
        requests=[('head11',tuple(63*i for i in range(1,12))),('key15',tuple(-16*i for i in range(1,16))),('edges',(0,-5,3,3,-1))]
        indices=sorted({i%slots for _,values in requests for i in values if i%slots})
        raw={};maps={};moduli=np.asarray(fullq+p,dtype=np.uint32)
        for index in indices:
            raw[index]={}
            for name in ('a','b'):
                arr=rng.integers(0,moduli[:,None],size=(8,len(moduli),N),dtype=np.uint32)
                raw[index][name]=np.frombuffer(arr.tobytes(),dtype=np.uint32).reshape(arr.shape)
            maps[index]=jnp.asarray(util.precompute_auto_map(N,util.find_automorphism_index_2n_complex(index,2*N)),dtype=jnp.int32)
        cache=SimpleNamespace(degree=N,batch=1,degree_layout=(r,c),r=r,c=c,dnum=8,q_towers=fullq,p_towers=p,per_level_rotation_keys=False,cache_rotation_keys=True,composite_degree=2,secret_key=None,raw_rotation_keys=raw,coef_maps=maps,ff_context_cls=finite_field.BarrettContext,_formatted_rotation_keys={})
        cache.q_moduli_at_level=lambda level:fullq[:2*(level+1)]
        cache.level_for_num_q=lambda n:n//2-1
        def get(index,level):
            n=2*(level+1);pair=[]
            for name in ('a','b'):
                a=raw[index][name];selected=np.concatenate((a[:,:n,:],a[:,len(fullq):,:]),axis=1)
                pair.append(jnp.asarray(np.transpose(selected,(0,2,1)).reshape(8,r,c,n+len(p))))
            cache._formatted_rotation_keys[index]=tuple(pair)
            return *pair,maps[index]
        cache.get_rot_key=get
        ctx=SimpleNamespace(degree=N,num_slots=slots,degree_layout=(r,c),dnum=8,composite_degree=2,_param_cache=cache,secret_key=None,parameters={'secret_key':None})
        class Accessor:
            def __init__(self):self._instances={}
            def __getitem__(self,key):
                level,index=key
                if key not in self._instances:
                    kernel=herot._HERotKernel(r,c,8,cache.q_moduli_at_level(level),p,finite_field_context=finite_field.BarrettContext)
                    kernel.control_gen(batch=1,degree_layout=(r,c),keygen_sizeQ=100)
                    a,b,m=get(index,level);parts=kernel.numPartQl;kernel.setup_rotation(a[:parts],b[:parts],m)
                    self._instances[key]=SimpleNamespace(_herot=kernel,_rotation_state=kernel._rotation_state)
                return self._instances[key]
            def take_explicit_rotation_fn(self,key):
                op=self[key];kernel=op._herot;kernel.evalkey_a_vector=kernel.evalkey_b_vector=kernel.coef_map=None;kernel._rotation_ready=False
                del self._instances[key];return kernel._rotate_array
        ctx.he_rot=Accessor()
        class Value:
            def __init__(self):
                self.polynomial=jnp.asarray(rng.integers(0,np.asarray(q,dtype=np.uint32),size=(1,2,r,c,Q),dtype=np.uint32))
                self.ntt_ctx=object();self.moduli=q;self.degree=N;self.degree_layout=(r,c);self.batch=1;self.num_elements=2;self.precision=32;self._ckks_scale=123.25;self._ckks_nsd=1
            @property
            def shape(self):return self.polynomial.shape
            @property
            def num_moduli(self):return len(self.moduli)
            def _clone_with_payload(self,payload):
                clone=object.__new__(Value);clone.__dict__=self.__dict__.copy();clone.polynomial=payload;return clone
        pins={i:tuple(id(a) for a in (raw[i]['a'],raw[i]['b'],maps[i])) for i in indices}
        def backing(index,a,b,m):
            if tuple(map(id,(a,b,m)))!=pins[index]:raise ValueError('public synthetic backing replaced')
        lease=AuthenticatedRotationLease(ctx,context_id='c'*64,backing_guard=backing)
        owner=GroupedCompiledRotation(ctx,source_lease=lease);ordinary=CompiledRotation(ctx);value=Value();saved=np.asarray(value.polynomial).copy()
        checks=[];decompositions=0
        for label,requested in requests:
            expected=tuple(value if i%slots==0 else ordinary.rotate(value,i%slots) for i in requested)
            observed=[];weak=[]
            def progress(ordinal,index,result):
                observed.append((ordinal,index,id(result)))
                if owner._group.digits is not None and not weak:
                    weak.extend((weakref.ref(owner._group),weakref.ref(owner._group.digits),weakref.ref(owner._group.kernel)))
            actual=owner.rotate_many(value,requested,progress=progress)
            assert len(actual)==len(requested)==len(observed)
            for i,left,right in zip(requested,actual,expected):
                np.testing.assert_array_equal(np.asarray(left.polynomial),np.asarray(right.polynomial))
                assert left._ckks_scale==value._ckks_scale and left._ckks_nsd==1 and left.moduli==q
                if not i%slots:assert left is value
            decompositions+=owner.last_group['decompositions']
            checks.append(dict(group=label,requests=list(requested),comparisons=len(actual),mode=owner.last_group['mode'],decompositions=owner.last_group['decompositions'],digit_bytes=owner.last_group['digit_bytes'],result_bytes=owner.last_group['result_bytes'],equal=True))
            assert owner._group is None and owner._failed is None
            gc.collect()
            assert all(ref()is None for ref in weak),'successful group/digit/kernel wrapper retained'
            checks[-1]['group_digit_kernel_wrappers_collected']=True
        np.testing.assert_array_equal(np.asarray(value.polynomial),saved)
        owner.close();ordinary.clear(executables=True);lease.retire()
        report['fixtures'].append(dict(N=N,Q=Q,P=len(p),checks=checks,input_unchanged=True,retired=True,elapsed_s=time.monotonic()-t))
        write();print(json.dumps(dict(N=N,Q=Q,comparisons=sum(x['comparisons'] for x in checks),decompositions=decompositions)),flush=True)
        del owner,ordinary,lease,ctx,cache,value,raw,maps,actual,expected
        jax.clear_caches()
    report.update(passed=True,elapsed_s=time.monotonic()-started,total_comparisons=sum(c['comparisons'] for row in report['fixtures'] for c in row['checks']),no_context_keygen_modules_loaded=not any(n in sys.modules for n in ('ckks_ctx','key_gen','jaxite_word.ckks_ctx','jaxite_word.key_gen')),jax_version=jax.__version__,numpy_version=np.__version__)
    write();return report
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--quick',action='store_true');main(**vars(parser.parse_args()))
