"""Secure full-accessor/compiled-rotation proof for isolated scoped NTT opt-in."""
from pathlib import Path
import gc,hashlib,json,math,sys,time,weakref
HERE=Path(__file__).resolve().parent;ROOT=HERE/'checkout'
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common'),str(HERE)]
import jax,numpy as np
from jaxite_word import CKKSContext,key_gen,he_params,ckks_ctx
from encrypted_ntt import enable_scoped_ntt_subsets
from encrypted_rotation import CompiledRotation
from encrypted_polynomial import EncryptedPolynomialEvaluator
import herot,he_ops,encrypted_reference,encrypted_ntt,ntt_mm,polynomial
jax.config.update('jax_enable_x64',True)
source_paths=[Path(__file__),*sorted((ROOT/'jaxite_word').glob('*.py')),*sorted((ROOT/'cadathlon/common').glob('*.py'))]
hashes=lambda:{str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
before=hashes();started=time.monotonic();q=[4079617,1196033,1589249,1376257,2277377,1720321,1769473,1785857];p=[8273921,8257537];N=8192;dnum=4;batch=2
assert he_params.qp_is_secure(q,p,N,he_params.SecurityTarget(128,'classical'))
keys=key_gen.gen_pke_pair(q,[],N);ek=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
indices=(3,4091);rotations={i:key_gen.gen_rotation_key(keys['secret_key'],q,p,i,dnum=dnum)[i] for i in indices}
S=float(q[-2])*q[-1];params=dict(degree=N,num_slots=N//2,scaling_factor=S,output_scale=S,q_towers=q,p_towers=p,composite_degree=2,degree_layout=(64,128),p=42,CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
client=CKKSContext(dict(params,**keys));server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(ek[n],dtype='uint32').transpose(0,2,1) for n in ('a','b')]))
server.program_initialization(list(indices),dnum=dnum,r=64,c=128,batch=batch,pregenerated_rotation_keys=rotations)
cache=server._param_cache;assert cache.rotation_polynomial_factory is None
values=np.stack([np.linspace(-.25,.25,N//2),.12*np.sin(np.arange(N//2)/13)])
ct=client.encrypt_slots_batch(values,scale=S*1.125);ct._ckks_nsd=1
client._polynomial_ntt_ctx=cache.ntt_q_max
evaluator=EncryptedPolynomialEvaluator(server,low_memory=True);compiler=CompiledRotation(server)
report={'scope':__doc__,'degree':N,'q':q,'p':p,'dnum':dnum,'batch':batch,'actual_log2_qp':sum(math.log2(x) for x in q+p),'security_ceiling':218,'source_sha256_before':before,'checks':[],'imported_paths':{name:str(Path(sys.modules[name].__file__).resolve()) for name in ('herot','he_ops','he_params','encrypted_ntt','encrypted_rotation','encrypted_reference')}}
assert all(Path(path).is_relative_to(ROOT) for path in report['imported_paths'].values())
for level in (3,1):
    value=ct if level==3 else server.he_level_reduce[3,level].level_reduce(ct)
    expected=[]
    for index in indices:
        expected.append(server.he_rot[level,index].rotate(value))
    jax.block_until_ready([x.polynomial for x in expected]);evaluator.checkpoint(expected,'baseline complete')
    provider=enable_scoped_ntt_subsets(server,max_entries=8)
    assert cache.rotation_polynomial_factory.__self__ is provider
    try:enable_scoped_ntt_subsets(server)
    except ValueError:pass
    else:raise AssertionError('duplicate provider installation accepted')
    for index,wanted in zip(indices,expected):
        actual=compiler.rotate(value,index)
        np.testing.assert_array_equal(actual.polynomial,wanted.polynomial)
        assert actual._ckks_scale==wanted._ckks_scale and actual._ckks_nsd==wanted._ckks_nsd and actual.moduli==wanted.moduli
        err=float(np.max(np.abs(client.decrypt_slots_batch(actual)-np.roll(values,-index,axis=-1))))
        assert err<1e-6
        report['checks'].append({'level':level,'index':index,'exact_payload_and_metadata':True,'max_abs_error':err})
    assert provider.stats['builds']>0
    compiled_count=compiler.compile_count;compiler.clear();evaluator.checkpoint(value,'clear and rebuild same level');gc.collect()
    repeated=compiler.rotate(value,indices[0]);np.testing.assert_array_equal(repeated.polynomial,expected[0].polynomial)
    assert compiler.compile_count==compiled_count
    report['checks'].append({'level':level,'same_executable_reused_after_cache_clear':True,'provider_stats':dict(provider.stats)})
    compiler.clear();del actual,repeated,wanted,expected
    evaluator.checkpoint(value,'provider retirement');gc.collect()
    assert provider.live_entries==0
    provider.close();assert cache.rotation_polynomial_factory is None
    # Existing CTs are still valid after the provider is disabled.
    ordinary=server.he_rot[level,indices[0]].rotate(value)
    assert server.he_rot[level,indices[0]]._herot._polynomial_factory is herot.Polynomial
    np.testing.assert_allclose(client.decrypt_slots_batch(ordinary),np.roll(values,-indices[0],axis=-1),atol=1e-6,rtol=0)
    del ordinary
    evaluator.checkpoint(value,'default path restored')
# A provider kept alive by its caller must not retain its server/cache/keys.
provider=enable_scoped_ntt_subsets(server);cache_ref=weakref.ref(cache);server_ref=weakref.ref(server);master_ref=weakref.ref(cache.ntt_q_max)
compiler.clear(executables=True);evaluator.checkpoint(ct,'final cleanup');provider.close()
ckks_ctx.evict_context_caches(client);ckks_ctx.evict_context_caches(server)
del value,ct,compiler,evaluator,server,cache,client
jax.clear_caches();polynomial._POLYNOMIAL_REGISTRY.clear();ntt_mm._barrett_ntt_context_cache.clear();gc.collect()
report['owner_contexts_collected']=server_ref() is None and cache_ref() is None and master_ref() is None
assert report['owner_contexts_collected']
report['source_sha256_after']=hashes();report['source_files_unchanged']=before==report['source_sha256_after'];report['passed']=report['source_files_unchanged'];report['elapsed_s']=time.monotonic()-started
(HERE/'integrated.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if not k.startswith('source_sha')}),flush=True)
raise SystemExit(0 if report['passed'] else 1)
