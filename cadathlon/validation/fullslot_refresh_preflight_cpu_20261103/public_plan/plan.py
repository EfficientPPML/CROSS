"""Public-only full-slot refresh planning; no context, keygen, HE or TPU."""
from pathlib import Path
import datetime,hashlib,io,json,math,os,shutil,sys,tarfile,time
ROOT=Path('/home/jianming_gatech/CROSS');HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
import numpy as np
from encrypted_refresh import RefreshProfile,refresh_plan
from jaxite_word import CKKSContext,key_gen,he_params
from jaxite_word.bootstrapping import Bootstrap,_select_layers

calls=[]
def forbidden(*args,**kwargs):
    calls.append('forbidden cryptographic construction')
    raise AssertionError(calls[-1])
CKKSContext.__init__=forbidden
Bootstrap.__init__=forbidden
for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key'):
    setattr(key_gen,name,forbidden)
for name,module in tuple(sys.modules.items()):
    if name=='key_gen':
        for function in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key'):setattr(module,function,forbidden)


def transform_upper(budget,degree,q_count,p_count):
    """Conservative existing full-packed hoisted payload inventory.

    Counts pre-pruning public diagonal positions, not measured peaks. Exact
    supported rotation inventory comes from Bootstrap.required_rotation_indices.
    Raw/hoisted coefficient arrays may contain zeros/aliases; upper bounds keep
    them all. Excludes compiler, controls, FFT construction and allocator costs.
    """
    slots=degree//2;layers,rows,remainder=_select_layers(int(math.log2(slots)),budget)
    collapses=[layers]*(budget-int(bool(remainder)))+([remainder] if remainder else [])
    counts=[(1<<(n+1))-1 for n in collapses]
    babies=[1<<(n//2+1+int(((1<<(n+1))-1)>7)) for n in collapses]
    mqp=q_count-2+p_count # top C2S begins at max_level-1
    qp_plain=degree*mqp*4;qp_ct=2*qp_plain
    return dict(collapse_layers=collapses,diagonal_positions_per_transform=counts,
        max_diagonal_positions=max(counts),max_baby_count=max(babies),
        top_c2s_qp_moduli=mqp,qp_plaintext_bytes=qp_plain,qp_ciphertext_bytes=qp_ct,
        peak_level_encoded_qp_plaintexts_upper_bytes=max(counts)*qp_plain,
        peak_level_retained_qp_babies_upper_bytes=max(babies)*qp_ct,
        raw_and_hoisted_complex128_diagonal_upper_bytes=4*sum(counts)*slots*16,
        four_full_ciphertext_workspaces_bytes=4*qp_ct,
        exclusions=['NTT/BAT/rotation/relinearization controls','hoisted digit expansion',
            'JAX executable constants/compiler/allocator overhead','keygen temporary arrays',
            'transient duplicate plaintext arrays during stack/transfer','FFT coefficient construction peak',
            'simultaneously live ApproxMod real/imag terms'])


def main():
    started=time.perf_counter()
    dependencies=[Path(__file__),*sorted((ROOT/'jaxite_word').glob('*.py')),*sorted((ROOT/'cadathlon/common').glob('*.py')),
        ROOT/'cadathlon/validation/check_secure_refresh.py',ROOT/'cadathlon/validation/secure_refresh_tpu_deep_validation.json',
        ROOT/'cadathlon/validation/qwen_bounded_test2_tpu.json']
    source={str(p):p.read_bytes() for p in dependencies};before={p:hashlib.sha256(v).hexdigest() for p,v in source.items()}
    with tarfile.open(HERE/'sources.tar.gz','w:gz') as archive:
        for name,data in source.items():
            info=tarfile.TarInfo(name.lstrip('/'));info.size=len(data);archive.addfile(info,io.BytesIO(data))
    profiles=[]
    for depth,dnum in ((49,4),(50,5),(52,8)):
        params=RefreshProfile(131072,depth,56,61,dnum).parameters()
        q,p=params['q_towers'],params['p_towers'];alpha=(len(q)+dnum-1)//dnum;parts=(len(q)+alpha-1)//alpha
        partition_products=[math.prod(q[i:i+alpha]) for i in range(0,len(q),alpha)]
        key_pair=2*parts*(len(q)+len(p))*131072*4
        row={'degree':131072,'slots':65536,'depth':depth,'dnum':dnum,'q':q,'p':p,
            'q_count':len(q),'p_count':len(p),'actual_partitions':parts,'partition_size':alpha,
            'log2_QP':math.log2(math.prod(q+p)),'QP_integer_bits':math.prod(q+p).bit_length(),
            'security_ceiling_bits':3523,'secure_128_classical':he_params.qp_is_secure(q,p,131072,he_params.SecurityTarget(128,'classical')),
            'P_product_bits':math.prod(p).bit_length(),'max_Q_partition_bits':max(partition_products).bit_length(),
            'P_covers_each_Q_partition':math.prod(p)>=max(partition_products),
            'scale':params['scaling_factor'],'rotation_pair_bytes':key_pair,
            'relinearization_pair_bytes':key_pair,'public_key_bytes':2*len(q)*131072*4,
            'client_secret_bytes':len(q)*131072*4,'top_ciphertext_bytes':2*len(q)*131072*4,'budgets':[]}
        for budget in (2,3,4,5,6,8):
            plan=refresh_plan(131072,65536,(budget,budget),max_level=depth)
            count=len(plan['rotations']);store=count*(key_pair+8192)
            row['budgets'].append(dict(level_budget=[budget,budget],rotations=list(plan['rotations']),
                rotation_count=count,conjugation_key_index=2*131072-1,
                contains_conjugation_key=(2*131072-1 in plan['rotations']),
                raw_rotation_key_bytes=count*key_pair,key_store_required_bytes=store,
                key_store_free_space_preflight_bytes=store+64*1024**2,
                bootstrap_depth=plan['boot_depth'],normalized_output_level=plan['output_level'],
                squared_continuation_output_level=plan['output_level']-1,input_reserved_level=2,
                projection_aliases=1,projection_rotations=0,
                payload_inventory=transform_upper(budget,131072,len(q),len(p))))
        profiles.append(row)
    filesystems={}
    for path in ('/tmp','/dev/shm'):
        usage=shutil.disk_usage(path);filesystems[path]={'total_bytes':usage.total,'free_bytes':usage.free,'used_bytes':usage.used}
    old=json.loads((ROOT/'cadathlon/validation/secure_refresh_tpu_deep_validation.json').read_text())
    qwen=json.loads((ROOT/'cadathlon/validation/qwen_bounded_test2_tpu.json').read_text())
    for row in profiles:
        for budget in row['budgets']:
            budget['fits_current_key_store_preflight']={path:entry['free_bytes']>=budget['key_store_free_space_preflight_bytes'] for path,entry in filesystems.items()}
    pattern=np.random.default_rng(20261103).permutation(np.linspace(-14.2,14.2,65536,dtype=np.float64))
    assert len(np.unique(pattern))==65536 and not np.array_equal(pattern[:128],pattern[128:256])
    selected=profiles[0]['budgets'][2]
    after={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in dependencies}
    result=dict(scope=__doc__,passed=before==after and not calls,created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes_start=before,source_hashes_end=after,source_files_unchanged=before==after,
        source_archive_sha256=hashlib.sha256((HERE/'sources.tar.gz').read_bytes()).hexdigest(),
        cryptographic_construction_calls=calls,profiles=profiles,filesystems=filesystems,
        previous_evidence={'period128':{k:old[k] for k in ('profile','level_budget','actual_ring','actual_output_level','refresh_max_abs_error','square_max_abs_error','source_archive_sha256')},
            'qwen2':{k:qwen[k] for k in ('passed','refresh_progress','source_count','source_files_unchanged','source_archive_sha256','scope')}},
        recommended_first_profile={'degree':131072,'depth':49,'dnum':4,'scale_bits':56,'first_bits':61,
            'period':65536,'level_budget':[4,4],'public_bound':16.,'amplitude':14.2,'meta_precision':10,
            'compiled':True,'low_memory':True,'ntt_subsets':True,'preserve_dispatch_cache':True,
            'key_directory':'/dev/shm','output_level':selected['normalized_output_level'],'launch_authorized':False},
        proposed_independent_pattern={'kind':'default_rng(20261103).permutation(linspace(-14.2,14.2,65536))',
            'shape':[65536],'unique_values':65536,'float64_sha256':hashlib.sha256(pattern.astype('<f8').tobytes()).hexdigest(),
            'no_short_period':True,'client_only':True},
        full_slot_execution_differences=['Dense C2S/S2C use the existing hoisted QP path; sparse period128 used a different path.',
            'Dense conjugation splits real and imaginary ApproxMod paths: two ApproxMod evaluations per bootstrap, four for meta10.',
            'No sparse gap fold or final rotate-by-period; BoundedSecureRefresh periodic projection has one alias and performs zero rotations.',
            'A successful repeated period128 pattern cannot validate full independent-slot accuracy or HBM/liveness.',
            'Compiled wrapper optimizes public primitive calls, but hoisted dense transforms retain their own JIT closures and stage cache clearing.'],
        limitations=['Public key/payload resource arithmetic is not measured peak host memory or TPU HBM.',
            'No encrypted output, approximation error, refresh latency, full model or file-stage completion is established.',
            'The existing check_secure_refresh disk-key path ignores CROSS_PUBLIC_KEY_DIRECTORY; a dedicated collector must pass directory=/dev/shm or set TMPDIR explicitly.',
            'Existing validator uses linspace rather than the proposed permutation and returns large samples; a source-bound wrapper should verify all65536 slots, enforce actual TPU, preserve failures and cleanup.',
            'Bound16 is the first direct comparison with prior proof, not coverage of BERT intermediates/softmax bounds32/64 or arbitrary ciphertext errors.',
            'Component rings N16384/32768/65536 cannot be composed by relabeling ring degree; subsequent same-ring component proof is required.'],
        proposed_gates={'refresh_absolute':1e-3,'square_absolute':1e-3,'all_physical_slots':65536,
            'source_unchanged':True,'keyless_evaluator':True,'forbid_server_codec_and_runtime_keygen':True,
            'fixed_before_encryption':True,'not_a_uniform_interval_certificate':True},
        elapsed_s=time.perf_counter()-started,cpu_affinity=sorted(os.sched_getaffinity(0)))
    (HERE/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':result['passed'],'profiles':[{k:r[k] for k in ('depth','dnum','q_count','p_count','log2_QP','rotation_pair_bytes')}|{'budgets':[{k:b[k] for k in ('level_budget','rotation_count','raw_rotation_key_bytes','normalized_output_level')} for b in r['budgets']]} for r in profiles], 'filesystems':filesystems,'elapsed_s':result['elapsed_s']},indent=2))

if __name__=='__main__':main()
