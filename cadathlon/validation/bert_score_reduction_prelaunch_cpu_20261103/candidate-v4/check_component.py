"""Isolated secure masked row-score reductions; no attention, refresh or model claim."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
import resource
import shutil
import signal
import sys
import tarfile
import tempfile
import time

if os.environ.get('JAX_PLATFORMS') != 'cpu':
    raise RuntimeError('explicit CPU backend required')
HERE = Path(__file__).resolve().parent
ROOT = HERE/'checkout'
sys.path[:0] = [str(ROOT), str(ROOT/'cadathlon/common')]
import jax
import numpy as np
from jaxite_word import CKKSContext, ckks_ctx, he_params, key_gen, util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_ntt import enable_scoped_ntt_subsets
from encrypted_packed_qwen import CompiledArithmetic
from encrypted_refresh import ClientRotationKeyStore
from encrypted_score_reduction import ScoreReductionLayout, ScoreReduction, validate_public_dummy, SLOTS
from fixtures import public_fixtures, make_client_values, make_client_scores, client_oracles, mean_variation
from support import require_secure_execution
from lifecycle import CleanupTracker
jax.config.update('jax_enable_x64', True)


def sources():
    paths = set(HERE.glob('*.py')) | {HERE/'source_origin.json', HERE/'baseline_manifest.json'}
    for folder in ('jaxite_word', 'cadathlon/common'):
        paths.update((ROOT/folder).glob('*.py'))
    return {str(p.relative_to(HERE)): p.read_bytes() for p in sorted(paths)}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def parameters():
    degree, depth, dnum = 131072, 4, 4
    q = composite_prime_gen(2, 2*(depth+1), 61, 56, 2*degree, 31)
    p = util.generate_p_towers(q, dnum, degree, aux_bits=30)
    if not he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical')):
        raise ValueError('insecure actual Q/P ring')
    he_params.assert_p_coverage(q, p, dnum)
    alpha = math.ceil(len(q)/dnum)
    parts = math.ceil(len(q)/alpha)
    shape = (parts, len(q)+len(p), degree)
    rotations = ScoreReductionLayout.rotation_indices()
    fixtures = public_fixtures()
    for item in fixtures: validate_public_dummy(item["inactive_dummy"], item["domain"])
    pair_bytes = 2*math.prod(shape)*4
    owned = (len(rotations)+1)*pair_bytes + 2*len(q)*degree*4
    return dict(degree=degree, depth=depth, dnum=dnum, q_towers=q, p_towers=p,
        degree_layout=[256,512], slots=SLOTS, actual_log2_qp=sum(map(math.log2,q+p)),
        security_bits=128, scale=float(q[-2])*q[-1], rotation_indices=rotations,
        actual_switch_partitions=parts, key_pair_shape=shape, raw_key_pair_bytes=pair_bytes,
        raw_all_owned_public_key_bytes=owned, key_storage_parent='/tmp',
        current_key_storage_free_bytes=shutil.disk_usage('/tmp').free,
        additional_storage_margin_bytes=4*1024**3, public_fixtures=fixtures,
        independent_slot_input=True, expected_total_rotations=56,
        expected_total_public_multiplications=8, expected_total_ciphertext_multiplications=2,
        expected_total_rescales=10, mean_path_levels=2, squared_denominator_path_levels=3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    before={}
    started = time.monotonic()
    report = dict(scope=__doc__,passed=False,source_hashes_before=before,phases=[],checks={},
        source_archive_sha256=None,
        started_utc=datetime.now(timezone.utc).isoformat(),fixed_absolute_error_gate=1e-6,
        cpu_affinity=sorted(os.sched_getaffinity(0)),process_pid=os.getpid(),
        versions=None,key_generation_started=False,encryption_started=False,
        environment={name:os.environ.get(name) for name in ('JAX_PLATFORMS','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS')},
        limitations=['Two signed-score/probability fixture pairs only; no composed inverse/softmax or model proof.',
            'Client and keyless server are separate objects in one dedicated process, not an OS sandbox.',
            'All codec APIs are blocked during server arithmetic; validation occurs only afterward.',
            'No model input, full attention, refresh, TPU timing or calibrated speed comparison.'])
    client=server=ops=provider=key_store=key_parent=encrypted=router=None
    keys=evaluation=value=None
    retained=[];patched=[];codec_patches=[];old_handlers={}
    encrypted_inputs={};client_vectors={};client_references={};fingerprints={}

    def phase(label, value=None, counts=None):
        if value is not None:
            value.polynomial.block_until_ready()
        row=dict(label=label,elapsed_s=time.monotonic()-started,
            peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        if value is not None:
            row.update(level=ops._level(value),scale=value._ckks_scale,nsd=value._ckks_nsd)
        if counts is not None:
            row['counts']=dict(counts)
        report['phases'].append(row)
        write(args.output/'report.json',report)
        print(json.dumps(row),flush=True)

    def interrupted(signum,frame):
        for number in old_handlers:
            signal.signal(number,signal.SIG_IGN)
        raise KeyboardInterrupt(f'interrupted by signal {signum}')

    def forbidden(*args,**kwargs):
        raise AssertionError('client codec or secret-key generation during server computation')

    try:
        write(args.output/'report.json',report)
        raw = sources()
        before = {name:sha(value) for name,value in raw.items()}
        with tarfile.open(args.output/'sources.tar.gz','w:gz') as archive:
            for name,data in raw.items():
                info=tarfile.TarInfo(name); info.size=len(data)
                archive.addfile(info,io.BytesIO(data))
        report['source_hashes_before']=before
        report['source_archive_sha256']=sha((args.output/'sources.tar.gz').read_bytes())
        report['versions']={name:importlib.metadata.version(name) for name in ('jax','jaxlib','numpy')}
        for number in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
            old_handlers[number]=signal.signal(number,interrupted)
        ring=parameters();report['ring']=ring
        if ring['current_key_storage_free_bytes'] < ring['raw_all_owned_public_key_bytes']+ring['additional_storage_margin_bytes']:
            raise ValueError('insufficient available storage for owned keys plus4GiB margin')
        if jax.default_backend()!='cpu' or any(d.platform!='cpu' for d in jax.devices()):
            raise ValueError('explicit CPU backend not honored')
        phase('actual-ring, P-coverage and resource preflight passed')
        if args.preflight:
            report.update(passed=True,preflight_only=True)
        else:
            q,p,N,dnum=ring['q_towers'],ring['p_towers'],ring['degree'],ring['dnum']
            scale=ring['scale'];rotations=ring['rotation_indices']
            key_parent=tempfile.TemporaryDirectory(prefix='cross-score-reduction-keys-',dir=ring['key_storage_parent'])
            report['owned_public_key_parent']=key_parent.name
            for item in ring['public_fixtures']:
                probabilities=make_client_values(item);scores=make_client_scores(item)
                client_vectors[item['name']+':mean_scores']=scores
                client_vectors[item['name']+':probabilities']=probabilities
                client_references[item['name']]=client_oracles(scores,probabilities,item)
            report['signed_mean_variation']={item['name']:mean_variation(client_references[item['name']]['mean'],item) for item in ring['public_fixtures']}
            phase('client fixture/domain checks passed; key generation starting')
            report['key_generation_started']=True
            keys=key_gen.gen_pke_pair(q,[],N)
            evaluation=key_gen.gen_evaluation_key(keys['secret_key'],q=q,P=p,dnum=dnum)
            key_store=ClientRotationKeyStore.generate(keys['secret_key'],q,p,degree=N,dnum=dnum,
                rotations=rotations,directory=key_parent.name)
            report['public_rotation_key_storage_bytes']=sum(path.stat().st_size for path in Path(key_parent.name).rglob('*.npy'))
            params=dict(degree=N,num_slots=SLOTS,scaling_factor=scale,output_scale=scale,
                q_towers=q,p_towers=p,composite_degree=2,degree_layout=(256,512),p=56,
                CKKS_M_FACTOR=1,max_bits_in_word=61,noise_scale_degree=1)
            client=CKKSContext(dict(params,**keys))
            server=CKKSContext(dict(params,evaluation_key=[jax.numpy.asarray(evaluation[name],dtype='uint32').transpose(0,2,1) for name in ('a','b')]))
            for module in {key_gen,sys.modules.get('key_gen')}-{None}:
                for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key','key_switch_gen'):
                    patched.append((module,name,getattr(module,name)));setattr(module,name,forbidden)
            report['key_generation_guards']=sorted({module.__name__ for module,_,_ in patched})
            if set(report['key_generation_guards'])!={'jaxite_word.key_gen','key_gen'}:
                raise RuntimeError('both key-generation aliases must be guarded')
            server.program_initialization(rotations,dnum=dnum,r=256,c=512,batch=1,pregenerated_rotation_keys=key_store.keys)
            if any(x is not None for x in (server.secret_key,server._param_cache.secret_key,server.parameters.get('secret_key'))):
                raise RuntimeError('server has secret key material')
            report['keyless_server']=True
            client._polynomial_ntt_ctx=server._param_cache.ntt_q_max
            provider=enable_scoped_ntt_subsets(server)
            ops=CompiledArithmetic(server,low_memory=True,preserve_dispatch_cache=True)
            keys=evaluation=None
            report['input_contracts']=[]
            for item in ring['public_fixtures']:
                for kind in ('mean_scores','probabilities'):
                    identity=item['name']+':'+kind
                    values=client_vectors[identity]
                    report['encryption_started']=True
                    encrypted=client.encrypt_slots(values);encrypted._ckks_nsd=1
                    require_secure_execution(server,encrypted)
                    encrypted_inputs[identity]=encrypted
                    fingerprints[identity]=(sha(np.asarray(encrypted.polynomial).tobytes()),
                        encrypted._ckks_scale,encrypted._ckks_nsd,ops._level(encrypted))
                    report['input_contracts'].append(dict(item,kind=kind,value_sha256=sha(values.astype('<f8').tobytes()),
                        hostile_invalid_key_and_head_and_upper_half_padding=True,
                        independent_physical_slots=SLOTS,minimum=float(values.min()),maximum=float(values.max())))
            encrypted=None;values=None
            for name in ('encrypt','decrypt','encrypt_slots','decrypt_slots','encrypt_slots_batch','decrypt_slots_batch','decode'):
                codec_patches.append((CKKSContext,name,getattr(CKKSContext,name)))
                setattr(CKKSContext,name,forbidden)
            report['class_codec_guards']=[name for _,name,_ in codec_patches]
            phase('all four inputs encrypted at level4; all client codecs guarded')
            counts={}
            for item in ring['public_fixtures']:
                key_mask=np.isin(np.arange(128),item['valid_key_indices'])
                layout=ScoreReductionLayout(key_mask,valid_rows=item['valid_rows'])
                def capture(label,value,observed,name=item['name']):
                    phase(name+' '+label,value,observed)
                router=ScoreReduction(ops,layout,progress=capture)
                value=encrypted_inputs[item['name']+':mean_scores']
                mean=router.mean(value)
                retained.append((item['name'],'mean',mean));mean=None
                value=encrypted_inputs[item['name']+':probabilities']
                denominator=router.squared_denominator(value,domain=item['domain'],inactive_dummy=item['inactive_dummy'])
                retained.append((item['name'],'squared_denominator',denominator));denominator=None
                for name,count in router.counts.items():counts[name]=counts.get(name,0)+count
            report['counts']=counts
            if counts!=dict(public_multiplications=8,actual_composite_rescales=10,path_levels=10,
                    rotations=56,ciphertext_additions=56,ciphertext_multiplications=2,public_bias_additions=2):
                raise RuntimeError('observed operation schedule differs')
            report['server_complete_before_client_validation']=True
            phase('all four encrypted outputs complete; enabling final client validation')
            for module,name,original in codec_patches:setattr(module,name,original)
            codec_patches.clear()
            expected_levels={'mean':2,'squared_denominator':1}
            for fixture,kind,value in retained:
                expected=client_references[fixture][kind]
                decoded=np.asarray(client.decrypt_slots(value))
                finite=decoded.shape==(SLOTS,) and np.isfinite(decoded).all()
                error=float(np.max(np.abs(decoded-expected))) if finite else None
                level=ops._level(value)
                metadata=level==expected_levels[kind] and value._ckks_nsd==1 and math.isclose(value._ckks_scale,scale,rel_tol=1e-12)
                report['checks'][fixture+':'+kind]=dict(passed=bool(finite and error<=1e-6 and metadata),
                    max_abs_error=error,finite_and_shape=bool(finite),slots_checked=SLOTS,level=level,
                    scale=value._ckks_scale,nsd=value._ckks_nsd,metadata_ok=metadata,
                    expected_minimum=float(expected.min()),expected_maximum=float(expected.max()),
                    execution_rings=require_secure_execution(server,value))
            report['input_payload_and_metadata_unchanged']=all(
                fingerprints[name]==(sha(np.asarray(value.polynomial).tobytes()),value._ckks_scale,
                    value._ckks_nsd,ops._level(value)) for name,value in encrypted_inputs.items())
            report['passed']=bool(len(report['checks'])==4 and all(row['passed'] for row in report['checks'].values())
                and report['input_payload_and_metadata_unchanged'])
            report['provider_stats']=dict(provider.stats)
            phase('final all-slot client validation complete')
    except BaseException as error:
        report.update(passed=False,error=f'{type(error).__name__}: {error}')
        raise
    finally:
        primary=sys.exception();tracker=CleanupTracker(primary)
        cleanup=tracker.errors;attempt=tracker.attempt
        for module,name,original in codec_patches:
            attempt('restore codec guard',lambda:setattr(module,name,original))
        if provider is not None:attempt('provider.close',provider.close)
        if ops is not None:
            attempt('rotation.clear',lambda:ops.rotation.clear(executables=True))
            for helper in (ops._plaintext_multiply,ops._rescale):
                attempt('compiled primitive clear',lambda:helper.clear(executables=True))
            attempt('multiplication compiler clear',ops._multiplication_compiler.clear_executables)
        for context in (server,client):
            if context is not None:attempt('evict context caches',lambda:ckks_ctx.evict_context_caches(context))
        context=keys=evaluation=encrypted=router=ops=server=client=value=mean=denominator=None
        retained.clear();encrypted_inputs.clear();client_vectors.clear();client_references.clear()
        attempt('jax.clear_caches',jax.clear_caches);attempt('gc.collect',gc.collect)
        if key_store is not None:attempt('key_store.close',key_store.close)
        if key_parent is not None:attempt('key_parent.cleanup',key_parent.cleanup)
        for module,name,original in patched:
            attempt('restore key generation guard',lambda:setattr(module,name,original))
        for number,handler in old_handlers.items():
            attempt('restore signal handler',lambda:signal.signal(number,handler))
        data=attempt('source verification',sources)
        after={name:sha(content) for name,content in data.items()} if data is not None else {}
        def verify_source_archive():
            with tarfile.open(args.output/'sources.tar.gz') as archive:
                members=archive.getmembers()
                if len(members)!=len(before) or {m.name for m in members}!=set(before):
                    raise ValueError('source inventory differs')
                for member in members:
                    if not member.isfile() or sha(archive.extractfile(member).read())!=before[member.name]:
                        raise ValueError('source archive bytes differ')
            return True
        verified=attempt('verify source archive',verify_source_archive)
        report.update(source_hashes_after=after,source_files_unchanged=bool(before) and before==after,
            source_archive_members_verified=verified is True,cleanup_errors=cleanup,
            public_key_store_removed=key_parent is None or not Path(key_parent.name).exists(),
            elapsed_s=time.monotonic()-started,peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            finished_utc=datetime.now(timezone.utc).isoformat())
        report['passed']=bool(report['passed'] and before==after and verified and not cleanup and report['public_key_store_removed'])
        try:write(args.output/'report.json',report)
        except BaseException as publication_error:
            preserved=primary if primary is not None else tracker.cancellation
            if preserved is None:raise
            preserved.add_note(f'Final report publication failed: {publication_error}')
        tracker.finish()
    return 0 if report['passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
