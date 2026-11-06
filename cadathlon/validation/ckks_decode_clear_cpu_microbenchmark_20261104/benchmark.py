"""Bounded clear CPU decoder microbenchmark; no HE/context/key/model calls."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent

def sha(raw):
    return hashlib.sha256(raw).hexdigest()

def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n')

def verify_sources(expected):
    got = {n: sha((ROOT / n).read_bytes()) for n in expected}
    if got != expected:
        raise RuntimeError('benchmark source closure differs')
    return got

def main(args):
    manifest = args.source_manifest.read_bytes()
    if sha(manifest) != args.source_manifest_sha256:
        raise ValueError('external source manifest differs')
    expected = json.loads(manifest)
    verify_sources(expected)
    plan = json.loads((ROOT / 'plan.json').read_bytes())
    for name, value in {'JAX_PLATFORMS':'cpu', 'JAX_PLATFORM_NAME':'cpu',
                        'JAX_ENABLE_X64':'True', 'PYTHONDONTWRITEBYTECODE':'1'}.items():
        if os.environ.get(name) != value:
            raise RuntimeError('CPU environment required before imports: ' + name)
    if sorted(os.sched_getaffinity(0)) != plan['cpu_affinity']:
        raise ValueError('fixed CPU affinity required')
    args.output.mkdir(exist_ok=False)
    identity = dict(pid=os.getpid(), pgid=os.getpgrp(), cpu_affinity=sorted(os.sched_getaffinity(0)))
    write(args.output / 'started.json', identity)
    report = dict(schema='clear-ckks-decode-microbenchmark-result-v1', **identity,
        passed=False, complete=False, scope=plan['scope'], cases=[],
        source_manifest_sha256=sha(manifest), source_hashes_start=expected,
        ciphertext_or_key_or_model_operation=False, actual_toy5_recovery=False,
        new_optimization=False, timing_scope='Only _fast_decode: CRT, conversion, noise check, FFT. Dataset construction, source checks, equality/hash checks, imports and warmup excluded from measured calls.',
        samples_per_decoder_per_case=4, gc_enabled=gc.isenabled(),
        limitations='One bounded run, shared host under unrelated load, no statistical speed claim; synthetic ordinary coefficients only.')
    primary = None
    try:
        # Exact immutable207 test module installs pre-JAX import/device guards,
        # imports the actual candidate module and compiles only the old decode AST.
        sys.path.insert(0, str(ROOT / 'runtime207'))
        import test_real_decode as proven
        np, ckks = proven.np, proven.ckks
        proven.test_cpu_and_source_origin()
        def forbidden(*unused, **keywords):
            raise AssertionError('HE/context/key/model operation forbidden in clear benchmark')
        for name in ('__init__','encrypt','encrypt_slots','encrypt_slots_batch','decrypt','decrypt_slots','decrypt_slots_batch'):
            setattr(ckks.CKKSContext, name, forbidden)
        for name in ('gen_pke_pair','gen_evaluation_key','gen_rotation_key'):
            setattr(ckks.kg, name, forbidden)
        ckks._fast_decrypt_to_rns_coeffs_array = forbidden
        assert all(sha((ROOT / 'runtime207' / n).read_bytes()) == h for n,h in plan['original_runtime207_sources'].items())
        slots = np.random.default_rng(plan['seed']).uniform(-0.5, 0.5, plan['slots'])
        slots.setflags(write=False)
        coeff = [complex(float(x), 0.0) for x in slots]
        ckks.FFTSpecialInv(coeff, 2 * plan['degree'])
        components = np.asarray([v.real for v in coeff] + [v.imag for v in coeff], dtype=np.float64)
        assert np.isfinite(components).all()
        components.setflags(write=False)
        dataset = args.output / 'synthetic_datasets'; dataset.mkdir()
        np.save(dataset / 'slots.npy', slots, allow_pickle=False)
        np.save(dataset / 'inverse_components.npy', components, allow_pickle=False)
        report['dataset'] = dict(seed=plan['seed'], slots_dtype=str(slots.dtype),
            slots_sha256=sha(slots.tobytes()), inverse_components_sha256=sha(components.tobytes()),
            degree=plan['degree'], slots=plan['slots'], original_inverse_fft_used=True,
            synthetic_real_slot_range=[float(slots.min()),float(slots.max())])
        decoders = {'old': proven.OLD, 'candidate': ckks._fast_decode}
        for case in plan['cases']:
            numbers = [ckks.nearest_int(float(v) * case['scale']) for v in components]
            assert all(abs(v) < 2**63 for v in numbers)
            ints = np.asarray(numbers, dtype=np.int64); ints.setflags(write=False)
            residues = np.remainder(ints[:,None], np.asarray(case['q'],dtype=np.int64)[None,:]).astype(np.uint64)
            residues.setflags(write=False)
            np.save(dataset / (case['name'] + '_coefficients.npy'), ints, allow_pickle=False)
            np.save(dataset / (case['name'] + '_residues.npy'), residues, allow_pickle=False)
            before = sha(residues.tobytes())
            ctx = SimpleNamespace(q_towers=case['q'], num_slots=plan['slots'])
            row = dict(name=case['name'], q=case['q'], num_moduli=len(case['q']), scale=case['scale'],
                scale_hex=case['scale'].hex(), scale_scope=case['scale_scope'],
                residues_shape=list(residues.shape), residues_dtype=str(residues.dtype),
                residues_sha256=before, integer_coefficients_sha256=sha(ints.tobytes()),
                maximum_absolute_integer_coefficient=str(max(abs(v) for v in numbers)),
                all_integer_to_float_conversions_representable=True, calls=[])
            # No overflow path is intended: these are ordinary quantized FFT coefficients.
            assert all(abs(v).bit_length() <= 63 for v in numbers)
            baseline = None
            schedule = [('warmup',None,name) for name in plan['warmup_order']]
            schedule += [('measured',pair,name) for pair,order in enumerate(plan['measured_pair_orders']) for name in order]
            for sequence,(phase,pair,name) in enumerate(schedule):
                start = time.perf_counter_ns()
                output = decoders[name](residues, ctx, case['scale'])
                elapsed = time.perf_counter_ns() - start
                assert output.shape == (plan['slots'],) and output.dtype == np.float64
                assert np.isfinite(output).all()
                if baseline is None:
                    baseline = output.copy(); baseline.setflags(write=False)
                assert np.array_equal(output.view(np.uint64), baseline.view(np.uint64))
                call = dict(sequence=sequence, phase=phase, pair=pair, decoder=name,
                    elapsed_ns=elapsed, elapsed_seconds=elapsed/1e9,
                    exact_binary64_agreement=True, output_sha256=sha(output.tobytes()))
                row['calls'].append(call)
                print(json.dumps({'case':case['name'],**call},sort_keys=True),flush=True)
                del output
            assert sha(residues.tobytes()) == before
            row['residues_unchanged'] = True
            row['max_abs_to_synthetic_slots_diagnostic'] = float(np.max(np.abs(baseline-slots)))
            report['cases'].append(row)
            write(args.output / 'progress.json', report)
            del residues, ints, numbers, baseline
        report['dataset_files'] = {p.name:dict(bytes=p.stat().st_size,sha256=sha(p.read_bytes())) for p in sorted(dataset.iterdir())}
        report['import_guard_denials'] = proven.IMPORT_GUARD_DENIALS
        report['device_access_denials'] = proven.DEVICE_ACCESS_DENIALS
        report['jax_backend'] = proven.jax.default_backend()
        report['device_platforms'] = [d.platform for d in proven.jax.devices()]
        from jax._src.lib import _jax
        report['tpu_plugin_loaded_at_end'] = bool(_jax.pjrt_plugin_loaded('tpu'))
        assert report['jax_backend']=='cpu' and report['device_platforms']==['cpu']
        assert not report['device_access_denials'] and report['tpu_plugin_loaded_at_end'] is False
        report['environment'] = dict(numpy_version=np.__version__,jax_version=proven.jax.__version__,python=sys.version)
        report['loaded_local_modules'] = {}
        for name,module in tuple(sys.modules.items()):
            path=getattr(module,'__file__',None)
            if path is not None:
                path=Path(path).resolve()
                if path.is_relative_to(ROOT/'runtime207') and path.is_file():
                    rel=str(path.relative_to(ROOT));digest=sha(path.read_bytes())
                    assert expected[rel]==digest
                    report['loaded_local_modules'][name]=dict(path=rel,sha256=digest)
        report['complete']=True;report['passed']=True
    except BaseException as error:
        primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        try:
            report['source_hashes_end']=verify_sources(expected)
            report['source_unchanged']=True
        except BaseException as error:
            report['passed']=False;report['source_unchanged']=False
            report['source_error']=dict(type=type(error).__name__,message=str(error))
            if primary is None:primary=error
            else:primary.add_note('source readback: '+repr(error))
        write(args.output/'report.json',report)
    if primary is not None:raise primary
    return 0 if report['passed'] else 1

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-manifest',type=Path,required=True)
    parser.add_argument('--source-manifest-sha256',required=True)
    raise SystemExit(main(parser.parse_args()))
