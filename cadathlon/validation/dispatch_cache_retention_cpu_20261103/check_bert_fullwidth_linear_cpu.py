"""ISOLATED cache-retention probe with secure pinned BERT projections; no HBM claim."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import tarfile
import tempfile
import time

if os.environ.get('JAX_PLATFORMS') != 'cpu':
    raise RuntimeError('This component proof requires explicit JAX_PLATFORMS=cpu')
ROOT = Path(os.environ['CROSS_ROOT']).resolve()
sys.path[:0] = [str(ROOT), str(ROOT / 'cadathlon/common')]
import jax
import numpy as np
from jaxite_word import CKKSContext, ckks_ctx, he_params, key_gen, util
from jaxite_word.composite_prime_gen import composite_prime_gen
from encrypted_packed_qwen import MatrixDiagonals
from encrypted_packed_transformer import PackedPairLayout, PackedTransformer
from encrypted_refresh import ClientRotationKeyStore
from support import require_secure_execution


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def source_bytes(extra):
    paths = {Path(__file__).resolve(), *extra}
    for folder in ('jaxite_word', 'cadathlon/common'):
        paths.update((ROOT / folder).rglob('*.py'))
    return {str(p): p.read_bytes() for p in sorted(paths)}


def load_pinned_matrices(args, manifest):
    revision = 'f150c1d609d1e50dd5e2e5408661cfac8339277c'
    checkpoint_hash = 'e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
    exporter_hash = '5b2c52d51ca1a2ef55d4fa2b2e1b878168e4b8b05d39b640d24dd073640efd0f'
    checkpoint = Path(manifest['checkpoint_file'])
    if (manifest['model_revision'] != revision or checkpoint.parent.name != revision
            or manifest['checkpoint_sha256'] != checkpoint_hash
            or digest(checkpoint) != checkpoint_hash
            or manifest['export_source_sha256'] != exporter_hash
            or digest(args.export_source) != exporter_hash
            or digest(args.weights) != manifest['export_sha256']):
        raise ValueError('Pinned checkpoint/export provenance mismatch')
    expected = {}
    for name, shape in [('query', (768, 768)), ('intermediate', (3072, 768)), ('output', (768, 3072))]:
        prefix = ('bert.encoder.layer.0.attention.self.query' if name == 'query'
                  else f'bert.encoder.layer.0.{name}.dense')
        for suffix, array_shape in [('weight', shape), ('bias', (shape[0],))]:
            expected[f'{name}_{suffix}'] = (prefix + '.' + suffix, array_shape)
    matrices = {}
    # Verify each exported array directly against the pinned safetensors bytes.
    with checkpoint.open('rb') as stream, np.load(args.weights, allow_pickle=False) as arrays:
        header_size = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_size))
        data_start = 8 + header_size
        if set(arrays.files) != set(expected) or set(manifest['arrays']) != set(expected):
            raise ValueError('Unexpected exported array inventory')
        for key, (tensor_name, shape) in expected.items():
            raw = arrays[key]
            spec = manifest['arrays'][key]
            tensor = header[tensor_name]
            if (raw.shape != shape or raw.dtype.str != '<f4'
                    or spec['shape'] != list(shape) or spec['dtype'] != raw.dtype.str
                    or manifest['original_tensor_names'][key] != tensor_name
                    or tensor['shape'] != list(shape) or tensor['dtype'] != 'F32'):
                raise ValueError(f'Pinned array metadata mismatch: {key}')
            raw_hash = hashlib.sha256(raw.tobytes(order='C')).hexdigest()
            start, stop = tensor['data_offsets']
            if stop - start != raw.nbytes:
                raise ValueError(f'Checkpoint tensor size mismatch: {key}')
            stream.seek(data_start + start)
            if (raw_hash != spec['sha256']
                    or raw_hash != hashlib.sha256(stream.read(stop - start)).hexdigest()
                    or not np.isfinite(raw).all()):
                raise ValueError(f'Pinned array byte mismatch: {key}')
            matrices[key] = raw.astype(np.float64)
    return {name: (matrices[name + '_weight'], matrices[name + '_bias'])
            for name in ('query', 'intermediate', 'output')}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--export-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    extra_sources = [args.manifest.resolve(), args.export_source.resolve()]
    sources = source_bytes(extra_sources)
    before = {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()}
    with tarfile.open(args.output / 'sources.tar.gz', 'w:gz') as archive:
        for name, data in sources.items():
            p = Path(name)
            member = str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else 'external/' + p.name
            info = tarfile.TarInfo(member)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    report = {'scope': __doc__, 'passed': False, 'cpu_only': True,
              'started_utc': datetime.now(timezone.utc).isoformat(),
              'full_model_evaluated': False, 'mrpc_accuracy_evaluated': False,
              'source_hashes_start': before, 'checks': {},
              'input_contract': 'One deterministic public vector per full-width matrix, repeated in two slot blocks; padding also checked.',
              'fixed_absolute_error_gate': 1e-6,
              'weights_npz_sha256': digest(args.weights),
              'weight_manifest': json.loads(args.manifest.read_text()),
              'versions': {name: importlib.metadata.version(name) for name in ('jax', 'jaxlib', 'numpy')},
              'cpu_affinity': sorted(os.sched_getaffinity(0))}
    write(args.output / 'report.json', report)
    client = server = ops = key_store = None
    patched = []
    key_parent = None
    started = time.perf_counter()
    try:
        jax.config.update('jax_enable_x64', True)
        degree, depth, dnum = 16384, 1, 2
        slots, features = degree // 2, 4096
        matrices = load_pinned_matrices(args, report['weight_manifest'])
        report['pinned_checkpoint_arrays_verified'] = True
        key_parent = tempfile.TemporaryDirectory(prefix='cross-bert-linear-keys-')
        layout = PackedPairLayout(1, 1, features, slots)
        rotations = sorted(set().union(*(MatrixDiagonals(w, slots, features).rotations for w, _ in matrices.values())))
        q = composite_prime_gen(2, 2 * (depth + 1), 61, 56, 2 * degree, 31)
        p = util.generate_p_towers(q, dnum, degree, aux_bits=30)
        assert he_params.qp_is_secure(q, p, degree, he_params.SecurityTarget(128, 'classical'))
        he_params.assert_p_coverage(q, p, dnum)
        scale = float(q[-2]) * q[-1]
        report['ring'] = dict(degree=degree, depth=depth, dnum=dnum, q_towers=q, p_towers=p,
                              actual_log2_qp=sum(map(math.log2, q + p)), security_bits=128,
                              slots=slots, features=features, rotations=rotations)
        print(f'Preparing secure full-width linear component with {len(rotations)} rotation keys', flush=True)
        keys = key_gen.gen_pke_pair(q, [], degree)
        evaluation = key_gen.gen_evaluation_key(keys['secret_key'], q=q, P=p, dnum=dnum)
        key_store = ClientRotationKeyStore.generate(keys['secret_key'], q, p, degree=degree,
                                                    dnum=dnum, rotations=rotations, directory=key_parent.name)
        params = dict(degree=degree, num_slots=slots, scaling_factor=scale, output_scale=scale,
                      q_towers=q, p_towers=p, composite_degree=2, degree_layout=(128, 128),
                      p=56, CKKS_M_FACTOR=1, max_bits_in_word=61, noise_scale_degree=1)
        client = CKKSContext(dict(params, **keys))
        server = CKKSContext(dict(params, evaluation_key=[
            jax.numpy.asarray(evaluation[name], dtype='uint32').transpose(0, 2, 1) for name in ('a', 'b')]))
        def forbidden(*a, **kw):
            raise AssertionError('Secret-key generation or server encryption/decryption is forbidden')
        for module in {key_gen, sys.modules.get('key_gen')} - {None}:
            for name in ('gen_pke_pair', 'gen_evaluation_key', 'gen_rotation_key', 'key_switch_gen'):
                patched.append((module, name, getattr(module, name)))
                setattr(module, name, forbidden)
        server.program_initialization(rotations, dnum=dnum, r=128, c=128, batch=1,
                                       pregenerated_rotation_keys=key_store.keys)
        client._polynomial_ntt_ctx = server._param_cache.ntt_q_max
        assert server.secret_key is None and server._param_cache.secret_key is None
        assert server.parameters.get('secret_key') is None
        server.encrypt = server.decrypt = server.encrypt_slots = server.decrypt_slots = forbidden
        server.encrypt_slots_batch = server.decrypt_slots_batch = forbidden
        ops = PackedTransformer(server, layout, low_memory=True)
        report['setup_s'] = time.perf_counter() - started
        report['keyless_server'] = True
        report['key_generation_guards'] = sorted({module.__name__ for module, _, _ in patched})
        rng = np.random.default_rng(94117)
        for name, (weight, bias) in matrices.items():
            values = rng.uniform(-.25, .25, size=(1, 1, weight.shape[1]))
            packed = layout.pack_client_embeddings(values)[0]
            encrypted = client.encrypt_slots(packed)
            require_secure_execution(server, encrypted)
            print(f'{name}: evaluating {weight.shape[0]}x{weight.shape[1]}', flush=True)
            begin = time.perf_counter()
            actual = ops.linear(encrypted, weight, bias)
            actual.polynomial.block_until_ready()
            elapsed = time.perf_counter() - begin
            rings = require_secure_execution(server, actual)
            decoded = np.asarray(client.decrypt_slots(actual))
            expected = layout.pack_client_embeddings(values @ weight.T + bias)[0]
            error = float(np.max(np.abs(decoded - expected)))
            level = ops.arithmetic._level(actual)
            metadata_ok = (level == 0 and actual._ckks_nsd == 1
                           and math.isclose(actual._ckks_scale, scale, rel_tol=1e-12))
            report['checks'][name] = dict(passed=bool(error < 1e-6 and metadata_ok),
                weight_shape=list(weight.shape), slots_checked=slots, max_abs_error=error,
                level=level, metadata_ok=metadata_ok, execution_rings=rings, server_cpu_s=elapsed)
            write(args.output / 'report.json', report)
            print(json.dumps(report['checks'][name]), flush=True)
            del encrypted, actual, decoded, expected, packed, values
            gc.collect()
        report['passed'] = (set(report['checks']) == {'query', 'intermediate', 'output'}
                            and all(row['passed'] for row in report['checks'].values()))
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        cleanup_errors = []
        def attempt(label, action):
            try:
                return action()
            except BaseException as error:
                cleanup_errors.append(f'{label}: {type(error).__name__}: {error}')
        for context in (server, client):
            if context is not None:
                attempt('evict_context_caches', lambda: ckks_ctx.evict_context_caches(context))
        context = None
        ops = server = client = None
        attempt('jax.clear_caches', jax.clear_caches)
        attempt('gc.collect', gc.collect)
        if key_store is not None:
            attempt('key_store.close', key_store.close)
        if key_parent is not None:
            attempt('key_parent.cleanup', key_parent.cleanup)
        for module, name, original in patched:
            attempt('restore ' + module.__name__ + '.' + name,
                    lambda: setattr(module, name, original))
        after_sources = attempt('source verification', lambda: source_bytes(extra_sources))
        after = ({name: hashlib.sha256(data).hexdigest() for name, data in after_sources.items()}
                 if after_sources is not None else {})
        weights_after = attempt('weight verification', lambda: digest(args.weights))
        report.update(source_hashes_end=after, source_files_unchanged=before == after,
                      finished_utc=datetime.now(timezone.utc).isoformat(),
                      cleanup_errors=cleanup_errors,
                      weights_unchanged=weights_after == report['weights_npz_sha256'],
                      public_key_store_removed=key_parent is None or not Path(key_parent.name).exists(),
                      elapsed_s=time.perf_counter() - started,
                      peak_host_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        report['passed'] = (report['passed'] and before == after and not cleanup_errors
                            and report['weights_unchanged'] and report['public_key_store_removed'])
        write(args.output / 'report.json', report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
