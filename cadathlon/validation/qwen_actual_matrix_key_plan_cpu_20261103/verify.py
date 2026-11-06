"""Offline public proof checks; optional current/source or public-matrix checks."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import math
import tarfile
import xml.etree.ElementTree as ET


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()


def file_sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def archive(path, expected):
    with tarfile.open(path) as source:
        members = source.getmembers()
        assert len(members) == len({m.name for m in members}) == len(expected) == 22
        assert all(m.isfile() and not PurePosixPath(m.name).is_absolute()
                   and '..' not in PurePosixPath(m.name).parts for m in members)
        raw = {m.name: source.extractfile(m).read() for m in members}
    assert {n: sha(data) for n, data in raw.items()} == expected
    return raw


def check_run(base, *, execute):
    report = json.loads((base / 'report.json').read_bytes())
    expected = report['source_hashes_before']
    assert expected == report['source_hashes_after'] == report['execution_source_hashes_after']
    assert type(report['execute']) is bool and report['execute'] is execute
    assert report['actual_execution_attempted'] is execute
    for key in ('passed', 'all_processes_reaped_and_owned_group_empty', 'sources_unchanged',
                'source_archive_verified', 'terminal_scope_completed', 'termination_scope_completed'):
        assert report[key] is True, key
    assert not report.get('error') and not report['cleanup_errors']
    assert report['he'] is False and report['model_forward'] is False
    process = report['process']
    assert type(process['direct_returncode']) is int and process['direct_returncode'] == 0
    assert all(process[k] is True for k in ('direct_child_reaped', 'group_empty_verified', 'cleanup_completed'))
    assert not process['cleanup_errors'] and not process['signals_sent']
    assert sha((base / 'sources.tar.gz').read_bytes()) == report['source_archive_sha256']
    raw = archive(base / 'sources.tar.gz', expected)
    inventory = json.loads(raw['source_inventory.json'])
    assert set(inventory['files']) == set(expected) and inventory['expected_tests'] == 21
    for name, origin in json.loads(raw['source_origins.json']).items():
        assert sha(raw[name]) == origin['sha256']
    return report, expected, raw


def expected_records(qkv, decoder):
    layer_names = {}
    for layer in range(28):
        prefix = f'model.layers.{layer}.'
        for role, shape in (('q_proj', [2048, 1024]), ('k_proj', [1024, 1024]), ('v_proj', [1024, 1024])):
            layer_names[prefix + 'self_attn.' + role + '.weight'] = shape
        layer_names[prefix + 'self_attn.o_proj.weight'] = [1024, 2048]
        for role, shape in (('gate_proj', [3072, 1024]), ('up_proj', [3072, 1024]), ('down_proj', [1024, 3072])):
            layer_names[prefix + 'mlp.' + role + '.weight'] = shape
    all_rows = qkv['matrices'] | decoder['matrices']
    assert len(qkv['matrices']) == 84 and len(decoder['matrices']) == 114 and len(all_rows) == 198
    assert set(all_rows) == set(layer_names) | {'lm_head.weight', 'model.embed_tokens.weight'}
    assert set(qkv['matrices']) == {n for n in layer_names if any(f'.{r}.' in n for r in ('q_proj', 'k_proj', 'v_proj'))}
    for name, shape in layer_names.items():
        assert all_rows[name]['shape'] == shape
    head, embedding = all_rows['lm_head.weight'], all_rows['model.embed_tokens.weight']
    assert head['shape'] == embedding['shape'] == [151936, 1024]
    assert embedding['storage_alias_of'] == 'lm_head.weight'
    assert all(embedding[k] == head[k] for k in ('file', 'file_sha256', 'sha256', 'source_raw_sha256'))
    assert decoder['tied_embedding_and_head_verified'] is True
    wanted = {(name, None): (shape, None) for name, shape in layer_names.items()}
    wanted.update({('lm_head.weight', i): ([min(4096, 151936 - start), 1024], [start, min(start + 4096, 151936)])
                   for i, start in enumerate(range(0, 151936, 4096))})
    return all_rows, wanted


def check_geometry(row):
    rows, cols = row['shape']
    assert row['absent_diagonals'] == []  # Actual measured matrices all have dense diagonal support.
    active = sorted({d % 65536 for d in range(1 - rows, cols)})
    assert row['active_diagonal_count'] == len(active) and row['active_diagonals_sha256'] == sha(canonical(active))
    target = max(1, round(math.sqrt(2 * len(active))))
    n1 = min((1 << bit for bit in range(17)), key=lambda divisor: (abs(divisor - target), divisor))
    assert (row['n1'], row['n2']) == (n1, 65536 // n1)
    babies = sorted({d % n1 for d in active} - {0})
    giants = sorted({d // n1 * n1 for d in active}, reverse=True)
    baby_steps = [b - a for a, b in zip([0] + babies, babies)]
    giant_steps = [a - b for a, b in zip(giants, giants[1:])]
    final = min(giants) if giants else 0
    calls = [v % 65536 for v in baby_steps + giant_steps + [final] if v % 65536]
    assert row['baby_steps'] == baby_steps and row['giant_steps'] == giant_steps and row['final_giant'] == final
    assert row['required_keys'] == sorted(set(calls)) and row['nonzero_rotation_calls'] == len(calls)
    assert row['weighted_giant_groups'] == len(giants) and row['ciphertext_depth'] == 1


def check_public_bodies(qkv_root, decoder_root, qkv, decoder, plan, recorded_files):
    # The optional check reads only separately exported, public, digest-pinned matrices.
    # No production modules, model loaders or backend modules are imported.
    import numpy as np
    checked = 0
    observed = {}
    records = {(r['name'], r['block']): r for r in plan['matrices']}
    for group, root, manifest in (('qkv', qkv_root, qkv), ('decoder', decoder_root, decoder)):
        for name, desc in manifest['matrices'].items():
            if name == 'model.embed_tokens.weight':
                continue
            assert type(desc['file']) is str and Path(desc['file']).name == desc['file']
            path = root / desc['file']
            assert file_sha(path) == desc['file_sha256']
            value = np.load(path, mmap_mode='r', allow_pickle=False, max_header_size=1024)
            try:
                assert value.dtype.str == '<f8' and list(value.shape) == desc['shape']
                payload = hashlib.sha256()
                for start in range(0, value.shape[0], 4096):
                    chunk = value[start:start + 4096]
                    assert np.isfinite(chunk).all()
                    payload.update(chunk.tobytes(order='C'))
                assert payload.hexdigest() == desc['sha256']
                parts = [(None, value)] if name != 'lm_head.weight' else [
                    (i, value[start:start + 4096]) for i, start in enumerate(range(0, 151936, 4096))]
                for block, part in parts:
                    row = records[name, block]
                    actual = sorted({d % 65536 for d in range(1 - part.shape[0], part.shape[1])
                                     if np.count_nonzero(np.diagonal(part, offset=d))})
                    assert sha(canonical(actual)) == row['active_diagonals_sha256']
                    if block is not None:
                        assert sha(part.tobytes(order='C')) == row['block_sha256']
                    checked += 1
                del part, parts, chunk
            finally:
                value._mmap.close()
            assert file_sha(path) == desc['file_sha256']
            observed[group + '/' + desc['file']] = desc['file_sha256']
    assert observed == recorded_files and len(observed) == 197 and checked == 234
    return {'public_files_checked': 197, 'actual_map_supports_checked': checked, 'backend_or_model_imported': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', type=Path)
    parser.add_argument('--public-bodies', action='store_true')
    parser.add_argument('--qkv-root', type=Path, default=Path('/tmp/qwen-qkv-attention-join-20261103/public_qkv_parameters'))
    parser.add_argument('--decoder-root', type=Path, default=Path('/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices'))
    args = parser.parse_args()
    base = Path(__file__).resolve().parent
    artifacts = json.loads((base / 'artifact_hashes.json').read_bytes())
    assert {p.relative_to(base).as_posix(): sha(p.read_bytes()) for p in base.rglob('*')
            if p.is_file() and p != base / 'artifact_hashes.json'} == artifacts
    report, expected, raw = check_run(base / 'actual-v1', execute=True)
    preflight, before, prior_raw = check_run(base / 'preflight-v1', execute=False)
    assert before == expected and prior_raw == raw
    counts = {key: sum(int(s.attrib.get(key, 0)) for s in ET.parse(base / 'preflight-v1/tests.xml').getroot().iter('testsuite'))
              for key in ('tests', 'errors', 'failures', 'skipped')}
    assert counts == preflight['tests'] == {'tests': 21, 'errors': 0, 'failures': 0, 'skipped': 0}
    worker_raw = (base / 'actual-v1/worker/report.json').read_bytes()
    worker = json.loads(worker_raw)
    assert sha(worker_raw) == report['worker_report_sha256']
    assert all(worker[k] is True for k in ('passed', 'operation_complete', 'termination_scope_completed'))
    assert all(worker[k] is False for k in ('HE', 'backend_imported', 'model_forward', 'key_generation'))
    plan_raw = (base / 'actual-v1/worker/plan.json').read_bytes()
    assert sha(plan_raw) == report['plan_sha256'] == worker['plan_sha256'] == 'bac55debfe95752a16a9a2e849e66488f4afbb576959403a2af4698358dd32ca'
    assert report['source_archive_sha256'] == '743723c927307ce4488860cb15a3fe2980b976f4760c3a9453dbf5e09ff09003'
    plan = json.loads(plan_raw)
    qkv, decoder = (json.loads(raw['inputs/' + n + '_parameters.json']) for n in ('qkv', 'decoder'))
    assert sha(raw['inputs/qkv_parameters.json']) == plan['qkv_manifest_sha256']
    assert sha(raw['inputs/decoder_parameters.json']) == plan['decoder_manifest_sha256']
    assert sha(raw['inputs/candidate_manifest.json']) == plan['candidate_manifest_sha256']
    for manifest in (qkv, decoder):
        assert manifest['candidate_sha256'] == plan['candidate_manifest_sha256']
        assert manifest['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert manifest['config_sha256'] == sha(raw['inputs/config.json'])
    descriptors, wanted = expected_records(qkv, decoder)
    records = {(r['name'], r['block']): r for r in plan['matrices']}
    assert len(records) == len(plan['matrices']) == 234 and set(records) == set(wanted)
    for key, row in records.items():
        shape, rows = wanted[key]
        assert row['shape'] == shape and row['rows'] == rows and row['full_tensor_sha256'] == descriptors[key[0]]['sha256']
        check_geometry(row)
    soft = json.loads(raw['inputs/softmax_plan.json'])
    rms = json.loads(raw['inputs/rms_plan.json'])
    assert len(rms['sites']) == 113
    nonmatrix = set(soft['keys']['combined_component_indices'])
    for site in rms['sites']:
        nonmatrix.update(site['norm_rotation_keys'])
    actual = {k for row in records.values() for k in row['required_keys']}
    full = json.loads(raw['inputs/full_plan.json'])
    dense = {k for row in [*full['linear_maps'].values(), *full['vocabulary']['maps']] for k in row['required_keys_dense_case']}
    assert len(nonmatrix) == 162 and 262143 in nonmatrix and len(nonmatrix | actual) == 166
    assert plan['nonmatrix_keys'] == sorted(nonmatrix) and plan['actual_matrix_keys'] == sorted(actual)
    assert plan['prior_dense_matrix_keys'] == sorted(dense)
    assert plan['new_matrix_keys'] == sorted(actual - dense) == [] and plan['prior_dense_matrix_keys_absent'] == sorted(dense - actual) == []
    assert plan['combined_keys'] == sorted(nonmatrix | actual)
    assert plan['matrix_maps_with_absent_diagonals'] == 0
    assert plan['layer_map_count'] == 196 and plan['vocabulary_blocks'] == 38
    assert plan['matrix_map_count'] == worker['matrix_map_count'] == report['matrix_map_count'] == 234
    assert plan['combined_key_count'] == worker['combined_key_count'] == report['combined_key_count'] == 166
    assert plan['all_layer_linear_rotation_calls_per_row_tile'] == sum(r['nonzero_rotation_calls'] for r in records.values() if r['block'] is None) == 22008
    assert plan['vocabulary_rotation_calls_last_row'] == sum(r['nonzero_rotation_calls'] for r in records.values() if r['block'] is not None) == 6226
    profile = next(p for p in json.loads(raw['inputs/profile_audit.json'])['profiles'] if p['name'] == 'depth49_p60_d8')
    assert plan['raw_rotation_pair_bytes'] == profile['rotation_pair_bytes'] == 2 * 8 * 114 * 131072 * 4
    assert plan['raw_rotation_keys_bytes'] == 166 * plan['raw_rotation_pair_bytes'] == 158746017792
    expected_files = {group + '/' + row['file']: row['file_sha256']
                      for group, manifest in (('qkv', qkv), ('decoder', decoder))
                      for name, row in manifest['matrices'].items() if name != 'model.embed_tokens.weight'}
    assert len(expected_files) == 197 and worker['public_matrix_file_hashes'] == expected_files
    terminal = json.loads((base / 'session_terminal.json').read_bytes())
    assert terminal['session_id'] == 39684 and terminal['observed_exit_code'] == 0 and terminal['tool_session_reaped'] is True
    assert terminal['attestation_source'] == 'Parent agent /root authoritative tool observation communicated to reviewer'
    if args.current:
        assert {n: sha((args.current / n).read_bytes()) for n in expected} == expected
        for sub in ('actual-v1', 'preflight-v1'):
            assert {n: sha((args.current / sub / 'execution' / n).read_bytes()) for n in expected} == expected
    result = dict(passed=True, source_count=22, preflight_tests=21, matrix_maps=234,
                  inherited_nonmatrix_keys=162, combined_keys=166, raw_rotation_key_bytes=158746017792,
                  current_checked=args.current is not None, public_body_readback=args.public_bodies,
                  scope='Public rotation planning only; no runtime key store, memory peak, model or HE proof.')
    if args.public_bodies:
        result.update(check_public_bodies(args.qkv_root, args.decoder_root, qkv, decoder, plan, expected_files))
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
