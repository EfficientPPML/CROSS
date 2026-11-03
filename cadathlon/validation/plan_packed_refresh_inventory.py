#!/usr/bin/env python3
"""Plan public packed refresh keys/storage without keys, ciphertexts or TPU.

All twelve fixtures are evaluated under one hypothetical shared profile:
N=131072, depth49, dnum4, scale56, first61, budgets(4,4). Default adapters
reject deep graphs; explicit experimental bounded adapters are available.
This inventory executes neither adapter and does not validate refresh accuracy.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu':
    raise RuntimeError('inventory planning requires JAX_PLATFORMS=cpu')
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'cadathlon/common')]

from encrypted_packed_transformer import plan_packed_transformer
from encrypted_reference import reference_ring_profile
from encrypted_refresh import RefreshProfile, refresh_plan
from jaxite_word import key_gen
from support import load_module

GIB = 2**30
DEGREE, DEPTH, DNUM = 131072, 49, 4
RESERVE = 64 * 2**20


def forbidden_key_generation(*args, **kwargs):
    raise AssertionError('this public inventory must not generate secret or evaluation keys')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=Path(__file__).with_suffix('.json'))
    args = parser.parse_args()
    # Enforce the no-keygen boundary even if a called planner later changes.
    for module in {key_gen, sys.modules.get('key_gen')} - {None}:
        for name in dir(module):
            if name.startswith('gen_') or name == 'key_switch_gen':
                setattr(module, name, forbidden_key_generation)
    paths = [Path(__file__)]
    paths += [ROOT/'cadathlon/common'/name for name in (
        'encrypted_refresh.py', 'encrypted_reference.py', 'encrypted_packed_transformer.py', 'encrypted_packed_qwen.py',
        'encrypted_transformer.py', 'encrypted_polynomial.py', 'support.py',
        'encrypted_bounded_arithmetic.py', 'encrypted_bounded_transformer.py')]
    paths += [ROOT/'jaxite_word'/name for name in (
        'bootstrapping.py', 'bsgs.py', 'key_gen.py', 'he_params.py', 'util.py', 'composite_prime_gen.py')]
    for problem in ('p2_bert', 'p4_qwen'):
        folder = ROOT/'cadathlon/agent_pset'/problem
        model = 'bert' if problem == 'p2_bert' else 'qwen'
        paths += [folder/'src/reference.py', folder/f'src/{model}_bounded_reference.py',
                  *sorted((folder/'test').glob('test*.in'))]

    def snapshot():
        return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}

    before = snapshot()
    params = RefreshProfile(DEGREE, DEPTH, 56, 61, DNUM).parameters()
    q, p = params['q_towers'], params['p_towers']
    alpha = math.ceil(len(q)/DNUM)
    parts = math.ceil(len(q)/alpha)
    key_shape = (parts, len(q)+len(p), DEGREE)
    pair_bytes = 2 * math.prod(key_shape) * 4
    filesystem = {}
    for path in ('/tmp', '/dev/shm'):
        usage = shutil.disk_usage(path)
        filesystem[path] = dict(total_bytes=usage.total, free_bytes=usage.free,
                                free_gib=usage.free/GIB, observed_utc=datetime.now(timezone.utc).isoformat())
    filesystem['nominal_80_gib_disk'] = dict(free_bytes=80*GIB, free_gib=80.,
        basis='Requested post-cleanup planning allowance; post-cleanup free space has not been measured.')
    filesystem['nominal_709_gib_tmpfs'] = dict(free_bytes=709*GIB, free_gib=709.,
        basis='Rounded planning allowance; use the separately measured /dev/shm snapshot before execution.')

    def storage(rotations):
        raw = len(rotations)*pair_bytes
        # Mirror ClientRotationKeyStore's conservative two-.npy/header allowance.
        store = raw + len(rotations)*8192
        return dict(raw_key_bytes=raw, raw_key_gib=raw/GIB,
                    disk_store_preflight_bytes=store, required_with_reserve_bytes=store+RESERVE,
                    fits={path: store+RESERVE <= item['free_bytes'] for path, item in filesystem.items()})

    cases, all_rotations = {}, set()
    for model, problem in (('bert', 'p2_bert'), ('qwen', 'p4_qwen')):
        folder = ROOT/'cadathlon/agent_pset'/problem
        oracle = load_module(folder/'src/reference.py')
        for number in range(1, 7):
            fixture = folder/f'test/test{number}.in'
            case = json.loads(fixture.read_text())
            ids = case['input_ids' if model == 'bert' else 'token_ids']
            batch, sequence = len(ids), len(ids[0])
            weights = oracle.toy_weights(case['config'], case['weight_seed'])
            graph = plan_packed_transformer(model, case['config'], weights, batch=batch,
                sequence=sequence, slots=DEGREE//2, attention_mask=case.get('attention_mask'))
            period = graph['minimum_slots']
            bootstrap = refresh_plan(DEGREE, period, (4, 4), max_level=DEPTH)
            graph_keys, bootstrap_keys = set(graph['rotations']), set(bootstrap['rotations'])
            projection_keys, shift = set(), period
            while shift < DEGREE//2:
                projection_keys.add(shift)
                shift *= 2
            assert projection_keys <= bootstrap_keys
            union = graph_keys | bootstrap_keys | projection_keys
            all_rotations.update(union)
            current_depth, current_dnum = reference_ring_profile(graph['depth'])
            current_profile = dict(status='default adapter rejects before keygen; explicit experimental bounded adapter available'
                if graph['depth'] > 52 else 'leveled reference; bootstrap keys are not required',
                chain_depth=current_depth, dnum=current_dnum, graph_rotation_count=len(graph_keys))
            cases[f'{model}/test{number}.in'] = dict(
                inventory_profile='hypothetical depth49/dnum4 graph-plus-bootstrap union',
                hypothetical_for_current_leveled_reference=graph['depth'] <= 52,
                current_reference_profile=current_profile,
                fixture=str(fixture.relative_to(ROOT)), batch=batch, sequence=sequence,
                features=graph['features'], period=period, aliases=DEGREE//2//period,
                sparse_bootstrap=period < DEGREE//2,
                graph_depth=graph['depth'], requires_refresh_at_depth49=graph['depth'] > DEPTH,
                default_reference_rejects_deep_graph=graph['depth'] > 52,
                experimental_bounded_adapter_available=True,
                bounded_adapter=str((folder/f'src/{model}_bounded_reference.py').relative_to(ROOT)),
                complete_encrypted_validation_established_by_this_inventory=False,
                bootstrap_depth=bootstrap['boot_depth'], normalized_output_level=bootstrap['output_level'],
                normalized_output_q_count=2*(bootstrap['output_level']+1),
                reusable_levels_after_reserving_two_for_refresh=bootstrap['output_level']-2,
                bootstrap_rotation_count=len(bootstrap_keys), graph_rotation_count=len(graph_keys),
                overlapping_rotation_count=len(graph_keys & bootstrap_keys), union_rotation_count=len(union),
                graph_rotation_indices=sorted(graph_keys), bootstrap_rotation_indices=sorted(bootstrap_keys),
                projection_rotation_indices=sorted(projection_keys), union_rotation_indices=sorted(union),
                **storage(union))
    retained_raw = sum(row['raw_key_bytes'] for row in cases.values())
    retained_store = sum(row['disk_store_preflight_bytes'] for row in cases.values())
    report = dict(scope=__doc__, encrypted_execution=False, key_generation_performed=False,
        graph_activation_execution=False, source_sha256=before,
        profile=dict(degree=DEGREE, num_slots=DEGREE//2, depth=DEPTH, dnum=DNUM,
                     scale_bits=56, first_bits=61, budget=[4, 4], q_count=len(q), p_count=len(p),
                     q_towers=q, p_towers=p, log2_qp=sum(math.log2(x) for x in q+p),
                     security_bits=128, actual_qp_security_and_p_coverage_checked=True,
                     partition_width=alpha, partition_count=parts, raw_key_component_shape=key_shape,
                     raw_key_pair_bytes=pair_bytes, raw_key_pair_gib=pair_bytes/GIB,
                     relinearization_pair_additional_host_bytes=pair_bytes),
        storage_snapshots=filesystem, cases=cases,
        all_twelve_distinct_client_stores_retained=dict(raw_key_bytes=retained_raw,
            raw_key_gib=retained_raw/GIB, disk_store_preflight_bytes=retained_store,
            fits={path: retained_store+RESERVE <= row['free_bytes'] for path, row in filesystem.items()}),
        hypothetical_one_client_reused_union=dict(rotation_count=len(all_rotations),
            rotation_indices=sorted(all_rotations), **storage(all_rotations)),
        next_proof=dict(period=512, aliases=128, representative_fixtures=['bert/test1.in', 'qwen/test3.in'],
            reason='First larger-period tier after128; four times fewer aliases,64 bootstrap keys vs46, and representative of the first packed BERT graph.',
            later_distinct_stress_cases=[
                'period8192: aliases8; BERT4/Qwen6 key stores exceed80GiB',
                'period32768: aliases2; BERT5 has the largest226-key union',
                'period65536: aliases1; BERT6 takes the full-slot bootstrap path and has no projection averaging']),
        limitations=[
            'All twelve rows use a hypothetical common refresh profile, including shallow graphs that currently use other leveled profiles. Explicit bounded adapters are available, but this inventory does not execute or validate them.',
            'Normalized output capacity25 is a static depth contract; bounded refresh requires input level2, leaving23 levels before another reserved refresh.',
            'No scheduler invocation count, actual HE error, bootstrap success, keygen time, runtime or device peak-memory claim is made.',
            'Aliases count repeated copies; errors may be correlated, so no square-root precision improvement is assumed. Coherent error survives averaging.',
            'Key sizes cover raw uint32 rotation/conjugation keys plus current store overhead and64MiB reserve; exclude public control tables, formatted device keys, ciphertexts, compilation and process memory.',
            'Tmpfs files occupy host RAM and can coexist with substantial runtime allocations; fitting the filesystem quota is not a memory-safety proof.',
            'The reused all-fixture union requires one compatible client key/profile and explicit lifetime reuse; current sessions generate distinct client stores.',
            'Free storage is a dated snapshot and must be checked immediately before actual key generation.'
        ])
    report['source_unchanged_at_recording'] = before == snapshot()
    if not report['source_unchanged_at_recording']:
        raise RuntimeError('source changed while planning; rerun against stable files')
    args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({case: {key: row[key] for key in ('period', 'aliases', 'union_rotation_count', 'raw_key_gib')}
                      for case, row in cases.items()}, indent=2))
    print(json.dumps({'all_distinct_gib': retained_raw/GIB, 'shared_union_keys': len(all_rotations),
                      'shared_union_gib': len(all_rotations)*pair_bytes/GIB}))


if __name__ == '__main__':
    main()
