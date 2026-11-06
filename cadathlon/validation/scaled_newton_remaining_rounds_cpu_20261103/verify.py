"""Read-only evidence audit. Does not run encryption or overwrite evidence."""
import ast
import hashlib
import json
import math
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()


def main():
    r = json.loads((HERE/'encrypted/report.json').read_text())
    audit = json.loads((HERE/'root_readback.json').read_text())
    assert sha((HERE/'encrypted/report.json').read_bytes()) == audit['report_sha256']
    assert sha((HERE/'encrypted/sources.tar.gz').read_bytes()) == audit['source_archive_sha256']
    assert r['passed'] is True and r['source_files_unchanged'] is True
    assert not r['cleanup_errors'] and not r.get('preflight_only', False)
    hashes = r['source_hashes_before']
    assert hashes == r['source_hashes_after'] and len(hashes) == 97
    with tarfile.open(HERE/'encrypted/sources.tar.gz') as archive:
        members = archive.getmembers()
        assert len(members) == len({m.name for m in members}) == 97
        assert all(m.isfile() for m in members)
        blobs = {m.name: archive.extractfile(m).read() for m in members}
    assert {n: sha(v) for n, v in blobs.items()} == hashes
    for name in ('check_remaining_rounds.py', 'public_schedule.py', 'baseline_manifest.json', 'test_newton.py'):
        assert sha((HERE/name).read_bytes()) == hashes[name]
    preflight = json.loads((HERE/'preflight_final/report.json').read_text())
    assert preflight['preflight_only'] and preflight['passed']
    assert preflight['source_hashes_before'] == preflight['source_hashes_after'] == hashes
    ring = r['ring']; q, p = ring['q_towers'], ring['p_towers']
    assert (ring['degree'], ring['depth'], ring['dnum'], len(q), len(p)) == (131072, 41, 3, 84, 27)
    assert math.prod(q+p).bit_length() == 3148 <= ring['security_bit_length_ceiling'] == 3523
    assert math.isclose(math.log2(math.prod(q+p)), ring['actual_log2_QP'], rel_tol=1e-14)
    assert math.prod(p) >= max(math.prod(q[i:i+28]) for i in range(0, len(q), 28))
    assert all(value % (2*ring['degree']) == 1 for value in q+p)
    tree = ast.parse(blobs['checkout/jaxite_word/he_params.py'].decode())
    table = next(n.value for n in tree.body if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == '_OPENFHE_CEILINGS_RAW' for t in n.targets))
    ceilings = next(ast.literal_eval(v) for k, v in zip(table.keys, table.values)
                    if isinstance(k, ast.Tuple) and isinstance(k.elts[0], ast.Attribute)
                    and k.elts[0].attr == 'CLASSICAL' and ast.literal_eval(k.elts[1]) == 128)
    assert ceilings[131072] == 3523
    assert r['keyless_server'] is True
    assert set(r['key_generation_guarded_modules']) == {'key_gen', 'jaxite_word.key_gen'}
    assert r['fixed_absolute_error_gate'] == 1e-6 and len(r['checks']) == 13
    level, scale = 40, ring['scale']
    for i, c in enumerate(r['checks']):
        if i:
            scale = scale*ring['scale']/math.prod(q[2*(level-2):2*(level-1)])
            level -= 3
        assert c['update'] == i+1 and c['passed'] and c['metadata_ok']
        assert c['level'] == level and c['nsd'] == 1 and c['all_slots_checked'] == 65536
        assert math.isclose(c['scale'], scale, rel_tol=1e-12)
        assert c['expected_scale'] == scale and c['max_abs_error_against_exact_float_circuit'] < 1e-6
        assert len(c['execution_rings']) == 1
        actual = c['execution_rings'][0]
        assert actual['q_towers'] == q and actual['p_towers'] == p and actual['degree'] == 131072
        assert actual['security_bits'] == 128 and actual['security_model'] == 'classical'
    assert level == 4
    assert len(r['public_contracts']) == len(r['declared_stage_outputs']) == 4
    for i, (contract, check) in enumerate(zip(r['public_contracts'], r['declared_stage_outputs'])):
        points = contract['input_points']; a, b = contract['a'], contract['b']
        assert contract['stage'] == check['stage'] == i+1
        assert contract['iterations'] == check['update'] == (9, 12, 13, 13)[i]
        assert contract['K'] == 128 and contract['s'] == 4. and contract['C'] == 32.
        assert contract['initial_z'] == 2./(32.*(a+b))
        assert contract['distinct_input_count'] == len(set(points)) == 258
        assert min(points) == a and max(points) == b and contract['inactive_dummy'] in points
        assert math.nextafter(a, b) in points and math.nextafter(b, a) in points
        assert check['segment_start'] == i*16384 and check['all_segment_slots_checked'] == 16384
        assert check['passed'] and check['max_abs_error_against_true_inverse'] < 1e-6
    logged = [json.loads(line) for line in (HERE/'encrypted.log').read_text().splitlines() if line.startswith('{')]
    assert logged == r['phases']
    names = [p['name'] for p in logged]
    assert names.index('Newton update13 complete') < names.index('entire thirteen-update server circuit complete; client validation begins') < names.index('client validation finished')
    assert audit['observed_exit_code'] == 0 and audit['session_reaped'] and audit['child_absent']
    print(json.dumps(dict(passed=True, archived_sources=97, encrypted_update_slot_checks=13*65536,
        fixed_absolute_gate=1e-6, worst_update_error=max(c['max_abs_error_against_exact_float_circuit'] for c in r['checks']),
        worst_true_inverse_error=max(c['max_abs_error_against_true_inverse'] for c in r['declared_stage_outputs']))))


if __name__ == '__main__':
    main()
