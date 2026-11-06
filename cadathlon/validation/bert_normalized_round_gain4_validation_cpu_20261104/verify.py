"""Portable public evidence verification; no runtime imports or payload reads."""
from pathlib import Path
import argparse
import hashlib
import io
import json
import math
import tarfile

HERE = Path(__file__).resolve().parent


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def parse(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate metadata key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def read(name):
    path = HERE / name
    assert path.is_file() and not path.is_symlink()
    return path.read_bytes()


def verify(current=None):
    inventory = parse(read('files.sha256.json'))
    actual = {p.name for p in HERE.iterdir() if p.is_file() and p.name != 'files.sha256.json'}
    assert actual == set(inventory)
    for name, pin in inventory.items():
        assert Path(name).name == name and sha(read(name)) == pin
    refs = parse(read('public_references.json'))
    assert set(refs) == {'server', 'server_sources', 'server_review', 'ready_sources',
                         'launch_grant', 'ready_review', 'ready_proof'}
    linked = {}
    for role, row in refs.items():
        parts = Path(row['path']).parts
        assert len(parts) == 3 and parts[0] == '..'
        assert parts[1] in {'bert_normalized_round_gain4_server_cpu_20261104',
                            'bert_normalized_round_gain4_validator_ready_cpu_20261104'}
        path = HERE / row['path']
        assert path.is_file() and not path.is_symlink()
        raw = path.read_bytes()
        assert sha(raw) == row['sha256']
        linked[role] = parse(raw)
    assert refs['server']['sha256'] == '897fa13c99005dff41d10e192d87417a6d31a155995c3e65805922fc5326d68a'
    assert refs['ready_sources']['sha256'] == 'e0bfcee1ab85f8750e576de24bdbdf13faf4f0b8abe66ecef55b3403a084cce0'
    assert refs['launch_grant']['sha256'] == 'c2493b48d5e38462b652004f7b7e856b4f20af6c9b83840dacce43843caecaa0'
    assert sha(read('report.json')) == '80599a5bdfa1cf531da3c165fa20cd607beb78a05e83127bf7962ce65ded0be2'
    parent = parse(read('report.json'))
    worker = parse(read('worker_report.json'))
    owner = parse(read('owner-terminal.json'))
    root = parse(read('root-review.json'))
    assert sha(read('root-review.json')) == '3b2082f28902e889c00ea117958feb20530d678e92ab23c718944d3d60d1062f'
    assert parent['worker_report_sha256'] == sha(read('worker_report.json')) == root['worker_report_sha256']
    assert root['report_sha256'] == owner['report_sha256'] == sha(read('report.json'))
    assert root['owner_terminal_sha256'] == sha(read('owner-terminal.json'))
    assert owner['owner_session'] == root['owner_session'] == 32501
    assert owner['owner_tool_chunk'] == root['owner_tool_chunk'] == '78b876'
    assert owner['owner_exit_code'] == root['owner_exit_code'] == 0
    assert owner['owner_reaped'] is root['owner_reaped'] is True
    assert root['promotion_authorized'] is False and root['retry_performed'] is False
    assert root['numerical_gates_changed'] is False
    expected = parse(read('source_manifest.json'))
    assert expected == linked['ready_sources'] and len(expected) == 187
    for report in (parent, worker):
        assert report['stage'] == 'validate'
        for name in ('passed', 'source_files_unchanged', 'terminal_scope_completed',
                     'termination_scope_completed'):
            assert report[name] is True
        assert report['source_hashes_start'] == report['source_hashes_end'] == expected
        assert not report.get('error')
        assert report['cpu_affinity'] == [124, 125, 126, 127]
    assert parent['execute'] is True and parent['cleanup_errors'] == []
    for name in ('all_direct_children_reaped', 'all_owned_groups_empty', 'process_cleanup_completed',
                 'worker_reaped', 'worker_group_empty_verified', 'workspace_retained', 'source_archive_verified'):
        assert parent[name] is True
    assert parent['worker_exit_code'] == 0
    for name in ('loader_entered', 'loader_cleanup_completed', 'phase_complete',
                 'all54_authenticated_before_private_read', 'numeric_validation_complete',
                 'external_stores_retained'):
        assert worker[name] is True
    assert worker['forbidden_attempts'] == []
    for field in ('loaded_local_modules_start', 'loaded_local_modules_end'):
        for row in worker[field].values():
            assert expected[row['path']] == row['sha256']
    for report, pin in ((parent, refs['ready_sources']['sha256']),
                        (worker, sha(read('source_manifest.json')))):
        guard = report['fresh_source_guard']
        assert guard['passed'] is guard['fresh'] is True
        assert guard['source_count'] == 187 and guard['manifest_sha256'] == pin
    archive = read('sources.tar.gz')
    assert sha(archive) == parent['source_archive_sha256'] == root['archive_sha256']
    with tarfile.open(fileobj=io.BytesIO(archive), mode='r:gz') as tar:
        members = tar.getmembers()
        assert len(members) == len({m.name for m in members}) == 187
        assert all(m.isfile() and 0 <= m.size <= 8 << 20 for m in members)
        assert sum(m.size for m in members) <= 16 << 20
        bodies = {m.name: tar.extractfile(m).read() for m in members}
    assert {name: sha(raw) for name, raw in bodies.items()} == expected
    assert parse(bodies['source_inventory.json']) == sorted(expected)
    assert parse(bodies['execution_authorization.json']) == {'ready': False}
    assert parse(bodies['producer_sources.json']) == linked['server_sources']
    assert sha(bodies['producer_server_root_review.json']) == refs['server_review']['sha256']
    bound = parse(bodies['producer_bindings.json'])
    assert bound['server_report_sha256'] == refs['server']['sha256']
    assert bound['server_root_review_sha256'] == refs['server_review']['sha256']
    assert parent['producer_terminal_pins'] == {'fixture': bound['fixture_report_sha256'],
                                              'server': bound['server_report_sha256']}
    assert parent['readback'] == worker['readback'] == linked['server']['readback']
    assert parent['request_contract'] == worker['request_contract'] == linked['server']['request_contract']
    assert parent['workspace_identity'] == worker['workspace_identity'] == [66305, 3370369]
    assert parent['readback']['tile_count'] == 54
    plan = parse(bodies['proof_plan.json'])
    assert sha(bodies['proof_plan.json']) == worker['request_contract']['numerical_plan_sha256']
    rows = worker['numeric_checkpoints']
    assert type(rows) is list and len(rows) == len(plan['checkpoints']) == 54

    def metric(row):
        for name in ('passed', 'shape_and_finite', 'error_arithmetic_finite'):
            assert row[name] is True
        for name in ('decoded_minimum', 'decoded_maximum', 'max_abs_error'):
            assert type(row[name]) is float and math.isfinite(row[name])
        assert 0 <= row['max_abs_error'] <= 1e-6
        assert row['decoded_minimum'] <= row['decoded_maximum']

    for index, (row, wanted) in enumerate(zip(rows, plan['checkpoints'])):
        values = dict(index=index, case=wanted['case'], label=wanted['label'], level=wanted['level'],
                      scale=wanted['scale'], slots=65536, absolute_gate=1e-6)
        for name, value in values.items():
            assert type(row[name]) is type(value) and row[name] == value
        metric(row)
    summary = dict(checkpoints=54, physical_slots=54 * 65536, absolute_gate=1e-6, passed=True,
                   maximum_absolute_error=max(row['max_abs_error'] for row in rows))
    assert parent['numeric_summary'] == worker['numeric_summary'] == root['numeric_summary'] == summary
    cases = {case['name']: case for case in plan['cases']}
    assert set(worker['derived_checks']) == set(cases)
    for name, case in cases.items():
        keys = {f'{prefix}{i:02}' for prefix in ('invariant', 'true_residual_vs_original')
                for i in range(1, case['updates'] + 1)}
        keys |= {'final_true_inverse', 'final_true_relative_residual', 'final_true_normalized_quotient'}
        checks = worker['derived_checks'][name]
        assert set(checks) == keys and len(checks) == root['derived_case_counts'][name]
        for row in checks.values():
            metric(row)
    assert sum(map(len, worker['derived_checks'].values())) == root['derived_gates'] == 52
    child, = parent['children']
    assert child['pid'] == child['process_group'] == worker['pid'] == 3859522
    assert child['direct_returncode'] == 0 and child['cleanup_errors'] == child['signals_sent'] == []
    for name in ('direct_child_reaped', 'group_empty_verified', 'cleanup_completed'):
        assert child[name] is True
    assert owner['observed_absent_pids'] == root['observed_absent_pids'] == [3859416, 3859417, 3859522]
    assert owner['observed_absent_groups'] == root['observed_absent_groups'] == [3859416, 3859522]
    grant = linked['launch_grant']
    assert grant['authorized'] is True and grant['phase'] == 'validate'
    assert grant['source_manifest_sha256'] == refs['ready_sources']['sha256']
    assert grant['readiness_review_sha256'] == refs['ready_review']['sha256']
    assert grant['preflight_report_sha256'] == refs['ready_proof']['sha256']
    if current is not None:
        current = Path(current).resolve()
        for name, pin in expected.items():
            assert not (current / name).is_symlink() and sha((current / name).read_bytes()) == pin
    print(json.dumps(dict(passed=True, sources=187, checkpoints=54, derived_gates=52,
                          maximum_absolute_error=summary['maximum_absolute_error'],
                          current_sources_compared=current is not None,
                          runtime_or_payload_reads=False), sort_keys=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--current', type=Path)
    verify(parser.parse_args().current)
