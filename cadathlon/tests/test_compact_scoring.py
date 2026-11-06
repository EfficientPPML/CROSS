"""The compact competition has a measured, bounded and reproducible rubric."""
import copy
import hashlib
import importlib.util
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / 'non_agent_pset/p3_cnn/src'
spec = importlib.util.spec_from_file_location('compact_score_test', SRC / 'score_compact.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def report(mode='baseline'):
    result = {'profile': 'compact-v1', 'gate_version': 'compact-v1', 'mode': mode,
              'passed': True, 'within_budget': True, 'worker_returncode': 0,
              'budget_s': 300., 'elapsed_s': 150., 'schedule': list(score.MODELS),
              'fixture_sha256': score.fixture_hashes(),
              'model_source_sha256': hashlib.sha256((SRC / 'compact_models.py').read_bytes()).hexdigest(),
              'versions': {key: 'tested' for key in score.VERSIONS},
              'devices': ['TPU_0'], 'device_kinds': ['TPU v6e'], 'source_sha256': {'runner.py': 'hash'},
              'cases': {}}
    for name, model in score.MODELS.items():
        result['cases'][name] = {'case': name, 'model': model, 'passed': True,
            'mode': 'HE-synthetic-smoke', 'batch': 1, 'device': 'TPU_0', 'cold_start_included': False,
            'server_s': [1., 1.1, .9], 'end_to_end_s': [2., 2.1, 1.9],
            'warmup_s': [5.], 'warmup_server_s': [4.], 'setup_s': 50.,
            'max_abs_error': 1e-7, 'nrmse': 1e-6, 'top1_agreement': 1.,
            'execution_rings': [{'degree': 8192, 'security_bits': 128, 'log2_QP': 212.456}]}
    return result


def test_reference_earns_correctness_only():
    r = report()
    result = score.score_reports(r, r)
    assert result['points'] == 10
    assert result['speed_points'] == 0


@pytest.mark.parametrize('factor,expected', [(2., 10.), (.5, 20.), (.25, 20.)])
def test_speed_points(factor, expected):
    candidate = report('candidate')
    for case in candidate['cases'].values():
        case['server_s'] = [v * factor for v in case['server_s']]
        case['end_to_end_s'] = [v * 4 for v in case['server_s']]
    assert score.score_reports(candidate, report())['points'] == expected


def test_partial_speed_credit_uses_server_medians():
    candidate = report('candidate')
    for case in candidate['cases'].values():
        # The median is sqrt(1/2); the outlier must not enter the score.
        # Client timing stays identical to the reference: server execution
        # is the published compact rubric's timing boundary.
        case['server_s'] = [.5, 2 ** -.5, 1.5]
    result = score.score_reports(candidate, report())
    assert result['points'] == 15
    assert all(case['speed_points'] == 2.5 for case in result['cases'].values())


def test_speed_credit_is_clamped_separately_for_each_model():
    candidate = report('candidate')
    candidate['cases']['student1.in']['server_s'] = [.4, .5, .6]
    candidate['cases']['student2.in']['server_s'] = [1.5, 2., 2.5]
    candidate['cases']['student2.in']['end_to_end_s'] = [3., 3., 3.]
    result = score.score_reports(candidate, report())
    assert result['points'] == 15
    assert result['cases']['student1.in']['speed_points'] == 5
    assert result['cases']['student2.in']['speed_points'] == 0


@pytest.mark.parametrize('mutate,match', [
    (lambda r: r.update(elapsed_s=301.), 'budget'),
    (lambda r: r.update(error='cleanup failed'), 'cleanup'),
    (lambda r: r.update(worker_returncode=1), 'cleanup'),
    (lambda r: r.update(worker_returncode=False), 'cleanup'),
    (lambda r: r.update(profile='legacy'), 'profile'),
    (lambda r: r['cases'].pop('student2.in'), 'both'),
    (lambda r: r.update(devices=['CPU_0']), 'TPU'),
    (lambda r: r.update(fixture_sha256={}), 'fixture'),
    (lambda r: r.update(model_source_sha256='different'), 'model'),
    (lambda r: r.update(source_sha256={}), 'source'),
    (lambda r: r['cases']['student1.in'].update(server_s=[1., float('nan'), 1.]), 'finite'),
    (lambda r: r['cases']['student1.in'].update(server_s=[1., 0., 1.]), 'positive'),
    (lambda r: r['cases']['student1.in'].update(server_s=[1.]), 'samples'),
    (lambda r: r['cases']['student1.in'].update(top1_agreement=.5), 'argmax'),
    (lambda r: r['cases']['student1.in'].update(execution_rings=[]), 'ring'),
    (lambda r: r['cases']['student1.in'].update(cold_start_included=True), 'warmed'),
])
def test_bad_evaluations_cannot_score(mutate, match):
    r = report('candidate')
    mutate(r)
    with pytest.raises(ValueError, match=match):
        score.score_reports(r, report())


def test_unmeasured_candidate_is_not_a_baseline():
    with pytest.raises(ValueError, match='calibration'):
        score.score_reports(report('candidate'), report('candidate'))


def test_different_environment_needs_calibration():
    r = report('candidate')
    r['versions']['jax'] = 'different'
    with pytest.raises(ValueError, match='versions'):
        score.score_reports(r, report())


def test_nonfinite_json_is_rejected_even_in_unused_fields(tmp_path):
    p = tmp_path / 'result.json'
    p.write_text('{"extra": NaN}')
    with pytest.raises(ValueError, match='nonfinite'):
        score.load_report(p)
