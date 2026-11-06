"""Student scoring is bound to one complete run and its frozen public assets."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
spec = importlib.util.spec_from_file_location('student_score_test', SUITE/'score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


@pytest.fixture
def measured(tmp_path, monkeypatch):
    suite = tmp_path/'problem/src/suite'
    suite.mkdir(parents=True)
    monkeypatch.setattr(score, 'SUITE_DIR', suite)
    config = suite/'suite_config.json'
    config.write_text(json.dumps(score.CONFIG))
    weights = suite.parent.parent/'weights'
    weights.mkdir()
    (weights/'mlp_784_50_10.pth').write_bytes(b'frozen-public-weights')
    (weights/'MANIFEST.json').write_text(json.dumps({'files': {'mlp_784_50_10.pth': {
        'sha256': hashlib.sha256(b'frozen-public-weights').hexdigest(), 'bytes': 21}}}))
    model = score.model_identity(suite.parent.parent)
    reports = {}
    for mode, run_id in (('baseline', 'a'*32), ('candidate', 'b'*32)):
        label = 'student-'+mode
        report = dict(profile='student', gate_version=score.STUDENT_GATE_VERSION, mode=mode,
            label=label, suite_commit=score.CONFIG['suite']['commit'], seed=score.CONFIG['determinism']['seed'],
            runs_per_case=3, require_tpu=True, passed=True, cleanup_passed=True, within_budget=True,
            elapsed_s=200., budget_s=300., evaluation_id=run_id, model_identity=model, cases={}, archive_sha256={})
        report['versions'] = {name: 'test-version' for name in ('jax', 'jaxlib', 'libtpu', 'numpy', 'torch', 'python')}
        hashes = {'test.py': 'f'*64, 'problem/suite/suite_config.json': score.sha256(config)}
        report.update(source_sha256=hashes, source_sha256_after=hashes.copy(), source_files_unchanged=True)
        for cid in ('C3', 'C4'):
            case = next(c for c in score.CONFIG['cases'] if c['id'] == cid)
            folder = suite/'results'/f'{cid}_{label}'
            folder.mkdir(parents=True)
            meta = dict(case, baseline_profile='student', label=label, evaluation_id=run_id,
                suite_commit=score.CONFIG['suite']['commit'], seed=score.CONFIG['determinism']['seed'],
                identity={'model': model, 'pixels_sha256': '1'*64, 'labels_sha256': '2'*64},
                execution={'batch': 100, 'devices': [f'TPU_{i}' for i in range(case['devices'])],
                    'device_kinds': ['TPU test']*case['devices'], 'execution_rings': [{'security_bits': 128, 'log2_QP': 212.45}]},
                verify={'status': 'PASS', 'complete': True, 'correct': 100,
                    'own_cleartext': {'status': 'ok', 'compared': 100, 'top1_agreement': 1., 'cleartext_correct': 100}})
            record = {'Timing': {'Total': '20s', 'Key Generation': '1s', 'Encrypted model preprocessing': '2s',
                                 'Encrypted computation': '0.5s'},
                      'Quality': {'Encrypted model quality': {'accuracy': 1., 'total_samples': 100},
                                  'Harness model quality': {'accuracy': 1.}}}
            (folder/'case.json').write_text(json.dumps(meta))
            for i in range(1, 4):
                (folder/f'results-{i}.json').write_text(json.dumps(record))
            report['cases'][cid] = dict(score.summarize([record]*3, 100, meta), passed=True)
            report['archive_sha256'][cid] = score.archive_hashes(folder, 3)
        path = score.evaluation_path(label)
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(report))
        if mode == 'baseline':
            snapshot = score.calibration_report_path(label, run_id)
            snapshot.parent.mkdir(parents=True)
            snapshot.write_text(json.dumps(report))
            for cid in ('C3', 'C4'):
                shutil.copytree(suite/'results'/f'{cid}_{label}', snapshot.parent/'archives'/cid)
            (suite/'baseline_manifest_student.json').write_text(json.dumps(score.calibration_payload(report['cases'], 'student', report)))
        reports[mode] = report
    return suite, reports


def run_score(monkeypatch, capsys, extra=()):
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'student', *extra])
    code = score.main()
    return code, json.loads(capsys.readouterr().out)


def test_complete_student_schedule_and_default_calibration(measured, monkeypatch, capsys):
    code, report = run_score(monkeypatch, capsys)
    assert code == 0 and report['passed']
    assert report['execution_points'] == 30
    assert report['speed_points'] == 0
    assert set(report['cases']) == {'C3', 'C4'}


@pytest.mark.parametrize('failure', ['missing', 'failed', 'cleanup', 'over_budget', 'inflated_budget', 'legacy', 'incomplete', 'old_gate'])
def test_incomplete_or_wrong_student_whole_report_scores_nothing(measured, monkeypatch, capsys, failure):
    suite, reports = measured
    path = score.evaluation_path('student-candidate')
    report = reports['candidate']
    if failure == 'missing':
        path.unlink()
    else:
        if failure == 'failed': report['passed'] = False
        if failure == 'cleanup': report['cleanup_passed'] = False
        if failure == 'over_budget': report['elapsed_s'] = 301
        if failure == 'inflated_budget': report['budget_s'] = 400
        if failure == 'legacy': report['profile'] = 'legacy'
        if failure == 'incomplete': del report['cases']['C4']
        if failure == 'old_gate': report['gate_version'] = 'student-v1'
        path.write_text(json.dumps(report))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and not result['passed']
    assert result['evaluation_error']
    assert result['subtotal_without_evidence'] == 0


def test_changed_archives_cannot_reuse_successful_whole_report(measured, monkeypatch, capsys):
    suite, _ = measured
    path = suite/'results/C3_student-candidate/results-1.json'
    record = json.loads(path.read_text())
    record['Timing']['Encrypted computation'] = '.25s'
    path.write_text(json.dumps(record))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and 'changed' in result['evaluation_error']
    assert result['subtotal_without_evidence'] == 0


def test_legacy_archive_profile_rejected_even_with_matching_hashes(measured, monkeypatch, capsys):
    suite, reports = measured
    folder = suite/'results/C3_student-candidate'
    path = folder/'case.json'
    meta = json.loads(path.read_text())
    meta['baseline_profile'] = 'legacy'
    path.write_text(json.dumps(meta))
    reports['candidate']['archive_sha256']['C3'] = score.archive_hashes(folder, 3)
    score.evaluation_path('student-candidate').write_text(json.dumps(reports['candidate']))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and result['subtotal_without_evidence'] == 0


def test_input_identity_cannot_mix_with_calibration(measured, monkeypatch, capsys):
    suite, reports = measured
    folder = suite/'results/C3_student-candidate'
    path = folder/'case.json'
    meta = json.loads(path.read_text())
    meta['identity']['pixels_sha256'] = '3'*64
    path.write_text(json.dumps(meta))
    reports['candidate']['cases']['C3']['identity'] = meta['identity']
    reports['candidate']['archive_sha256']['C3'] = score.archive_hashes(folder, 3)
    score.evaluation_path('student-candidate').write_text(json.dumps(reports['candidate']))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1
    assert result['cases']['C3']['points'] == 0
    assert 'input identity' in ' '.join(result['cases']['C3']['reasons'])


def test_model_bytes_must_match_authoritative_manifest(measured):
    suite, _ = measured
    (suite.parent.parent/'weights/mlp_784_50_10.pth').write_bytes(b'other-public-weights!')
    with pytest.raises(ValueError, match='MANIFEST'):
        score.model_identity(suite.parent.parent)


def test_failed_later_baseline_attempt_does_not_erase_published_calibration(measured, monkeypatch, capsys):
    suite, _ = measured
    score.evaluation_path('student-baseline').write_text(json.dumps({'passed': False}))
    shutil.rmtree(suite/'results/C3_student-baseline')
    code, result = run_score(monkeypatch, capsys)
    assert code == 0 and result['passed']


def test_historical_manifest_is_not_relabelled_as_current(measured, monkeypatch, capsys):
    suite, _ = measured
    target = suite/'baseline_manifest_student.json'
    payload = json.loads(target.read_text())
    del payload['evaluation_id']
    del payload['gate_version']
    target.write_text(json.dumps(payload))
    before = target.read_bytes()
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and 'newly measured' in result['baseline_error']
    assert result['speed_points'] == 0 and target.read_bytes() == before


@pytest.mark.parametrize('field', ['versions', 'source_files_unchanged', 'device_kinds'])
def test_software_hardware_and_stable_sources_bind_the_evaluation(measured, monkeypatch, capsys, field):
    suite, reports = measured
    report = reports['candidate']
    if field == 'versions':
        report['versions']['jax'] = 'different-version'
    elif field == 'source_files_unchanged':
        report[field] = False
    else:
        folder = suite/'results/C3_student-candidate'
        path = folder/'case.json'
        meta = json.loads(path.read_text())
        meta['execution']['device_kinds'] = ['TPU other']
        path.write_text(json.dumps(meta))
        report['cases']['C3']['execution'] = meta['execution']
        report['archive_sha256']['C3'] = score.archive_hashes(folder, 3)
    score.evaluation_path('student-candidate').write_text(json.dumps(report))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and not result['passed']
    assert result['cases']['C3']['points'] == 0


def test_changed_label_cannot_borrow_another_attempt(measured, monkeypatch, capsys):
    _, _ = measured
    code, result = run_score(monkeypatch, capsys, ('--evaluation-report', str(score.evaluation_path('student-baseline'))))
    assert code == 1
    assert 'label' in result['evaluation_error']
    assert result['subtotal_without_evidence'] == 0


def test_student_manifest_cannot_bypass_transactional_evaluator(measured, monkeypatch):
    suite, _ = measured
    path = suite/'baseline_manifest_student.json'
    previous = path.read_bytes()
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'student', '--emit-baseline', '--baseline', str(path)])
    with pytest.raises(SystemExit, match='complete evaluator transaction'):
        score.main()
    assert path.read_bytes() == previous


def test_gate_configuration_cannot_change_after_measurement(measured, monkeypatch, capsys):
    suite, _ = measured
    config = suite/'suite_config.json'
    contents = json.loads(config.read_text())
    contents['gates']['max_batch_s'] = 10000
    config.write_text(json.dumps(contents))
    code, result = run_score(monkeypatch, capsys)
    assert code == 1 and result['subtotal_without_evidence'] == 0
    assert 'schedule/gates' in result['evaluation_error']


def test_default_make_pipeline_uses_one_profile():
    output = subprocess.check_output(['make', '-n', '-C', str(SUITE.parent),
        'suite-evaluate-baseline', 'suite-evaluate', 'suite-score'], text=True)
    assert output.count('--profile student') == 2
    assert '--label student-candidate' in output
    assert '--evaluation-profile student' in output
    legacy = subprocess.check_output(['make', '-n', '-C', str(SUITE.parent),
        'suite-evaluate-baseline', 'suite-evaluate', 'suite-score', 'P1_BASELINE_PROFILE=legacy'], text=True)
    assert legacy.count('--profile legacy') == 2
    assert '--evaluation-profile legacy' in legacy
