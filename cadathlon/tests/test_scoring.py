"""Scoring must reject corrupt runs instead of publishing misleading scores."""
import copy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
spec = importlib.util.spec_from_file_location('p1_score_test', SUITE / 'score.py')
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)


def record():
    return {'Timing': {'Total': '10s', 'Key Generation': '1s',
                       'Encrypted model preprocessing': '2s',
                       'Encrypted computation': '0.5s'},
            'Quality': {'Encrypted model quality': {'accuracy': 1., 'total_samples': 100},
                        'Harness model quality': {'accuracy': 1.}}}


@pytest.mark.parametrize('bad', ['NaNs', '-1s', '0s', 'infs'])
def test_bad_repetition_cannot_hide_in_median(bad):
    records = [record() for _ in range(3)]
    records[0]['Timing']['Encrypted computation'] = bad
    with pytest.raises(ValueError, match='timing'):
        score.summarize(records, 100)


def test_short_repetition_cannot_hide_in_median():
    records = [record() for _ in range(3)]
    records[0]['Quality']['Encrypted model quality']['total_samples'] = 1
    with pytest.raises(ValueError, match='incomplete'):
        score.summarize(records, 100)


def test_label_never_falls_back_to_different_run(tmp_path, monkeypatch):
    monkeypatch.setattr(score, 'SUITE_DIR', tmp_path)
    other = tmp_path / 'results/C3'
    other.mkdir(parents=True)
    (other / 'results-1.json').write_text(json.dumps(record()))
    with pytest.raises(FileNotFoundError, match='candidate'):
        score.read_case(tmp_path, 1, 1, 'C3', 'candidate')


def test_archive_hardware_must_match_case(tmp_path, monkeypatch):
    monkeypatch.setattr(score, 'SUITE_DIR', tmp_path)
    folder = tmp_path / 'results/C3_candidate'
    folder.mkdir(parents=True)
    (folder / 'results-1.json').write_text(json.dumps(record()))
    meta = {**score.CONFIG['cases'][2], 'devices': 8}
    (folder / 'case.json').write_text(json.dumps(meta))
    with pytest.raises(ValueError, match='devices'):
        score.read_case(tmp_path, 1, 1, 'C3', 'candidate')


def test_failed_baseline_does_not_overwrite_manifest(tmp_path, monkeypatch):
    target = tmp_path / 'baseline.json'
    target.write_text('existing calibration')
    monkeypatch.setattr(sys, 'argv', ['score', '--cases', 'C3', '--emit-baseline', '--baseline', str(target)])
    monkeypatch.setattr(score, 'read_case', lambda *a: ([record()], {'verify': {'status': 'FAIL'}}, tmp_path))
    with pytest.raises(SystemExit, match='refusing'):
        score.main()
    assert target.read_text() == 'existing calibration'


def test_unknown_case_is_an_error(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['score', '--cases', 'TYPO'])
    with pytest.raises(SystemExit) as exc:
        score.main()
    assert exc.value.code == 2


def test_truncated_ground_truth_cannot_pass(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('p1_archive_test', SUITE / 'archive.py')
    archive = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(archive)
    truth = tmp_path / 'datasets/small/intermediate/test_labels.txt'
    truth.parent.mkdir(parents=True)
    truth.write_text('1\n' * 99)
    encrypted = tmp_path / 'io/small/encrypted_model_predictions.txt'
    encrypted.parent.mkdir(parents=True)
    encrypted.write_text('1\n' * 100)
    monkeypatch.setattr(archive, 'own_cleartext_predictions', lambda *a: (['1'] * 100, None))
    verdict = archive.freeze_verdict(tmp_path, {'instance': 1, 'batch': 100})
    assert verdict['status'] == 'FAIL'
    assert not verdict['complete']
    assert verdict['labels_written'] == 99


@pytest.mark.parametrize('field,value', [('Server Reported', []), ('Quality', []), ('Timing', [])])
def test_malformed_measurement_objects_are_rejected(field, value):
    sample = record()
    sample[field] = value
    with pytest.raises(ValueError, match='JSON object'):
        score.summarize([sample], 100)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), '-1s', 'NaNs', True])
def test_optional_server_timing_is_validated(bad):
    sample = record()
    sample['Server Reported'] = {'server_execute_s': bad}
    with pytest.raises(ValueError):
        score.summarize([sample], 100)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1, True, []])
def test_optional_harness_accuracy_is_validated(bad):
    sample = record()
    sample['Quality']['Harness model quality']['accuracy'] = bad
    with pytest.raises(ValueError):
        score.summarize([sample], 100)


@pytest.mark.parametrize('bad', [[], {'verify': []}, {'verify': {'own_cleartext': []}}])
def test_malformed_archive_objects_are_rejected(bad):
    with pytest.raises(ValueError, match='JSON object'):
        score.summarize([record()], 100, bad)


@pytest.mark.parametrize('bad', [[], 'invalid', None])
def test_malformed_case_json_is_rejected(tmp_path, monkeypatch, bad):
    monkeypatch.setattr(score, 'SUITE_DIR', tmp_path)
    folder = tmp_path / 'results/C3_candidate'
    folder.mkdir(parents=True)
    (folder / 'results-1.json').write_text(json.dumps(record()))
    (folder / 'case.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='JSON object'):
        score.read_case(tmp_path, 1, 1, 'C3', 'candidate')


@pytest.mark.parametrize('bad', [[], {'cases': []}, {'cases': {'C3': []}}])
def test_malformed_baseline_reports_error_without_crashing(tmp_path, monkeypatch, capsys, bad):
    if isinstance(bad, dict):
        bad.update({'suite_commit': score.CONFIG['suite']['commit'],
                    'seed': score.CONFIG['determinism']['seed'],
                    'runs_per_case': score.CONFIG['determinism']['num_runs'],
                    'statistic': score.CONFIG['determinism']['statistic']})
    baseline = tmp_path / 'baseline.json'
    baseline.write_text(json.dumps(bad))
    monkeypatch.setattr(sys, 'argv', ['score', '--cases', 'C3', '--baseline', str(baseline)])
    monkeypatch.setattr(score, 'read_case', lambda *a: ([record()], None, tmp_path))
    score.main()
    report = json.loads(capsys.readouterr().out)
    assert 'JSON object' in report['baseline_error']
    assert report['speed_points'] == 0


def test_different_ring_profiles_never_share_speed_baseline(monkeypatch, capsys):
    meta = json.loads((SUITE / 'results/C3_baseline/case.json').read_text())
    meta['baseline_profile'] = 'student'
    data = record()
    data['Quality']['Encrypted model quality']['accuracy'] = .96
    monkeypatch.setattr(sys, 'argv', ['score', '--cases', 'C3'])
    monkeypatch.setattr(score, 'read_case', lambda *a: ([data], meta, SUITE))
    score.main()
    result = json.loads(capsys.readouterr().out)
    assert result['cases']['C3']['passed'] == 0
    assert result['cases']['C3']['throughput_ratio'] is None
    assert result['speed_points'] == 0


def test_old_student_calibration_needs_complete_evaluation_evidence(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(score, 'SUITE_DIR', tmp_path)
    meta = json.loads((SUITE / 'results/C3_baseline/case.json').read_text())
    meta['baseline_profile'] = 'student'
    calibration = {
        'suite_commit': score.CONFIG['suite']['commit'],
        'seed': score.CONFIG['determinism']['seed'],
        'runs_per_case': score.CONFIG['determinism']['num_runs'],
        'statistic': score.CONFIG['determinism']['statistic'],
        'cases': {cid: {'verified': 'PASS', 'images': 100, 'baseline_profile': 'student',
                        'throughput_images_per_s': 200.} for cid in ('C3', 'C4')},
    }
    (tmp_path / 'baseline_manifest_student.json').write_text(json.dumps(calibration))
    monkeypatch.setattr(score, 'read_case', lambda *a: ([record()], meta, tmp_path))
    monkeypatch.setattr(sys, 'argv', ['score', '--evaluation-profile', 'student'])
    score.main()
    report = json.loads(capsys.readouterr().out)
    assert 'student-v2' in report['baseline_error']
    assert report['evaluation_error']
    assert report['execution_points'] == report['speed_points'] == 0


@pytest.mark.parametrize('subset,profile', [(True, 'student'), (False, 'legacy')])
def test_incomplete_or_wrong_profile_calibration_preserves_manifest(monkeypatch, tmp_path, subset, profile):
    meta = json.loads((SUITE / 'results/C3_baseline/case.json').read_text())
    meta['baseline_profile'] = profile
    monkeypatch.setattr(score, 'read_case', lambda *a: ([record()], meta, tmp_path))
    manifest = tmp_path / 'baseline.json'
    manifest.write_text('previous calibration')
    args = ['score', '--evaluation-profile', 'student', '--emit-baseline', '--baseline', str(manifest)]
    if subset:
        args += ['--cases', 'C3']
    monkeypatch.setattr(sys, 'argv', args)
    with pytest.raises(SystemExit, match='refusing'):
        score.main()
    assert manifest.read_text() == 'previous calibration'
