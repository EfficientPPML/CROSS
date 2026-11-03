"""Score organizer-rerun compact-v1 reports (20 points; never the legacy SOTA rubric).

Each model earns five correctness points and up to five speed points, reaching
full speed points at twice the measured reference throughput. A complete run
must fit 300 seconds. Reports are evidence from the protected runner, not a
sandbox or permission to accept contestant-reported timings without rerunning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SRC = Path(__file__).resolve().parent
MODELS = {'student1.in': 'compact_gray_cnn', 'student2.in': 'compact_rgb_cnn'}
VERSIONS = ('jax', 'jaxlib', 'libtpu', 'numpy', 'torch', 'python')


def number(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if value < 0 or (positive and value == 0):
        raise ValueError(f'{name} must be {"positive" if positive else "nonnegative"}')
    return value


def load_report(path):
    def reject(value):
        raise ValueError(f'nonfinite JSON value {value}')
    result = json.loads(Path(path).read_text(), parse_constant=reject)
    if not isinstance(result, dict):
        raise ValueError('report must be an object')
    return result


def fixture_hashes():
    return {name: {suffix: hashlib.sha256((SRC.parent / 'test' / name).with_suffix(suffix).read_bytes()).hexdigest()
                   for suffix in ('.in', '.out')} for name in MODELS}


def validate(report, *, baseline=False):
    if not isinstance(report, dict):
        raise ValueError('report must be an object')
    if report.get('profile') != 'compact-v1' or report.get('gate_version') != 'compact-v1':
        raise ValueError('expected compact-v1 profile and gates')
    if report.get('mode') not in ('baseline', 'candidate') or (baseline and report['mode'] != 'baseline'):
        raise ValueError('calibration requires an actual baseline evaluation')
    if (report.get('passed') is not True or report.get('within_budget') is not True
            or type(report.get('worker_returncode')) is not int or report['worker_returncode'] != 0
            or report.get('error') or report.get('cleanup_error')):
        raise ValueError('evaluation did not pass all gates and cleanup')
    budget = number(report.get('budget_s'), 'budget_s', positive=True)
    elapsed = number(report.get('elapsed_s'), 'elapsed_s', positive=True)
    if budget > 300 or elapsed > budget:
        raise ValueError('complete evaluation exceeds the 300-second budget')
    if report.get('schedule') != list(MODELS) or set(report.get('cases', {})) != set(MODELS):
        raise ValueError('both compact models must be evaluated exactly once')
    if report.get('fixture_sha256') != fixture_hashes():
        raise ValueError('fixture identity differs from the frozen compact cases')
    model_hash = hashlib.sha256((SRC / 'compact_models.py').read_bytes()).hexdigest()
    if report.get('model_source_sha256') != model_hash:
        raise ValueError('model definition differs from the frozen compact models')
    versions = report.get('versions')
    if not isinstance(versions, dict) or any(not isinstance(versions.get(k), str) or not versions[k] for k in VERSIONS):
        raise ValueError('software versions are missing')
    devices, kinds = report.get('devices'), report.get('device_kinds')
    if (not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], str)
            or not devices[0].startswith('TPU') or not isinstance(kinds, list) or len(kinds) != 1
            or not isinstance(kinds[0], str) or 'TPU' not in kinds[0]):
        raise ValueError('compact evaluation requires one identified TPU device')
    sources = report.get('source_sha256')
    if not isinstance(sources, dict) or not sources:
        raise ValueError('source identity is missing')
    medians = {}
    for name, model in MODELS.items():
        case = report['cases'][name]
        if not isinstance(case, dict) or case.get('passed') is not True or case.get('model') != model:
            raise ValueError(f'{name}: missing verified model result')
        if case.get('case') != name or case.get('mode') != 'HE-synthetic-smoke':
            raise ValueError(f'{name}: encrypted case identity missing')
        if type(case.get('batch')) is not int or case['batch'] != 1 or case.get('cold_start_included') is not False:
            raise ValueError(f'{name}: expected warmed batch-one timing')
        if case.get('device') != devices[0]:
            raise ValueError(f'{name}: device identity differs from the evaluation')
        for field, count in (('server_s', 3), ('end_to_end_s', 3), ('warmup_s', 1), ('warmup_server_s', 1)):
            samples = case.get(field)
            if not isinstance(samples, list) or len(samples) != count:
                raise ValueError(f'{name}: {field} needs {count} samples')
            for sample in samples:
                number(sample, f'{name}.{field}', positive=True)
        if any(server > total for server, total in zip(case['server_s'], case['end_to_end_s'])):
            raise ValueError(f'{name}: server timing exceeds end-to-end timing')
        number(case.get('setup_s'), f'{name}.setup_s')
        number(case.get('max_abs_error'), f'{name}.max_abs_error')
        number(case.get('nrmse'), f'{name}.nrmse')
        if number(case.get('top1_agreement'), f'{name}.top1_agreement') != 1:
            raise ValueError(f'{name}: argmax gate failed')
        rings = case.get('execution_rings')
        if not isinstance(rings, list) or not rings:
            raise ValueError(f'{name}: no actual-ring verification evidence')
        for ring in rings:
            if (not isinstance(ring, dict) or type(ring.get('security_bits')) is not int
                    or ring['security_bits'] < 128 or type(ring.get('degree')) is not int
                    or ring['degree'] < 1024):
                raise ValueError(f'{name}: missing 128-bit actual-ring verification')
            number(ring.get('log2_QP'), f'{name}.log2_QP', positive=True)
        medians[name] = statistics.median(case['server_s'])
    return medians


def score_reports(result, baseline):
    reference = validate(baseline, baseline=True)
    measured = validate(result)
    for key in ('versions', 'device_kinds'):
        if result[key] != baseline[key]:
            raise ValueError(f'{key} differs from calibration; recalibrate on judging hardware')
    cases = {}
    for name in MODELS:
        # Subtract logs rather than divide first: finite positive samples can
        # still overflow a floating-point ratio.
        log_speedup = math.log2(reference[name]) - math.log2(measured[name])
        speed_points = 5 * min(max(log_speedup, 0), 1)
        cases[name] = {'correctness_points': 5., 'speed_points': round(speed_points, 6),
                       'reference_median_server_s': reference[name],
                       'median_server_s': measured[name]}
    return {'passed': True, 'profile': 'compact-v1', 'cases': cases,
            'correctness_points': 10.,
            'speed_points': round(sum(c['speed_points'] for c in cases.values()), 6),
            'points': round(10 + sum(c['speed_points'] for c in cases.values()), 6),
            'maximum_points': 20,
            'note': 'Organizer arithmetic over protected reruns; this is not the legacy SOTA bonus.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    args = parser.parse_args()
    try:
        result = score_reports(load_report(args.result), load_report(args.baseline))
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        print(json.dumps({'passed': False, 'points': 0., 'error': str(error)}, allow_nan=False))
        return 1
    print(json.dumps(result, indent=2, allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
