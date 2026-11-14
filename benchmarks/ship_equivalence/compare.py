"""Reject incomplete or unmatched four-way reports before comparing latency."""
import argparse
import json
import math
from pathlib import Path
import statistics

VARIANTS = {'cross-native', 'cross-ship', 'jaxite-patched', 'jaxite-optimized'}
COMMON = ('degree', 'q', 'p', 'scale', 'dnum', 'seed', 'magnitude', 'error_gate')


def compare(reports, require_timing=True):
    if len(reports) != 4 or {r['variant'] for r in reports} != VARIANTS:
        raise ValueError('Exactly one report for each of the four variants is required')
    reference = reports[0]
    ship_reference = next(r for r in reports if r['variant'] == 'cross-ship')
    rows = []
    for report in reports:
        variant = report['variant']
        if not report['passed'] or report.get('stage') != 'complete':
            raise ValueError(f'{variant}: correctness run did not complete')
        for field in COMMON:
            if report['manifest'][field] != reference['manifest'][field]:
                raise ValueError(f'{variant}: mismatched {field}')
        if variant != 'cross-native':
            for field in ('weight', 'theta', 'gamma'):
                if report['manifest'][field] != ship_reference['manifest'][field]:
                    raise ValueError(f'{variant}: mismatched SHIP {field}')
        for field in ('backend', 'device_kind', 'jax_version', 'numpy_version',
                      'scope', 'output_q', 'output_scale', 'output_level', 'output_nsd',
                      'slots', 'retained_multiplication_levels'):
            if report[field] != reference[field]:
                raise ValueError(f'{variant}: mismatched {field}')
        if require_timing:
            for field in ('jaxlib_version', 'python_version', 'host', 'visible_chips', 'harness_sha256',
                          'libtpu_version', 'xla_flags', 'stack_limit_bytes', 'persistent_cache_enabled'):
                if field not in report or report[field] != reference[field]:
                    raise ValueError(f'{variant}: missing or mismatched {field}')
        if require_timing:
            for field in ('commit', 'patch_sha256'):
                if report['cross_source'][field] != reference['cross_source'][field]:
                    raise ValueError(f'{variant}: mismatched CROSS source {field}')
        if require_timing and report['backend'] != 'tpu':
            raise ValueError('CPU correctness runs cannot supply comparative latency')
        for field in ('errors', 'post_square_errors'):
            errors = report[field]
            if set(errors) != {'ramp', 'random'} or any(
                not math.isfinite(v) or v > report['manifest']['error_gate']
                for v in errors.values()
            ):
                raise ValueError(f'{variant}: invalid {field}')
        row = dict(variant=variant,
                   bootstrap_max_error=max(report['errors'].values()),
                   square_max_error=max(report['post_square_errors'].values()))
        if not require_timing:
            rows.append(row)
            continue
        wall = report['warm_wall_ms']
        profiles = report.get('device_profiles', [])
        active = [p['device_active_ms'] for p in profiles]
        if len(wall) < 3 or len(active) != len(wall) or any(
            not math.isfinite(v) or v <= 0 for v in wall + active
        ):
            raise ValueError(f'{variant}: need at least three wall and device samples')
        row.update(warm_wall_median_ms=statistics.median(wall),
                   device_active_median_ms=statistics.median(active))
        rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs=4)
    parser.add_argument('--correctness-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(compare([json.loads(path.read_text()) for path in args.reports],
                             require_timing=not args.correctness_only), indent=2))
