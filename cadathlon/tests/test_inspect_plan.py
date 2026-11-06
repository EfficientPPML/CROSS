"""Plan inspection reflects the selected reference's secure policy."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('student,degree,digits', [(False, 16384, 2), (True, 8192, 4)])
def test_plan_matches_selected_baseline(tmp_path, student, degree, digits):
    root = Path(__file__).resolve().parents[2]
    script = root / 'cadathlon/non_agent_pset/p1_benchmark/src/inspect_plan.py'
    command = [sys.executable, str(script)] + (['--student-baseline'] if student else [])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True,
                            env=dict(os.environ, JAX_PLATFORMS='cpu', CROSS_ROOT=str(root)), check=True)
    plans = [json.loads(line) for line in result.stdout.splitlines()]
    assert [plan['case'] for plan in plans] == ['test1.in', 'test6.in']
    assert all(plan['degree'] == degree and plan['dnum'] == digits for plan in plans)
    assert all(plan['security_bits'] == 128 for plan in plans)
    assert all(plan['profile'] == ('student' if student else 'legacy') for plan in plans)
