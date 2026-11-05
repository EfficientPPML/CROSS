"""Actual upstream WISE schedules against independently evaluated neural nets."""
import copy
import json
import os
from pathlib import Path
import sys
import numpy as np
import pytest

P5 = Path(__file__).resolve().parents[1] / "agent_pset/p5_wise"
sys.path.insert(0, str(P5 / "src"))
from schedule import build_schedule, execute, NumpyArithmetic, pack_inputs
from workloads import input_values, reference


@pytest.mark.skipif(not os.environ.get("WISE_ROOT"), reason="requires pinned upstream WISE checkout")
@pytest.mark.parametrize("case_number,slots", [(1, 8), (1, 64), (2, 64), (3, 1024), (4, 16), (5, 128), (6, 1024)])
def test_real_wise_planner_matches_independent_network(case_number, slots):
    case = json.loads((P5 / "test" / f"test{case_number}.in").read_text())
    schedule = build_schedule(case, slots)
    before = schedule.identity()
    for seed in [case["input_seed"], 7919]:
        x = input_values(case, seed)
        expected = reference(case, x)
        for strategy in ["direct", "bsgs"]:
            output = execute(schedule, pack_inputs(schedule, x), NumpyArithmetic(), strategy=strategy)
            actual = np.concatenate(output)[:schedule.output_size].reshape(expected.shape)
            np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)
    assert before == schedule.identity()
    altered = copy.deepcopy(schedule)
    altered.layers[0][1].mat[0][0].diags[0].data[0] += .25
    assert altered.identity() != before


def test_invalid_slot_count_rejected_before_planning():
    for slots in [0, 1, 3, True]:
        with pytest.raises(ValueError, match="power of two"):
            build_schedule({}, slots)
