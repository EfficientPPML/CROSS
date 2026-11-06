"""Client key planning and server selection must agree on the requested ring."""
from pathlib import Path
import sys

import pytest

SUITE = Path(__file__).resolve().parents[1] / 'non_agent_pset/p1_benchmark/src/suite'
sys.path.insert(0, str(SUITE))
import stage_lib


def test_profile_typo_fails_before_setup(monkeypatch):
    monkeypatch.setenv('P1_BASELINE_PROFILE', 'studnet')
    with pytest.raises(ValueError, match='P1_BASELINE_PROFILE'):
        stage_lib.baseline_profile()


def test_default_profile_preserves_legacy(monkeypatch):
    monkeypatch.delenv('P1_BASELINE_PROFILE', raising=False)
    assert stage_lib.baseline_profile() == 'legacy'


def test_student_server_and_key_planner_share_policy(monkeypatch):
    monkeypatch.setenv('P1_BASELINE_PROFILE', 'student')
    monkeypatch.setenv('P3_BASELINE', '1')
    stage_lib.cross_root()
    module = stage_lib.implementation()
    from support import load_module
    reference = load_module(SUITE.parent / "reference.py")
    packed = stage_lib.pack_program(reference.build_model(50).double().eval())
    assert packed.ring_config.degree == 8192
    assert packed.ring_config.dnum == 4


def test_other_problem_import_cannot_change_p1_ring(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setenv('P1_BASELINE_PROFILE', 'student')
    monkeypatch.setitem(sys.modules, 'baseline_student', SimpleNamespace())
    monkeypatch.setitem(sys.modules, 'reference', SimpleNamespace())
    model = stage_lib.load_model()
    packed = stage_lib.pack_program(model)
    assert packed.ring_config.degree == 8192
    assert packed.ring_config.dnum == 4
