"""Planning metadata identifies the actual selected polynomial schedule."""
from pathlib import Path
import importlib.util
from types import SimpleNamespace


def test_runtime_statistics_match_selected_depth_and_public_polynomial():
    path = Path(__file__).parents[1] / 'validation/plan_transformer_polynomials.py'
    spec = importlib.util.spec_from_file_location('runtime_polynomial_statistics', path)
    planning = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(planning)
    from encrypted_polynomial import EncryptedPolynomialEvaluator
    from encrypted_transformer import public_toy_approximations
    reports = planning.runtime_approximation_statistics()
    for problem, statistics in reports.items():
        for name, approximation in public_toy_approximations(problem).items():
            row = statistics[name]
            actual_depth = EncryptedPolynomialEvaluator.polynomial_depth(
                SimpleNamespace(strategy='power'), approximation)
            assert row['selected_balanced_power_depth'] == actual_depth
            assert row['input_power_basis'] == approximation.input_power_basis
            assert row['sampled_selected_monomial_vs_chebyshev_deviation'] < 1e-11
    silu = reports['qwen']['silu']
    assert silu['input_power_basis'] is True
    assert silu['selected_balanced_power_depth'] == 2
    assert silu['balanced_monomial_depth_including_affine_map_and_final_sum'] == 3
    assert reports['qwen']['inverse_sqrt']['input_power_basis'] is False
