"""Focused owned-child boundaries; runtime imports occur only in children."""
from test_support import run_case

def test_actual_setup_cli_denies_before_helper_body_or_backend(tmp_path):
    run_case('test_actual_setup_cli_denies_before_helper_body_or_backend', tmp_path)

def test_actual_server_cli_denies_before_helper_body_or_backend(tmp_path):
    run_case('test_actual_server_cli_denies_before_helper_body_or_backend', tmp_path)

def test_actual_client_cli_denies_before_helper_body_or_backend(tmp_path):
    run_case('test_actual_client_cli_denies_before_helper_body_or_backend', tmp_path)

def test_help_and_usage_keep_zero_and_two_without_dispatch(tmp_path):
    run_case('test_help_and_usage_keep_zero_and_two_without_dispatch', tmp_path)

def test_recorded_compose_normal_zero_one_follow_original_evidence_close(tmp_path):
    run_case('test_recorded_compose_normal_zero_one_follow_original_evidence_close', tmp_path)

def test_recorded_compose_late_close_is_infrastructure_even_after_report(tmp_path):
    run_case('test_recorded_compose_late_close_is_infrastructure_even_after_report', tmp_path)
