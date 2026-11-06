"""Focused owned-child boundaries; runtime imports occur only in children."""
from test_support import run_case

def test_recorded_203_generation_preserves_original_secret_and_literal_rotations(tmp_path):
    run_case('test_recorded_203_generation_preserves_original_secret_and_literal_rotations', tmp_path)

def test_generation_source_denial_precedes_store_and_rng(tmp_path):
    run_case('test_generation_source_denial_precedes_store_and_rng', tmp_path)

def test_generation_security_denial_precedes_store_and_rng(tmp_path):
    run_case('test_generation_security_denial_precedes_store_and_rng', tmp_path)

def test_generation_capacity_denial_precedes_store_and_rng(tmp_path):
    run_case('test_generation_capacity_denial_precedes_store_and_rng', tmp_path)

def test_partial_store_acquisition_keeps_original_owner_for_cleanup(tmp_path):
    run_case('test_partial_store_acquisition_keeps_original_owner_for_cleanup', tmp_path)

def test_generation_postclose_failure_does_not_return_ready_result(tmp_path):
    run_case('test_generation_postclose_failure_does_not_return_ready_result', tmp_path)

def test_setup_encrypt_success_barriers_original_moduli_and_ntt_tables(tmp_path):
    run_case('test_setup_encrypt_success_barriers_original_moduli_and_ntt_tables', tmp_path)

def test_setup_encrypt_attached_partial_failure_barriers_before_key_exit(tmp_path):
    run_case('test_setup_encrypt_attached_partial_failure_barriers_before_key_exit', tmp_path)

def test_setup_late_ff_owner_replacement_rejects_before_foreign_barrier(tmp_path):
    run_case('test_setup_late_ff_owner_replacement_rejects_before_foreign_barrier', tmp_path)

def test_setup_late_table_array_replacement_rejects_before_foreign_barrier(tmp_path):
    run_case('test_setup_late_table_array_replacement_rejects_before_foreign_barrier', tmp_path)
