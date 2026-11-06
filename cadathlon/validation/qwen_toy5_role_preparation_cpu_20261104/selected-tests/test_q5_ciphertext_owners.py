"""Focused owned-child boundaries; runtime imports occur only in children."""
from test_support import run_case

def test_six_constructor_lookup_is_exact_and_body_scope_precedes_imports(tmp_path):
    run_case('test_six_constructor_lookup_is_exact_and_body_scope_precedes_imports', tmp_path)

def test_server_cache_passes_three_argument_polynomial_abi(tmp_path):
    run_case('test_server_cache_passes_three_argument_polynomial_abi', tmp_path)

def test_cache_replacement_after_getter_rejects_before_polynomial(tmp_path):
    run_case('test_cache_replacement_after_getter_rejects_before_polynomial', tmp_path)

def test_client_explicit_ntt_acquires_before_original_initializers(tmp_path):
    run_case('test_client_explicit_ntt_acquires_before_original_initializers', tmp_path)

def test_barrett_initializer_partial_arrays_are_retained_and_barred(tmp_path):
    run_case('test_barrett_initializer_partial_arrays_are_retained_and_barred', tmp_path)

def test_ntt_initializer_partial_arrays_are_retained_and_barred(tmp_path):
    run_case('test_ntt_initializer_partial_arrays_are_retained_and_barred', tmp_path)

def test_returned_moduli_array_is_barred_after_validation_failure(tmp_path):
    run_case('test_returned_moduli_array_is_barred_after_validation_failure', tmp_path)

def test_returned_moduli_array_is_barred_after_security_failure(tmp_path):
    run_case('test_returned_moduli_array_is_barred_after_security_failure', tmp_path)

def test_barrier_failure_attempts_remaining_original_owners_and_keeps_primary(tmp_path):
    run_case('test_barrier_failure_attempts_remaining_original_owners_and_keeps_primary', tmp_path)

def test_input_close_tombstone_is_idempotent_and_preserves_primary(tmp_path):
    run_case('test_input_close_tombstone_is_idempotent_and_preserves_primary', tmp_path)

def test_input_partial_consumer_failure_closes_original_children_inside_key_scope(tmp_path):
    run_case('test_input_partial_consumer_failure_closes_original_children_inside_key_scope', tmp_path)

def test_clean_ct_ntt_release_never_disposes_borrowed_context(tmp_path):
    run_case('test_clean_ct_ntt_release_never_disposes_borrowed_context', tmp_path)
