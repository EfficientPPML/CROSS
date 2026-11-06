"""Focused owned-child boundaries; runtime imports occur only in children."""
from test_support import run_case

def test_all47_recorded_public_reads_precede_private_factory(tmp_path):
    run_case('test_all47_recorded_public_reads_precede_private_factory', tmp_path)

def test_last_public_artifact_rejection_keeps_private_unentered(tmp_path):
    run_case('test_last_public_artifact_rejection_keeps_private_unentered', tmp_path)

def test_default_decoder_abi_and_full_scratch_snapshots_are_original(tmp_path):
    run_case('test_default_decoder_abi_and_full_scratch_snapshots_are_original', tmp_path)

def test_decoder_failure_stops_without_relaxed_retry_and_reaches_key_scope(tmp_path):
    run_case('test_decoder_failure_stops_without_relaxed_retry_and_reaches_key_scope', tmp_path)

def test_nonfinite_unused_slot_fails_full_decode_gate(tmp_path):
    run_case('test_nonfinite_unused_slot_fails_full_decode_gate', tmp_path)

def test_returned_consumer_load_failure_retires_inside_private_scope(tmp_path):
    run_case('test_returned_consumer_load_failure_retires_inside_private_scope', tmp_path)

def test_codec_or_context_replacement_is_rejected_before_next_decode(tmp_path):
    run_case('test_codec_or_context_replacement_is_rejected_before_next_decode', tmp_path)

def test_server_recorded47_finish_and_all_consumers_retire_before_keys(tmp_path):
    run_case('test_server_recorded47_finish_and_all_consumers_retire_before_keys', tmp_path)

def test_server_finish_callback_mutation_reaches_original_cleanup_and_key_failure(tmp_path):
    run_case('test_server_finish_callback_mutation_reaches_original_cleanup_and_key_failure', tmp_path)

def test_server_partial_owner_acquisition_preserves_prior_cleanup(tmp_path):
    run_case('test_server_partial_owner_acquisition_preserves_prior_cleanup', tmp_path)
