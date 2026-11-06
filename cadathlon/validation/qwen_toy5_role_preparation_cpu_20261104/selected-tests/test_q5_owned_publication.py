"""Focused owned-child boundaries; runtime imports occur only in children."""
from test_support import run_case

def test_bounded_write_and_writelines_reject_before_growth(tmp_path):
    run_case('test_bounded_write_and_writelines_reject_before_growth', tmp_path)

def test_bounded_seek_and_truncate_preserve_extent_on_rejection(tmp_path):
    run_case('test_bounded_seek_and_truncate_preserve_extent_on_rejection', tmp_path)

def test_tiny_sixfield_npz_publish_matches_exact_bytes_after_buffer_close(tmp_path):
    run_case('test_tiny_sixfield_npz_publish_matches_exact_bytes_after_buffer_close', tmp_path)

def test_partial_positive_raw_writes_make_exact_forward_progress(tmp_path):
    run_case('test_partial_positive_raw_writes_make_exact_forward_progress', tmp_path)

def test_zero_raw_write_rejects_and_closes_original_once(tmp_path):
    run_case('test_zero_raw_write_rejects_and_closes_original_once', tmp_path)

def test_write_failure_preserves_primary_while_attempting_close(tmp_path):
    run_case('test_write_failure_preserves_primary_while_attempting_close', tmp_path)

def test_late_output_close_failure_never_returns_success(tmp_path):
    run_case('test_late_output_close_failure_never_returns_success', tmp_path)

def test_owned_root_replacement_is_rejected_without_deleting_replacement(tmp_path):
    run_case('test_owned_root_replacement_is_rejected_without_deleting_replacement', tmp_path)

def test_real_public_assets_tiny_load_check_and_clean_retirement(tmp_path):
    run_case('test_real_public_assets_tiny_load_check_and_clean_retirement', tmp_path)

def test_real_public_assets_fifo_rejects_without_blocking_or_body_read(tmp_path):
    run_case('test_real_public_assets_fifo_rejects_without_blocking_or_body_read', tmp_path)

def test_real_exporter_clean_registry_mirrors_and_repeatclose(tmp_path):
    run_case('test_real_exporter_clean_registry_mirrors_and_repeatclose', tmp_path)
