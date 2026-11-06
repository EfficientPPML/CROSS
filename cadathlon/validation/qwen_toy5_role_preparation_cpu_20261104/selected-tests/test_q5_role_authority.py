"""Focused owned-child boundaries; runtime imports occur only in children."""
from test_support import run_case

def test_fixed_q5_profile_and_203_descriptors(tmp_path):
    run_case('test_fixed_q5_profile_and_203_descriptors', tmp_path)

def test_profile_substitution_reaches_exact_contract_rejection(tmp_path):
    run_case('test_profile_substitution_reaches_exact_contract_rejection', tmp_path)

def test_complete_synthetic_setup_chain_admits_distinct_role_maps(tmp_path):
    run_case('test_complete_synthetic_setup_chain_admits_distinct_role_maps', tmp_path)

def test_internal_generated_manifests_never_supply_later_terminal_authority(tmp_path):
    run_case('test_internal_generated_manifests_never_supply_later_terminal_authority', tmp_path)

def test_repinned_outer_and_root_fail_at_semantic_join(tmp_path):
    run_case('test_repinned_outer_and_root_fail_at_semantic_join', tmp_path)

def test_phase_source_and_original_proof_identity_reauthenticate(tmp_path):
    run_case('test_phase_source_and_original_proof_identity_reauthenticate', tmp_path)
