"""Re-pinned malformed proof semantics must not become key readiness."""
import pytest
from synthetic_keygen_chain import fake_hash


@pytest.mark.parametrize('case', ['context_id', 'public_manifest_sha256',
                                  'private_manifest_sha256', 'stores_sha256'])
def test_generated_identity_must_match_independent_postterminal_pin(chain, case):
    with pytest.raises(ValueError,match='differs'):
        chain(change_ready_pins=lambda row:row.__setitem__(case,fake_hash('different '+case))).ready()


@pytest.mark.parametrize('case', ['incomplete_public_count', 'missing_secret_count', 'missing_array', 'boolean_array_shape',
                                  'cleanup_incomplete', 'wrong_p_coverage', 'non_cpu_backend',
                                  'changed_source_end', 'wrong_role_link', 'wrong_store_mode',
                                  'changed_store_parent', 'model_scope_claim'])
def test_repinned_report_or_manifest_semantics_fail_closed(chain, case):
    options = {}
    if case == 'missing_array':
        options['change_public'] = lambda row:row['arrays'].pop('rot_65535_a.npy')
        message = 'complete335 public array inventory'
    elif case == 'boolean_array_shape':
        options['change_public'] = lambda row:row['arrays']['public_key.npy']['shape'].__setitem__(0,True)
        message = 'bounded exact key array descriptor'
    else:
        def change(row):
            if case == 'incomplete_public_count': row['counts']['public_arrays'] = 334
            elif case == 'missing_secret_count': row['counts']['private_arrays'] = 0
            elif case == 'cleanup_incomplete': row['checks']['cleanup_completed'] = False
            elif case == 'wrong_p_coverage': row['security']['p_coverage'] = False
            elif case == 'non_cpu_backend': row['backend']['observed_device_platforms'] = ['tpu']
            elif case == 'changed_source_end': row['source_hashes_after']['qwen_key_store.py'] = fake_hash('changed writer')
            elif case == 'wrong_role_link': row['response_consumer_source_manifest_sha256'] = fake_hash('other role')
            elif case == 'wrong_store_mode': row['stores']['private']['mode'] = 0o755
            elif case == 'changed_store_parent': row['stores']['public']['parent']['inode'] += 1
            elif case == 'model_scope_claim': row['scope']['model_bodies_read'] = True
            else: raise AssertionError(case)
        options['change_report'] = change
        message = {
            'incomplete_public_count':'complete generation counts',
            'missing_secret_count':'complete generation counts',
            'cleanup_incomplete':'complete real generation/retirement checks',
            'wrong_p_coverage':'genuine fixed security evidence',
            'non_cpu_backend':'only actual CPU devices',
            'changed_source_end':'fixed actual prerequisite source',
            'wrong_role_link':'report/intent response_consumer',
            'wrong_store_mode':'retained key root mode',
            'changed_store_parent':'retained/granted parent identity',
            'model_scope_claim':'generation-only scope',
        }[case]
    with pytest.raises(ValueError,match=message): chain(**options).ready()


@pytest.mark.parametrize('case', ['exit70', 'not_normal_return', 'unreaped', 'live_group', 'boolean_pid'])
def test_repinned_outer_requires_normal_zero_and_independent_retirement(chain, case):
    def change(row):
        if case == 'exit70': row['exit_code'] = 70
        elif case == 'not_normal_return': row['normal_return'] = False
        elif case == 'unreaped': row['reaped'] = False
        elif case == 'live_group': row['process_group_absent'] = False
        elif case == 'boolean_pid': row['pid'] = True
        else: raise AssertionError(case)
    with pytest.raises(ValueError,match='independent normal-zero/reap outer'):
        chain(change_outer=change).ready()


@pytest.mark.parametrize('case', ['archive_unreviewed', 'stores_unreviewed', 'model_proof_changed', 'downstream_enabled'])
def test_repinned_root_review_cannot_weaken_readiness_scope(chain, case):
    def change(row):
        if case == 'archive_unreviewed': row['source_archive_members_verified'] = False
        elif case == 'stores_unreviewed': row['current_owned_stores_verified'] = False
        elif case == 'model_proof_changed': row['model_evidence']['outer_sha256'] = fake_hash('different model outer')
        elif case == 'downstream_enabled': row['downstream_execution_authorized'] = True
        else: raise AssertionError(case)
    with pytest.raises(ValueError,match='independent root key readiness'):
        chain(change_root=change).ready()
