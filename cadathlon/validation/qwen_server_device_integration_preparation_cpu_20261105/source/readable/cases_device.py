"""Original owner/controller tests with recorded native effects, no backend."""
from types import SimpleNamespace

def _failure(f,call,message=None):
    error=f.expect((ValueError,RuntimeError,PermissionError,TypeError),call,message)
    assert f.row()['failure']is not None and f.row()['quarantined']
    f.sticky_owner();return error

def run(f,case):
    checks=[]
    if case=='owner_original_native_denial':
        f.prepare_owner(acquire=False,synchronize=False)
        _failure(f,f.device.start,'genuine exclusive physical lease')
        assert f.jax is None and f.row()['core']is None and f.row()['lease_attempted']
        assert f.counts['core']==f.counts['native']==0
        assert not f.audit.initialization_enabled and not f.audit.keys_enabled and not f.audit.bodies_enabled
        checks.append('original unavailable native acquire rejects before core/default/key/body')
    elif case in('owner_array_factory_replacement','owner_array_enter_replacement','owner_default_enter_failure'):
        f.mode={'owner_array_factory_replacement':'factory_array','owner_array_enter_replacement':'enter_array','owner_default_enter_failure':'enter_failure'}[case]
        f.prepare_owner();error=f.expect((ValueError,RuntimeError),f.device.start)
        row=f.row();assert row['array_type']is f.jax.ORIGINAL_ARRAY
        assert f.jax.REPLACEMENT_CALLS==0 and len(f.jax.SCOPES)==1
        assert row['default_scope']is f.jax.SCOPES[0]and row['default_scope'].exits==0
        if case=='owner_default_enter_failure':assert error is f.jax.MARKER and not row['default_enter_completed']
        f.sticky_owner();checks.append('Array/default owner captured before foreign factory/enter and no replacement instancecheck')
    elif case=='owner_native_sync_denial':
        f.prepare_owner(sync_denial_after_load=True);f.start()
        f.audit.keys_enabled=True;transfers=tuple(f.array('denial-setup'+str(i))for i in range(4))
        f.device.register_loaded(f.handle,transfers)
        _failure(f,lambda:f.device.post_keys(f.handle),'genuine initialization/NTT/cache/async')
        assert all(value.waits==1 for value in transfers)
        assert f.row()['context']is f.ctx and f.row()['transfers']is transfers
        checks.append('recorded pre-init sync, then original native synchronization denial after all explicit original barriers')
    else:
        f.prepare_owner();f.start();row=f.row()
        if case=='owner_clean_four_stages':
            f.register();f.evaluate_boundary()
            evidence=f.device.evidence();assert [r['stage']for r in evidence['observations']]==['post_init','post_keys','pre_evaluation']
            f.clean_owner()
            assert len(row['host'])==5 and len(row['observations'])==4
            assert row['default_scope'].enters==row['default_scope'].exits==1
            assert all(array.waits>=3 for array in f.transfers)
            checks.append('four original ordered observations, five recorded hosts, original clean scope exit')
        elif case=='owner_hook_export_mutation':
            def replacement(*args,**kwargs):f.counts['replacement']+=1;raise AssertionError('changed hook invoked')
            f.mutate(f.process,'device_capacity_check_lease',replacement)
            _failure(f,f.device.check);assert f.counts['replacement']==0
            checks.append('captured hook export rejected before sentinel; original owner sticky after restore')
        elif case=='owner_policy_equal_replacement':
            policy=f.device_packet_value.policy;original=policy.__dict__;replacement=dict(original)
            assert replacement==original and replacement is not original
            f.mutate(policy,'__dict__',replacement);_failure(f,f.device.check)
            assert policy.__dict__ is original;checks.append('equal replacement dictionary rejected and original identity restored without reset')
        elif case=='owner_original_audit_accessor_replacement':
            def replacement():f.counts['replacement']+=1;raise AssertionError('changed audit invoked')
            f.mutate(f.audit,'check',replacement);_failure(f,f.device.check)
            assert f.counts['replacement']==0;checks.append('original audit accessor checked before invocation')
        elif case=='owner_wrong_singleton_array':
            f.audit.keys_enabled=True;wrong=f.array('wrong',f.jax.Device())
            _failure(f,lambda:f.device.register_loaded(f.handle,(wrong,)),'retained selected singleton')
            assert row['transfers']==(wrong,)and wrong.waits==0
            checks.append('wrong singleton device rejected with original transfer retained')
        elif case=='owner_array_descriptor_replacement':
            f.register();original=f.jax.Array.block_until_ready
            def replacement(value):f.counts['replacement']+=1;raise AssertionError('foreign wait invoked')
            f.mutate(f.jax.Array,'block_until_ready',replacement)
            _failure(f,lambda:f.evaluate_boundary(),'array device/barrier binding')
            assert f.counts['replacement']==0 and f.jax.Array.block_until_ready is original
            checks.append('original descriptor compared before any changed wait')
        elif case in('owner_ciphertext_payload_replacement','owner_ciphertext_moduli_replacement'):
            value=f.ciphertext('ciphertext');f.device.ciphertext(value)
            field='polynomial'if case.endswith('payload_replacement')else'moduli_array'
            original=getattr(value,field);foreign=f.array('foreign')
            f.mutate(value,field,foreign);_failure(f,f.device.check,'ciphertext payload/moduli identity')
            assert any(original is a for saved in row['ciphertexts']for a in saved[1:])and foreign.waits==0
            checks.append('standalone ciphertext original triple retained, field replacement rejected')
        elif case in('owner_prepared_payload_replacement','owner_prepared_moduli_replacement'):
            f.register();f.evaluate_boundary();value=f.prepared[0]
            field='polynomial'if case.endswith('payload_replacement')else'moduli_array'
            original=getattr(value,field);foreign=f.array('foreign-prepared')
            f.mutate(value,field,foreign);_failure(f,f.device.check,'ciphertext payload/moduli identity')
            assert row['evaluation'][3]is f.prepared and any(original is a for saved in row['ciphertexts']for a in saved[1:])
            assert foreign.waits==0;checks.append('prepared original triple and immutable prepared tuple retained')
        elif case=='owner_partial_moduli_getter':
            f.register();f.audit.bodies_enabled=True;good=f.ciphertext('earlier');payload=f.array('partial.payload')
            class Partial:
                @property
                def polynomial(self):return payload
                @property
                def moduli_array(self):raise f.marker
            partial=Partial();prepared=(good,partial)
            error=f.expect(ValueError,lambda:f.device.before_evaluation(f.ctx,f.handle,object(),prepared))
            assert error is f.marker and row['evaluation'][3]is prepared
            assert any(x is partial for x in row['raw_returns'])and any(x is payload for x in row['raw_returns'])
            assert row['ciphertexts'][0][0]is good
            f.sticky_owner();checks.append('all returned partial owners retained before the failing getter')
        elif case=='owner_caught_transition_reentry':
            f.audit.keys_enabled=True;transfers=tuple(f.array('reentry'+str(i))for i in range(4))
            f.device.register_loaded(f.handle,transfers)
            f.mode='caught_reentry';_failure(f,lambda:f.device.post_keys(f.handle))
            assert f.caught is not None;checks.append('caught nested device transition leaves original owner failed without registry reset')
        elif case=='owner_default_exit_failure':
            f.register();f.evaluate_boundary();f.handle.ctx=None;f.device.terminal();f.device.retire_guard()
            f.jax.MODE='exit_failure';error=f.expect(RuntimeError,lambda:f.owner_module.close_default_scope(f.device))
            assert error is f.jax.MARKER and row['default_scope'].exits==1 and row['default_exit_attempted']
            f.owner_module.close_default_scope(f.device,primary=error)
            assert row['default_scope'].exits==1;f.sticky_owner();checks.append('uncertain default exit is attempted once and original failed graph retained')
        elif case=='capacity_immutable_snapshot_after_hook_mutation':
            f.audit.keys_enabled=True;transfers=tuple(f.array('stats-setup'+str(i))for i in range(4))
            f.device.register_loaded(f.handle,transfers)
            original=f.jax.STATS;before=dict(original);f.stats_sample_target=f.jax.SAMPLES+1
            f.mode='mutate_stats';f.device.post_keys(f.handle)
            observed=f.policy.parse(row['observations'][-1],'observation')
            assert original!=before and observed['statistics']==before
            permit=f.capacity._REGISTRY[id(row['permit'])]
            assert any(x is original for x in permit['returned'])
            assert any(type(x)is tuple and dict(x)==before for x in permit['returned'])
            assert observed['required_additional_bytes']==10 and observed['comparison_passed']is True
            f.jax.STATS=before;f.evaluate_boundary();f.clean_owner()
            checks.append('real402 comparison and serialized report use the same retained immutable sample')
        else:raise AssertionError('unselected device case '+case)
    return dict(passed=True,case=case,checks=checks,recording_only=True,actual_authority=False)
