"""Independent finite assertions around actual selected runtime/control paths."""
import json
from types import SimpleNamespace
from integration_fixture import raises

def run_direct(f,scenario):
    backend_case=scenario.startswith('backend_')or scenario.startswith('export_backend_')or scenario=='publisher_digest_mismatch'
    if backend_case:f.backend_seams()
    f.capture_runtime();r=f.runtime
    assert f.counts['ids']==f.counts['digest']==f.counts['ready_ids']==1
    assert json.loads(r.expected_ids_bytes)['private_manifest_sha256']is None
    if scenario=='backend_before_key_scope':
        before=f.counts['core_load'];raises(PermissionError,f.build_backend)
        assert f.counts['core_load']==before==0;f.sticky(r.check);return
    if scenario=='runtime_caught_transition':
        raises((PermissionError,ValueError),r.allow_public_key_reads)
        assert isinstance(f.caught_nested,ValueError)and 'keys.enabled'not in f.events
        f.sticky(r.check);return
    r.allow_public_key_reads()
    assert f.audit.keys_enabled is True
    if backend_case:f.build_backend()
    if scenario in('runtime_hot_clean','runtime_pending_checks','backend_forwarding'):
        names=('full','phase','pins','ready_ids','ids','digest')
        before=tuple(f.counts[n]for n in names)
        for _ in range(3):
            r.check();r.require_public_key_read_scope()
            if backend_case:f.backend_instance.check();f.forward_map()
        assert tuple(f.counts[n]for n in names)==before
        assert ('keys.enter'in f.events and 'keys.enabled'in f.events)
        if scenario=='runtime_pending_checks':
            r.admit_server_bodies(f.handle);assert f.audit.bodies_enabled is True and 'bodies.enabled'in f.events
        full_before=f.counts['full'];f.full_boundary();assert f.counts['full']==full_before+1
        assert f.counts['ids']==f.counts['digest']==1
        f.clean_retire();return
    if scenario=='publisher_digest_mismatch':
        r.admit_server_bodies(f.handle)
        before=(f.counts['map'],f.counts['evidence'])
        error=raises(ValueError,lambda:f.backend_instance.factories.publisher(f.handle,'0'*64))
        assert str(error)=='original admitted server input digest differs'and before==(f.counts['map'],f.counts['evidence'])
        f.clean_retire();return
    route=r.check
    def sentinel(*args,**kwargs):f.counts['replacement']+=1;raise AssertionError('replacement executed before original guard')
    if scenario=='export_require_runtime':f.mutate(f.process,'require_runtime',sentinel)
    elif scenario=='export_runtime_action':f.mutate(f.process,'_runtime_action',sentinel)
    elif scenario=='export_backend_runtime_check':
        route=f.backend_instance.check;f.mutate(f.backend,'_runtime',sentinel)
    elif scenario=='export_backend_runtime_constructor':
        route=lambda:f.backend.ServerBackend(r);f.mutate(f.backend,'_runtime',sentinel)
    elif scenario=='cleanup_is_cleanup':f.mutate(f.cleanup_origin,'is_cleanup',sentinel)
    elif scenario=='cleanup_mark':f.mutate(f.cleanup_origin,'mark',sentinel)
    elif scenario=='runtime_ids_bytes':
        old=r._expected_ids_bytes;new=bytes(bytearray(old));assert new==old and new is not old
        f.mutate(r,'_expected_ids_bytes',new)
    elif scenario=='runtime_input_digest':
        old=r._input_digest;new=old.encode().decode();assert new==old and new is not old
        f.mutate(r,'_input_digest',new)
    elif scenario=='derived_ids_dictionary':
        old=f.ids.__dict__;new=dict(old);assert new==old and new is not old
        f.mutate(f.ids,'__dict__',new)
    elif scenario=='runtime_control':f.mutate(r,'_control',sentinel)
    elif scenario=='audit_flags':f.mutate(f.audit,'bodies_enabled',True)
    elif scenario=='guard_check_descriptor':f.mutate(f.guard.AdmissionSnapshot,'check',sentinel)
    elif scenario=='source_read_export':f.mutate(f.process,'read',sentinel)
    elif scenario=='environment_callback_code':
        original=f.process.exact_environment
        def make(self):
            def replacement(dep):self.counts['replacement']+=1;raise AssertionError('changed callback code executed')
            return replacement
        replacement=make(f)
        assert original.__code__.co_freevars==replacement.__code__.co_freevars
        f.mutate(original,'__code__',replacement.__code__)
    elif scenario=='snapshot_caught_reentry':f.environment_mode='reenter'
    elif scenario=='wrong_handle':
        route=lambda:r.admit_server_bodies(SimpleNamespace(admission=object()))
    elif scenario in('secret_direct','secret_cache','secret_parameters'):
        if scenario=='secret_direct':f.mutate(f.ctx,'secret_key',object())
        elif scenario=='secret_cache':f.mutate(f.ctx._param_cache,'secret_key',object())
        else:f.mutate(f.ctx,'parameters',{'secret_key':object()})
        route=lambda:r.admit_server_bodies(f.handle)
    else:raise AssertionError('unselected direct case')
    before=(f.counts['map'],f.counts['core_load'],f.counts['replacement'])
    raises((PermissionError,ValueError),route)
    assert (f.counts['map'],f.counts['core_load'],f.counts['replacement'])==before
    if scenario=='snapshot_caught_reentry':assert isinstance(f.caught_nested,(PermissionError,ValueError))
    assert 'bodies.enabled'not in f.events
    f.sticky(r.check)


def run_process(f,scenario):
    f.process_seams();code=f.execute_process();events=f.events
    assert f.counts['ids']==f.counts['digest']==f.counts['ready_ids']==1
    assert 'audit.close'in events and 'shield.exit'in events
    if scenario=='process_default_capacity_denial':
        assert code==70 and type(f.reported_error)is PermissionError
        assert f.counts['core_load']==f.counts['stage']==f.counts['evidence']==0
        assert not f.temporary and 'core.install'not in events and 'keys.enabled'not in events
    elif scenario=='process_clean':
        assert code==0 and f.reported_error is None and f.counts['guard_retire']==1
        assert f.counts['stage']==f.counts['evidence']==1
        assert events.index('loader.context.retired')<events.index('termination.exit')<events.index('evidence.close')<events.index('audit.close')<events.index('shield.exit')<events.index('guard.retire')
        assert 'full.packet'in events[events.index('loader.context.retired'):events.index('evidence.close')]
        assert f.guard._REGISTRY=={}and f.process._RUNTIME_CONTROLS=={}and f.process._QUARANTINE==[]
        assert json.loads(f.publication['report.json'])['passed']is True
        return
    elif scenario=='process_public_proof_changed':
        assert code==70 and type(f.reported_error)is ValueError
        assert str(f.reported_error)=='public input digest differs'
        assert 'report.json'not in f.publication and 'evidence.close'in events
    elif scenario=='process_evidence_close_failure':
        assert code==70 and f.reported_error is f.cleanup_error and 'report.json'in f.publication
        assert events.index('evidence.close')<events.index('audit.close')<events.index('shield.exit')
    elif scenario=='process_body_and_cleanup_failure':
        assert code==70 and f.reported_error is f.body_error
        assert 'loader.context.retired'in events and 'evidence.close'in events and 'failure.json'in f.publication
    elif scenario=='process_late_guard_retirement_failure':
        assert code==70 and f.reported_error is f.late_error and f.counts['guard_retire']==1
        assert events.index('audit.close')<events.index('shield.exit')<events.index('guard.retire')
        assert 'report.json'in f.publication
    else:raise AssertionError('unselected process case')
    if scenario!='process_late_guard_retirement_failure':assert f.counts['guard_retire']==0
    assert len(f.process._QUARANTINE)==1
    owners=f.process._QUARANTINE[0]
    assert owners[0]is f.packet and owners[-1]is f.reported_error
    retained=owners[8]
    assert retained['packet']is f.packet and retained['guard']is not None and retained['status']!='retired'
    # A Finalizer-only failure can leave the unretired guard status active;
    # the actual process still raises and quarantines it. Do not infer a new
    # status transition that the selected source does not perform.
    assert f.guard._REGISTRY[id(retained['guard'])][1]is f.packet


def terminal(entry,scenario):
    if scenario=='terminal_zero':call=lambda:0;wanted=0
    elif scenario=='terminal_one':call=lambda:1;wanted=1
    else:
        error=SystemExit(0)if scenario=='terminal_systemexit_zero'else SystemExit(1)if scenario=='terminal_systemexit_one'else ValueError('owned synthetic terminal failure')
        def call():raise error
        wanted=70
    result=entry.terminal_status(call)
    assert result==wanted
    return dict(terminal_status=result,function_boundary_only=True,genuine_bootstrap=False)
