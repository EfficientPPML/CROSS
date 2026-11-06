"""Fresh-process synthetic device-state tests; no real device or host probes."""
import hashlib
import importlib
import importlib.abc
import importlib.util
import json
import os
from pathlib import Path
import sys
import threading

FORBIDDEN=frozenset(('numpy','jax','jaxlib','libtpu','jax_plugins','jaxite_word','torch','tensorflow','transformers','tokenizers','safetensors'))
DENIED=[]
RECORDING_JAX=None
def absent():
    found={n:m for n,m in sys.modules.items()if n.split('.')[0]in FORBIDDEN}
    assert found==({}if RECORDING_JAX is None else{'jax':RECORDING_JAX})
    assert not DENIED
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIED.append(('discovery',fullname));raise ImportError('genuine backend discovery forbidden')
        return None
def audit(event,args):
    if event=='import'and args and str(args[0]).split('.')[0]in FORBIDDEN:
        DENIED.append(('import',str(args[0])));raise ImportError('genuine backend import forbidden')
    if event=='open'and args and type(args[0])in(str,bytes):
        name=os.fsdecode(args[0])
        if any(name==p or name.startswith(p+'/')for p in('/dev','/proc','/sys','/synthetic-device-fixture')):
            DENIED.append(('forbidden_metadata_or_device_open',name));raise PermissionError('fixture paths must never be accessed')
def raises(kinds,fn):
    try:fn()
    except kinds as error:return error
    raise AssertionError('required rejection did not occur')

def main():
    global RECORDING_JAX
    assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
    scenario,root_arg,result_arg=sys.argv[1:];root=Path(root_arg).resolve();runtime=root/'runtime'
    runtime_sources=json.loads((root/'runtime_sources.json').read_bytes())
    assert len(runtime_sources)==34
    def sources():
        assert not any(p.is_symlink()for p in runtime.rglob('*'))
        actual={p.relative_to(runtime).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()for p in runtime.rglob('*')if p.is_file()}
        assert actual==runtime_sources
    absent();sources();finder=DenyBackend();sys.meta_path.insert(0,finder);sys.addaudithook(audit)
    assert not any(n in sys.modules for n in('qwen_device_policy','qwen_device_capacity','qwen_server_process','qwen_server_resource','qwen_server_admission','device_fixtures'))
    sys.path[:0]=[str(runtime),str(root)]
    import device_cases
    import device_fixtures as fixtures
    assert scenario in device_cases.CASES
    policy_module=importlib.import_module('qwen_device_policy')
    capacity=importlib.import_module('qwen_device_capacity')
    phase_module=importlib.import_module('qwen_server_admission')
    resource=importlib.import_module('qwen_server_resource')
    process=importlib.import_module('qwen_server_process')
    fixture=fixtures.build(runtime,runtime_sources,phase_module,scenario)
    phase=fixture['phase'];blobs=fixture['blobs'];pins=fixture['pins'];pp=fixture['phase_pins']
    original_phase_check=phase_module.ServerPhaseAdmission.check
    original_pins_check=phase_module.ExpectedServerPhasePins.check
    original_resource=resource.authenticate_policy;original_logical=resource.logical_facts
    assert sys.getprofile()is None
    counts=dict(phase=0,pins=0,resource=0,logical=0)
    def phase_check(value):
        assert value is phase;counts['phase']+=1;return value
    def pins_check(value):
        assert value is phase.expected_pins;counts['pins']+=1;return pp
    def observe_calls(frame,event,arg):
        if event=='call':
            if frame.f_code is original_resource.__code__:counts['resource']+=1
            if frame.f_code is original_logical.__code__:counts['logical']+=1
    def source_binding():
        sources();absent();assert sys.meta_path[0]is finder
        assert resource.authenticate_policy is original_resource and resource.logical_facts is original_logical
        for name in runtime_sources:
            if not name.endswith('.py'):continue
            module_name=name[:-3]
            if module_name=='recording_jax':continue
            module=sys.modules.get(module_name)
            if module is not None:
                expected=runtime/name
                assert Path(module.__file__)==expected and Path(module.__spec__.origin)==expected
    phase_module.ServerPhaseAdmission.check=phase_check
    phase_module.ExpectedServerPhasePins.check=pins_check
    sys.setprofile(observe_calls)
    restores=[];result=None;record=None;policy=None
    try:
        source_binding()
        assert capacity._ISSUED is False and capacity._REGISTRY=={}
        if scenario.startswith('document_'):
            errors={
                'document_role_order':'exact immutable ordered device proof tuple required',
                'document_noncanonical':'canonical compact evidence required',
                'document_source_phase_join':'device/phase source_archive_sha256 differs',
                'document_origin_join':'JAX spec origin differs',
                'document_workspace_bound':'stage exceeds independently reviewed resource envelope',
                'document_review_coverage':'exact independent factual review required',
                'document_lease_ownership':'exclusive independently authorized initialization required',
            }
            call=lambda:policy_module.authenticate(phase,blobs,expected_pins=pins)
            if scenario=='document_clean':
                admitted=call();assert type(admitted)is policy_module.DevicePolicyAdmission
                assert admitted.phase is phase and admitted.blobs is blobs and admitted.expected_pins is pins
                assert admitted.execution_authorized is False and admitted.check().blobs is blobs
                assert counts['resource']>=2 and counts['logical']>=2
            else:
                error=raises(ValueError,call);assert str(error)==errors[scenario]
                expected_resource=scenario in('document_origin_join','document_workspace_bound','document_review_coverage','document_lease_ownership')
                assert(counts['resource']>0)is expected_resource
                assert(counts['logical']>0)is expected_resource
            assert capacity._ISSUED is False and capacity._REGISTRY=={}and RECORDING_JAX is None
        else:
            policy=policy_module.authenticate(phase,blobs,expected_pins=pins)
            original_dict=policy.__dict__;original_fields=(policy.phase,policy.blobs,policy.expected_pins)
            marker=RuntimeError('owned synthetic hook marker')
            state=dict(mode=scenario,log=[],marker=marker,policy=policy,capacity=capacity,initializations=0,partial_owners=[],caught_reentry=None,returned_device=None,replacement_calls=0)
            process.STATE=state
            def arguments(got_phase,got_blobs):
                assert got_phase is phase and got_blobs is blobs
            def host(stage):
                return fixtures.canonical(dict(schema='qwen-device-host-observation-v1',pid=os.getpid(),stage=stage,
                    resource_policy_sha256=fixtures.sha(fixture['documents']['resource_policy']),
                    observation=dict(after_import=stage!='before_init',borrowed_body_reads=False,capacity_reserved=False,
                        device_capacity_checked=False,resource_admission_complete=False,synthetic_recording_only=True)))
            def install_recording_jax():
                global RECORDING_JAX
                assert 'jax'not in sys.modules and RECORDING_JAX is None
                path=runtime/'recording_jax.py'
                assert fixtures.sha(path.read_bytes())==runtime_sources['recording_jax.py']
                spec=importlib.util.spec_from_file_location('jax',path)
                module=importlib.util.module_from_spec(spec)
                # Explicit source-owned double, never import_module('jax') or
                # backend discovery. Exact real origins are this fixture file.
                RECORDING_JAX=module;sys.modules['jax']=module
                spec.loader.exec_module(module)
                assert Path(module.__file__)==path and Path(module.__spec__.origin)==path
                return module
            state.update(arguments=arguments,host=host,sources=source_binding,install_recording_jax=install_recording_jax)
            def retained_failed():
                assert len(capacity._REGISTRY)==1
                row=next(iter(capacity._REGISTRY.values()))
                assert row['state']=='failed'and row['policy']is policy
                assert row['policy_dict']is original_dict
                assert all(a is b for a,b in zip(row['policy_fields'],original_fields,strict=True))
                # Restore the original identities and benign recording behavior
                # before testing public stickiness, not merely the private flag
                # or global one-use issuance rule. Never reset guard state.
                while restores:restores.pop()()
                state['mode']='benign'
                if RECORDING_JAX is not None:
                    RECORDING_JAX.STATS={'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':90}
                assert policy.__dict__ is original_dict
                assert all(policy.__dict__[name]is value for name,value in zip(('phase','blobs','expected_pins'),original_fields,strict=True))
                before=(len(state['log']),state['initializations'],0 if RECORDING_JAX is None else RECORDING_JAX.CALLS)
                original_permit=row['permit']
                if row['device']is None:
                    raises(PermissionError,lambda:capacity.initialize_once(original_permit))
                else:
                    next_index=row['stage_index']+1
                    assert 0<=next_index<len(fixtures.STAGES)
                    raises(PermissionError,lambda:capacity.observe_capacity(original_permit,fixtures.STAGES[next_index]))
                assert (len(state['log']),state['initializations'],0 if RECORDING_JAX is None else RECORDING_JAX.CALLS)==before
                assert capacity._REGISTRY[id(original_permit)]is row and row['state']=='failed'
                raises(PermissionError,lambda:capacity.issue_initialization(policy))
                assert (len(state['log']),state['initializations'],0 if RECORDING_JAX is None else RECORDING_JAX.CALLS)==before
                assert row['state']=='failed'
                return row
            if scenario in('binding_failure','lease_failure','reentry_caught','reentry_uncaught'):
                error=raises((RuntimeError,PermissionError),lambda:capacity.issue_initialization(policy))
                if scenario in('binding_failure','lease_failure'):assert error is marker
                if scenario=='reentry_caught':assert type(state['caught_reentry'])is PermissionError
                record=retained_failed();assert state['initializations']==0 and RECORDING_JAX is None
            else:
                permit=capacity.issue_initialization(policy);record=capacity._REGISTRY[id(permit)]
                assert permit.execution_authorized is False and record['state']=='permitted'
                assert state['initializations']==0 and RECORDING_JAX is None
                if scenario=='initialization_failure':
                    error=raises(RuntimeError,lambda:capacity.initialize_once(permit));assert error is marker
                    assert state['initializations']==1 and len(state['partial_owners'])==1
                    assert record['device']is None and RECORDING_JAX is None
                    assert record is retained_failed()
                elif scenario=='policy_blobs':
                    restores.append(lambda:object.__setattr__(policy,'blobs',blobs))
                    raises(ValueError,lambda:capacity.initialize_once(permit))
                    assert state['returned_device']is RECORDING_JAX.DEVICE
                    assert any(v is RECORDING_JAX.DEVICE for v in record['returned'])
                    assert record['device']is None and state['initializations']==1
                    assert record is retained_failed()
                else:
                    assert capacity.initialize_once(permit)is permit
                    fake=RECORDING_JAX;device=fake.DEVICE
                    assert record['device']is device and state['initializations']==1 and record['state']=='initialized'
                    assert len(state['partial_owners'])==1 and fake.CALLS==0
                    if scenario=='state_clean':
                        observations=[]
                        for number,stage in enumerate(fixtures.STAGES,1):
                            value=capacity.observe_capacity(permit,stage);assert value.execution_authorized is False
                            assert fake.CALLS==number
                            raw=capacity.require_current(permit,value,stage);assert fake.CALLS==number
                            observed=json.loads(raw);observations.append(raw)
                            assert observed['stage']==stage and observed['required_additional_bytes']==10
                            assert observed['statistics']=={'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':90}
                            assert observed['comparison_passed']is True
                            for name in('capacity_reserved','global_free_memory_measured','actual_server_role_authorized'):assert observed[name]is False
                            assert observed['sufficient_only_under_independently_reviewed_evidence']is True
                        assert [s for n,s in state['log']if n=='host']==['before_init',*fixtures.STAGES]
                        assert [s for n,s in state['log']if n=='synchronize']==list(fixtures.STAGES)
                        assert capacity.retire(permit)==tuple(observations)and capacity._REGISTRY=={}
                        assert sys.modules['jax']is fake and fake.config.jax_default_device is device
                        assert len(state['partial_owners'])==1
                        raises(PermissionError,lambda:capacity.issue_initialization(policy))
                        raises(PermissionError,lambda:capacity.require_current(permit,value,'terminal'))
                    else:
                        def next_stage():return capacity.observe_capacity(permit,'post_init')
                        if scenario=='policy_dictionary':
                            replacement=dict(original_dict);assert replacement==original_dict and replacement is not original_dict
                            object.__setattr__(policy,'__dict__',replacement)
                            restores.append(lambda:object.__setattr__(policy,'__dict__',original_dict))
                        elif scenario=='policy_phase':
                            replacement=phase_module.ServerPhaseAdmission(**phase.__dict__);assert replacement==phase and replacement is not phase
                            object.__setattr__(policy,'phase',replacement);restores.append(lambda:object.__setattr__(policy,'phase',phase))
                        elif scenario=='policy_pins':
                            replacement=bytes(bytearray(pins));assert replacement==pins and replacement is not pins
                            object.__setattr__(policy,'expected_pins',replacement);restores.append(lambda:object.__setattr__(policy,'expected_pins',pins))
                        elif scenario=='hook_export':
                            original=process.device_capacity_check_lease
                            def replacement(*args):state['replacement_calls']+=1;raise AssertionError('replacement hook executed')
                            process.device_capacity_check_lease=replacement
                            restores.append(lambda:setattr(process,'device_capacity_check_lease',original))
                        elif scenario=='stats_descriptor':
                            original=fake.Device.memory_stats
                            def replacement(self):state['replacement_calls']+=1;raise AssertionError('replacement descriptor executed')
                            fake.Device.memory_stats=replacement;restores.append(lambda:setattr(fake.Device,'memory_stats',original))
                        elif scenario=='default_device':
                            fake.config.jax_default_device=fake.Device()
                            restores.append(lambda:setattr(fake.config,'jax_default_device',device))
                        elif scenario.startswith('stats_'):
                            fake.STATS={
                                'stats_none':None,'stats_missing':{'bytes_in_use':90},
                                'stats_boolean':{'bytes_in_use':True,'bytes_limit':100},
                                'stats_negative':{'bytes_in_use':-1,'bytes_limit':100},
                                'stats_inconsistent':{'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':89},
                            }[scenario]
                        elif scenario=='insufficient_headroom':fake.STATS={'bytes_in_use':91,'bytes_limit':100}
                        elif scenario=='stage_order':next_stage=lambda:capacity.observe_capacity(permit,'post_keys')
                        elif scenario=='stale_admission':
                            older=capacity.observe_capacity(permit,'post_init');current=capacity.observe_capacity(permit,'post_keys')
                            assert current is record['current']and older is not current and fake.CALLS==2
                            next_stage=lambda:capacity.require_current(permit,older,'post_init')
                        elif scenario=='wrong_thread':pass
                        else:raise AssertionError('unselected state case')
                        prior_log=len(state['log']);prior_calls=fake.CALLS
                        if scenario=='wrong_thread':
                            errors=[]
                            def target():
                                try:next_stage()
                                except BaseException as error:errors.append(error)
                            thread=threading.Thread(target=target);thread.start();thread.join()
                            assert len(errors)==1 and type(errors[0])is PermissionError
                        else:
                            error=raises((ValueError,TypeError,PermissionError,MemoryError),next_stage)
                            if scenario=='insufficient_headroom':assert type(error)is MemoryError
                        if scenario in('policy_dictionary','policy_phase','policy_pins','hook_export','wrong_thread','stage_order','stale_admission'):
                            assert len(state['log'])==prior_log and fake.CALLS==prior_calls
                        elif scenario in('stats_descriptor','default_device'):assert fake.CALLS==prior_calls
                        else:assert fake.CALLS==prior_calls+1
                        assert state['replacement_calls']==0 and record is retained_failed()
                        assert record['device']is device and state['initializations']==1
            assert counts['resource']>0 and counts['logical']>0
        raises(PermissionError,lambda:capacity.create_actual_server(object()))
        result=dict(scenario=scenario,assertions_completed=True,pid=os.getpid(),pgid=os.getpgrp(),
            recording_phase_validation=True,resource_validator_is_real=True,resource_calls=counts['resource'],logical_calls=counts['logical'],
            temporary_recording_jax_used=RECORDING_JAX is not None,genuine_backend_imported=False,genuine_capacity_measured=False,
            actual_policy_or_grant=False,scientific_admission_tested=False,device_sources_modified=False)
    finally:
        while restores:restores.pop()()
        sys.setprofile(None);assert sys.getprofile()is None
        phase_module.ServerPhaseAdmission.check=original_phase_check
        phase_module.ExpectedServerPhasePins.check=original_pins_check
        assert phase_module.ServerPhaseAdmission.check is original_phase_check
        assert phase_module.ExpectedServerPhasePins.check is original_pins_check
        assert resource.authenticate_policy is original_resource and resource.logical_facts is original_logical
        if RECORDING_JAX is not None:
            assert sys.modules.get('jax')is RECORDING_JAX
            del sys.modules['jax'];RECORDING_JAX=None
        absent();sources()
    assert result is not None
    result.update(original_phase_identities_restored=True,original_resource_identities_preserved=True,
        profile_observer_restored=True,temporary_jax_removed=True,backend_denials=list(DENIED))
    with Path(result_arg).open('x')as stream:json.dump(result,stream,sort_keys=True);stream.write('\n')

if __name__=='__main__':main()
