"""Fresh-process recording guard checks; never scientific proof authentication."""
import ast
import hashlib
import importlib
import importlib.abc
import importlib.machinery
import json
import os
from pathlib import Path
import sys
import threading

FORBIDDEN=frozenset(('numpy','jax','jaxlib','libtpu','jax_plugins','jaxite_word','torch','tensorflow','transformers','tokenizers','safetensors'))
HELPERS=frozenset(('client_role_common','freeze_admission','freeze_issuer','keygen_authority','keygen_boundary_pins','model_admission','public_source_pins','qwen_file_contract','qwen_key_contract','qwen_public_parse','qwen_scientific_admission','qwen_server_admission','qwen_server_authority_packet','scientific_source_pins','qwen_admission_snapshot'))
DENIALS=[]

def absent():
    assert not any(n.split('.')[0]in FORBIDDEN for n in sys.modules)
    assert not DENIALS

class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:
            DENIALS.append(fullname);raise ImportError('backend excluded from snapshot state tests')
        return None

def blocked_import(event,args):
    if event=='import'and args and str(args[0]).split('.')[0]in FORBIDDEN:
        DENIALS.append(str(args[0]))
        raise ImportError('backend excluded from snapshot state tests')

def raises(kind,fn):
    try:fn()
    except kind as error:return error
    raise AssertionError('required rejection did not occur')

def main():
    assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
    scenario,root_arg,result_arg=sys.argv[1:]
    root=Path(root_arg).resolve();runtime=root/'runtime'
    source_map=json.loads((root/'runtime_sources.json').read_bytes())
    assert len(source_map)==28
    def sources():
        actual={str(p.relative_to(runtime)):hashlib.sha256(p.read_bytes()).hexdigest()for p in runtime.rglob('*')if p.is_file()}
        assert actual==source_map
        assert not any(p.is_symlink()for p in runtime.rglob('*'))
    absent();sources();finder=DenyBackend();sys.meta_path.insert(0,finder);sys.addaudithook(blocked_import)
    assert not any(n in sys.modules for n in HELPERS)
    sys.path.insert(0,str(runtime))
    snapshot=importlib.import_module('qwen_admission_snapshot')
    packet_module=importlib.import_module('qwen_server_authority_packet')
    phase_module=importlib.import_module('qwen_server_admission')
    science_module=importlib.import_module('qwen_scientific_admission')
    file_module=importlib.import_module('qwen_file_contract')
    model_module=importlib.import_module('model_admission')
    key_module=importlib.import_module('keygen_authority')
    def source_binding():
        sources();absent();assert sys.meta_path[0]is finder
        for name in HELPERS:
            module=sys.modules[name];wanted=runtime/(name+'.py')
            assert Path(module.__file__)==wanted and Path(module.__spec__.origin)==wanted
    source_binding()
    public=file_module.PublicContract((('synthetic-public-marker',b'synthetic non-proof metadata'),))
    generation_model=model_module.ModelAdmission(public,b'synthetic generation manifest',b'synthetic generation report',b'synthetic generation outer',b'synthetic generation root',model_module.ExpectedModelPins(b'synthetic generation pins'))
    server_model=model_module.ModelAdmission(public,b'synthetic server manifest',b'synthetic server report',b'synthetic server outer',b'synthetic server root',model_module.ExpectedModelPins(b'synthetic server pins'))
    intent=key_module.ActualKeygenIntent(public,b'synthetic intent')
    launch=key_module.LaunchAdmission(intent,b'synthetic grant',generation_model,key_module.ExpectedLaunchPins(b'synthetic launch pins'))
    ready=key_module.GeneratedKeyReadiness(launch,b'synthetic public manifest',b'synthetic key report',b'synthetic key outer',b'synthetic key root',key_module.ExpectedReadyPins(b'synthetic ready pins'))
    proofs=tuple((name,('synthetic non-proof '+name).encode())for name in sorted(science_module.PROOF_LINKS))
    assert len(proofs)==25
    scientific=science_module.ScientificAdmission(public,server_model,ready,b'synthetic request',b'synthetic upload',b'synthetic statement',proofs,science_module.ExpectedScientificPins(b'synthetic scientific pins'))
    phase=phase_module.ServerPhaseAdmission(public,server_model,ready,b'synthetic request',b'synthetic upload',b'synthetic phase grant',phase_module.ExpectedServerPhasePins(b'synthetic phase pins'),scientific)
    packet=packet_module.ServerPacket(public,b'synthetic packet','a'*64,phase,proofs,b'synthetic dependencies',b'synthetic role',b'synthetic resources',b'synthetic resource review')
    # Independent traversal observes the real dataclass graph. No check method
    # is used here and these marker bytes intentionally are not valid evidence.
    def graph_types(value,seen=None):
        if seen is None:seen=set()
        if type(value)in(str,bytes)or id(value)in seen:return set()
        seen.add(id(value))
        if type(value)is tuple:
            out=set()
            for child in value:out|=graph_types(child,seen)
            return out
        out={type(value).__module__+'.'+type(value).__name__}
        for child in object.__getattribute__(value,'__dict__').values():out|=graph_types(child,seen)
        return out
    expected_types={
        'qwen_server_authority_packet.ServerPacket','qwen_server_admission.ServerPhaseAdmission','qwen_server_admission.ExpectedServerPhasePins',
        'qwen_scientific_admission.ScientificAdmission','qwen_scientific_admission.ExpectedScientificPins','qwen_file_contract.PublicContract',
        'model_admission.ModelAdmission','model_admission.ExpectedModelPins','keygen_authority.ActualKeygenIntent',
        'keygen_authority.ExpectedLaunchPins','keygen_authority.LaunchAdmission','keygen_authority.ExpectedReadyPins','keygen_authority.GeneratedKeyReadiness'}
    assert graph_types(packet)==expected_types
    state={'full':0,'binding':0,'mode':None,'guard':None,'reentry':None,'replacement_calls':0}
    marker=RuntimeError('owned recording validation marker')
    restores=[]
    original_report=server_model.report_bytes
    def field(owner,name,value):
        before=object.__getattribute__(owner,'__dict__').get(name)
        existed=name in object.__getattribute__(owner,'__dict__')
        object.__setattr__(owner,name,value)
        restores.append(lambda:object.__setattr__(owner,name,before)if existed else object.__delattr__(owner,name))
    def recording_validator(value):
        assert value is packet
        state['full']+=1
        if state['mode']=='validator_failure':raise marker
        if state['mode']=='validator_mutation':field(server_model,'report_bytes',b'changed recording bytes')
        return value
    def binding():
        source_binding();state['binding']+=1
        if state['mode']=='initial_callback_failure'and state['binding']==1:raise marker
        if state['mode']=='final_callback_failure'and state['binding']==2:raise marker
        if state['mode']in('reentry_uncaught','reentry_caught')and state['guard']is not None:
            if state['mode']=='reentry_uncaught':state['guard'].check(packet)
            else:state['reentry']=raises(PermissionError,lambda:state['guard'].check(packet))
    original_packet_check=packet_module.ServerPacket.check
    original_saved_check=snapshot._FULL_PACKET_CHECK
    assert original_packet_check is original_saved_check
    packet_module.ServerPacket.check=recording_validator
    snapshot._FULL_PACKET_CHECK=recording_validator
    def sticky(guard):
        assert snapshot._REGISTRY[id(guard)][-1]=='failed'
        before=state['binding'];raises(PermissionError,lambda:guard.check(packet))
        assert state['binding']==before and snapshot._REGISTRY[id(guard)][-1]=='failed'
    def attempt_retained(with_graph):
        attempt=snapshot._CAPTURE_ATTEMPT
        assert attempt[0]is packet
        assert (attempt[2]is not None)is with_graph
        if with_graph:
            row=next(row for row in attempt[2]if row[0]is server_model)
            assert dict(row[3])['report_bytes']is original_report
        raises(PermissionError,lambda:snapshot.capture_admitted(packet,check_bindings=binding))
        assert snapshot._CAPTURE_ATTEMPT is attempt
    result=None
    try:
        if scenario in('validator_failure','initial_callback_failure','final_callback_failure','validator_mutation','mutable_graph'):
            state['mode']=scenario
            if scenario=='mutable_graph':field(public,'blobs',list(public.blobs))
            error=raises((RuntimeError,ValueError,TypeError),lambda:snapshot.capture_admitted(packet,check_bindings=binding))
            if scenario.endswith('_failure'):assert error is marker
            assert state['full']==(0 if scenario in('initial_callback_failure','mutable_graph')else 1)
            attempt_retained(scenario not in('initial_callback_failure','mutable_graph'))
            assert not snapshot._REGISTRY
        else:
            guard=snapshot.capture_admitted(packet,check_bindings=binding);state['guard']=guard
            assert state['full']==1 and state['binding']==3 and snapshot._CAPTURE_ATTEMPT is None
            if scenario=='clean_graph':
                assert guard.check(packet)is packet and guard.check(packet)is packet
                assert state['full']==1 and state['binding']==5
                guard.retire(packet)
                assert id(guard)not in snapshot._REGISTRY and state['binding']==6
                raises(PermissionError,lambda:guard.check(packet));raises(PermissionError,lambda:guard.retire(packet))
                raises(PermissionError,lambda:snapshot.capture_admitted(packet,check_bindings=binding))
            elif scenario=='wrong_thread':
                errors=[]
                def other():
                    try:guard.check(packet)
                    except BaseException as error:errors.append(error)
                thread=threading.Thread(target=other);thread.start();thread.join()
                assert len(errors)==1 and isinstance(errors[0],PermissionError)
                sticky(guard)
            else:
                checked=packet
                if scenario=='byte_replace':
                    replacement=bytes(bytearray(original_report));assert replacement==original_report and replacement is not original_report
                    field(server_model,'report_bytes',replacement)
                elif scenario=='tuple_replace':
                    replacement=tuple(list(public.blobs));assert replacement==public.blobs and replacement is not public.blobs
                    field(public,'blobs',replacement)
                elif scenario=='extra_field':field(phase.expected_pins,'unexpected',b'extra')
                elif scenario=='dict_replace':
                    original=server_model.__dict__;object.__setattr__(server_model,'__dict__',dict(original))
                    restores.append(lambda:object.__setattr__(server_model,'__dict__',original))
                elif scenario in('helper_code','helper_export'):
                    original=file_module.hash_value
                    def replacement(*args,**kwargs):state['replacement_calls']+=1;raise AssertionError('replacement helper invoked')
                    if scenario=='helper_code':
                        # Use a zero-closure body, matching the original helper.
                        def zero_closure(*args,**kwargs):raise AssertionError('replacement helper invoked')
                        old_code=original.__code__;original.__code__=zero_closure.__code__
                        restores.append(lambda:setattr(original,'__code__',old_code))
                    else:
                        file_module.hash_value=replacement;restores.append(lambda:setattr(file_module,'hash_value',original))
                elif scenario=='keyword_default':
                    fn=file_module.ciphertext_binding;kw=fn.__kwdefaults__;old=kw['actual_scale'];kw['actual_scale']='changed default'
                    restores.append(lambda:kw.__setitem__('actual_scale',old))
                elif scenario=='descriptor':
                    original=model_module.ModelAdmission.model_id
                    def replacement(owner):state['replacement_calls']+=1;raise AssertionError('replacement descriptor invoked')
                    model_module.ModelAdmission.model_id=property(replacement)
                    restores.append(lambda:setattr(model_module.ModelAdmission,'model_id',original))
                elif scenario=='wrong_packet':checked=packet_module.ServerPacket(**packet.__dict__);assert checked is not packet
                elif scenario in('reentry_uncaught','reentry_caught'):state['mode']=scenario
                else:raise AssertionError('unselected scenario')
                raises((ValueError,PermissionError),lambda:guard.check(checked))
                state['mode']=None
                while restores:restores.pop()()
                assert state['replacement_calls']==0
                if scenario=='reentry_caught':assert isinstance(state['reentry'],PermissionError)
                sticky(guard)
            assert state['full']==1
        result={'scenario':scenario,'assertions_completed':True,'graph_type_count':13,'recording_full_validation_seam':True,
            'scientific_admission_tested':False,'actual_proof_instances':False,'backend_imported':False,'backend_denials':list(DENIALS),
            'recorded_full_check_calls':state['full'],'binding_callback_calls':state['binding'],
            'pid':os.getpid(),'pgid':os.getpgrp()}
    finally:
        try:
            while restores:restores.pop()()
        finally:
            packet_module.ServerPacket.check=original_packet_check
            snapshot._FULL_PACKET_CHECK=original_saved_check
            assert packet_module.ServerPacket.check is original_packet_check
            assert snapshot._FULL_PACKET_CHECK is original_saved_check
            absent();sources()
    assert result is not None
    result['original_validator_identities_restored']=True
    with Path(result_arg).open('x')as stream:json.dump(result,stream,sort_keys=True);stream.write('\n')

if __name__=='__main__':main()
