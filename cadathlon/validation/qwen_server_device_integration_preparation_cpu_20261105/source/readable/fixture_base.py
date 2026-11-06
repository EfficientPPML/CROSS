"""Shared declared Python-only seams. Real policy/source/owner functions remain.

All input documents and owner objects are synthetic. No candidate import occurs
until the isolated, reviewed child deliberately constructs Fixture in a later run.
"""
from contextlib import contextmanager
import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace
from fixture_documents import build as build_documents
_ABSENT_DESCRIPTOR=object()

def _saved_attribute(owner,name):
    return vars(owner).get(name,_ABSENT_DESCRIPTOR)if isinstance(owner,type)else getattr(owner,name)

def _restore_attribute(owner,name,value):
    if isinstance(owner,type):
        if value is _ABSENT_DESCRIPTOR:
            delattr(owner,name);assert name not in vars(owner)
        else:
            setattr(owner,name,value);assert vars(owner)[name]is value
    else:
        object.__setattr__(owner,name,value)if not isinstance(owner,ModuleType)else setattr(owner,name,value)
        assert getattr(owner,name)is value

def expect(kinds,call,message=None):
    try:call()
    except kinds as error:
        if message is not None:assert message in str(error),(message,type(error).__name__,str(error))
        return error
    raise AssertionError('expected rejection did not occur')

class Fixture:
    def __init__(self,test_root,runtime,entry,expected,blobs,owned,loader,temporary,case):
        self.test_root,self.root,self.entry,self.expected,self.blobs,self.owned=test_root,runtime,entry,expected,blobs,owned
        self.loader,self.temporary,self.case=loader,temporary,case
        self.events=[];self.seams=[];self.restores=[];self.mutations=[];self.details={}
        self.counts={n:0 for n in('policy','resource','logical','issue','native','open','hook','sample','full','phase','pins','ids','digest','core','replacement','host','sync','evaluate','publish')}
        self.process=self.module('qwen_server_process');self.policy=self.module('qwen_device_policy')
        self.capacity=self.module('qwen_device_capacity');self.resource=self.module('qwen_server_resource')
        self.device_packet=self.module('qwen_server_device_packet');self.owner_module=self.module('qwen_server_device_owner')
        self.original_device_close=self.owner_module.close_default_scope
        self.native=self.module('qwen_server_native_authority');self.phase_module=self.module('qwen_server_admission')
        self.guard=self.module('qwen_admission_snapshot');self.key_module=self.module('keygen_authority')
        self.key_contract=self.module('qwen_key_contract');self.file_module=self.module('qwen_file_contract')
        self.model_module=self.module('model_admission');self.science_module=self.module('qwen_scientific_admission')
        self.packet_module=self.module('qwen_server_authority_packet');self.backend_module=self.module('qwen_server_backend')
        self.canonical,self.sha=self.process.canonical,self.process.sha
        self.phase_rows={};self.runtime=None;self.snapshot_check=None;self.snapshot_retire=None
        self.gate=SimpleNamespace(core_enabled=False);self.retained={};self.device=None;self.jax=None
        self.mode='clean';self.marker=ValueError('owned synthetic operation marker');self.caught=None
        self.stats_sample_target=None
        self._profile_before=sys.getprofile();assert self._profile_before is None
        self._profile_codes={self.policy.authenticate.__code__:'policy',self.resource.authenticate_policy.__code__:'resource',
            self.resource.logical_facts.__code__:'logical',self.capacity.issue_initialization.__code__:'issue',
            self.phase_module.ServerPhaseAdmission.expected_ids.__code__:'ids',self.phase_module.ServerPhaseAdmission.input_digest.fget.__code__:'digest'}
        for name in self.policy.HOOKS:self._profile_codes[getattr(self.process,name).__code__]='hook'
        def profile(frame,event,arg):
            if event=='call'and frame.f_code in self._profile_codes:self.counts[self._profile_codes[frame.f_code]]+=1
        self._profile=profile;sys.setprofile(profile)
        def phase_check(value):
            assert id(value)in self.phase_rows and self.phase_rows[id(value)]['phase']is value
            self.counts['phase']+=1;return value
        def pins_check(value):
            matches=[b for b in self.phase_rows.values()if b['phase'].expected_pins is value]
            assert len(matches)==1;self.counts['pins']+=1;return matches[0]['phase_pins']
        self.patch(self.phase_module.ServerPhaseAdmission,'check',phase_check,'P/S: phase complete-validator recording')
        self.patch(self.phase_module.ExpectedServerPhasePins,'check',pins_check,'P/S: expected phase pin validator recording')
    def module(self,name):return importlib.import_module(name)
    expect=staticmethod(expect)
    def event(self,name):self.events.append(name)
    def patch(self,owner,name,value,label):
        original=getattr(owner,name)
        # Preserve descriptors themselves. Saving getattr(class, classmethod)
        # would restore a bound-method object instead of the original descriptor.
        saved=_saved_attribute(owner,name)
        self.restores.append((owner,name,saved));setattr(owner,name,value)
        self.seams.append(dict(owner=getattr(owner,'__name__',type(owner).__name__),export=name,label=label))
        return original
    def mutate(self,owner,name,value):
        original=getattr(owner,name);self.mutations.append((owner,name,_saved_attribute(owner,name)))
        object.__setattr__(owner,name,value)if not isinstance(owner,(ModuleType,type))else setattr(owner,name,value)
        return original
    def restore_mutations(self):
        while self.mutations:
            owner,name,value=self.mutations.pop()
            _restore_attribute(owner,name,value)
    def new_packet(self,*,flags=None,mutate_document=None,request_value=None,grant_extra=None):
        b=build_documents(self.root,self.expected,self.phase_module,self.test_root/'recording_jax.py',
            flags=flags,mutate_document=mutate_document,request_value=request_value,grant_extra=grant_extra)
        b['raw']=self.canonical(dict(schema='qwen-server-device-packet-v1',expected_pins=b['pins'].decode(),documents={n:raw.decode()for n,raw in b['blobs']}))
        b['digest']=self.sha(b['raw']);self.phase_rows[id(b['phase'])]=b;return b
    def packet_auth(self,bundle,*,raw=None,digest=None,original_policies=None):
        originals=bundle['documents']if original_policies is None else original_policies
        return self.device_packet.authenticate_packet(bundle['phase'],bundle['raw']if raw is None else raw,
            expected_sha256=bundle['digest']if digest is None else digest,
            resource_policy=originals['resource_policy'],resource_review=originals['resource_review'],dependency_policy=originals['dependency_policy'])
    def rebuild_request(self,request,*,mutate_document=None):
        """Pre-capture metadata rebuild; keep the same thirteen-graph phase."""
        assert self.snapshot_check is None and self.device is None
        old_phase=self.phase
        extra={k:v for k,v in self.bundle['grant'].items()if k not in('source_root','source_manifest_sha256','source_archive_sha256','policies')}
        bundle=self.new_packet(request_value=request,grant_extra=extra,mutate_document=mutate_document)
        temporary_phase=bundle['phase']
        for name in('request_bytes','grant_bytes','expected_pins'):object.__setattr__(old_phase,name,getattr(temporary_phase,name))
        del self.phase_rows[id(temporary_phase)]
        bundle['phase']=old_phase;bundle['phase_pins']['input_digest']=self.input_digest
        self.phase_rows[id(old_phase)]=bundle;self.bundle=bundle;self.pin_values=bundle['phase_pins']
        docs=bundle['documents'];self.dep=json.loads(docs['dependency_policy']);self.res=json.loads(docs['resource_policy'])
        for name,key in(('dependency_policy','dependency_policy'),('resource_policy','resource_policy'),('resource_review','resource_review')):
            object.__setattr__(self.packet,name,docs[key])
        object.__setattr__(self.phase.scientific,'request_bytes',self.phase.request_bytes)
        self.device_packet_value=self.packet_auth(bundle);self.resource_pins=self.process.resource_pins(self.phase)
    def no_device_work(self):
        assert not self.capacity._ISSUED and self.capacity._REGISTRY=={}and self.owner_module._OWNERS=={}
        assert all(self.counts[n]==0 for n in('issue','native','open','hook','sample'))
        assert 'jax'not in sys.modules
    def source_check(self):
        assert self.entry.snapshot(self.expected)==self.blobs
        self.entry.origins(self.expected,core_enabled=self.gate.core_enabled)
        self.loader.check()
    def _graph(self):
        self.ids=self.key_contract.ExpectedIds('1'*64,'2'*64,self.key_contract.KEYGEN,'3'*64,'4'*64,'5'*64,None)
        self.ids.check('server');self.input_digest='6'*64
        self.identity={n:getattr(self.ids,n)for n in('context_id','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256')}
        request=dict(public=self.identity,input={'input_tile_count':1})
        parent=self.owned/'outputs';parent.mkdir(mode=0o700)
        info=parent.stat();parent_state=dict(path=str(parent),device=info.st_dev,inode=info.st_ino,mode=0o700)
        # Fabricated public-key descriptor identities are consumed only by the
        # L/E exact fake-FD/stat recording. No key root exists or is observed.
        key_parent_info=self.owned.lstat()
        key_parent=dict(path=str(self.owned),device=key_parent_info.st_dev,inode=key_parent_info.st_ino,mode=key_parent_info.st_mode&0o7777)
        key_store=dict(path=str(self.owned/'unopened-public-keys'),device=201,inode=999,mode=0o700,parent=key_parent,
            manifest=dict(name='public_manifest.json',sha256='a'*64,state=dict(device=201,inode=998,size=1,mtime_ns=9,ctime_ns=10,mode=0o600)))
        grant_extra=dict(role='full28_server',public_key_store=key_store,
            input=dict(root=dict(path=str(self.owned/'unopened-input')),request_manifest_sha256='b'*64),
            outputs={n:dict(path=str(parent/n),parent=parent_state)for n in('response','evidence')})
        self.bundle=self.new_packet(request_value=request,grant_extra=grant_extra)
        self.phase=self.bundle['phase'];self.pin_values=self.bundle['phase_pins'];self.pin_values['input_digest']=self.input_digest
        self.public=object.__new__(self.file_module.PublicContract)
        object.__setattr__(self.public,'blobs',(('synthetic marker',b'unadmitted metadata; no model proof'),))
        generation=self.model_module.ModelAdmission(self.public,b'gen manifest',b'gen report',b'gen outer',b'gen root',self.model_module.ExpectedModelPins(b'gen pins'))
        self.model=self.model_module.ModelAdmission(self.public,b'model manifest',b'model report',b'model outer',b'model root',self.model_module.ExpectedModelPins(b'model pins'))
        intent=self.key_module.ActualKeygenIntent(self.public,b'synthetic intent')
        launch=self.key_module.LaunchAdmission(intent,b'synthetic grant',generation,self.key_module.ExpectedLaunchPins(b'launch pins'))
        self.ready=self.key_module.GeneratedKeyReadiness(launch,self.canonical({'synthetic_public_manifest':True}),b'key report',b'key outer',b'key root',self.key_module.ExpectedReadyPins(b'ready pins'))
        proofs=tuple((n,('not-proof '+n).encode())for n in sorted(self.science_module.PROOF_LINKS))
        scientific=self.science_module.ScientificAdmission(self.public,self.model,self.ready,self.phase.request_bytes,b'upload',b'statement',proofs,self.science_module.ExpectedScientificPins(b'science pins'))
        for name,value in dict(public=self.public,model=self.model,readiness=self.ready,request_manifest_bytes=b'upload',scientific=scientific).items():object.__setattr__(self.phase,name,value)
        docs=self.bundle['documents'];self.dep=json.loads(docs['dependency_policy']);self.res=json.loads(docs['resource_policy'])
        self.packet=self.packet_module.ServerPacket(self.public,b'packet','9'*64,self.phase,proofs,docs['dependency_policy'],b'role',docs['resource_policy'],docs['resource_review'])
        def full(value):
            assert value is self.packet;self.counts['full']+=1;self.event('full.packet');return value
        def ready_ids(value,*,role):assert value is self.ready and role=='server';return self.ids
        def environment(dep):assert dep is self.dep
        self.patch(self.packet_module.ServerPacket,'check',full,'S: complete scientific ServerPacket check recording')
        self.patch(self.guard,'_FULL_PACKET_CHECK',full,'S: same captured full validator recording')
        self.patch(self.key_module.GeneratedKeyReadiness,'expected_ids',ready_ids,'S: generated-key ID recording; original phase derivation retained')
        self.patch(self.process,'exact_environment',environment,'S: environment recording; no installed backend/host observation')
        self.device_packet_value=self.packet_auth(self.bundle)
        self.resource_pins=self.process.resource_pins(self.phase)
        self._audit_and_core()
    def _audit_and_core(self):
        f=self
        self.core_modules={n:SimpleNamespace()for n in('he_params','key_gen','ckks_ctx','polynomial')}
        class Core:
            def check(self):f.event('core.check')
            def load(self,names):f.counts['core']+=1;f.event('core.load');return {n:f.core_modules[n]for n in names}
            def retain(self,name,fields):return SimpleNamespace(check=lambda:f.event('core.lease:'+name))
        class Audit:
            def __init__(self,*args,**kwargs):
                self.enabled=False;self.active=True;self.keys_enabled=False;self.bodies_enabled=False
                self.initialization_enabled=False;self.native_binding=None;self.original_open=os.open;self.wrapper=os.open;self.denials=[]
                f.audit=self
            def check(self):
                f.event('audit.check')
                if self.denials:raise PermissionError('recorded sticky audit denial')
            def install(self):f.event('audit.install')
            def enable_initialization(self,owner,raw):
                f.event('audit.initialization');owner.require_initialization_scope()
                self.native_binding=(owner,raw);self.initialization_enabled=True
            def enable_public_keys(self,runtime):
                f.event('keys.enter');runtime.require_public_key_read_scope();runtime.check();self.keys_enabled=True;runtime.check();f.event('keys.enabled')
            def enable_server_bodies(self,runtime):
                f.event('bodies.enter');runtime.require_server_body_scope();runtime.check();self.bodies_enabled=True;runtime.check();f.event('bodies.enabled')
            def close(self):f.event('audit.close');self.active=False
        self.Core,self.Audit=Core,Audit;self.core=Core();self.audit=Audit()
        self.audit_methods={n:getattr(self.audit,n)for n in('check','install','close','enable_initialization','enable_public_keys','enable_server_bodies')}
        self.ctx=SimpleNamespace(secret_key=None,_param_cache=SimpleNamespace(secret_key=None),parameters={'secret_key':None})
        self.key_public=SimpleNamespace(raw=self.ready.public_manifest_bytes,expected=self.ids,role='server')
        self.keys=SimpleNamespace(public=self.key_public,private=None,check=lambda:self.identity)
        self.files=self.module('qwen_key_files_admission').FilesAdmission(self.public,self.keys)
        self.handle=SimpleNamespace(admission=self.files,public=self.key_public,public_manifest=json.loads(self.ready.public_manifest_bytes),ctx=self.ctx)
    def _install_jax(self):
        assert 'jax'not in sys.modules
        path=self.test_root/'recording_jax.py';spec=importlib.util.spec_from_file_location('jax',path)
        value=importlib.util.module_from_spec(spec);self.temporary['jax']=value;sys.modules['jax']=value
        # SourceFileLoader probes a bytecode cache even with -B. Compile the
        # exact selected recording source directly so the real audit observes
        # only the declared public source, retaining the honest file/spec.
        raw=path.read_bytes();assert self.sha(raw)==self.loader.full_map['recording_jax.py']
        exec(compile(raw,str(path),'exec',dont_inherit=True),value.__dict__)
        self.jax=value;value.MODE=self.mode;value.EVENTS=self.events
        self._profile_codes[value.Device.memory_stats.__code__]='sample'
        self.seams.append(dict(owner='sys.modules',export='jax',label='owned recording_jax with honest file/spec/source hash; no backend import'))
    def native_seams(self,*,acquire=True,synchronize=True,sync_denial_after_load=False):
        original_native_sync=self.native.synchronize
        def native_acquire(phase,blobs):
            assert phase is self.phase and blobs is self.device_packet_value.policy.blobs
            self.counts['native']+=1;self.event('native.acquire');return self.native.NativeLease()
        def native_check(lease,phase,blobs):
            assert type(lease)is self.native.NativeLease and phase is self.phase
            self.event('native.check')
        def native_sync(lease,stage,context,owners):
            assert type(lease)is self.native.NativeLease;self.counts['sync']+=1;self.event('native.sync:'+stage)
            if sync_denial_after_load and context is not None:return original_native_sync(lease,stage,context,owners)
        if acquire:
            self.patch(self.native,'acquire',native_acquire,'O: explicit unbacked NativeLease construction')
            self.patch(self.native,'check',native_check,'O: lease continuity recording')
        if synchronize:self.patch(self.native,'synchronize',native_sync,'O: unreturned-work synchronization recording')
        def core(root,blobs,expected,*,gate):
            assert root==str(self.root)and blobs is self.blobs and expected is self.expected
            self.event('core.install');self._install_jax();return self.core
        def installed(dep):assert dep is self.dep;return{'synthetic-installed':True}
        def origins(dep,root):assert dep is self.dep and root==str(self.root);return{'jax':{'file':str(self.test_root/'recording_jax.py')}}
        def host(phase,policy,review,*,expected_pins,after_import):
            assert phase is self.phase;self.counts['host']+=1;self.event('host:'+str(after_import))
            return dict(after_import=after_import,borrowed_body_reads=False,capacity_reserved=False,device_capacity_checked=False,resource_admission_complete=False,synthetic_recording_only=True)
        for module,name,fn,label in((self.owner_module,'install_core',core,'O: core construction/load recording'),
             (self.owner_module,'hash_installed',installed,'O: installed source hash recording'),
             (self.owner_module,'imported_origins',origins,'O: installed backend origins recording'),
             (self.owner_module,'check_host_capacity',host,'O: host capacity observation recording')):
            self.patch(module,name,fn,label)
        self.installed_recording,self.origins_recording=installed,origins
    def capture_snapshot(self):
        self.snapshot_check,self.snapshot_retire=self.process._capture_snapshot(self.packet,str(self.root),self.expected,self.blobs,self.dep,self.gate,self.entry.snapshot,self.entry.origins,self.retained)
        self.snapshot_check()
    def device_bindings(self):
        if self.snapshot_check is not None:self.snapshot_check()
        self.entry.origins(self.expected,core_enabled=self.gate.core_enabled);self.audit.check()
        if self.mode=='caught_reentry'and self.device is not None and self.jax is not None and not self.caught and self.row()['pending']is not None:
            self.mode='clean';self.caught=expect((RuntimeError,PermissionError),lambda:self.device.post_keys(self.handle))
        if self.mode=='mutate_stats'and self.jax is not None and self.stats_sample_target is not None and self.jax.SAMPLES>=self.stats_sample_target:
            self.jax.STATS.update(bytes_in_use=999999,bytes_limit=1);self.mode='clean';self.event('stats.mutated')
    def full_boundary(self):
        self.source_check();self.packet.check()
        if self.snapshot_check is not None:self.snapshot_check()
        if self.runtime is not None:self.runtime.check()
    def prepare_owner(self,*,acquire=True,synchronize=True,capture=True,before_capture=None,sync_denial_after_load=False):
        self._graph();self.native_seams(acquire=acquire,synchronize=synchronize,sync_denial_after_load=sync_denial_after_load)
        if before_capture is not None:before_capture(self)
        if capture:self.capture_snapshot()
        def binding_check():return self.device_bindings()
        def full_boundary():return self.full_boundary()
        self.device=self.owner_module.create(self.device_packet_value,root=str(self.root),expected=self.expected,source_blobs=self.blobs,
            dep=self.dep,gate=self.gate,binding_check=binding_check,full_boundary=full_boundary,audit=self.audit,audit_methods=self.audit_methods,resource_pins=self.resource_pins)
        return self.device
    def enable_numpy(self):
        assert 'numpy'not in sys.modules
        module=ModuleType('numpy');module.__spec__=None
        self.temporary['numpy']=module;sys.modules['numpy']=module;self.gate.core_enabled=True
        self.seams.append(dict(owner='sys.modules',export='numpy',label='empty source-owned definition-only name; no array or backend origin'))
        return module
    def activate_runtime(self):
        self.audit.enabled=True
        ids=self.phase.expected_ids();digest=self.phase.input_digest
        self.derived=(ids,self.process._expected_ids_bytes(ids),digest)
        def source_check():return self.full_boundary()
        def capacity_check(stage):return self.device.require_stage(stage)
        self.runtime,self.runtime_retire=self.process._runtime(self.packet,self.core,self.audit,self.audit_methods,
            source_check,capacity_check,self.files,self.snapshot_check,self.derived,self.device)
        return self.runtime
    def row(self):return self.owner_module._OWNERS[id(self.device)]
    def start(self):
        result=self.device.start();assert result[0]is self.core
        assert self.row()['stage']=='post_init';return result
    def array(self,label,device=None):return self.jax.Array(label,device)
    def ciphertext(self,label):return SimpleNamespace(polynomial=self.array(label+'.payload'),moduli_array=self.array(label+'.moduli'))
    def register(self):
        self.audit.keys_enabled=True;self.transfers=tuple(self.array('setup'+str(i))for i in range(4))
        self.device.register_loaded(self.handle,self.transfers);self.device.post_keys(self.handle)
        return self.transfers
    def evaluate_boundary(self,prepared=None):
        self.audit.bodies_enabled=True
        self.prepared=(self.ciphertext('prepared'),)if prepared is None else prepared
        self.full_owner=SimpleNamespace(synthetic=True)
        self.device.before_evaluation(self.ctx,self.handle,self.full_owner,self.prepared)
    def clean_owner(self):
        self.handle.ctx=None;self.device.terminal();rows=self.device.retire_guard()
        self.owner_module.close_default_scope(self.device)
        assert len(rows)==4 and self.row()['mode']=='retired'and self.row()['default_exit_attempted']
        if self.snapshot_retire is not None:self.snapshot_retire()
    def finish_runtime_owner(self):
        row=self.row();assert row['handle']is not None and row['handle'].ctx is None
        self.device.terminal()
        if self.runtime is not None:self.runtime_retire();self.event('runtime.retired')
        self.device.retire_guard();self.event('device.retired')
        self.original_device_close(self.device)
        if self.snapshot_retire is not None:self.snapshot_retire();self.event('snapshot.retired')
    def sticky_owner(self):
        row=self.row();assert row['failure']is not None and row['quarantined']
        self.restore_mutations();self.mode='clean'
        if self.jax is not None:
            self.jax.Array=self.jax.ORIGINAL_ARRAY;self.jax.MODE='clean';self.jax.STATS={'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':90}
        before=(len(self.events),self.counts['native'],self.counts['core'],0 if self.jax is None else self.jax.SAMPLES)
        expect(PermissionError,self.device.check)
        assert before==(len(self.events),self.counts['native'],self.counts['core'],0 if self.jax is None else self.jax.SAMPLES)
        assert self.owner_module._OWNERS[id(self.device)]is row
    def close(self):
        self.restore_mutations()
        if self.device is not None and self.row()['mode']!='retired':
            row=self.row()
            self.original_device_close(self.device,primary=row['failure']or RuntimeError('partial synthetic owner intentionally retained until child exit'))
        while self.restores:
            owner,name,value=self.restores.pop();_restore_attribute(owner,name,value)
        for name,value in list(self.temporary.items()):
            assert sys.modules.get(name)is value;del sys.modules[name]
        self.temporary.clear();sys.setprofile(self._profile_before)
