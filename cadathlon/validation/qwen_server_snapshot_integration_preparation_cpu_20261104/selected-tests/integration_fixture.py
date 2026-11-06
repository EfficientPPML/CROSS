"""Explicit unadmitted graph/owner seams for selected real391 functions.

No arrays, backend contexts, existing input files, or actual policy authority.
The child installs the real entry definitions under an explicit __main__ seam.
"""
from contextlib import contextmanager
import importlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace


def raises(kinds,call):
    try:call()
    except kinds as error:return error
    raise AssertionError('required rejection did not occur')


class Fixture:
    def __init__(self,root,entry,expected,blobs,owned,*,mode='direct',scenario='runtime_hot_clean'):
        self.root,self.entry,self.expected,self.blobs,self.owned=root,entry,expected,blobs,owned
        self.mode,self.scenario=mode,scenario
        self.process=importlib.import_module('qwen_server_process')
        self.backend=importlib.import_module('qwen_server_backend')
        self.guard=importlib.import_module('qwen_admission_snapshot')
        self.phase_module=importlib.import_module('qwen_server_admission')
        self.key_module=importlib.import_module('keygen_authority')
        self.file_module=importlib.import_module('qwen_file_contract')
        self.model_module=importlib.import_module('model_admission')
        self.science_module=importlib.import_module('qwen_scientific_admission')
        self.packet_module=importlib.import_module('qwen_server_authority_packet')
        self.key_contract=importlib.import_module('qwen_key_contract')
        self.cleanup_origin=importlib.import_module('cleanup_origin')
        self.restores=[];self.mutations=[];self.temporary={};self.seams=[];self.events=[]
        self.counts=dict(full=0,phase=0,pins=0,ready_ids=0,ids=0,digest=0,environment=0,core_load=0,
            map=0,owner_check=0,capacity=0,stage=0,evidence=0,guard_retire=0,replacement=0)
        self.runtime=None;self.runtime_retire=None;self.retained={};self.snapshot_check=None;self.snapshot_retire=None
        self.environment_mode=None;self.audit_closed=False;self.body_error=ValueError('owned synthetic body failure')
        self.cleanup_error=OSError('owned synthetic close failure');self.late_error=RuntimeError('owned synthetic late retirement failure')
        self.caught_nested=None;self.reported_error=None;self.publication={}
        self.dep={'synthetic_environment':'no actual configuration'}
        self.gate=SimpleNamespace(core_enabled=False)
        self._graph()
        self._admission_seams()
        self._owners()
        self.profile_original=sys.getprofile();assert self.profile_original is None
        self.ids_code=self.phase_module.ServerPhaseAdmission.expected_ids.__code__
        self.digest_code=self.phase_module.ServerPhaseAdmission.input_digest.fget.__code__
        self.retire_code=self.guard.AdmissionSnapshot.retire.__code__
        def observe(frame,event,arg):
            if event=='call':
                if frame.f_code is self.ids_code:self.counts['ids']+=1
                elif frame.f_code is self.digest_code:self.counts['digest']+=1
                elif frame.f_code is self.retire_code:self.counts['guard_retire']+=1;self.events.append('guard.retire')
        sys.setprofile(observe)

    def patch(self,owner,name,value,label):
        before=getattr(owner,name);self.restores.append((owner,name,before));setattr(owner,name,value);self.seams.append(label)
        return before

    def mutate(self,owner,name,value):
        before=getattr(owner,name);self.mutations.append((owner,name,before))
        object.__setattr__(owner,name,value)if not isinstance(owner,(ModuleType,type))else setattr(owner,name,value)
        return before

    def restore_mutations(self):
        while self.mutations:
            owner,name,value=self.mutations.pop()
            object.__setattr__(owner,name,value)if not isinstance(owner,(ModuleType,type))else setattr(owner,name,value)

    def native_names(self,*,process_api=False):
        assert not self.temporary and 'jax'not in sys.modules and 'numpy'not in sys.modules
        for name in('numpy','jax'):
            module=ModuleType(name);module.__spec__=None
            self.temporary[name]=module;sys.modules[name]=module
        self.gate.core_enabled=True
        if process_api:
            self.temporary['jax'].config=SimpleNamespace(jax_enable_x64=True)
            self.temporary['jax'].devices=lambda:[SimpleNamespace(platform='tpu')]
        self.seams.append('source-owned temporary numpy/jax names; no installed imports or arrays')

    def _graph(self):
        canonical=self.process.canonical;sha=self.process.sha
        self.ids=self.key_contract.ExpectedIds('1'*64,'2'*64,self.key_contract.KEYGEN,'3'*64,'4'*64,'5'*64,None)
        self.ids.check('server');self.input_digest='6'*64
        self.public=object.__new__(self.file_module.PublicContract)
        object.__setattr__(self.public,'blobs',(('synthetic marker',b'not a scientific or model proof'),))
        self.generation_model=self.model_module.ModelAdmission(self.public,b'gen manifest',b'gen report',b'gen outer',b'gen root',self.model_module.ExpectedModelPins(b'gen pins'))
        self.model=self.model_module.ModelAdmission(self.public,b'server manifest',b'server report',b'server outer',b'server root',self.model_module.ExpectedModelPins(b'server pins'))
        intent=self.key_module.ActualKeygenIntent(self.public,b'synthetic intent')
        launch=self.key_module.LaunchAdmission(intent,b'synthetic grant',self.generation_model,self.key_module.ExpectedLaunchPins(b'launch pins'))
        self.ready=self.key_module.GeneratedKeyReadiness(launch,canonical({'synthetic_public_manifest':True}),b'key report',b'key outer',b'key root',self.key_module.ExpectedReadyPins(b'ready pins'))
        self.identity={n:getattr(self.ids,n)for n in('context_id','model_id','producer_source_manifest_sha256','consumer_source_manifest_sha256')}
        self.request=canonical({'public':self.identity,'input':{'synthetic':True}})
        self.grant=dict(source_root=str(self.root),public_key_store=dict(path=str(self.owned/'unopened-public-keys')),
            outputs=dict(evidence=dict(path=str(self.owned/'recorded-evidence'),parent={'synthetic':True}),response=dict(path=str(self.owned/'unopened-response'),parent={'synthetic':True})))
        policies={n:'7'*64 for n in('dependency_policy_sha256','resource_policy_sha256','resource_review_sha256')}
        self.pin_values=dict(source_manifest_sha256=sha(canonical(self.expected)),source_archive_sha256='8'*64,
            phase_grant_sha256=sha(canonical(self.grant)),input_digest=self.input_digest,policies=policies)
        proofs=tuple((name,('not-proof '+name).encode())for name in sorted(self.science_module.PROOF_LINKS))
        scientific=self.science_module.ScientificAdmission(self.public,self.model,self.ready,self.request,b'upload',b'statement',proofs,self.science_module.ExpectedScientificPins(b'science pins'))
        self.phase=self.phase_module.ServerPhaseAdmission(self.public,self.model,self.ready,self.request,b'upload',canonical(self.grant),self.phase_module.ExpectedServerPhasePins(b'phase pins'),scientific)
        self.packet=self.packet_module.ServerPacket(self.public,b'packet','9'*64,self.phase,proofs,b'dep',b'role',b'resource',b'resource review')
        def types(value,seen):
            if type(value)in(str,bytes)or id(value)in seen:return set()
            seen.add(id(value))
            if type(value)is tuple:
                return set().union(*(types(child,seen)for child in value))
            return{type(value)}|set().union(*(types(child,seen)for child in object.__getattribute__(value,'__dict__').values()))
        assert types(self.packet,set())=={cls for cls,fields in self.guard._FIELDS}
        self.seams.append('PublicContract constructor bypass; thirteen real admission types with unadmitted marker bytes')

    def _admission_seams(self):
        def full(value):
            assert value is self.packet;self.counts['full']+=1;self.events.append('full.packet');return value
        def phase(value):
            assert value is self.phase;self.counts['phase']+=1;return value
        def pins(value):
            assert value is self.phase.expected_pins;self.counts['pins']+=1;return self.pin_values
        def ready_ids(value,*,role):
            assert value is self.ready and role=='server';self.counts['ready_ids']+=1;return self.ids
        def environment(dep):
            assert dep is self.dep;self.counts['environment']+=1
            if self.environment_mode=='reenter'and self.runtime is not None:
                self.environment_mode=None
                self.caught_nested=raises((ValueError,PermissionError),self.runtime.check)
            if self.scenario=='process_late_guard_retirement_failure'and self.audit_closed:raise self.late_error
        self.patch(self.packet_module.ServerPacket,'check',full,'ServerPacket.check recording full validator')
        self.patch(self.guard,'_FULL_PACKET_CHECK',full,'same recording guard full-validator reference')
        self.patch(self.phase_module.ServerPhaseAdmission,'check',phase,'phase validation recording')
        self.patch(self.phase_module.ExpectedServerPhasePins,'check',pins,'phase-pin validation recording')
        self.patch(self.key_module.GeneratedKeyReadiness,'expected_ids',ready_ids,'readiness ID derivation recording; phase method/property original')
        self.patch(self.process,'exact_environment',environment,'environment binding recording; no host observation')

    def _owners(self):
        fixture=self
        class Core:
            def check(owner):fixture.events.append('core.check')
            def load(owner,names):
                fixture.counts['core_load']+=1;fixture.events.append('core.load')
                return {name:SimpleNamespace()for name in names}
            def retain(owner,name,fields):
                return SimpleNamespace(check=lambda:fixture.events.append('lease.check:'+name))
        class Audit:
            def __init__(owner,*args,**kwargs):
                owner.enabled=fixture.mode!='process';owner.active=True;owner.keys_enabled=False;owner.bodies_enabled=False
                owner.original_open=os.open;owner.wrapper=os.open;fixture.audit=owner
            def check(owner):fixture.events.append('audit.check')
            def install(owner):fixture.events.append('audit.install')
            def enable_public_keys(owner,value):
                fixture.events.append('keys.enter');value.require_public_key_read_scope();value.check()
                if fixture.scenario=='runtime_caught_transition':
                    fixture.caught_nested=raises(ValueError,value.allow_public_key_reads)
                owner.keys_enabled=True;value.check();fixture.events.append('keys.enabled')
            def enable_server_bodies(owner,value):
                fixture.events.append('bodies.enter');value.require_server_body_scope();value.check()
                owner.bodies_enabled=True;value.check();fixture.events.append('bodies.enabled')
            def close(owner):
                fixture.events.append('audit.close');owner.active=False;fixture.audit_closed=True
        self.Core=Core;self.Audit=Audit;self.core=Core();self.audit=Audit()
        self.key_public=SimpleNamespace(raw=self.ready.public_manifest_bytes,expected=self.ids,role='server')
        self.keys=SimpleNamespace(public=self.key_public,private=None,check=lambda:self.identity)
        file_module=importlib.import_module('qwen_key_files_admission')
        self.files=file_module.FilesAdmission(self.public,self.keys)
        self.ctx=SimpleNamespace(secret_key=None,_param_cache=SimpleNamespace(secret_key=None),parameters={'secret_key':None})
        self.handle=SimpleNamespace(admission=self.files,public=self.key_public,public_manifest=json.loads(self.ready.public_manifest_bytes),ctx=self.ctx)

    def capacity(self):self.counts['capacity']+=1;self.events.append('recorded.capacity')

    def full_boundary(self):
        self.events.append('boundary.begin');self.snapshot_check()
        assert self.entry.snapshot(self.expected)==self.blobs
        self.entry.origins(self.expected,core_enabled=self.gate.core_enabled)
        self.packet.check();self.snapshot_check()
        if self.runtime is not None:self.runtime.check()
        self.events.append('boundary.end')

    def capture_runtime(self):
        self.snapshot_check,self.snapshot_retire=self.process._capture_snapshot(self.packet,str(self.root),self.expected,self.blobs,self.dep,self.gate,self.entry.snapshot,self.entry.origins,self.retained)
        self.snapshot_check();ids=self.phase.expected_ids();digest=self.phase.input_digest
        self.derived=(ids,self.process._expected_ids_bytes(ids),digest);self.snapshot_check()
        callbacks={name:getattr(self.audit,name)for name in('check','install','close','enable_public_keys','enable_server_bodies')}
        # Match the actual process's plain-function callback contract. These
        # only forward to the same fixture methods; runtime requirements and
        # every case assertion remain unchanged.
        def source_check():return self.full_boundary()
        def capacity_check():return self.capacity()
        self.runtime,self.runtime_retire=self.process._runtime(self.packet,self.core,self.audit,callbacks,source_check,capacity_check,self.files,self.snapshot_check,self.derived)
        return self.runtime

    def backend_seams(self):
        self.native_names()
        readers=importlib.import_module('qwen_key_readers')
        composition=importlib.import_module('qwen_backend_composition')
        def mapping(owner,name,description,shape,moduli):
            self.counts['map']+=1;self.events.append('map.call');self.last_map=(owner,name,description,shape,moduli);return self.map_result
        def forbidden_engine(*args,**kwargs):raise AssertionError('unselected numerical engine callback')
        self.engine=composition.Dependencies(*(forbidden_engine for _ in self.backend.ENGINE_NAMES))
        def engine():return self.engine
        self.map_result=object();self.last_map=None
        self.patch(readers,'map_array',mapping,'actual reader module recording map export; no file reads')
        self.patch(composition,'_runtime_dependencies',engine,'actual composition module recording factory; real Dependencies dataclass')

    def build_backend(self):
        self.backend_instance=self.backend.ServerBackend(self.runtime);return self.backend_instance

    def forward_map(self):
        fixture=self
        class Owner:
            def check(owner):fixture.counts['owner_check']+=1;fixture.events.append('map.owner.check')
        owner=Owner();name='synthetic-only.npy';description={'synthetic':True};shape=(1,2);moduli=(17,)
        before=self.counts['owner_check'];start=len(self.events)
        result=self.backend_instance.dependencies.map_array(owner,name,description,shape,moduli)
        assert result is self.map_result and self.counts['owner_check']==before+1
        assert all(a is b for a,b in zip(self.last_map,(owner,name,description,shape,moduli),strict=True))
        assert self.events[start:].index('map.owner.check')<self.events[start:].index('map.call')

    def sticky(self,route):
        before=(self.counts['map'],self.counts['core_load'],self.counts['replacement'],tuple(self.events).count('keys.enabled'),tuple(self.events).count('bodies.enabled'))
        self.restore_mutations();self.environment_mode=None
        raises((PermissionError,ValueError),route)
        after=(self.counts['map'],self.counts['core_load'],self.counts['replacement'],tuple(self.events).count('keys.enabled'),tuple(self.events).count('bodies.enabled'))
        assert after==before and self.process._RUNTIME_CONTROLS[id(self.runtime)][0]is self.runtime
        assert self.retained['packet']is self.packet and self.retained['guard']is not None

    def clean_retire(self):
        self.handle.ctx=None;self.runtime_retire();self.events.append('runtime.retired')
        self.snapshot_retire();self.events.append('snapshot.retired')
        assert self.guard._REGISTRY=={}and self.process._RUNTIME_CONTROLS=={}

    def process_seams(self):
        """Full run/Finalizer bodies, with explicit pre-capture recordings."""
        assert self.mode=='process'
        fixture=self;p=self.process
        self.proof_path=self.owned/'tiny-public-proof.json'
        self.proof_bytes=p.canonical({'synthetic_public_observation':True})
        self.proof_path.write_bytes(self.proof_bytes)
        self.records={str(self.proof_path):(p.sha(self.proof_bytes),1024,self.proof_bytes)}
        self.args=SimpleNamespace(archive_sha256='8'*64,packet_sha256='9'*64)
        self.raw_map=p.canonical(self.expected)
        def admit(args,raw_map,expected,blobs):
            assert args is fixture.args and raw_map is fixture.raw_map and expected is fixture.expected and blobs is fixture.blobs
            fixture.events.append('recording.admit')
            return fixture.packet,object(),fixture.dep,object(),fixture.records,(),(),{'synthetic_memory':True}
        def host(*args,**kwargs):
            fixture.events.append('recorded.host');return {'after_import':kwargs['after_import'],'synthetic_only':True}
        def capacity(*args,**kwargs):fixture.capacity()
        def installed(dep):
            assert dep is fixture.dep;fixture.events.append('recorded.installed');return {'synthetic':True}
        def origins(dep,root):
            assert dep is fixture.dep and root==str(fixture.root);return {'synthetic_origins':True}
        def authenticate_public(raw,profile,ids,*,role):
            assert raw is fixture.ready.public_manifest_bytes and ids is fixture.ids and role=='server'
            return fixture.key_public
        def bind_files(public,keys,identity):
            assert public is fixture.public and keys is fixture.key_public and identity==p.canonical(fixture.identity)
            return fixture.files
        class Evidence:
            def __init__(owner,path,*,expected_parent):
                fixture.counts['evidence']+=1;fixture.evidence=owner;fixture.events.append('evidence.create')
                assert path==fixture.grant['outputs']['evidence']['path']
                assert expected_parent==p.canonical(fixture.grant['outputs']['evidence']['parent'])
                owner.closed=False
            def check(owner):
                assert not owner.closed;fixture.events.append('evidence.check')
            def write(owner,name,raw):
                owner.check();assert name in('report.json','failure.json')and type(raw)is bytes
                fixture.publication[name]=raw;fixture.events.append('evidence.write:'+name)
            def close(owner,*,primary=None):
                fixture.events.append('evidence.close');assert not owner.closed;owner.closed=True
                if fixture.scenario in('process_evidence_close_failure','process_body_and_cleanup_failure'):
                    raise fixture.cleanup_error
        @contextmanager
        def termination():
            fixture.events.append('termination.enter')
            try:yield
            finally:fixture.events.append('termination.exit')
        @contextmanager
        def shield():
            fixture.events.append('shield.enter')
            try:yield
            finally:fixture.events.append('shield.exit')
        for name,value,label in(
            ('admit',admit,'complete admission recording; no public proof authentication'),
            ('check_host_capacity',host,'host resource observation recording; no probes'),
            ('hash_installed',installed,'installed dependency hash recording'),
            ('imported_origins',origins,'installed backend origin recording; local entry.origins remains real'),
            ('AccessScope',self.Audit,'audit transition/close recording; no syscall wrapper or body capability'),
            ('Output',Evidence,'evidence owner recording; no output FD/byte integrity claim'),
            ('authenticate_public',authenticate_public,'public key binding recording'),
            ('bind_files',bind_files,'FilesAdmission construction recording'),
            ('termination_signals',termination,'termination signal-scope recording'),
            ('cleanup_signals',shield,'cleanup signal-scope recording')):
            self.patch(p,name,value,label)
        core_module=importlib.import_module('qwen_backend_identity')
        def install_core(root,blobs,expected,*,gate):
            assert root==str(fixture.root)and blobs is fixture.blobs and expected is fixture.expected and gate is fixture.gate
            fixture.events.append('core.install');return fixture.core
        self.patch(core_module,'install_core',install_core,'core install recording; no alias finder or runtime imports')
        if self.scenario=='process_default_capacity_denial':
            # The actual source-fixed complete-capacity callback is untouched.
            # No native name or stage module is installed for this case.
            self.seams.append('default complete capacity denial exact and unmodified')
            return
        self.native_names(process_api=True)
        self.patch(p,'require_complete_capacity',capacity,'complete capacity positive recording only; no factual capacity admission')
        stage_module=importlib.import_module('qwen_stage_composition')
        class Backend:
            def __init__(owner,runtime):
                fixture.runtime=runtime;runtime.check();owner.dependencies=object();owner.factories=object()
                fixture.recording_backend=owner;fixture.events.append('recorded.backend')
        def stage(files,public_keys,request,*,key_dependencies,factories):
            fixture.counts['stage']+=1;fixture.events.append('loader.enter')
            assert files is fixture.files and public_keys==fixture.grant['public_key_store']['path']
            assert key_dependencies is fixture.recording_backend.dependencies and factories is fixture.recording_backend.factories
            assert request==json.loads(fixture.request)
            fixture.runtime.admit_server_bodies(fixture.handle)
            try:
                fixture.runtime.require_server_body_scope();fixture.events.append('recorded.stage')
                if fixture.scenario=='process_body_and_cleanup_failure':raise fixture.body_error
                if fixture.scenario=='process_public_proof_changed':
                    before=fixture.counts['full'];fixture.runtime.check();assert fixture.counts['full']==before
                    fixture.proof_path.write_bytes(p.canonical({'synthetic_public_observation':'changed after hot check'}))
                return p.canonical({'synthetic_response_only':True})
            finally:
                fixture.handle.ctx=None;fixture.events.append('loader.context.retired')
        self.patch(self.backend,'ServerBackend',Backend,'process orchestration backend recording; real constructor tested separately')
        self.patch(stage_module,'_server_for_stage',stage,'loader/stage/context retirement recording; no reader/engine execution')

    def execute_process(self):
        def invoke():
            try:return self.process.run(self.args,self.raw_map,self.expected,self.blobs,self.gate,self.entry.snapshot,self.entry.origins)
            except BaseException as error:self.reported_error=error;raise
        return self.entry.terminal_status(invoke)

    def close(self):
        self.restore_mutations();sys.setprofile(self.profile_original)
        while self.restores:
            owner,name,value=self.restores.pop();setattr(owner,name,value);assert getattr(owner,name)is value
        for name,module in self.temporary.items():
            assert sys.modules.get(name)is module;del sys.modules[name]
        assert sys.getprofile()is self.profile_original
