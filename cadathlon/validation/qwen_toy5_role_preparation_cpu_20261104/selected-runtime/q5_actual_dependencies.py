"""Source-fixed genuine Q5 call bindings behind the default-denied runtime.

No backend import occurs at module import. Setup generation, key readers,
context construction and code execution require the exact RuntimeAdmission;
that authority is intentionally unavailable in this source preparation.
"""
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path
import importlib,os,stat
from q5_key_contract import canonical,parse,bindings,KEYGEN
from q5_key_producer import same,CORE_MANIFEST_SHA256
from q5_key_admission import FilesAdmission

def _runtime(value):
    from q5_actual_process import RuntimeAdmission
    if type(value)is not RuntimeAdmission:raise PermissionError('exact Q5 runtime authority required')
    value.check();return value

def _same(a,b):
    return(getattr(a,'__self__',None)is getattr(b,'__self__',None)and getattr(a,'__func__',a)is getattr(b,'__func__',b))

def _root(info,path):return dict(path=str(path),device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode),uid=info.st_uid)

@dataclass(frozen=True)
class ActualDependencies:
    backend:object
    security_check:object
    runtime_factory:object
    directory_factory:object
    map_array:object
    def require_key_scope(self,admission,client_private):
        if type(self)is not ActualDependencies or type(self.backend)is not ActualBackend or self.backend.dependencies is not self:
            raise PermissionError('original source-fixed Q5 loader dependencies required')
        self.backend.check();self.backend.runtime.require_key_scope(admission,client_private)

@dataclass(frozen=True)
class ActualGenerationDependencies:
    backend:object
    source_check:object
    security_check:object
    capacity_check:object
    store_factory:object
    gen_pke_pair:object
    gen_evaluation_key:object
    gen_rotation_key:object
    shield:object
    synthetic_only:bool=False
    def check(self):
        if type(self)is not ActualGenerationDependencies or type(self.backend)is not ActualBackend or self.synthetic_only is not False:
            raise PermissionError('exact actual setup generation dependencies required')
        self.backend.check();self.backend.runtime.require_generation_scope()
        if self.backend.generation is not self:raise ValueError('original setup generation dependencies changed')
        for name,original in self.backend._generation_callbacks:
            if not _same(getattr(self,name),original):raise ValueError('original setup generation callback changed')

class ActualBackend:
    def __init__(self,runtime):
        _runtime(runtime);runtime.require_backend_scope()
        self.runtime=runtime;self.phase=runtime.phase;self.profile=runtime.phase.profile;self.core=runtime.core
        modules=self.core.load(('he_params','key_gen','ckks_ctx','polynomial','finite_field','ntt_mm'))
        self.he_params=modules['he_params'];self.key_gen=modules['key_gen'];self.ckks=modules['ckks_ctx'];self.polynomial=modules['polynomial']
        self.finite_field=modules['finite_field'];self.ntt_mm=modules['ntt_mm']
        self.identities=(self.core.retain('he_params',('DEFAULT_SIGMA','SECRET_DISTRIBUTION','SecurityTarget','qp_is_secure','assert_p_coverage')),
            self.core.retain('ckks_ctx',('CKKSContext','evict_context_caches')),
            self.core.retain('polynomial',('Polynomial','_POLYNOMIAL_REGISTRY')),
            self.core.retain('finite_field',('BarrettContext',)),
            self.core.retain('ntt_mm',('NTTCiphertextBarrettContext','ntt_ciphertext_context_for')))
        import jax
        import numpy as np
        from support import require_secure_execution
        from cleanup import cleanup_signals
        from q5_key_readers import ReadDirectory,map_array
        self.jax=jax;self.np=np;self.secure=require_secure_execution;self.shield=cleanup_signals
        self.encrypt_slots=self.ckks.CKKSContext.encrypt_slots;self.decrypt_slots=self.ckks.CKKSContext.decrypt_slots
        backend=self
        class GrantedDirectory(ReadDirectory):
            def __init__(owner,path,*,private=False):
                backend.check();expected=backend.runtime.expected_key_root(private=private)
                if str(path)!=expected['path']or Path(path).resolve()!=Path(path):raise ValueError('source/grant key root differs')
                owner._expected=canonical(expected);owner._original_parent=Path(path).parent
                named=Path(path).lstat()
                if not stat.S_ISDIR(named.st_mode)or _root(named,path)!=expected or named.st_uid!=os.geteuid()or stat.S_IMODE(named.st_mode)!=0o700:
                    raise ValueError('original granted key root differs before open')
                super().__init__(path,private=private)
            def check(owner):
                super().check();expected=parse(owner._expected,8192)
                if expected!=backend.runtime.expected_key_root(private=owner.private):raise ValueError('original key-root expectation changed')
                for info in(os.fstat(owner.fd),owner.path.lstat()):
                    if not stat.S_ISDIR(info.st_mode)or _root(info,owner.path)!=expected or info.st_uid!=os.geteuid():raise ValueError('original held/named key root differs')
        def security_check(p):
            backend.check();same(p,backend.profile.parameters,'original Q5 physical profile differs');h=backend.he_params
            if p['sigma']!=h.DEFAULT_SIGMA or p['secret_distribution']!=h.SECRET_DISTRIBUTION:raise ValueError('original security distribution differs')
            if not h.qp_is_secure(p['q_towers'],p['p_towers'],p['degree'],h.SecurityTarget(128,'classical')):raise ValueError('actual full QP fails128bit')
            h.assert_p_coverage(p['q_towers'],p['p_towers'],p['dnum']);backend.check()
        def runtime_factory():
            backend.check()
            return SimpleNamespace(Context=backend.ckks.CKKSContext,jax=backend.jax,ckks=backend.ckks,polynomial=backend.polynomial,
                key_modules={'key_gen':backend.key_gen,'jaxite_word.key_gen':backend.key_gen})
        def bound_map(owner,name,description,shape,moduli):
            backend.check();owner.check();return map_array(owner,name,description,shape,moduli)
        self.dependencies=ActualDependencies(self,security_check,runtime_factory,GrantedDirectory,bound_map)
        self._callbacks=(security_check,runtime_factory,GrantedDirectory,bound_map)
        self.generation=None;self._generation_callbacks=()
        self._fixed=(runtime,self.phase,self.profile,self.core,self.he_params,self.key_gen,self.ckks,self.polynomial,self.finite_field,self.ntt_mm,
            self.identities,self.jax,self.np,self.secure,self.shield,self.dependencies,self.encrypt_slots,self.decrypt_slots)
        self.check()
    def check(self):
        _runtime(self.runtime)
        current=(self.runtime,self.phase,self.profile,self.core,self.he_params,self.key_gen,self.ckks,self.polynomial,self.finite_field,self.ntt_mm,
            self.identities,self.jax,self.np,self.secure,self.shield,self.dependencies,self.encrypt_slots,self.decrypt_slots)
        if any(a is not b for a,b in zip(current,self._fixed)):raise ValueError('original Q5 backend binding changed')
        self.core.check()
        for binding in self.identities:binding.check()
        for name,original in zip(('security_check','runtime_factory','directory_factory','map_array'),self._callbacks):
            if getattr(self.dependencies,name)is not original:raise ValueError('original loader factory changed')
        # Server codecs are deliberately patched by the original loader before
        # context construction. Client roles retain the original unbound ABI.
        if self.runtime.role!='server'and(self.ckks.CKKSContext.encrypt_slots is not self.encrypt_slots or self.ckks.CKKSContext.decrypt_slots is not self.decrypt_slots):
            raise ValueError('original default client codec changed')
        return self
    def generation_dependencies(self,intent):
        self.check();self.runtime.require_generation_scope(intent)
        if self.generation is not None:raise ValueError('one setup generation dependency binding required')
        from q5_key_store import KeyStore
        backend=self
        def source_check(value):
            backend.check();backend.runtime.require_generation_scope(value);v,_=value.check()
            return dict(source_manifest_sha256=v['generation_source_manifest_sha256'],core_source_manifest_sha256=CORE_MANIFEST_SHA256,keygen_source_sha256=KEYGEN,source_unchanged=True,synthetic_only=False)
        def security(p):
            backend.dependencies.security_check(p)
            return dict(qp_secure=True,p_coverage=True,sigma=p['sigma'],secret_distribution=p['secret_distribution'],synthetic_only=False)
        def capacity(v):
            backend.runtime.require_generation_capacity(v)
            return dict(public_root=v['public_root'],private_root=v['private_root'],fresh_disjoint_roots=True,storage_capacity=True,descriptor_capacity=True,synthetic_only=False)
        def store(path,specs,*,expected_parent,private=False,shield):
            backend.check();backend.runtime.require_generation_scope(intent)
            return KeyStore(path,specs,expected_parent=expected_parent,private=private,shield=shield)
        values=(source_check,security,capacity,store,self.key_gen.gen_pke_pair,self.key_gen.gen_evaluation_key,self.key_gen.gen_rotation_key,self.shield)
        self.generation=ActualGenerationDependencies(self,*values)
        self._generation_callbacks=tuple(zip(('source_check','security_check','capacity_check','store_factory','gen_pke_pair','gen_evaluation_key','gen_rotation_key','shield'),values))
        return self.generation
    def arithmetic(self):
        self.check();self.runtime.require_server_body_scope()
        table=(('BoundedArithmetic','encrypted_bounded_arithmetic'),('BoundedPackedTransformer','encrypted_bounded_transformer'),
               ('BoundedSecureRefresh','encrypted_refresh'),('PackedPairLayout','encrypted_packed_transformer'),
               ('PolynomialApproximation','encrypted_polynomial'),('enable_scoped_ntt_subsets','encrypted_ntt'))
        result={name:getattr(importlib.import_module(module),name)for name,module in table}
        self.runtime.check_common_definitions(result);self.check();return result
    def client_handle(self,handle):
        self.check();self.runtime.check_loaded_handle(handle)
        if self.runtime.role=='server':raise PermissionError('server cannot acquire a private decoder adapter')
        from q5_ciphertext_owner import CiphertextOwner
        from q5_checkpoint_contract import DEFAULT_DECODER_SHA256
        backend=self;ctx=handle.ctx;admission=handle.admission
        def check():
            backend.check();backend.runtime.check_loaded_handle(handle)
            if handle.ctx is not ctx or handle.admission is not admission:raise ValueError('original borrowed client key handle changed')
        def owner(payload,row):check();return CiphertextOwner(backend.np,backend,ctx,payload,row,check=check)
        return SimpleNamespace(ctx=ctx,check=check,new_ciphertext_owner=owner,decrypt_slots=self.decrypt_slots,
                               decoder_source_sha256=DEFAULT_DECODER_SHA256,key_owner=handle)
