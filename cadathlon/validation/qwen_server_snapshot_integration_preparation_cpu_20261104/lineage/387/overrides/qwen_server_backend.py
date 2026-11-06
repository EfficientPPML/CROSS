"""Source-fixed full-model server dependencies; no public execution entry.

The prospective process must supply its exact runtime and phase admission.
This module cannot grant a body read or choose a resource policy. Context/key
ownership remains in the original loader, including the seven-codec denial.
"""
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json
import os
import stat
from client_role_common import raw_object
from qwen_file_contract import canonical,same

ENGINE_NAMES=('enable','arithmetic','linear','layout','refresh','contract','parameters','owner','secure','materialize')


def _runtime(value):
    from qwen_server_process import RuntimeAdmission
    if type(value)is not RuntimeAdmission:raise PermissionError('exact source-fixed server runtime required')
    value.check()
    return value


def _file_state(info):
    return dict(device=info.st_dev,inode=info.st_ino,size=info.st_size,
        mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns,mode=stat.S_IMODE(info.st_mode))


@dataclass(frozen=True)
class ActualServerDependencies:
    backend:object
    security_check:object
    runtime_factory:object
    directory_factory:object
    map_array:object

    def check(self,admission,public_keys,client_private):
        if type(self)is not ActualServerDependencies or type(self.backend)is not ServerBackend:
            raise PermissionError('exact actual server dependency binding required')
        self.backend.check()
        self.backend.runtime.require_public_key_read_scope()
        if self.backend.dependencies is not self:raise ValueError('original actual server dependencies changed')
        from qwen_key_files_admission import FilesAdmission
        if type(admission)is not FilesAdmission:raise TypeError('complete file admission required before key reader')
        admission.check()
        phase=self.backend.phase;grant=raw_object(phase.grant_bytes,1<<16,'server phase grant')
        if client_private is not None or str(public_keys)!=grant['public_key_store']['path']:
            raise ValueError('exact granted public-only key path required')
        if (admission.public_contract is not phase.public
                or admission.keys.public.raw!=phase.readiness.public_manifest_bytes
                or admission.keys.public.expected!=phase.expected_ids()
                or admission.keys.public.role!='server' or admission.keys.private is not None):
            raise ValueError('keyless server file admission differs from generated readiness')
        same(admission.keys.check(),json.loads(phase.request_bytes)['public'],'actual server request identity differs')


@dataclass(frozen=True)
class ActualServerFactories:
    backend:object
    request_files:object
    parameter_files:object
    publisher:object
    engine_dependencies:object
    shield:object

    def check(self,admission,request,key_dependencies):
        if type(self)is not ActualServerFactories or type(self.backend)is not ServerBackend:
            raise PermissionError('exact actual server factories required')
        backend=self.backend;backend.check()
        if backend.factories is not self or key_dependencies is not backend.dependencies:
            raise ValueError('original server factory/dependency binding changed')
        if canonical(request)!=backend.phase.request_bytes:raise ValueError('actual server request differs from phase grant')
        grant=raw_object(backend.phase.grant_bytes,1<<16,'server phase grant')
        key_dependencies.check(admission,grant['public_key_store']['path'],None)
        return self


class ServerBackend:
    """One process-bound set of original loader and full-model callbacks."""
    def __init__(self,runtime):
        _runtime(runtime);runtime.require_public_key_read_scope()
        self.runtime=runtime;self.phase=runtime.phase;self.core=runtime.core
        from qwen_server_admission import ServerPhaseAdmission
        if type(self.phase)is not ServerPhaseAdmission:raise TypeError('exact complete server phase required')
        self.phase.check()
        modules=self.core.load(('he_params','key_gen','ckks_ctx','polynomial'))
        self.he_params=modules['he_params'];self.key_gen=modules['key_gen']
        self.ckks=modules['ckks_ctx'];self.polynomial=modules['polynomial']
        self.security_identity=self.core.retain('he_params',('DEFAULT_SIGMA','SECRET_DISTRIBUTION','SecurityTarget','qp_is_secure','assert_p_coverage'))
        self.context_identity=self.core.retain('ckks_ctx',('CKKSContext','evict_context_caches'))
        self.polynomial_identity=self.core.retain('polynomial',('Polynomial','_POLYNOMIAL_REGISTRY'))
        import jax
        from cleanup import cleanup_signals
        from qwen_key_readers import ReadDirectory,map_array
        from qwen_backend_composition import _runtime_dependencies
        self.jax=jax;self.shield=cleanup_signals
        self._base_directory=ReadDirectory;self._base_map_array=map_array
        self.engine=_runtime_dependencies()
        self._engine_callbacks=tuple(getattr(self.engine,n)for n in ENGINE_NAMES)
        runtime.check()
        phase=self.phase;backend=self
        store_bytes=canonical(raw_object(phase.grant_bytes,1<<16,'server phase grant')['public_key_store'])

        class GrantedPublicDirectory(ReadDirectory):
            def __init__(owner,path,*,private=False):
                backend.check();runtime.require_public_key_read_scope()
                expected=json.loads(store_bytes)
                if private is not False or str(path)!=expected['path']or Path(path).resolve()!=Path(path):
                    raise ValueError('exact granted public-only original key root required')
                owner.expected=store_bytes
                super().__init__(path,private=False)
            def check(owner):
                super().check()
                if owner.private is not False or owner.expected is not store_bytes:
                    raise ValueError('original server public key binding changed')
                expected=json.loads(store_bytes)
                for info in(os.fstat(owner.fd),owner.path.lstat()):
                    same(dict(path=str(owner.path),device=info.st_dev,inode=info.st_ino,mode=stat.S_IMODE(info.st_mode)),
                        {n:expected[n]for n in('path','device','inode','mode')},'independently retained public key root differs')
                    if info.st_uid!=os.geteuid():raise ValueError('retained public key root owner differs')
                parent=owner.path.parent.lstat()
                same(dict(path=str(owner.path.parent),device=parent.st_dev,inode=parent.st_ino,mode=stat.S_IMODE(parent.st_mode)),
                    expected['parent'],'independently retained public key parent differs')
                manifest=expected['manifest'];named=os.stat(manifest['name'],dir_fd=owner.fd,follow_symlinks=False)
                if not stat.S_ISREG(named.st_mode)or named.st_uid!=os.geteuid():raise ValueError('original public key manifest required')
                same(_file_state(named),manifest['state'],'retained public manifest identity differs before body')

        def security_check(parameters):
            backend.check();p=phase.public.profile();same(parameters,p,'actual loaded profile differs')
            h=backend.he_params
            if type(p['sigma'])is not float or p['sigma']!=h.DEFAULT_SIGMA or p['secret_distribution']!=h.SECRET_DISTRIBUTION:
                raise ValueError('fixed security distributions differ')
            if not h.qp_is_secure(p['q_towers'],p['p_towers'],p['degree'],h.SecurityTarget(128,'classical')):
                raise ValueError('actual full QP ring is insecure')
            h.assert_p_coverage(p['q_towers'],p['p_towers'],p['dnum']);backend.check()

        def runtime_factory():
            backend.check();runtime.require_public_key_read_scope()
            return SimpleNamespace(Context=backend.ckks.CKKSContext,jax=backend.jax,ckks=backend.ckks,
                polynomial=backend.polynomial,key_modules={'key_gen':backend.key_gen,'jaxite_word.key_gen':backend.key_gen})

        def bound_map_array(owner,name,description,shape,moduli):
            backend.check();runtime.require_public_key_read_scope();owner.check()
            return map_array(owner,name,description,shape,moduli)

        self.dependencies=ActualServerDependencies(self,security_check,runtime_factory,GrantedPublicDirectory,bound_map_array)
        self._dependency_callbacks=(security_check,runtime_factory,GrantedPublicDirectory,bound_map_array)

        def request_files(handle):
            backend.check();runtime.require_public_key_read_scope()
            backend.dependencies.check(handle.admission,json.loads(store_bytes)['path'],None)
            # The original loader has authenticated all335 public arrays and
            # constructed its keyless context before yielding this handle.
            runtime.admit_server_bodies(handle);runtime.require_server_body_scope()
            from qwen_file_lifecycle import RequestFiles
            grant=raw_object(phase.grant_bytes,1<<16,'server phase grant');row=grant['input']
            return RequestFiles(phase.public,json.loads(phase.request_bytes),phase.request_manifest_bytes,row['root']['path'],
                request_manifest_sha256=row['request_manifest_sha256'],expected_root=canonical(row['root']))

        def parameter_files(handle):
            backend.check();runtime.require_server_body_scope()
            from qwen_public_parameters import open_identity_bound_parameters
            return open_identity_bound_parameters(phase.public,phase.model.stores)

        def publisher(handle,input_digest):
            backend.check();runtime.require_server_body_scope()
            if input_digest!=phase.input_digest:raise ValueError('original admitted server input digest differs')
            from qwen_publisher import Publisher
            output=raw_object(phase.grant_bytes,1<<16,'server phase grant')['outputs']['response']
            return Publisher(phase.public,json.loads(phase.request_bytes),'response',output['path'],
                expected_parent=canonical(output['parent']),input_digest=input_digest,shield=backend.shield)

        def engine_dependencies(ctx):
            backend.check();runtime.require_server_body_scope()
            if ctx.secret_key is not None or ctx._param_cache.secret_key is not None or ctx.parameters.get('secret_key')is not None:
                raise ValueError('keyless server context required before engine factory')
            return backend.engine

        self.factories=ActualServerFactories(self,request_files,parameter_files,publisher,engine_dependencies,self.shield)
        self._factory_callbacks=(request_files,parameter_files,publisher,engine_dependencies,self.shield)
        self._fixed=(runtime,self.phase,self.core,self.he_params,self.key_gen,self.ckks,self.polynomial,
            self.security_identity,self.context_identity,self.polynomial_identity,self.jax,self.shield,
            self._base_directory,self._base_map_array,self.engine,self.dependencies,self.factories)
        self.check()

    def check(self):
        _runtime(self.runtime)
        current=(self.runtime,self.phase,self.core,self.he_params,self.key_gen,self.ckks,self.polynomial,
            self.security_identity,self.context_identity,self.polynomial_identity,self.jax,self.shield,
            self._base_directory,self._base_map_array,self.engine,self.dependencies,self.factories)
        if any(a is not b for a,b in zip(current,self._fixed)):raise ValueError('original server backend binding changed')
        self.phase.check();self.core.check()
        for retained in(self.security_identity,self.context_identity,self.polynomial_identity):retained.check()
        for name,original in zip(('security_check','runtime_factory','directory_factory','map_array'),self._dependency_callbacks):
            if getattr(self.dependencies,name)is not original:raise ValueError('original actual server key callback changed')
        for name,original in zip(('request_files','parameter_files','publisher','engine_dependencies','shield'),self._factory_callbacks):
            if getattr(self.factories,name)is not original:raise ValueError('original actual server stage callback changed')
        from qwen_backend_composition import Dependencies
        if type(self.engine)is not Dependencies:raise TypeError('exact original full-model dependencies required')
        for name,original in zip(ENGINE_NAMES,self._engine_callbacks):
            if getattr(self.engine,name)is not original:raise ValueError('original full-model callback changed')
        return self
