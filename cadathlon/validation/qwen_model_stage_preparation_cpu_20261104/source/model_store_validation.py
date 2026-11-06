"""Complete model-store visit/retirement through trusted recording factories.

No actual entry or backend/model factory is selected. This seam does not create
an admissible validation report, execution grant or independent terminal proof.
"""
from dataclasses import dataclass
from cleanup import cleanup_signals
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from qwen_file_contract import PublicContract,canonical,sha,same
from qwen_public_parse import parse
from qwen_public_parameters import ParameterCatalog
from model_admission import Aggregate,ExpectedStoreIdentity,MODEL_ID,AGGREGATE_SHA256,CATALOG_SHA256,_json,_source_map

_QUARANTINE=[]


def create_actual_model_validator(*args,**kwargs):
    raise PermissionError('actual current-store validation has no admitted source/grant entry')


def _accessor(fn):return(getattr(fn,'__self__',None),getattr(fn,'__func__',fn))
def _same_accessor(a,b):return all(x is y for x,y in zip(_accessor(a),_accessor(b)))
def _catalog(c):return dict(matrices=parse(c.matrices),gammas=parse(c.gammas),gamma_container_sha256=c.gamma_container_sha256,provenance=parse(c.provenance))


@dataclass(frozen=True)
class RecordingValidationIntent:
    aggregate:Aggregate
    stores:ExpectedStoreIdentity
    producer_source_map:bytes
    def check(self):
        if type(self)is not RecordingValidationIntent or type(self.aggregate)is not Aggregate or type(self.stores)is not ExpectedStoreIdentity:raise TypeError('exact immutable recording validation intent required')
        self.aggregate.check();self.stores.check()
        if self.stores.aggregate is not self.aggregate:raise ValueError('store intent belongs to another aggregate')
        source=_json(self.producer_source_map,1<<20,'recording producer source map');_source_map(source)
        return source
    @property
    def execution_authorized(self):return False


@dataclass(frozen=True)
class RecordingDependencies:
    """Trusted injected callbacks only, not capabilities or actual admission."""
    source_check:object
    parameter_factory:object
    shield:object=cleanup_signals
    synthetic_only:bool=True
    def check(self):
        if type(self)is not RecordingDependencies or self.synthetic_only is not True:raise PermissionError('explicit synthetic model validation dependencies required')
        if not all(callable(x)for x in(self.source_check,self.parameter_factory,self.shield)):raise TypeError('recording dependency callables required')


@dataclass(frozen=True)
class RecordingCompletion:
    raw:bytes
    @property
    def body_integrity_admitted(self):return False
    @property
    def execution_authorized(self):return False
    @property
    def terminal_ready(self):return False


def validate_store_for_recording(intent,dependencies):
    if type(intent)is not RecordingValidationIntent or type(dependencies)is not RecordingDependencies:raise TypeError('exact recording intent/dependencies required')
    source_map=intent.check();dependencies.check();aggregate=intent.aggregate;stores=intent.stores
    public=aggregate.public;original=(intent.aggregate,intent.stores,intent.producer_source_map,aggregate.raw,aggregate.public,public.blobs,stores.raw)
    source_check,factory,shield=dependencies.source_check,dependencies.parameter_factory,dependencies.shield
    # The existing selected catalog is rederived before even the factory seam.
    catalog=ParameterCatalog.from_public(public);catalog_bytes=canonical(_catalog(catalog))
    same(_catalog(catalog),aggregate.catalog,'selected catalog/public preview differs before acquisition')
    logical=parse(catalog.matrices);gammas=parse(catalog.gammas)
    physical={(row['family'],row['descriptor']['file'])for row in logical.values()}
    if len(logical)!=198 or len(physical)!=197 or len(gammas)!=113:raise ValueError('complete fixed aggregate catalog required')
    owner=None;owner_catalog=None;owner_catalog_state=None;callbacks={};primary=None;result=None;cleanup_report={}
    def guard():
        current=(intent.aggregate,intent.stores,intent.producer_source_map,aggregate.raw,aggregate.public,public.blobs,stores.raw)
        if any(a is not b for a,b in zip(current,original)):raise ValueError('original model validation intent changed')
        intent.check();dependencies.check()
        if any(not _same_accessor(a,b)for a,b in zip((dependencies.source_check,dependencies.parameter_factory,dependencies.shield),(source_check,factory,shield))):raise ValueError('original recording validation dependency changed')
        if owner is not None:
            for name,fn in callbacks.items():
                if not _same_accessor(getattr(owner,name,None),fn):raise ValueError('original parameter owner callback changed: '+name)
            if owner_catalog is not None:
                if owner.catalog is not owner_catalog or owner.expected_store is not stores:raise ValueError('original parameter owner admission changed')
                if(owner_catalog.matrices,owner_catalog.gammas,owner_catalog.gamma_container_sha256,owner_catalog.provenance)!=owner_catalog_state:raise ValueError('original parameter owner catalog changed')
    def source():
        guard();value=source_check(intent);guard()
        same(value,dict(source_hashes=source_map,source_manifest_sha256=sha(intent.producer_source_map),source_unchanged=True,synthetic_only=True),'recording source prerequisite differs')
    try:
        source()
        owner=factory(public,stores)
        # Capture close first; an incomplete later interface remains retained.
        callbacks['close']=owner.close
        for name in('check','matrix','gamma','readback'):callbacks[name]=getattr(owner,name)
        if not all(callable(fn)for fn in callbacks.values()):raise TypeError('complete parameter owner callbacks required')
        owner_catalog=owner.catalog
        if type(owner_catalog)is not ParameterCatalog:raise TypeError('existing exact ParameterCatalog required')
        owner_catalog.check();owner_catalog_state=(owner_catalog.matrices,owner_catalog.gammas,owner_catalog.gamma_container_sha256,owner_catalog.provenance)
        same(_catalog(owner_catalog),parse(catalog_bytes),'returned parameter catalog differs')
        guard();callbacks['check']();guard()
        # Discard each returned view immediately. The owner caches each physical
        # lease once; the tied embedding/head names share that existing lease.
        for name in sorted(logical):
            guard();callbacks['matrix'](name);guard();callbacks['check']();guard()
        if type(owner.matrices)is not dict or set(owner.matrices)!=physical:raise ValueError('all197 physical leases required after198 logical visits')
        for name in sorted(gammas):
            guard();callbacks['gamma'](name);guard();callbacks['check']();guard()
        if type(owner.gamma_values)is not dict or set(owner.gamma_values)!=set(gammas):raise ValueError('all113 gamma members required')
        callbacks['readback']();guard();source()
        result=dict(schema='qwen-model-store-recording-completion-v1',model_id=MODEL_ID,aggregate_manifest_sha256=AGGREGATE_SHA256,catalog_sha256=CATALOG_SHA256,expected_store_binding_sha256=sha(stores.raw),producer_source_manifest_sha256=sha(intent.producer_source_map),logical_matrix_calls=198,physical_descriptors=197,gamma_calls=113,cleanup_completed=False,source_postcheck_completed=False,synthetic_only=True,body_integrity_admitted=False,actual_execution_enabled=False,terminal_ready=False,independent_outer_receipt_sha256=None)
    except BaseException as error:primary=protect_body(error)
    finally:
        original_owner=owner;original_callbacks=dict(callbacks);final=Finalizer(cleanup_report,primary)
        def retire():
            final.attempt('original model ownership before cleanup',guard)
            readback=original_callbacks.get('readback')
            if readback is not None:final.attempt('original model final readback',readback)
            final.attempt('original model ownership after readback',guard)
            close=original_callbacks.get('close')
            if close is not None:final.attempt('original model backing close',lambda:close(primary=final.selected()))
            final.attempt('original model ownership after close',guard)
            final.attempt('model terminal source postcheck',source)
        final.shielded(shield,retire)
        chosen=final.selected()
        if chosen is not None:
            _QUARANTINE.append((intent,dependencies,original,source_check,factory,shield,original_owner,original_callbacks,owner_catalog,owner_catalog_state,result,cleanup_report,chosen));raise chosen
    if result is None:raise RuntimeError('recording model validation did not complete')
    result['cleanup_completed']=True;result['source_postcheck_completed']=True
    return RecordingCompletion(canonical(result))
