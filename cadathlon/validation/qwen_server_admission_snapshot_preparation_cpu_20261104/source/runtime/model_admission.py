"""Pure fixed-model evidence binding; no archive, filesystem or NumPy access.

External expected pins are trusted caller arguments, never supplied by a report.
A returned binding is model-integrity evidence, not a body-read/execution grant.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from qwen_file_contract import PublicContract,canonical,sha,hash_value,same
from qwen_public_parse import parse
from public_source_pins import PUBLIC_PINS

AGGREGATE_SHA256='18189e5f2f2de3a4939c6afca19d79822465d917caf06c4f6e36668bd37c1f55'
MODEL_ID='ee3daf8b07141134466d02a6e1c141365ac4e0f0117d62f887c416a050b04bfb'
CATALOG_SHA256='6faa4630169f055b853a62a64088952e2ead48dc2ace5b61b6aa438c0106a0c0'
ROOT_FIELDS={'path','device','inode','mode'}
STATE_FIELDS={'device','inode','size','mtime_ns','ctime_ns','mode'}
REPORT_FIELDS={'schema','mode','passed','complete','error_type','model_id','aggregate_manifest_sha256','catalog_sha256','execution_grant_sha256','source_manifest_sha256','source_archive_sha256','source_hashes_before','source_hashes_after','process','stores','counts','checks','scope','audit_denials'}
CHECKS=('public_provenance_authenticated','complete_catalog_rederived','all_matrix_headers_extents_hashes_finite','all_gamma_members_headers_hashes_finite','original_held_identity_unchanged','readback_completed','source_postcheck_completed','cleanup_completed','signals_restored','borrowed_stores_retained')
SCOPE_FALSE=('new_export_performed','model_forward','backend_context_constructed','keys_read','ciphertexts_read','token_inputs_read','coefficient_selection','scientific_calibration_pass_claimed')


def create_actual_model_admission(*args,**kwargs):
    raise PermissionError('actual model-stage role admission is not implemented')


def _object(value,fields,label):
    if type(value)is not dict or set(value)!=set(fields):raise ValueError('exact '+label+' fields required')
    return value


def _json(raw,maximum,label):
    if type(raw)is not bytes or not 0<len(raw)<=maximum:raise ValueError('bounded immutable '+label+' bytes required')
    value=parse(raw)
    if type(value)is not dict or canonical(value)!=raw:raise ValueError('canonical '+label+' object required')
    return value


def _integer(value,minimum,label,maximum=None):
    if type(value)is not int or value<minimum or(maximum is not None and value>maximum):raise ValueError('exact bounded '+label+' integer required')
    return value


def _path(value):
    if type(value)is not str or not 1<len(value)<=4096 or value.startswith('//')or '\x00'in value or '\\'in value:raise ValueError('canonical absolute store path required')
    p=PurePosixPath(value)
    if not p.is_absolute()or str(p)!=value or '..'in p.parts or p==p.parent:raise ValueError('canonical non-root store path required')
    return value


def _flat(value):
    if type(value)is not str or not 0<len(value)<=255 or value in('.','..')or any(c in value for c in('/','\\','\x00')):raise ValueError('flat ordinary filename required')
    return value


def _source_map(value):
    if type(value)is not dict or not 0<len(value)<=512:raise ValueError('bounded explicit source map required')
    for name,digest in value.items():
        if type(name)is not str or not 0<len(name)<=512 or '\\'in name or '\x00'in name:raise ValueError('ordinary relative source name required')
        p=PurePosixPath(name)
        if p.is_absolute()or str(p)!=name or '..'in p.parts or name in('.','..'):raise ValueError('canonical relative source name required')
        hash_value(digest)
    return value


@dataclass(frozen=True)
class Aggregate:
    public:PublicContract
    raw:bytes
    def check(self):
        if type(self)is not Aggregate or type(self.public)is not PublicContract:raise TypeError('exact aggregate/public contract required')
        values=self.public.check();v=_json(self.raw,1<<20,'aggregate manifest')
        if sha(self.raw)!=AGGREGATE_SHA256:raise ValueError('fixed reviewed aggregate manifest differs')
        if v['model_id']!=MODEL_ID or sha(canonical(v['content']))!=MODEL_ID:raise ValueError('content-derived model identity differs')
        if v['catalog_sha256']!=CATALOG_SHA256 or sha(canonical(v['catalog']))!=CATALOG_SHA256:raise ValueError('catalog identity differs')
        same(v['public_source_sha256'],{n:sha(raw)for n,raw in values.items()},'aggregate/public metadata pins differ')
        return v
    @property
    def catalog(self):return self.check()['catalog']
    @property
    def execution_authorized(self):return False


@dataclass(frozen=True)
class ExpectedStoreIdentity:
    """Immutable external metadata; this data object is not a read capability."""
    aggregate:Aggregate
    raw:bytes
    def check(self):
        if type(self)is not ExpectedStoreIdentity or type(self.aggregate)is not Aggregate:raise TypeError('exact immutable expected store identity required')
        catalog=self.aggregate.catalog;v=_json(self.raw,2<<20,'store identity');_object(v,('roots','files'),'store identity')
        _object(v['roots'],('qkv','decoder','gamma'),'store roots')
        for family,row in v['roots'].items():
            _object(row,ROOT_FIELDS,'store root');_path(row['path']);_integer(row['device'],0,'root device');_integer(row['inode'],1,'root inode');_integer(row['mode'],0,'root mode',4095)
        expected={}
        for row in catalog['matrices'].values():
            d=row['descriptor'];key=(row['family'],d['file'])
            if key in expected:
                for field in ('shape','file_sha256','sha256'):
                    if expected[key][field]!=d[field]:raise ValueError('tied physical matrix descriptor differs')
            else:expected[key]=d
        if len(expected)!=197:raise ValueError('fixed197 physical matrix inventory required')
        if type(v['files'])is not list or len(v['files'])!=198:raise ValueError('exact197 matrices plus gamma container required')
        seen={};order=[];gamma=[]
        for row in v['files']:
            _object(row,('family','name','state','file_sha256','raw_sha256'),'held file')
            if type(row['family'])is not str or row['family']not in('qkv','decoder','gamma'):raise ValueError('exact file family required')
            _flat(row['name']);key=(row['family'],row['name'])
            if key in seen:raise ValueError('duplicate held file record')
            seen[key]=row;order.append(key);state=_object(row['state'],STATE_FIELDS,'held file state')
            for n in ('device','mtime_ns','ctime_ns'):_integer(state[n],0,'file '+n)
            for n in ('inode','size'):_integer(state[n],1,'file '+n)
            _integer(state['mode'],0,'file mode',4095);hash_value(row['file_sha256'])
            if row['family']=='gamma':
                gamma.append(row)
                if row['file_sha256']!=catalog['gamma_container_sha256']or row['raw_sha256']is not None or state['size']>4<<20:raise ValueError('gamma container binding differs')
            else:
                if key not in expected:raise ValueError('unexpected matrix file')
                d=expected[key];hash_value(row['raw_sha256']);size=8
                for dimension in d['shape']:size*=dimension
                if row['file_sha256']!=d['file_sha256']or row['raw_sha256']!=d['sha256']or not size<state['size']<=size+16384:raise ValueError('matrix file/hash/extent binding differs')
        if order!=sorted(order)or set(seen)-{('gamma',x['name'])for x in gamma}!=set(expected)or len(gamma)!=1:raise ValueError('sorted complete physical store inventory required')
        return v
    def root(self,family):return self.check()['roots'][family]
    def file(self,family,name):
        found=[r for r in self.check()['files']if r['family']==family and r['name']==name]
        if len(found)!=1:raise ValueError('one externally expected file required')
        return found[0]
    @property
    def gamma(self):return next(r for r in self.check()['files']if r['family']=='gamma')
    @property
    def execution_authorized(self):return False


@dataclass(frozen=True)
class ExpectedModelPins:
    """Trusted external caller argument; never infer it from incoming evidence."""
    raw:bytes
    def check(self):
        if type(self)is not ExpectedModelPins:raise TypeError('exact externally trusted expected pins required')
        v=_json(self.raw,1<<20,'expected model pins')
        fields={'schema','model_id','aggregate_manifest_sha256','catalog_sha256','producer_source_manifest_sha256','producer_source_archive_sha256','producer_source_map','execution_grant_sha256','validation_report_sha256','validation_outer_sha256','root_review_sha256','consumer_source_manifest_sha256'}
        _object(v,fields,'expected pins')
        if v['schema']!='qwen-model-expected-pins-v1':raise ValueError('expected-pin schema differs')
        for n in fields-{'schema','producer_source_map'}:hash_value(v[n])
        if(v['model_id'],v['aggregate_manifest_sha256'],v['catalog_sha256'])!=(MODEL_ID,AGGREGATE_SHA256,CATALOG_SHA256):raise ValueError('fixed aggregate expectation differs')
        _source_map(v['producer_source_map'])
        if sha(canonical(v['producer_source_map']))!=v['producer_source_manifest_sha256']:raise ValueError('canonical expected producer map digest differs')
        return v


@dataclass(frozen=True)
class ModelAdmission:
    public:PublicContract
    manifest_bytes:bytes
    report_bytes:bytes
    outer_bytes:bytes
    review_bytes:bytes
    expected_pins:ExpectedModelPins
    @property
    def execution_authorized(self):return False
    @property
    def model_id(self):return MODEL_ID
    def check(self):
        if type(self)is not ModelAdmission:raise TypeError('exact model admission required')
        return authenticate_model(self.public,self.manifest_bytes,self.report_bytes,self.outer_bytes,self.review_bytes,expected_pins=self.expected_pins)
    @property
    def stores(self):
        self.check();return ExpectedStoreIdentity(Aggregate(self.public,self.manifest_bytes),canonical(parse(self.report_bytes)['stores']))


def authenticate_model(public,manifest_bytes,validation_report_bytes,validation_outer_bytes,root_review_bytes,*,expected_pins):
    """Authenticate immutable public evidence only; no archive or body IO."""
    if type(public)is not PublicContract or type(expected_pins)is not ExpectedModelPins:raise TypeError('public contract and external expected pins required')
    expected=expected_pins.check();aggregate=Aggregate(public,manifest_bytes);aggregate.check()
    report=_json(validation_report_bytes,2<<20,'validation report');outer=_json(validation_outer_bytes,1<<16,'validation outer');review=_json(root_review_bytes,1<<16,'root review')
    for raw,name in ((validation_report_bytes,'validation_report_sha256'),(validation_outer_bytes,'validation_outer_sha256'),(root_review_bytes,'root_review_sha256')):
        if sha(raw)!=expected[name]:raise ValueError('external '+name+' differs')
    _object(report,REPORT_FIELDS,'validation report')
    if report['schema']!='qwen-model-store-validation-v1'or report['mode']!='reuse_existing_exports'or report['passed']is not True or report['complete']is not True or report['error_type']is not None or report['audit_denials']!=[]:raise ValueError('complete clean readonly model validation required')
    same(report['checks'],{n:True for n in CHECKS},'complete model validation checks required')
    same(report['scope'],dict(parameter_body_reads_performed=True,**{n:False for n in SCOPE_FALSE}),'model-only readonly validation scope differs')
    same(report['counts'],dict(logical_matrices=198,physical_matrices=197,gamma_vectors=113,physical_matrix_raw_bytes=4767875072,gamma_raw_bytes=524288),'complete fixed model counts required')
    for n in ('model_id','aggregate_manifest_sha256','catalog_sha256','execution_grant_sha256'):
        if report[n]!=expected[n]:raise ValueError('report '+n+' differs')
    for name,field in (('source_manifest_sha256','producer_source_manifest_sha256'),('source_archive_sha256','producer_source_archive_sha256')):
        if report[name]!=expected[field]:raise ValueError('report source binding differs')
    for n in ('source_hashes_before','source_hashes_after'):same(_source_map(report[n]),expected['producer_source_map'],'source before/end map differs')
    process=_object(report['process'],('pid','pgid'),'validation process')
    for n in process:_integer(process[n],1,'process '+n)
    stores=ExpectedStoreIdentity(aggregate,canonical(report['stores']));stores.check()
    outer_expected=dict(schema='qwen-model-store-validation-outer-v1',report_sha256=sha(validation_report_bytes),aggregate_manifest_sha256=AGGREGATE_SHA256,model_id=MODEL_ID,source_manifest_sha256=expected['producer_source_manifest_sha256'],source_archive_sha256=expected['producer_source_archive_sha256'],execution_grant_sha256=expected['execution_grant_sha256'],pid=process['pid'],pgid=process['pgid'],exit_code=0,normal_return=True,reaped=True,process_group_absent=True)
    same(outer,outer_expected,'independent validation outer receipt differs')
    review_expected=dict(schema='qwen-model-stage-root-admission-v1',passed=True,model_id=MODEL_ID,aggregate_manifest_sha256=AGGREGATE_SHA256,catalog_sha256=CATALOG_SHA256,validation_report_sha256=sha(validation_report_bytes),validation_outer_sha256=sha(validation_outer_bytes),source_manifest_sha256=expected['producer_source_manifest_sha256'],source_archive_sha256=expected['producer_source_archive_sha256'],execution_grant_sha256=expected['execution_grant_sha256'],current_store_binding_sha256=sha(stores.raw),all_required_links_verified=True,source_archive_members_verified=True,actual_server_execution_authorized=False)
    same(review,review_expected,'externally trusted root admission differs')
    return ModelAdmission(public,manifest_bytes,validation_report_bytes,validation_outer_bytes,root_review_bytes,expected_pins)


def require_admitted(value,public,*,consumer_source_manifest_sha256):
    if type(value)is not ModelAdmission or type(public)is not PublicContract:raise TypeError('exact retained model admission required')
    hash_value(consumer_source_manifest_sha256);public.check();value.check()
    if public.blobs!=value.public.blobs:raise ValueError('consumer public metadata differs')
    if value.expected_pins.check()['consumer_source_manifest_sha256']!=consumer_source_manifest_sha256:raise ValueError('selected model consumer source differs')
    return value
