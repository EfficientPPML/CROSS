"""Pure Qwen keygen evidence contracts; no I/O, imports of crypto, or runner.

Expected pin objects are independently trusted caller inputs. Merely creating
these ordinary Python objects grants no resource, body, RNG or role authority.
"""
from dataclasses import dataclass
from pathlib import PurePosixPath
import re
from qwen_file_contract import PublicContract, canonical, sha, hash_value, same
from qwen_public_parse import parse
from qwen_key_contract import PublicProfile, ExpectedIds, KEYGEN, bindings, authenticate_public
from model_admission import ModelAdmission, MODEL_ID, require_admitted
from keygen_boundary_pins import CORE_MANIFEST_SHA256, REQUIRED_ACTUAL_SOURCES

NONCE = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
MODEL_FIELDS = {'manifest_sha256','report_sha256','outer_sha256','root_review_sha256','expected_pins_sha256'}
POLICY_FIELDS = {'dependency_policy_sha256','role_policy_sha256','resource_policy_sha256','resource_review_sha256'}
ROLE_FIELDS = {'source_manifest_sha256','source_archive_sha256','readiness_review_sha256','execution_grant_sha256'}
INTENT_FIELDS = {'schema','nonce','model_id','generation_source_manifest_sha256','core_source_manifest_sha256','keygen_source_sha256','request_producer_source_manifest_sha256','response_consumer_source_manifest_sha256','bindings','parameters','paths','logical_storage'}
LAUNCH_PIN_FIELDS = {'schema','intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','source_map','model_evidence','role_evidence','prelaunch_review_sha256','policies'}
GRANT_FIELDS = {'schema','role','execute','intent_sha256','source_root','source_manifest_sha256','source_archive_sha256','model_evidence','role_evidence','prelaunch_review_sha256','policies','parents','scope'}
READY_PIN_FIELDS = {'schema','intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','report_sha256','outer_sha256','root_review_sha256','context_id','public_manifest_sha256','private_manifest_sha256','stores_sha256'}
REPORT_FIELDS = {'schema','passed','complete','error_type','intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','source_hashes_before','source_hashes_after','model_id','model_evidence','role_evidence','core_source_manifest_sha256','keygen_source_sha256','bindings','request_producer_source_manifest_sha256','response_consumer_source_manifest_sha256','context_id','public_manifest_sha256','private_manifest_sha256','stores','counts','security','backend','policies','checks','scope','process','audit_denials'}
OUTER_FIELDS = {'schema','report_sha256','intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','context_id','public_manifest_sha256','private_manifest_sha256','stores_sha256','pid','pgid','exit_code','normal_return','reaped','process_group_absent'}
ROOT_FIELDS = {'schema','passed','report_sha256','outer_sha256','intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','context_id','public_manifest_sha256','private_manifest_sha256','stores_sha256','model_id','model_evidence','role_evidence','policies','all_required_links_verified','source_archive_members_verified','current_owned_stores_verified','downstream_execution_authorized'}
CHECK_NAMES = ('model_admission_authenticated','source_precheck_completed','real_security_predicates_completed','resource_admission_completed','capacity_checks_completed','external_parent_identity_checked_before_creation','pre_rng_checks_completed','original_secret_operand_preserved','all335_public_arrays_authenticated','secret_array_authenticated','public_private_manifest_link_authenticated','all_owned_files_fsynced','root_and_parent_fsync_completed','final_readback_completed','original_owned_identity_unchanged','cleanup_completed','source_postcheck_completed','signals_restored')
SCOPE_FALSE = ('preexisting_key_bodies_read','model_bodies_read','token_or_calibration_bodies_read','ciphertexts_read','ckks_context_constructed','model_forward','scientific_acceptance_claimed','coefficient_selection_or_revision','downstream_execution_authorized')
GRANT_SCOPE = dict(fresh_key_generation=True,new_owned_key_readback=True,**{n:False for n in SCOPE_FALSE})


def create_actual_keygen_process(*args, **kwargs):
    raise PermissionError('actual key generation awaits reviewed process, parent binding and resource admission')


def _object(value, fields, label):
    if type(value) is not dict or set(value) != set(fields):
        raise ValueError('exact '+label+' fields required')
    return value


def _bytes(raw, limit, label):
    if type(raw) is not bytes or not 0 < len(raw) <= limit:
        raise ValueError('bounded immutable '+label+' bytes required')
    value = parse(raw)
    if type(value) is not dict or canonical(value) != raw:
        raise ValueError('canonical '+label+' object required')
    return value


def _integer(value, minimum, label, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError('exact bounded '+label+' integer required')
    return value


def _path(value):
    if type(value) is not str or not 1 < len(value) <= 4096 or '\x00' in value or '\\' in value or value.startswith('//'):
        raise ValueError('canonical absolute path required')
    path = PurePosixPath(value)
    if not path.is_absolute() or str(path) != value or '..' in path.parts or path == path.parent:
        raise ValueError('canonical non-root path required')
    return path


def _digest_object(value, fields, label):
    _object(value, fields, label)
    for digest in value.values(): hash_value(digest)
    return value


def _source_map(value):
    if type(value) is not dict or not 0 < len(value) <= 512:
        raise ValueError('bounded complete external source map required')
    for name, digest in value.items():
        if type(name) is not str or not 0 < len(name) <= 512 or '\\' in name or '\x00' in name:
            raise ValueError('ordinary relative source name required')
        path = PurePosixPath(name)
        if path.is_absolute() or str(path) != name or '..' in path.parts or name in ('.','..'):
            raise ValueError('canonical relative source name required')
        hash_value(digest)
    for name, digest in REQUIRED_ACTUAL_SOURCES.items():
        if value.get(name) != digest:
            raise ValueError('fixed actual prerequisite source differs: '+name)
    return value


def _profile(public):
    if type(public) is not PublicContract: raise TypeError('exact public contract required')
    blobs = public.check()
    profile = PublicProfile.from_bytes(blobs['inputs/profile_audit.json'], blobs['inputs/actual_key_plan.json'])
    same(profile.parameters, public.profile(), 'key/file profile differs')
    return profile


def _storage(profile):
    specs = profile.arrays
    return dict(public_arrays=335,private_arrays=1,public_files=336,private_files=2,
        public_raw_bytes=sum(s.raw_bytes for s in specs),private_raw_bytes=profile.secret.raw_bytes,
        public_required_bytes=sum(s.raw_bytes+4096 for s in specs)+(1<<30),
        private_required_bytes=profile.secret.raw_bytes+4096+(1<<20),
        retained_store_descriptors=342,store_temporary_allowance=8,
        scope='inherited logical store allowance only; actual memory/runtime/evidence headroom requires separate reviewed policy')


def model_proof_pins(admission, public, consumer_sources):
    require_admitted(admission, public, consumer_source_manifest_sha256=consumer_sources)
    return dict(manifest_sha256=sha(admission.manifest_bytes),report_sha256=sha(admission.report_bytes),
        outer_sha256=sha(admission.outer_bytes),root_review_sha256=sha(admission.review_bytes),
        expected_pins_sha256=sha(admission.expected_pins.raw))


@dataclass(frozen=True)
class ActualKeygenIntent:
    public: PublicContract
    raw: bytes
    def check(self):
        if type(self) is not ActualKeygenIntent: raise TypeError('exact immutable actual intent required')
        profile = _profile(self.public); value = _bytes(self.raw, 1<<16, 'actual keygen intent')
        _object(value, INTENT_FIELDS, 'actual intent')
        if value['schema'] != 'qwen-keygen-actual-intent-v1' or type(value['nonce']) is not str or not NONCE.fullmatch(value['nonce']):
            raise ValueError('actual intent schema/nonce differs')
        for name in ('model_id','generation_source_manifest_sha256','core_source_manifest_sha256','keygen_source_sha256','request_producer_source_manifest_sha256','response_consumer_source_manifest_sha256'):
            hash_value(value[name])
        if (value['model_id'],value['core_source_manifest_sha256'],value['keygen_source_sha256']) != (MODEL_ID,CORE_MANIFEST_SHA256,KEYGEN):
            raise ValueError('fixed content/core/keygen identity differs')
        same(value['bindings'],bindings(),'fixed candidate/profile bindings differ')
        same(value['parameters'],profile.parameters,'actual keygen physical profile differs')
        same(value['logical_storage'],_storage(profile),'inherited logical storage differs')
        _object(value['paths'],('public_root','private_root','evidence_root'),'owned leaf paths')
        paths = [_path(value['paths'][name]) for name in ('public_root','private_root','evidence_root')]
        for i, path in enumerate(paths):
            if any(path==other or path in other.parents or other in path.parents for other in paths[:i]):
                raise ValueError('owned key/evidence leaves must be disjoint')
        return value
    @property
    def execution_authorized(self): return False


@dataclass(frozen=True)
class ExpectedLaunchPins:
    """Independent caller pins, not inferred from incoming grant/evidence."""
    raw: bytes
    def check(self):
        if type(self) is not ExpectedLaunchPins: raise TypeError('external launch pins required')
        value = _bytes(self.raw, 1<<20, 'expected launch pins'); _object(value, LAUNCH_PIN_FIELDS, 'launch pins')
        if value['schema'] != 'qwen-keygen-launch-pins-v1': raise ValueError('launch-pin schema differs')
        for name in ('intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','prelaunch_review_sha256'):
            hash_value(value[name])
        source = _source_map(value['source_map'])
        if sha(canonical(source)) != value['source_manifest_sha256']: raise ValueError('canonical compact source-map digest differs')
        _digest_object(value['model_evidence'],MODEL_FIELDS,'model proof pins')
        _digest_object(value['policies'],POLICY_FIELDS,'external policy/review pins')
        _object(value['role_evidence'],('request_producer','response_consumer'),'future actual role proofs')
        for row in value['role_evidence'].values(): _digest_object(row,ROLE_FIELDS,'independently reviewed actual role pins')
        return value


def _parent(value, path):
    _object(value, ('path','device','inode','mode'), 'granted parent')
    if type(value['path']) is not str or value['path'] != str(path.parent): raise ValueError('granted parent path differs')
    _integer(value['device'],0,'parent device');_integer(value['inode'],1,'parent inode');_integer(value['mode'],0,'parent mode',4095)
    # Parent ancestry is permitted: existing /tmp may contain protected siblings.
    return value


@dataclass(frozen=True)
class LaunchAdmission:
    intent: ActualKeygenIntent
    grant_bytes: bytes
    model: ModelAdmission
    expected_pins: ExpectedLaunchPins
    def check(self):
        if type(self) is not LaunchAdmission: raise TypeError('exact retained launch binding required')
        return authenticate_launch(self.intent,self.grant_bytes,self.model,expected_pins=self.expected_pins)
    @property
    def execution_authorized(self): return False


def authenticate_launch(intent, grant_bytes, model_admission, *, expected_pins):
    """Pure public authority binding; no file opens, capacity check or RNG."""
    if type(intent) is not ActualKeygenIntent or type(expected_pins) is not ExpectedLaunchPins:
        raise TypeError('exact intent and external launch pins required')
    value = intent.check(); pins = expected_pins.check()
    grant = _bytes(grant_bytes, 1<<16, 'actual keygen grant');_object(grant, GRANT_FIELDS, 'actual grant')
    if sha(intent.raw) != pins['intent_sha256'] or sha(grant_bytes) != pins['grant_sha256']:
        raise ValueError('external intent/grant digest differs')
    if value['generation_source_manifest_sha256'] != pins['source_manifest_sha256']:
        raise ValueError('actual intent source differs')
    for role in ('request_producer','response_consumer'):
        if value[role+'_source_manifest_sha256'] != pins['role_evidence'][role]['source_manifest_sha256']:
            raise ValueError('intent/externally reviewed actual role source differs')
    model = model_proof_pins(model_admission,intent.public,pins['source_manifest_sha256'])
    same(model,pins['model_evidence'],'external ModelAdmission proof differs')
    if model_admission.model_id != value['model_id']: raise ValueError('actual intent/model admission differs')
    if grant['schema'] != 'qwen-keygen-actual-grant-v1' or grant['role'] != 'generate_fresh_qwen166_keys' or grant['execute'] is not True:
        raise ValueError('explicit actual keygen grant required')
    for name in ('intent_sha256','source_manifest_sha256','source_archive_sha256','prelaunch_review_sha256'):
        if grant[name] != pins[name]: raise ValueError('granted '+name+' differs')
    same(grant['model_evidence'],model,'grant/model evidence differs')
    same(grant['role_evidence'],pins['role_evidence'],'grant/independently reviewed actual role proof differs')
    same(grant['policies'],pins['policies'],'independently reviewed policies differ')
    same(grant['scope'],GRANT_SCOPE,'keygen-only grant scope differs')
    source = _path(grant['source_root'])
    _object(grant['parents'],('public_root','private_root','evidence_root'),'parent inventory')
    for name, raw in value['paths'].items():
        leaf = _path(raw)
        if leaf==source or leaf in source.parents or source in leaf.parents: raise ValueError('owned leaf/source overlap')
        _parent(grant['parents'][name],leaf)
    return LaunchAdmission(intent,grant_bytes,model_admission,expected_pins)


@dataclass(frozen=True)
class ExpectedReadyPins:
    """Post-terminal externally supplied generated identities and proof pins."""
    raw: bytes
    def check(self):
        if type(self) is not ExpectedReadyPins: raise TypeError('external post-terminal pins required')
        value = _bytes(self.raw,1<<16,'expected readiness pins');_object(value,READY_PIN_FIELDS,'readiness pins')
        if value['schema'] != 'qwen-keygen-ready-pins-v1': raise ValueError('readiness-pin schema differs')
        for name in READY_PIN_FIELDS-{'schema'}: hash_value(value[name])
        return value


def _stores(value, intent, grant, pins, public_raw):
    _object(value,('public','private'),'retained generated stores')
    for role in ('public','private'):
        row = _object(value[role],('path','device','inode','mode','parent','manifest'),'retained store')
        if row['path'] != intent['paths'][role+'_root']: raise ValueError('retained key root differs')
        _integer(row['device'],0,'root device');_integer(row['inode'],1,'root inode')
        same(row['mode'],0o700,'retained key root mode differs')
        same(row['parent'],grant['parents'][role+'_root'],'retained/granted parent identity differs')
        manifest = _object(row['manifest'],('name','sha256','state'),'retained manifest')
        if manifest['name'] != role+'_manifest.json' or manifest['sha256'] != pins[role+'_manifest_sha256']:
            raise ValueError('retained manifest name/external digest differs')
        state = _object(manifest['state'],('device','inode','size','mtime_ns','ctime_ns','mode'),'retained manifest state')
        for name in ('device','mtime_ns','ctime_ns'): _integer(state[name],0,'manifest '+name)
        _integer(state['inode'],1,'manifest inode');_integer(state['size'],1,'manifest size',1<<20 if role=='public' else 1<<16)
        same(state['mode'],0o600,'retained manifest mode differs')
        if state['device'] != row['device']: raise ValueError('new manifest/root device differs')
        if role=='public' and state['size'] != len(public_raw): raise ValueError('public manifest extent differs')
    if sha(canonical(value)) != pins['stores_sha256']: raise ValueError('external retained-store digest differs')
    return value


def _expected_ids(intent, pins, role):
    if type(role) is not str or role not in ('client','server'): raise ValueError('exact future key role required')
    result = ExpectedIds(pins['context_id'],pins['public_manifest_sha256'],KEYGEN,intent['model_id'],
        intent['request_producer_source_manifest_sha256'],intent['response_consumer_source_manifest_sha256'],
        pins['private_manifest_sha256'] if role=='client' else None)
    result.check(role);return result


@dataclass(frozen=True)
class GeneratedKeyReadiness:
    launch: LaunchAdmission
    public_manifest_bytes: bytes
    report_bytes: bytes
    outer_bytes: bytes
    root_review_bytes: bytes
    expected_pins: ExpectedReadyPins
    def check(self):
        if type(self) is not GeneratedKeyReadiness: raise TypeError('exact retained key readiness required')
        return authenticate_readiness(self.launch,self.public_manifest_bytes,self.report_bytes,self.outer_bytes,self.root_review_bytes,expected_pins=self.expected_pins)
    def expected_ids(self, *, role):
        self.check()
        return _expected_ids(self.launch.intent.check(),self.expected_pins.check(),role)
    @property
    def execution_authorized(self): return False


def authenticate_readiness(launch,public_manifest_bytes,report_bytes,outer_bytes,root_review_bytes,*,expected_pins):
    """No private manifest/key body input; authenticate public terminal bytes."""
    if type(launch) is not LaunchAdmission or type(expected_pins) is not ExpectedReadyPins:
        raise TypeError('retained launch and independent readiness pins required')
    launch.check();intent=launch.intent.check();grant=_bytes(launch.grant_bytes,1<<16,'retained grant')
    start=launch.expected_pins.check();pins=expected_pins.check()
    for name in ('intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256'):
        if pins[name] != start[name]: raise ValueError('readiness/original launch '+name+' differs')
    for raw,name,limit in ((report_bytes,'report_sha256',2<<20),(outer_bytes,'outer_sha256',1<<16),(root_review_bytes,'root_review_sha256',1<<16),(public_manifest_bytes,'public_manifest_sha256',1<<20)):
        _bytes(raw,limit,name)
        if sha(raw)!=pins[name]:raise ValueError('external '+name+' differs')
    report=_bytes(report_bytes,2<<20,'keygen report');_object(report,REPORT_FIELDS,'keygen report')
    if report['schema']!='qwen-keygen-actual-report-v1' or report['passed']is not True or report['complete']is not True or report['error_type']is not None or report['audit_denials']!=[]:
        raise ValueError('complete clean genuine keygen report required')
    for name in ('intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','context_id','public_manifest_sha256','private_manifest_sha256'):
        if report[name]!=pins[name]:raise ValueError('report external '+name+' differs')
    for name in ('model_id','core_source_manifest_sha256','keygen_source_sha256','request_producer_source_manifest_sha256','response_consumer_source_manifest_sha256'):
        if report[name]!=intent[name]:raise ValueError('report/intent '+name+' differs')
    same(report['bindings'],intent['bindings'],'report fixed bindings differ')
    same(report['model_evidence'],start['model_evidence'],'report model proof differs')
    same(report['role_evidence'],start['role_evidence'],'report future actual role proof differs')
    same(report['policies'],start['policies'],'report independently reviewed policy differs')
    for name in ('source_hashes_before','source_hashes_after'):
        same(_source_map(report[name]),start['source_map'],'complete producer source before/end differs')
    same(report['counts'],dict(pke_pairs=1,evaluation_key_pairs=1,rotation_key_pairs=166,primitive_calls=168,public_arrays=335,private_arrays=1,public_files=336,private_files=2),'complete generation counts differ')
    same(report['security'],dict(target_bits=128,attack='classical',degree=131072,q_towers=100,p_towers=14,dnum=8,qp_bits=3403,qp_inclusive_ceiling=3523,p_product_bits=415,largest_q_partition_bits=390,sigma=3.190000057220458984375,secret_distribution='uniform_ternary',noise_scale=1,key_generation_version=3,qp_is_secure=True,p_coverage=True),'genuine fixed security evidence differs')
    backend=_object(report['backend'],('dependency_policy_sha256','observed_device_platforms','jax_enable_x64','genuine_backend_imports_completed'),'actual CPU backend evidence')
    if backend['dependency_policy_sha256']!=start['policies']['dependency_policy_sha256'] or backend['jax_enable_x64']is not True or backend['genuine_backend_imports_completed']is not True:
        raise ValueError('actual selected CPU dependency evidence differs')
    platforms=backend['observed_device_platforms']
    if type(platforms)is not list or not 0<len(platforms)<=64 or any(type(x)is not str or x!='cpu'for x in platforms):
        raise ValueError('only actual CPU devices are selected for key generation')
    same(report['checks'],{name:True for name in CHECK_NAMES},'complete real generation/retirement checks required')
    same(report['scope'],dict(fresh_key_generation=True,new_owned_key_readback=True,**{name:False for name in SCOPE_FALSE}),'generation-only scope differs')
    profile=_profile(launch.intent.public)
    authenticate_public(public_manifest_bytes,profile,_expected_ids(intent,pins,'server'),role='server')
    stores=_stores(report['stores'],intent,grant,pins,public_manifest_bytes)
    process=_object(report['process'],('pid','pgid'),'generation process')
    for name in process:_integer(process[name],1,'process '+name)
    outer=_bytes(outer_bytes,1<<16,'generation outer');_object(outer,OUTER_FIELDS,'generation outer')
    linked={name:pins[name]for name in ('report_sha256','intent_sha256','grant_sha256','source_manifest_sha256','source_archive_sha256','context_id','public_manifest_sha256','private_manifest_sha256','stores_sha256')}
    same(outer,dict(schema='qwen-keygen-actual-outer-v1',**linked,**process,exit_code=0,normal_return=True,reaped=True,process_group_absent=True),'independent normal-zero/reap outer differs')
    review=_bytes(root_review_bytes,1<<16,'root key readiness');_object(review,ROOT_FIELDS,'root key readiness')
    same(review,dict(schema='qwen-keygen-root-readiness-v1',passed=True,**linked,outer_sha256=pins['outer_sha256'],model_id=intent['model_id'],model_evidence=start['model_evidence'],role_evidence=start['role_evidence'],policies=start['policies'],all_required_links_verified=True,source_archive_members_verified=True,current_owned_stores_verified=True,downstream_execution_authorized=False),'independent root key readiness differs')
    return GeneratedKeyReadiness(launch,public_manifest_bytes,report_bytes,outer_bytes,root_review_bytes,expected_pins)
