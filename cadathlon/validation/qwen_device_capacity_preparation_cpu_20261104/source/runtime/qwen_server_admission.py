"""Pure admission for a prospective full28 public-key server phase.

All inputs are independently trusted public proof bytes or retained admissions.
No file, backend, resource, key or model access occurs here. This boundary does
not issue a grant, manufacture scientific evidence or accept an HE result.
"""
from dataclasses import dataclass
from qwen_file_contract import (PublicContract,canonical,sha,same,hash_value,
    validate_request,validate_request_manifest)
from qwen_key_contract import PublicProfile,authenticate_public
from client_role_common import (obj,raw_object,integer,path,source_map,hashes,
    model_pins,key_pins,output,disjoint,MODEL_FIELDS,KEY_FIELDS,POLICY_FIELDS)

ENTRY='qwen_server_entry.py:run_server'
REQUIRED_MODULES=('qwen_server_entry.py','qwen_server_process.py','qwen_server_admission.py',
    'qwen_scientific_admission.py','qwen_server_dependencies.py','qwen_server_resource.py',
    'qwen_server_backend.py','qwen_key_loader.py','qwen_stage_composition.py',
    'qwen_file_lifecycle.py','qwen_public_parameters.py','qwen_publisher.py',
    'qwen_server_dependencies.json','qwen_server_role_policy.json')
PIN_FIELDS={'schema','source_map','source_manifest_sha256','source_archive_sha256',
    'readiness_review_sha256','phase_grant_sha256','request_contract_sha256',
    'request_manifest_sha256','input_digest','model_store_sha256','model_evidence',
    'key_evidence','policies','scientific_authorization_sha256'}
GRANT_FIELDS={'schema','role','entry','execute','source_root','source_manifest_sha256',
    'source_archive_sha256','readiness_review_sha256','request_contract_sha256',
    'model_store_sha256','model_evidence','key_evidence','policies',
    'scientific_authorization_sha256','public_key_store','input','outputs','scope'}
SCOPE=dict(full28_server_forward=True,public_model_body_reads=True,public_key_reads=True,
    request_ciphertext_reads=True,response_ciphertext_writes=True,key_generation=False,
    private_key_reads=False,client_encoding=False,client_decoding=False,
    token_or_calibration_reads=False,scientific_acceptance=False)


def create_actual_server(*args,**kwargs):
    raise PermissionError('full server process, scientific and resource admission remain unselected')


@dataclass(frozen=True)
class ExpectedServerPhasePins:
    raw:bytes
    def check(self):
        if type(self)is not ExpectedServerPhasePins:raise TypeError('independent server phase pins required')
        v=raw_object(self.raw,1<<20,'server expected pins');obj(v,PIN_FIELDS,'server expected pins')
        if v['schema']!='qwen-server-phase-expected-pins-v1':raise ValueError('server pin schema differs')
        for n in PIN_FIELDS-{'schema','source_map','model_evidence','key_evidence','policies'}:hash_value(v[n])
        sources=source_map(v['source_map'])
        if not set(REQUIRED_MODULES)<=set(sources):raise ValueError('actual server source closure incomplete')
        if sha(canonical(sources))!=v['source_manifest_sha256']:raise ValueError('external complete server map differs')
        hashes(v['model_evidence'],MODEL_FIELDS,'server model proof')
        hashes(v['key_evidence'],KEY_FIELDS,'server generated-key proof')
        hashes(v['policies'],POLICY_FIELDS,'server policies')
        for field,name in (('dependency_policy_sha256','qwen_server_dependencies.json'),('role_policy_sha256','qwen_server_role_policy.json')):
            if v['policies'][field]!=sources[name]:raise ValueError('server selected policy differs')
        return v


@dataclass(frozen=True)
class ServerPhaseAdmission:
    public:PublicContract
    model:object
    readiness:object
    request_bytes:bytes
    request_manifest_bytes:bytes
    grant_bytes:bytes
    expected_pins:ExpectedServerPhasePins
    scientific:object
    def check(self):
        return authenticate_server_phase(self.public,self.model,self.readiness,self.request_bytes,
            self.request_manifest_bytes,self.grant_bytes,expected_pins=self.expected_pins,scientific=self.scientific)
    @property
    def input_digest(self):
        self.check();return self.expected_pins.check()['input_digest']
    def expected_ids(self):
        self.check();return self.readiness.expected_ids(role='server')
    @property
    def execution_authorized(self):return False


def authenticate_server_phase(public,model,readiness,request_bytes,request_manifest_bytes,
        grant_bytes,*,expected_pins,scientific):
    if type(public)is not PublicContract or type(expected_pins)is not ExpectedServerPhasePins:
        raise TypeError('exact public contract and independent server pins required')
    public.check();pins=expected_pins.check()
    model_evidence=model_pins(model,public,pins['source_manifest_sha256'])
    same(model_evidence,pins['model_evidence'],'actual server-consumer model proof differs')
    stores=model.stores;stores.check()
    if sha(stores.raw)!=pins['model_store_sha256']:raise ValueError('independently granted parameter store identity differs')
    keys=key_pins(readiness);same(keys,pins['key_evidence'],'independent generated-key proof differs')
    if readiness.launch.intent.public.blobs!=public.blobs:raise ValueError('server and generation public metadata differ')
    intent=readiness.launch.intent.check()
    if model.model_id!=intent['model_id']:raise ValueError('server/generation aggregate model differs')
    generated_model=readiness.launch.expected_pins.check()['model_evidence']
    for name in MODEL_FIELDS-{'expected_pins_sha256'}:
        if generated_model[name]!=model_evidence[name]:raise ValueError('server and keygen original model proof differs')
    request=validate_request(public,raw_object(request_bytes,1<<20,'server request'))
    if sha(request_bytes)!=pins['request_contract_sha256']:raise ValueError('independent request contract differs')
    ids=readiness.expected_ids(role='server');ids.require_identity(request['public'],'server')
    metadata=public.check();profile=PublicProfile.from_bytes(metadata['inputs/profile_audit.json'],metadata['inputs/actual_key_plan.json'])
    authenticate_public(readiness.public_manifest_bytes,profile,ids,role='server')
    upload=raw_object(request_manifest_bytes,1<<20,'request upload')
    validate_request_manifest(public,request,upload)
    if sha(request_manifest_bytes)!=pins['request_manifest_sha256']or upload['input_digest']!=pins['input_digest']:
        raise ValueError('independent request upload/input digest differs')

    from qwen_scientific_admission import ScientificAdmission
    if type(scientific)is not ScientificAdmission:raise TypeError('separate full scientific PASS admission required')
    scientific.check()
    if sha(scientific.statement_bytes)!=pins['scientific_authorization_sha256']:
        raise ValueError('independent scientific authorization differs')
    binding=scientific.bindings()
    expected=dict(server_source_manifest_sha256=pins['source_manifest_sha256'],
        server_source_archive_sha256=pins['source_archive_sha256'],
        request_contract_sha256=pins['request_contract_sha256'],request_manifest_sha256=pins['request_manifest_sha256'],
        input_digest=pins['input_digest'],context_id=ids.context_id,
        public_manifest_sha256=keys['public_manifest_sha256'],model_id=model.model_id)
    for name,value in expected.items():
        if binding.get(name)!=value:raise ValueError('scientific/server '+name+' differs')

    grant=raw_object(grant_bytes,1<<16,'server phase grant');obj(grant,GRANT_FIELDS,'server phase grant')
    if sha(grant_bytes)!=pins['phase_grant_sha256']:raise ValueError('external server execution grant differs')
    if (grant['schema']!='qwen-server-phase-grant-v1'or grant['role']!='full28_server'
            or grant['entry']!=ENTRY or grant['execute']is not True):
        raise ValueError('explicit exact full-model server phase required')
    source_root=str(path(grant['source_root']))
    for name in ('source_manifest_sha256','source_archive_sha256','readiness_review_sha256',
            'request_contract_sha256','model_store_sha256','scientific_authorization_sha256'):
        if grant[name]!=pins[name]:raise ValueError('server phase '+name+' differs')
    same(grant['model_evidence'],model_evidence,'server grant model proof differs')
    same(grant['key_evidence'],keys,'server grant generated-key proof differs')
    same(grant['policies'],pins['policies'],'server grant policies differ')
    generated_stores=raw_object(readiness.report_bytes,2<<20,'generated key report')['stores']
    same(grant['public_key_store'],generated_stores['public'],'server public key store differs')
    row=obj(grant['input'],('kind','root','request_manifest_sha256','input_digest'),'server input')
    if row['kind']!='fresh_encrypted_request':raise ValueError('original encrypted request required')
    root=obj(row['root'],('path','device','inode','mode'),'request root');path(root['path'])
    integer(root['device'],0,'request root device');integer(root['inode'],1,'request root inode')
    same(root['mode'],0o700,'retained request root mode differs')
    for name in ('request_manifest_sha256','input_digest'):
        if row[name]!=pins[name]:raise ValueError('server granted '+name+' differs')
    obj(grant['outputs'],('response','evidence'),'server fresh outputs')
    for value in grant['outputs'].values():output(value)
    model_roots=[v['path']for v in stores.check()['roots'].values()]
    key_roots=[v['path']for v in generated_stores.values()]
    disjoint(key_roots,[source_root,*model_roots])
    protected=[source_root,*model_roots,*key_roots]
    disjoint([root['path']],protected)
    disjoint([v['path']for v in grant['outputs'].values()],protected+[root['path']])
    same(grant['scope'],SCOPE,'exact server-only scope differs')
    return ServerPhaseAdmission(public,model,readiness,request_bytes,request_manifest_bytes,
        grant_bytes,expected_pins,scientific)
