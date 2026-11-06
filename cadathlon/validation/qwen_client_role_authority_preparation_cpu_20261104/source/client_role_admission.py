"""Pure conditional and post-keygen client admission. All actual entry denies.

Expected pin arguments are independently trusted caller inputs. These public
objects cannot open files, import a backend, reserve resources or launch a role.
"""
from dataclasses import dataclass
from qwen_file_contract import canonical,sha,same,hash_value,validate_request
from qwen_key_contract import PublicProfile,authenticate_public
from keygen_authority import ActualKeygenIntent
from client_role_common import (obj,raw_object,integer,hashes,role,path,source_map,
    model_pins,key_pins,parent,output,state,disjoint,MODEL_FIELDS,KEY_FIELDS,POLICY_FIELDS)
from client_role_source_pins import FIXED_CLIENT_SOURCES,ACTUAL_CLIENT_MODULES

ENTRY={'request_producer':'qwen_client_entry.py:produce_request','response_consumer':'qwen_client_entry.py:consume_response'}
CONDITIONAL_PIN_FIELDS={'schema','role','source_manifest_sha256','source_archive_sha256','source_map','readiness_review_sha256','conditional_grant_sha256','generation_intent_sha256','generation_source_manifest_sha256','generation_source_archive_sha256','model_evidence','policies'}
CONDITIONAL_FIELDS={'schema','role','entry','source_root','source_manifest_sha256','source_archive_sha256','readiness_review_sha256','generation_intent_sha256','generation_source_manifest_sha256','generation_source_archive_sha256','model_id','model_evidence','policies','conditions','scope'}
CONDITIONS=dict(generated_key_readiness_required=True,phase_instance_grant_required=True,current_owned_store_checks_required=True,actual_dependency_and_capacity_checks_required=True)
CONDITIONAL_SCOPE=dict(conditional_role_admission=True,body_entry_by_this_document_alone=False,key_generation=False,model_forward=False,scientific_acceptance=False)
PHASE_PIN_FIELDS={'schema','role','conditional_grant_sha256','phase_grant_sha256','request_contract_sha256','key_evidence'}
PHASE_FIELDS={'schema','role','entry','execute','conditional_grant_sha256','source_manifest_sha256','source_archive_sha256','model_evidence','key_evidence','request_contract_sha256','policies','key_stores','input','outputs','scope'}

def create_actual_client_input(*args,**kwargs):
    raise PermissionError('actual client process/dependencies/resources remain unimplemented')

def create_actual_client_output(*args,**kwargs):
    raise PermissionError('actual client process/dependencies/resources remain unimplemented')

def _role_source(value):
    source_map(value)
    for name,digest in FIXED_CLIENT_SOURCES.items():
        if value.get(name)!=digest:raise ValueError('fixed client dependency differs: '+name)
    if not set(ACTUAL_CLIENT_MODULES)<=set(value):raise ValueError('actual client implementation closure incomplete')
    return value

@dataclass(frozen=True)
class ExpectedRolePins:
    raw:bytes
    def check(self):
        if type(self)is not ExpectedRolePins:raise TypeError('external role pins required')
        v=raw_object(self.raw,1<<20,'role pins');obj(v,CONDITIONAL_PIN_FIELDS,'role pins')
        if v['schema']!='qwen-client-role-expected-pins-v1':raise ValueError('role pin schema differs')
        role(v['role'])
        for n in CONDITIONAL_PIN_FIELDS-{'schema','role','source_map','model_evidence','policies'}:hash_value(v[n])
        if sha(canonical(_role_source(v['source_map'])))!=v['source_manifest_sha256']:raise ValueError('canonical client source map differs')
        hashes(v['model_evidence'],MODEL_FIELDS,'model evidence');hashes(v['policies'],POLICY_FIELDS,'role policies')
        for field,name in (('dependency_policy_sha256','qwen_client_dependencies.json'),('role_policy_sha256','qwen_client_role_policy.json')):
            if v['policies'][field]!=v['source_map'][name]:raise ValueError('source-bound client policy differs')
        return v

@dataclass(frozen=True)
class ConditionalRoleAdmission:
    intent:ActualKeygenIntent
    model:object
    grant_bytes:bytes
    expected_pins:ExpectedRolePins
    def check(self):return authenticate_conditional_role(self.intent,self.model,self.grant_bytes,expected_pins=self.expected_pins)
    @property
    def execution_authorized(self):return False

def authenticate_conditional_role(intent,model,grant_bytes,*,expected_pins):
    if type(intent)is not ActualKeygenIntent or type(expected_pins)is not ExpectedRolePins:raise TypeError('exact intent and independently trusted role pins required')
    v=intent.check();pins=expected_pins.check();which=pins['role']
    # This checks the caller's exact actual consumer, never consumer28 by default.
    evidence=model_pins(model,intent.public,pins['source_manifest_sha256'])
    same(evidence,pins['model_evidence'],'independent client ModelAdmission pins differ')
    if model.model_id!=v['model_id']:raise ValueError('client/generation model differs')
    if sha(intent.raw)!=pins['generation_intent_sha256']or v['generation_source_manifest_sha256']!=pins['generation_source_manifest_sha256']:
        raise ValueError('intended generation source/intent differs')
    if v[which+'_source_manifest_sha256']!=pins['source_manifest_sha256']:raise ValueError('intended named client source differs')
    grant=raw_object(grant_bytes,1<<16,'conditional role grant');obj(grant,CONDITIONAL_FIELDS,'conditional role grant')
    if sha(grant_bytes)!=pins['conditional_grant_sha256']:raise ValueError('external conditional grant digest differs')
    if grant['schema']!='qwen-client-conditional-role-grant-v1'or grant['role']!=which or grant['entry']!=ENTRY[which]:raise ValueError('conditional role/entry differs')
    path(grant['source_root'])
    for n in ('source_manifest_sha256','source_archive_sha256','readiness_review_sha256','generation_intent_sha256','generation_source_manifest_sha256','generation_source_archive_sha256'):
        if grant[n]!=pins[n]:raise ValueError('conditional '+n+' differs')
    if grant['model_id']!=v['model_id']:raise ValueError('conditional model differs')
    same(grant['model_evidence'],evidence,'conditional model proof differs');same(grant['policies'],pins['policies'],'conditional role policy differs')
    same(grant['conditions'],CONDITIONS,'all later gates remain required');same(grant['scope'],CONDITIONAL_SCOPE,'conditional scope differs')
    return ConditionalRoleAdmission(intent,model,grant_bytes,expected_pins)

@dataclass(frozen=True)
class ExpectedPhasePins:
    raw:bytes
    def check(self):
        if type(self)is not ExpectedPhasePins:raise TypeError('independent phase pins required')
        v=raw_object(self.raw,1<<16,'phase pins');obj(v,PHASE_PIN_FIELDS,'phase pins')
        if v['schema']!='qwen-client-phase-expected-pins-v1':raise ValueError('phase pin schema differs')
        role(v['role'])
        for n in ('conditional_grant_sha256','phase_grant_sha256','request_contract_sha256'):hash_value(v[n])
        hashes(v['key_evidence'],KEY_FIELDS,'independent generated-key evidence');return v

def _input(value,which,request):
    if which=='request_producer':
        obj(value,('kind','path','sha256','state','dtype','shape'),'embedding input')
        if value['kind']!='original_embeddings'or value['dtype']!='float64':raise ValueError('original float64 embeddings required')
        path(value['path']);hash_value(value['sha256']);state(value['state'])
        same(value['shape'],[1,request['input']['sequence_length'],1024],'embedding shape differs')
        raw=request['input']['sequence_length']*1024*8
        if not raw<value['state']['size']<=raw+16384:raise ValueError('bounded embedding extent differs')
        return value['path']
    obj(value,('kind','root','response_manifest_sha256','input_digest','server_evidence'),'response input')
    if value['kind']!='complete38_response':raise ValueError('complete response input required')
    root=obj(value['root'],('path','device','inode','mode'),'response root');path(root['path'])
    integer(root['device'],0,'response device');integer(root['inode'],1,'response inode');same(root['mode'],0o700,'owned response mode differs')
    for n in ('response_manifest_sha256','input_digest'):hash_value(value[n])
    hashes(value['server_evidence'],('envelope_sha256','outer_sha256','root_review_sha256','expected_pins_sha256'),'server evidence')
    return root['path']

@dataclass(frozen=True)
class PhaseAdmission:
    conditional:ConditionalRoleAdmission
    readiness:object
    request_bytes:bytes
    grant_bytes:bytes
    expected_pins:ExpectedPhasePins
    server:object=None
    def check(self):return authenticate_phase(self.conditional,self.readiness,self.request_bytes,self.grant_bytes,expected_pins=self.expected_pins,server=self.server)
    def expected_ids(self):
        self.check();return self.readiness.expected_ids(role='client')
    @property
    def execution_authorized(self):return False

def authenticate_phase(conditional,readiness,request_bytes,grant_bytes,*,expected_pins,server=None):
    if type(conditional)is not ConditionalRoleAdmission or type(expected_pins)is not ExpectedPhasePins:raise TypeError('exact retained conditional and independent phase pins required')
    conditional.check();role_pins=conditional.expected_pins.check();pins=expected_pins.check();which=role_pins['role']
    if pins['role']!=which or pins['conditional_grant_sha256']!=sha(conditional.grant_bytes):raise ValueError('phase/conditional role differs')
    keys=key_pins(readiness);same(keys,pins['key_evidence'],'independent generated-key proof differs')
    if readiness.launch.intent.raw!=conditional.intent.raw:raise ValueError('different original generation intent')
    if(keys['source_manifest_sha256']!=role_pins['generation_source_manifest_sha256']or keys['source_archive_sha256']!=role_pins['generation_source_archive_sha256']):raise ValueError('actual generation source differs from conditional grant')
    # Keygen's model consumer differs, but its four actual producer proof bytes
    # must be the same body-integrity proof as the separately repinned client.
    generated_model=readiness.launch.expected_pins.check()['model_evidence']
    for n in MODEL_FIELDS-{'expected_pins_sha256'}:
        if generated_model[n]!=role_pins['model_evidence'][n]:raise ValueError('generation/client model body proof differs')
    evidence=readiness.launch.expected_pins.check()['role_evidence'][which]
    same(evidence,dict(source_manifest_sha256=role_pins['source_manifest_sha256'],source_archive_sha256=role_pins['source_archive_sha256'],readiness_review_sha256=role_pins['readiness_review_sha256'],execution_grant_sha256=sha(conditional.grant_bytes)),'generated keys do not bind this actual client role')
    public=conditional.intent.public
    if readiness.launch.intent.public.blobs!=public.blobs:raise ValueError('key/client public metadata differs')
    request=validate_request(public,raw_object(request_bytes,1<<20,'request'))
    if sha(request_bytes)!=pins['request_contract_sha256']:raise ValueError('independent request differs')
    expected=readiness.expected_ids(role='client');expected.require_identity(request['public'],'client')
    profile=PublicProfile.from_bytes(public.check()['inputs/profile_audit.json'],public.check()['inputs/actual_key_plan.json'])
    authenticate_public(readiness.public_manifest_bytes,profile,expected,role='client')
    grant=raw_object(grant_bytes,1<<16,'phase grant');obj(grant,PHASE_FIELDS,'phase grant')
    if sha(grant_bytes)!=pins['phase_grant_sha256']:raise ValueError('external phase grant differs')
    if grant['schema']!='qwen-client-phase-grant-v1'or grant['role']!=which or grant['entry']!=ENTRY[which]or grant['execute']is not True:raise ValueError('explicit phase role/entry grant required')
    for n in ('conditional_grant_sha256','request_contract_sha256'):
        if grant[n]!=pins[n]:raise ValueError('phase '+n+' differs')
    for n in ('source_manifest_sha256','source_archive_sha256'):
        if grant[n]!=role_pins[n]:raise ValueError('phase source differs')
    same(grant['key_evidence'],keys,'phase key evidence differs');same(grant['model_evidence'],role_pins['model_evidence'],'phase model proof differs');same(grant['policies'],role_pins['policies'],'phase policies differ')
    stores=raw_object(readiness.report_bytes,2<<20,'retained key report')['stores']
    same(grant['key_stores'],stores,'independent retained key store/manifest identities differ')
    borrowed_input=_input(grant['input'],which,request)
    obj(grant['outputs'],('evidence','result'),'fresh outputs')
    for v in grant['outputs'].values():output(v)
    source_root=raw_object(conditional.grant_bytes,1<<16,'conditional grant')['source_root']
    model_roots=[v['path']for v in conditional.model.stores.check()['roots'].values()]
    protected=[source_root,stores['public']['path'],stores['private']['path']]+model_roots
    disjoint([borrowed_input],protected)
    disjoint([v['path']for v in grant['outputs'].values()],protected+[borrowed_input])
    same(grant['scope'],dict(request_encryption=which=='request_producer',response_decryption=which=='response_consumer',key_generation=False,model_body_reads=False,model_forward=False,token_or_calibration_reads=False,scientific_acceptance=False),'exact phase scope differs')
    if which=='request_producer':
        if server is not None:raise ValueError('request cannot carry response admission')
    else:
        from client_server_terminal import ServerAdmission
        if type(server)is not ServerAdmission:raise TypeError('separate actual server terminal admission required')
        server.check()
        if server.readiness is not readiness or server.public.blobs!=public.blobs or server.request_bytes!=request_bytes:raise ValueError('server/client retained request or key evidence differs')
        same(grant['input']['server_evidence'],server.evidence(),'phase/server evidence differs')
        if(sha(server.response_bytes)!=grant['input']['response_manifest_sha256']or server.input_digest!=grant['input']['input_digest']):raise ValueError('phase/server response differs')
    return PhaseAdmission(conditional,readiness,request_bytes,grant_bytes,expected_pins,server)
