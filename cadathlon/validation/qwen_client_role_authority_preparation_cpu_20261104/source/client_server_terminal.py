"""Pure separate actual-server evidence; preserves existing inner file schemas.

This module opens no response/key/model files and invokes no producer. Independent
expected pins and outer/root observations remain external trusted inputs.
"""
from dataclasses import dataclass
from qwen_file_contract import canonical,sha,same,hash_value,validate_request,validate_response
from client_role_common import (obj,raw_object,integer,hashes,source_map,model_pins,key_pins,MODEL_FIELDS,KEY_FIELDS)

PIN_FIELDS={'schema','server_source_manifest_sha256','server_source_archive_sha256','server_source_map','server_grant_sha256','server_role_review_sha256','scientific_authorization_sha256','model_evidence','key_evidence','request_contract_sha256','input_digest','response_manifest_sha256','inner_report_sha256','inner_outer_sha256','envelope_sha256','outer_sha256','root_review_sha256'}
CHECKS=('source_precheck_completed','source_postcheck_completed','model_admission_authenticated','key_readiness_authenticated','all335_public_arrays_authenticated','private_key_absent','request_payloads_authenticated','parameters_rechecked','all38_security_and_barriers_completed','all38_published','original_owners_retired','cleanup_completed','signals_restored')

def create_actual_server_consumer(*args,**kwargs):
    raise PermissionError('server or response-body execution is not supplied by pure terminal evidence')

@dataclass(frozen=True)
class ExpectedServerPins:
    raw:bytes
    def check(self):
        if type(self)is not ExpectedServerPins:raise TypeError('independent server pins required')
        v=raw_object(self.raw,1<<20,'server pins');obj(v,PIN_FIELDS,'server pins')
        if v['schema']!='qwen-actual-server-expected-pins-v1':raise ValueError('server pin schema differs')
        for n in PIN_FIELDS-{'schema','server_source_map','model_evidence','key_evidence'}:hash_value(v[n])
        if sha(canonical(source_map(v['server_source_map'])))!=v['server_source_manifest_sha256']:raise ValueError('canonical actual server map differs')
        hashes(v['model_evidence'],MODEL_FIELDS,'server model evidence');hashes(v['key_evidence'],KEY_FIELDS,'server key evidence');return v

def _inner(public,request,input_digest,response_bytes,report_bytes,outer_bytes,pins):
    # Pure equivalent of373 ProducerTerminal.authenticate. The producer_source
    # below is REQUEST provenance, not the separately authenticated server map.
    if sha(report_bytes)!=pins['inner_report_sha256']or sha(outer_bytes)!=pins['inner_outer_sha256']:raise ValueError('external inner file-proof digest differs')
    report=raw_object(report_bytes,1<<16,'inner producer report');outer=raw_object(outer_bytes,1<<16,'inner producer outer')
    expected=dict(schema='qwen-organizer-server-report-v1',
        producer_source_manifest_sha256=request['public']['producer_source_manifest_sha256'],
        consumer_source_manifest_sha256=request['public']['consumer_source_manifest_sha256'],
        context_id=request['public']['context_id'],model_id=request['public']['model_id'],request_contract_sha256=sha(canonical(request)),
        input_digest=input_digest,response_manifest_sha256=sha(response_bytes),output_count=38,
        body_completed=True,source_unchanged=True,parameters_rechecked=True,cleanup_completed=True,termination_scope_completed=True,passed=True)
    same(report,expected,'unchanged inner request-provenance report differs')
    for n in ('server_pid','server_pgid'):integer(outer.get(n),1,'inner '+n)
    same(outer,dict(schema='qwen-organizer-server-outer-v1',report_sha256=pins['inner_report_sha256'],
        producer_source_manifest_sha256=expected['producer_source_manifest_sha256'],request_contract_sha256=sha(canonical(request)),
        server_pid=outer['server_pid'],server_pgid=outer['server_pgid'],exit_code=0,reaped=True,process_group_absent=True),'unchanged inner producer outer differs')
    return dict(pid=outer['server_pid'],pgid=outer['server_pgid'])

@dataclass(frozen=True)
class ServerAdmission:
    public:object
    model:object
    readiness:object
    request_bytes:bytes
    input_digest:str
    response_bytes:bytes
    inner_report_bytes:bytes
    inner_outer_bytes:bytes
    envelope_bytes:bytes
    outer_bytes:bytes
    root_review_bytes:bytes
    expected_pins:ExpectedServerPins
    def check(self):
        return authenticate_server(self.public,self.model,self.readiness,self.request_bytes,self.input_digest,self.response_bytes,
            self.inner_report_bytes,self.inner_outer_bytes,self.envelope_bytes,self.outer_bytes,self.root_review_bytes,expected_pins=self.expected_pins)
    def evidence(self):
        self.check();return dict(envelope_sha256=sha(self.envelope_bytes),outer_sha256=sha(self.outer_bytes),root_review_sha256=sha(self.root_review_bytes),expected_pins_sha256=sha(self.expected_pins.raw))
    @property
    def execution_authorized(self):return False

def authenticate_server(public,model,readiness,request_bytes,input_digest,response_bytes,
                        inner_report_bytes,inner_outer_bytes,envelope_bytes,outer_bytes,root_review_bytes,*,expected_pins):
    if type(expected_pins)is not ExpectedServerPins:raise TypeError('independent actual server expectations required')
    pins=expected_pins.check();model_evidence=model_pins(model,public,pins['server_source_manifest_sha256'])
    same(model_evidence,pins['model_evidence'],'independent server model proof differs')
    keys=key_pins(readiness);same(keys,pins['key_evidence'],'independent server key proof differs')
    if readiness.launch.intent.public.blobs!=public.blobs:raise ValueError('server/key public metadata differs')
    prior_model=readiness.launch.expected_pins.check()['model_evidence']
    for n in MODEL_FIELDS-{'expected_pins_sha256'}:
        if prior_model[n]!=model_evidence[n]:raise ValueError('server/keygen model integrity proof differs')
    c=validate_request(public,raw_object(request_bytes,1<<20,'request'))
    readiness.expected_ids(role='server').require_identity(c['public'],'server')
    hash_value(input_digest)
    if(sha(request_bytes)!=pins['request_contract_sha256']or input_digest!=pins['input_digest']or sha(response_bytes)!=pins['response_manifest_sha256']):raise ValueError('independent server input/output differs')
    validate_response(public,c,input_digest,raw_object(response_bytes,1<<20,'complete response'))
    process=_inner(public,c,input_digest,response_bytes,inner_report_bytes,inner_outer_bytes,pins)
    for raw,field,limit in ((envelope_bytes,'envelope_sha256',2<<20),(outer_bytes,'outer_sha256',1<<16),(root_review_bytes,'root_review_sha256',1<<16)):
        raw_object(raw,limit,field)
        if sha(raw)!=pins[field]:raise ValueError('external actual server '+field+' differs')
    links={n:pins[n]for n in ('server_source_manifest_sha256','server_source_archive_sha256','server_grant_sha256','server_role_review_sha256','scientific_authorization_sha256','request_contract_sha256','input_digest','response_manifest_sha256','inner_report_sha256','inner_outer_sha256')}
    links|=dict(context_id=c['public']['context_id'],model_id=c['public']['model_id'],model_evidence=model_evidence,key_evidence=keys)
    envelope=raw_object(envelope_bytes,2<<20,'actual server envelope')
    same(envelope,dict(schema='qwen-actual-server-envelope-v1',mode='full28_keyless_server',passed=True,complete=True,error_type=None,
        **links,source_hashes_before=pins['server_source_map'],source_hashes_after=pins['server_source_map'],process=process,
        checks={n:True for n in CHECKS},audit_denials=[],
        scope=dict(full28_encrypted_evaluation=True,private_key_read=False,client_codec_called=False,key_generation=False,coefficient_revision=False,scientific_acceptance_claimed=False)),
        'complete actual server source/role/retirement envelope differs')
    outer=raw_object(outer_bytes,1<<16,'actual server outer')
    outer_links={n:links[n]for n in ('server_source_manifest_sha256','server_source_archive_sha256','server_grant_sha256','request_contract_sha256','response_manifest_sha256')}
    same(outer,dict(schema='qwen-actual-server-outer-v1',envelope_sha256=sha(envelope_bytes),**outer_links,**process,
        exit_code=0,normal_return=True,reaped=True,process_group_absent=True),'normal-zero independently observed server terminal differs')
    review=raw_object(root_review_bytes,1<<16,'actual server root review')
    same(review,dict(schema='qwen-actual-server-root-admission-v1',passed=True,envelope_sha256=sha(envelope_bytes),outer_sha256=sha(outer_bytes),
        **links,source_archive_members_verified=True,all_required_links_verified=True,current_response_store_verified=True,
        scientific_authorization_verified=True,client_execution_authorized=False),'independent actual server root links differ')
    return ServerAdmission(public,model,readiness,request_bytes,input_digest,response_bytes,inner_report_bytes,inner_outer_bytes,envelope_bytes,outer_bytes,root_review_bytes,expected_pins)
