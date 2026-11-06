"""Metadata-only fabricated client chains. Never actual files, grants or readiness.

Every runtime source-map addition, role grant, store/process identity and output
artifact hash is explicitly SYNTHETIC. Public fixed metadata alone is historical.
No arrays, payload/header reads, original model receipt or backend are used.
"""
import json
from types import SimpleNamespace
from qwen_file_contract import canonical,sha,request,artifact,response,DELTA
from client_role_source_pins import FIXED_CLIENT_SOURCES,ACTUAL_CLIENT_MODULES
from client_role_admission import (ExpectedRolePins,ExpectedPhasePins,authenticate_conditional_role,
    authenticate_phase,ENTRY,CONDITIONS,CONDITIONAL_SCOPE)
from client_server_terminal import ExpectedServerPins,authenticate_server,CHECKS
from client_role_common import key_pins,model_pins
from synthetic_keygen_chain import fake_hash,clone,synthetic_model
from synthetic_keygen_chain_client import build_chain


def role_map(label):
    value=dict(FIXED_CLIENT_SOURCES)
    for name in ACTUAL_CLIENT_MODULES:value.setdefault(name,fake_hash('SYNTHETIC '+label+' '+name))
    return value


def build_client_chain(public,aggregate,*,same_maps=False,change_role_source=None,
                       change_conditional=None,change_role_pins=None,
                       client_model_consumer=None,change_client_model=None,
                       change_generation_roles=None,**key_changes):
    maps={r:role_map('shared'if same_maps else r)for r in ENTRY}
    if change_role_source:
        for role,value in maps.items():change_role_source(role,value)
    hashes={r:sha(canonical(m))for r,m in maps.items()};conditional={}
    def role_builder(intent,generation_source,generation_archive,model):
        result={}
        for which in ENTRY:
            consumer=client_model_consumer or hashes[which]
            change=None if change_client_model is None else lambda row:change_client_model(which,row)
            client_model=synthetic_model(public,aggregate,consumer,change)
            evidence=dict(manifest_sha256=sha(client_model.manifest_bytes),report_sha256=sha(client_model.report_bytes),
                outer_sha256=sha(client_model.outer_bytes),root_review_sha256=sha(client_model.review_bytes),expected_pins_sha256=sha(client_model.expected_pins.raw))
            policies=dict(dependency_policy_sha256=maps[which]['qwen_client_dependencies.json'],
                role_policy_sha256=maps[which]['qwen_client_role_policy.json'],resource_policy_sha256=fake_hash(which+' resource policy'),resource_review_sha256=fake_hash(which+' resource review'))
            grant=dict(schema='qwen-client-conditional-role-grant-v1',role=which,entry=ENTRY[which],
                source_root='/synthetic-unopened-source/client-'+which,source_manifest_sha256=hashes[which],
                source_archive_sha256=fake_hash(which+' source archive'),readiness_review_sha256=fake_hash(which+' readiness review'),
                generation_intent_sha256=sha(intent.raw),generation_source_manifest_sha256=generation_source,
                generation_source_archive_sha256=generation_archive,model_id=client_model.model_id,
                model_evidence=evidence,policies=policies,conditions=clone(CONDITIONS),scope=clone(CONDITIONAL_SCOPE))
            if change_conditional:change_conditional(which,grant)
            raw=canonical(grant)
            pins=dict(schema='qwen-client-role-expected-pins-v1',role=which,source_map=maps[which],
                source_manifest_sha256=hashes[which],source_archive_sha256=grant['source_archive_sha256'],
                readiness_review_sha256=grant['readiness_review_sha256'],conditional_grant_sha256=sha(raw),
                generation_intent_sha256=sha(intent.raw),generation_source_manifest_sha256=generation_source,
                generation_source_archive_sha256=generation_archive,model_evidence=evidence,policies=policies)
            if change_role_pins:change_role_pins(which,pins)
            expected=ExpectedRolePins(canonical(pins))
            conditional[which]=SimpleNamespace(intent=intent,model=client_model,grant=grant,raw=raw,pins=pins,expected=expected)
            result[which]=dict(source_manifest_sha256=hashes[which],source_archive_sha256=pins['source_archive_sha256'],
                readiness_review_sha256=pins['readiness_review_sha256'],execution_grant_sha256=sha(raw))
        if change_generation_roles:change_generation_roles(result)
        return result
    key=build_chain(public,aggregate,role_sources=hashes,role_builder=role_builder,**key_changes)
    def admit(which):
        c=conditional[which]
        return authenticate_conditional_role(c.intent,c.model,c.raw,expected_pins=c.expected)
    def make_request(ready,positions=(2,7,11,19,25,38,50,61,75,90,101,115,126,139,153,170,201)):
        ids=ready.expected_ids(role='client')
        identity=public.identity(context_id=ids.context_id,model_id=ids.model_id,
            producer_sources=ids.producer_source_manifest_sha256,consumer_sources=ids.consumer_source_manifest_sha256)
        return canonical(request(public,identity,'synthetic-request',list(positions)))
    return SimpleNamespace(public=public,aggregate=aggregate,maps=maps,key=key,conditional=conditional,admit=admit,request=make_request)


def server_bundle(chain,ready,request_raw,*,change_response=None,change_inner=None,change_inner_outer=None,
                  change_envelope=None,change_outer=None,change_root=None,change_model=None,change_pins=None):
    public=chain.public;c=json.loads(request_raw);input_digest=fake_hash('SYNTHETIC input payload inventory')
    rows=[artifact(public,c,'response',i,payload_sha256=fake_hash('SYNTHETIC unopened ciphertext '+str(i)),
                   actual_scale=DELTA,input_digest=input_digest)for i in range(38)]
    res=response(public,c,input_digest,rows)
    if change_response:change_response(res)
    response_raw=canonical(res)
    source={'synthetic_actual_server.py':fake_hash('SYNTHETIC independently reviewed server source')};source_hash=sha(canonical(source))
    model=synthetic_model(public,chain.aggregate,source_hash,change_model)
    model_evidence=dict(manifest_sha256=sha(model.manifest_bytes),report_sha256=sha(model.report_bytes),outer_sha256=sha(model.outer_bytes),root_review_sha256=sha(model.review_bytes),expected_pins_sha256=sha(model.expected_pins.raw))
    keys=key_pins(ready)
    inner=dict(schema='qwen-organizer-server-report-v1',producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],consumer_source_manifest_sha256=c['public']['consumer_source_manifest_sha256'],context_id=c['public']['context_id'],model_id=c['public']['model_id'],request_contract_sha256=sha(request_raw),input_digest=input_digest,response_manifest_sha256=sha(response_raw),output_count=38,body_completed=True,source_unchanged=True,parameters_rechecked=True,cleanup_completed=True,termination_scope_completed=True,passed=True)
    if change_inner:change_inner(inner)
    inner_raw=canonical(inner)
    inner_outer=dict(schema='qwen-organizer-server-outer-v1',report_sha256=sha(inner_raw),producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],request_contract_sha256=sha(request_raw),server_pid=707,server_pgid=706,exit_code=0,reaped=True,process_group_absent=True)
    if change_inner_outer:change_inner_outer(inner_outer)
    inner_outer_raw=canonical(inner_outer)
    links=dict(server_source_manifest_sha256=source_hash,server_source_archive_sha256=fake_hash('SYNTHETIC server archive'),server_grant_sha256=fake_hash('SYNTHETIC server grant'),server_role_review_sha256=fake_hash('SYNTHETIC server role review'),scientific_authorization_sha256=fake_hash('SYNTHETIC independent scientific authorization'),request_contract_sha256=sha(request_raw),input_digest=input_digest,response_manifest_sha256=sha(response_raw),inner_report_sha256=sha(inner_raw),inner_outer_sha256=sha(inner_outer_raw),context_id=c['public']['context_id'],model_id=c['public']['model_id'],model_evidence=model_evidence,key_evidence=keys)
    process=dict(pid=707,pgid=706)
    envelope=dict(schema='qwen-actual-server-envelope-v1',mode='full28_keyless_server',passed=True,complete=True,error_type=None,**links,source_hashes_before=clone(source),source_hashes_after=clone(source),process=process,checks={n:True for n in CHECKS},audit_denials=[],scope=dict(full28_encrypted_evaluation=True,private_key_read=False,client_codec_called=False,key_generation=False,coefficient_revision=False,scientific_acceptance_claimed=False))
    if change_envelope:change_envelope(envelope)
    envelope_raw=canonical(envelope)
    outer_links={n:links[n]for n in ('server_source_manifest_sha256','server_source_archive_sha256','server_grant_sha256','request_contract_sha256','response_manifest_sha256')}
    outer=dict(schema='qwen-actual-server-outer-v1',envelope_sha256=sha(envelope_raw),**outer_links,**process,exit_code=0,normal_return=True,reaped=True,process_group_absent=True)
    if change_outer:change_outer(outer)
    outer_raw=canonical(outer)
    root=dict(schema='qwen-actual-server-root-admission-v1',passed=True,envelope_sha256=sha(envelope_raw),outer_sha256=sha(outer_raw),**links,source_archive_members_verified=True,all_required_links_verified=True,current_response_store_verified=True,scientific_authorization_verified=True,client_execution_authorized=False)
    if change_root:change_root(root)
    root_raw=canonical(root)
    pins=dict(schema='qwen-actual-server-expected-pins-v1',**{n:links[n]for n in links if n not in ('context_id','model_id')},server_source_map=source,envelope_sha256=sha(envelope_raw),outer_sha256=sha(outer_raw),root_review_sha256=sha(root_raw))
    if change_pins:change_pins(pins)
    expected=ExpectedServerPins(canonical(pins))
    def admit():return authenticate_server(public,model,ready,request_raw,input_digest,response_raw,inner_raw,inner_outer_raw,envelope_raw,outer_raw,root_raw,expected_pins=expected)
    return SimpleNamespace(model=model,input_digest=input_digest,response=res,response_raw=response_raw,inner=inner,inner_raw=inner_raw,inner_outer=inner_outer,inner_outer_raw=inner_outer_raw,envelope=envelope,envelope_raw=envelope_raw,outer=outer,outer_raw=outer_raw,root=root,root_raw=root_raw,pins=pins,expected=expected,admit=admit)


def phase_bundle(chain,which,*,ready=None,request_raw=None,server=None,change_grant=None,change_pins=None):
    ready=chain.key.ready()if ready is None else ready
    request_raw=chain.request(ready)if request_raw is None else request_raw
    conditional=chain.admit(which);rp=conditional.expected_pins.check();keys=key_pins(ready);c=json.loads(request_raw)
    if which=='request_producer':
        length=c['input']['sequence_length'];inp=dict(kind='original_embeddings',path='/synthetic-unopened-input/original.npy',sha256=fake_hash('SYNTHETIC original embedding bytes'),state=dict(device=8,inode=801,size=length*1024*8+128,mtime_ns=1,ctime_ns=2,mode=0o600),dtype='float64',shape=[1,length,1024])
    else:
        if server is None:server=server_bundle(chain,ready,request_raw).admit()
        inp=dict(kind='complete38_response',root=dict(path='/synthetic-unopened-response/result',device=8,inode=802,mode=0o700),response_manifest_sha256=sha(server.response_bytes),input_digest=server.input_digest,server_evidence=server.evidence())
    outputs={name:dict(path='/synthetic-unopened-phase/'+name,parent=dict(path='/synthetic-unopened-phase',device=8,inode=810,mode=0o700))for name in ('result','evidence')}
    grant=dict(schema='qwen-client-phase-grant-v1',role=which,entry=ENTRY[which],execute=True,conditional_grant_sha256=sha(conditional.grant_bytes),source_manifest_sha256=rp['source_manifest_sha256'],source_archive_sha256=rp['source_archive_sha256'],model_evidence=rp['model_evidence'],key_evidence=keys,request_contract_sha256=sha(request_raw),policies=rp['policies'],key_stores=json.loads(ready.report_bytes)['stores'],input=inp,outputs=outputs,scope=dict(request_encryption=which=='request_producer',response_decryption=which=='response_consumer',key_generation=False,model_body_reads=False,model_forward=False,token_or_calibration_reads=False,scientific_acceptance=False))
    if change_grant:change_grant(grant)
    raw=canonical(grant)
    pins=dict(schema='qwen-client-phase-expected-pins-v1',role=which,conditional_grant_sha256=sha(conditional.grant_bytes),phase_grant_sha256=sha(raw),request_contract_sha256=sha(request_raw),key_evidence=keys)
    if change_pins:change_pins(pins)
    expected=ExpectedPhasePins(canonical(pins))
    def admit():return authenticate_phase(conditional,ready,request_raw,raw,expected_pins=expected,server=server)
    return SimpleNamespace(ready=ready,conditional=conditional,request_raw=request_raw,server=server,grant=grant,raw=raw,pins=pins,expected=expected,admit=admit)
