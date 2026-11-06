"""Pure two-step server packet admission; never opens a proof or body path.

The packet's digest comes from an independent launch review. Expected pins
inside it inherit that external authority; none are derived from outcomes.
Preliminary inspection is metadata, not scientific or filesystem permission.
"""
from dataclasses import dataclass
from qwen_file_contract import (PublicContract,canonical,sha,hash_value,same,
    validate_request,validate_request_manifest)
from public_source_pins import PUBLIC_PINS
from client_role_common import raw_object,obj,path,integer,output,disjoint,model_pins,key_pins,MODEL_FIELDS
from model_admission import ExpectedModelPins,authenticate_model
from keygen_authority import (ActualKeygenIntent,ExpectedLaunchPins,ExpectedReadyPins,
    authenticate_launch,authenticate_readiness)
from qwen_server_admission import (ExpectedServerPhasePins,ServerPhaseAdmission,
    authenticate_server_phase,GRANT_FIELDS,ENTRY,SCOPE as SERVER_SCOPE)
from qwen_scientific_admission import (ExpectedScientificPins,ScientificAdmission,
    authenticate_scientific,PROOF_LINKS,PROOF_LIMITS,PROOF_BUNDLE_LIMIT,
    CHECKS as SCIENTIFIC_CHECKS,SCOPE as SCIENTIFIC_SCOPE)

PACKET_LIMIT=16<<20
FIELDS={'schema','generation_intent','generation_model','generation_launch_grant','generation_launch_pins',
    'generated_public_manifest','generated_report','generated_outer','generated_root_review','generated_ready_pins',
    'server_model','request','request_manifest','phase_grant','phase_pins','scientific_statement','scientific_expected_pins',
    'scientific_proofs','dependency_policy','role_policy','resource_policy','resource_review'}
POLICIES=('dependency_policy','role_policy','resource_policy','resource_review')
DESCRIPTOR_FIELDS={'name','path','sha256','limit'}


def _bytes(value,label,limit=2<<20):
    if type(value)is not str:raise TypeError('exact UTF-8 public '+label+' bytes required')
    raw=value.encode('utf-8')
    if not 0<len(raw)<=limit:raise ValueError('bounded public '+label+' required')
    return raw


def _model(public,value):
    obj(value,('manifest','report','outer','root_review','expected_pins'),'public model proof bundle')
    return authenticate_model(public,_bytes(value['manifest'],'model manifest'),_bytes(value['report'],'model report'),
        _bytes(value['outer'],'model outer',1<<16),_bytes(value['root_review'],'model root review',1<<16),
        expected_pins=ExpectedModelPins(_bytes(value['expected_pins'],'model external pins',1<<20)))


def public_from_sources(blobs):
    if type(blobs)is not dict:raise TypeError('already authenticated complete source bytes required')
    return PublicContract.from_blobs({name:blobs['public_inputs/'+name]for name in PUBLIC_PINS})


def _packet(public,raw,expected_sha256):
    if type(public)is not PublicContract:raise TypeError('exact source-bound public contract required')
    public.check();hash_value(expected_sha256)
    if type(raw)is not bytes or not 0<len(raw)<=PACKET_LIMIT:raise ValueError('bounded immutable server authority packet required')
    if sha(raw)!=expected_sha256:raise ValueError('independently supplied server packet digest differs')
    packet=raw_object(raw,PACKET_LIMIT,'server public authority packet');obj(packet,FIELDS,'server public authority packet')
    if packet['schema']!='qwen-server-authority-packet-v1':raise ValueError('server public authority packet schema differs')
    return packet


def _model_fields(value):
    obj(value,('manifest','report','outer','root_review','expected_pins'),'public model proof bundle')
    for name,limit in(('manifest',2<<20),('report',2<<20),('outer',1<<16),('root_review',1<<16),('expected_pins',1<<20)):
        _bytes(value[name],'public model '+name,limit)


def _embedded_bounds(packet):
    _model_fields(packet['generation_model']);_model_fields(packet['server_model'])
    bounds={'generation_intent':1<<16,'generation_launch_grant':1<<16,'generation_launch_pins':1<<20,
        'generated_public_manifest':1<<20,'generated_report':2<<20,'generated_outer':1<<16,
        'generated_root_review':1<<16,'generated_ready_pins':1<<16,
        'request':1<<20,'request_manifest':1<<20,'phase_grant':1<<16,'phase_pins':1<<20,
        'scientific_statement':1<<16,'scientific_expected_pins':2<<20,
        **{name:1<<20 for name in POLICIES}}
    for name,limit in bounds.items():_bytes(packet[name],name,limit)


def _preliminary_grant(raw,pins):
    """Structural/grant/source binding only; full phase admission comes later."""
    grant=raw_object(raw,1<<16,'server phase grant');obj(grant,GRANT_FIELDS,'server phase grant')
    if sha(raw)!=pins['phase_grant_sha256']:raise ValueError('independent server phase grant differs')
    if(grant['schema']!='qwen-server-phase-grant-v1'or grant['role']!='full28_server'
       or grant['entry']!=ENTRY or grant['execute']is not True):raise ValueError('exact independently granted server role required')
    path(grant['source_root'])
    for name in('source_manifest_sha256','source_archive_sha256','readiness_review_sha256','request_contract_sha256','model_store_sha256','scientific_authorization_sha256'):
        same(grant[name],pins[name],'server grant/source prerequisite differs')
    for name in('model_evidence','key_evidence','policies'):same(grant[name],pins[name],'server grant/public proof pins differ')
    same(grant['scope'],SERVER_SCOPE,'exact server-only scope required')
    row=obj(grant['input'],('kind','root','request_manifest_sha256','input_digest'),'server input')
    if row['kind']!='fresh_encrypted_request':raise ValueError('original encrypted request required')
    root=obj(row['root'],('path','device','inode','mode'),'request root');path(root['path'])
    integer(root['device'],0,'request root device');integer(root['inode'],1,'request root inode');same(root['mode'],0o700,'request root mode differs')
    for name in('request_manifest_sha256','input_digest'):same(row[name],pins[name],'server granted request differs')
    obj(grant['outputs'],('response','evidence'),'server output roots')
    for value in grant['outputs'].values():output(value)
    # The actual key/model store schema and all root relations are checked by
    # the complete phase. No borrowed store is opened during inspection.
    if type(grant['public_key_store'])is not dict:raise ValueError('public key store metadata required')
    return grant


@dataclass(frozen=True)
class ProofDescriptor:
    name:str
    path:str
    sha256:str
    limit:int
    def check(self):
        if type(self)is not ProofDescriptor or type(self.name)is not str or self.name not in PROOF_LINKS:raise TypeError('exact named scientific proof descriptor required')
        path(self.path);hash_value(self.sha256);integer(self.limit,PROOF_LIMITS[self.name],'fixed scientific proof limit',PROOF_LIMITS[self.name])
        return self


def _descriptors(value,scientific_pins):
    obj(value,PROOF_LINKS,'exact25 separate scientific proof descriptors')
    rows=[];paths={}
    for name in sorted(PROOF_LINKS):
        row=obj(value[name],DESCRIPTOR_FIELDS,'scientific proof descriptor')
        if row['name']!=name:raise ValueError('scientific descriptor name differs')
        desc=ProofDescriptor(**row);desc.check()
        group,field=PROOF_LINKS[name]
        if desc.sha256!=scientific_pins['evidence'][group][field]:raise ValueError('scientific descriptor lacks independent expected digest')
        if desc.path in paths and paths[desc.path]!=(desc.sha256,desc.limit):raise ValueError('one public proof path has inconsistent bindings')
        paths[desc.path]=(desc.sha256,desc.limit);rows.append(desc)
    if sum(row.limit for row in rows)!=PROOF_BUNDLE_LIMIT:raise ValueError('scientific descriptor total bound differs')
    return tuple(rows)


def _base_admissions(public,packet,pins):
    """Authenticate small public model/key evidence, without scientific files."""
    intent=ActualKeygenIntent(public,_bytes(packet['generation_intent'],'generation intent',1<<16))
    launch=authenticate_launch(intent,_bytes(packet['generation_launch_grant'],'generation grant',1<<16),
        _model(public,packet['generation_model']),expected_pins=ExpectedLaunchPins(_bytes(packet['generation_launch_pins'],'generation launch pins',1<<20)))
    ready=authenticate_readiness(launch,_bytes(packet['generated_public_manifest'],'public key manifest',1<<20),
        _bytes(packet['generated_report'],'generated-key report'),_bytes(packet['generated_outer'],'generated-key outer',1<<16),
        _bytes(packet['generated_root_review'],'generated-key root review',1<<16),
        expected_pins=ExpectedReadyPins(_bytes(packet['generated_ready_pins'],'generated-key external pins',1<<16)))
    model=_model(public,packet['server_model']);current=model_pins(model,public,pins['source_manifest_sha256'])
    same(current,pins['model_evidence'],'independent actual server model proof differs')
    same(key_pins(ready),pins['key_evidence'],'independent generated-key readiness differs')
    prior=ready.launch.expected_pins.check()['model_evidence']
    for name in MODEL_FIELDS-{'expected_pins_sha256'}:same(current[name],prior[name],'keygen/server actual model proof differs')
    if ready.launch.intent.public.blobs!=public.blobs or model.model_id!=intent.check()['model_id']:raise ValueError('one source-bound original public model required')
    return model,ready


@dataclass(frozen=True)
class ServerPacketInspection:
    public:PublicContract
    raw:bytes
    expected_sha256:str
    proof_descriptors:tuple
    phase_pins_bytes:bytes
    scientific_statement_bytes:bytes
    scientific_expected_pins_bytes:bytes
    source_root:str
    model_roots:tuple
    key_roots:tuple
    request_root:str
    output_roots:tuple
    dependency_policy:bytes
    role_policy:bytes
    resource_policy:bytes
    resource_review:bytes
    def check(self):
        if type(self)is not ServerPacketInspection:raise TypeError('exact preliminary packet inspection required')
        rebuilt=inspect_packet(self.public,self.raw,expected_sha256=self.expected_sha256)
        if type(self.proof_descriptors)is not tuple or len(self.proof_descriptors)!=len(PROOF_LINKS):raise ValueError('immutable complete descriptor tuple required')
        for original,current in zip(self.proof_descriptors,rebuilt.proof_descriptors):
            if type(original)is not ProofDescriptor:raise TypeError('exact retained proof descriptor required')
            original.check()
            if(original.name,original.path,original.sha256,original.limit)!=(current.name,current.path,current.sha256,current.limit):raise ValueError('retained scientific descriptor differs')
        for name in('phase_pins_bytes','scientific_statement_bytes','scientific_expected_pins_bytes',*POLICIES):
            if type(getattr(self,name))is not bytes or getattr(self,name)!=getattr(rebuilt,name):raise ValueError('retained preliminary public metadata differs')
        for name in('source_root','request_root'):
            if type(getattr(self,name))is not str or getattr(self,name)!=getattr(rebuilt,name):raise ValueError('retained source/request path scope differs')
        for name in('model_roots','key_roots','output_roots'):
            if type(getattr(self,name))is not tuple or any(type(x)is not str for x in getattr(self,name))or getattr(self,name)!=getattr(rebuilt,name):raise ValueError('retained borrowed/output path scope differs')
        return self
    @property
    def forbidden_proof_roots(self):
        self.check();return(self.source_root,*self.model_roots,*self.key_roots,self.request_root,*self.output_roots)
    @property
    def scientific_proof_bytes_ceiling(self):return PROOF_BUNDLE_LIMIT
    @property
    def execution_authorized(self):return False
    @property
    def proof_reads_authorized(self):return False


def inspect_packet(public,raw,*,expected_sha256):
    """Authenticate small metadata before the separate reader/resource boundary.

    Does not authenticate the large scientific proofs, authorize reading their
    paths, inspect capacity, or substitute for full ServerPhaseAdmission.
    """
    packet=_packet(public,raw,expected_sha256);_embedded_bounds(packet)
    phase_raw=_bytes(packet['phase_pins'],'independent server pins',1<<20)
    pins=ExpectedServerPhasePins(phase_raw).check()
    if 'qwen_server_authority_packet.py'not in pins['source_map']:raise ValueError('exact public packet module must be selected')
    scientific_raw=_bytes(packet['scientific_expected_pins'],'independent scientific pins',2<<20)
    scientific_pins=ExpectedScientificPins(scientific_raw).check()
    statement_raw=_bytes(packet['scientific_statement'],'scientific statement',1<<16)
    if sha(statement_raw)!=scientific_pins['statement_sha256']or sha(statement_raw)!=pins['scientific_authorization_sha256']:raise ValueError('independent scientific statement differs')
    statement=raw_object(statement_raw,1<<16,'scientific statement');obj(statement,('schema','binding','evidence','checks','scope'),'scientific statement')
    if statement['schema']!='qwen-full28-scientific-admission-v1':raise ValueError('scientific statement schema differs')
    same(statement['binding'],scientific_pins['bindings'],'scientific external binding differs')
    same(statement['evidence'],scientific_pins['evidence'],'scientific external evidence differs')
    same(statement['checks'],{name:True for name in SCIENTIFIC_CHECKS},'complete scientific statement checks required')
    same(statement['scope'],SCIENTIFIC_SCOPE,'scientific scope differs')
    same(scientific_pins['server_source_map'],pins['source_map'],'scientific and phase complete source maps differ')
    bindings=scientific_pins['bindings']
    for name in('source_manifest_sha256','source_archive_sha256'):
        same(bindings['server_'+name],pins[name],'scientific and phase source identity differs')
    for name in('request_contract_sha256','request_manifest_sha256','input_digest'):same(bindings[name],pins[name],'scientific and phase request identity differs')
    request_raw=_bytes(packet['request'],'server request',1<<20);request=raw_object(request_raw,1<<20,'server request');validate_request(public,request)
    upload_raw=_bytes(packet['request_manifest'],'request upload',1<<20);upload=raw_object(upload_raw,1<<20,'request upload');validate_request_manifest(public,request,upload)
    if sha(request_raw)!=pins['request_contract_sha256']or sha(upload_raw)!=pins['request_manifest_sha256']or upload['input_digest']!=pins['input_digest']:raise ValueError('independent request/upload/input differs')
    grant=_preliminary_grant(_bytes(packet['phase_grant'],'server phase grant',1<<16),pins)
    model,ready=_base_admissions(public,packet,pins)
    ids=ready.expected_ids(role='server');ids.require_identity(request['public'],'server')
    for name,value in(('context_id',ids.context_id),('public_manifest_sha256',ids.public_manifest_sha256),('model_id',model.model_id)):
        same(bindings[name],value,'scientific statement and actual model/key identity differ')
    current_model=model_pins(model,public,pins['source_manifest_sha256'])
    same({name:scientific_pins['evidence']['model'][name]for name in current_model},current_model,'scientific and actual server model evidence differs')
    stores=model.stores;store_rows=stores.check()
    if sha(stores.raw)!=pins['model_store_sha256']:raise ValueError('independent model store identity differs')
    generated_stores=raw_object(ready.report_bytes,2<<20,'generated key report')['stores']
    same(grant['public_key_store'],generated_stores['public'],'server public key store differs')
    source_root=str(path(grant['source_root']));request_root=str(path(grant['input']['root']['path']))
    model_roots=tuple(str(path(v['path']))for k,v in sorted(store_rows['roots'].items()))
    key_roots=tuple(str(path(v['path']))for k,v in sorted(generated_stores.items()))
    output_roots=tuple(str(path(v['path']))for k,v in sorted(grant['outputs'].items()))
    disjoint(key_roots,[source_root,*model_roots]);disjoint([request_root],[source_root,*model_roots,*key_roots])
    disjoint(output_roots,[source_root,*model_roots,*key_roots,request_root])
    descriptors=_descriptors(packet['scientific_proofs'],scientific_pins)
    forbidden=tuple(path(x)for x in(source_root,*model_roots,*key_roots,request_root,*output_roots))
    for descriptor in descriptors:
        p=path(descriptor.path)
        if any(p==r or p in r.parents or r in p.parents for r in forbidden):raise ValueError('public scientific proof overlaps source, borrowed body or output scope')
    policy_bytes={name:_bytes(packet[name],name,1<<20)for name in POLICIES}
    for name,policy in policy_bytes.items():
        if sha(policy)!=pins['policies'][name+'_sha256']:raise ValueError('independently pinned server policy differs')
    return ServerPacketInspection(public,raw,expected_sha256,descriptors,phase_raw,statement_raw,scientific_raw,
        source_root,model_roots,key_roots,request_root,output_roots,**policy_bytes)


def _model_identity(model):
    return(model.public.blobs,model.manifest_bytes,model.report_bytes,model.outer_bytes,model.review_bytes,model.expected_pins.raw)


def _readiness_identity(ready):
    launch=ready.launch
    return(launch.intent.public.blobs,launch.intent.raw,launch.grant_bytes,launch.expected_pins.raw,_model_identity(launch.model),
        ready.public_manifest_bytes,ready.report_bytes,ready.outer_bytes,ready.root_review_bytes,ready.expected_pins.raw)


def _phase_identity(phase):
    scientific=phase.scientific
    if type(phase)is not ServerPhaseAdmission or type(scientific)is not ScientificAdmission:raise TypeError('exact retained server/scientific admission required')
    return(phase.public.blobs,phase.request_bytes,phase.request_manifest_bytes,phase.grant_bytes,phase.expected_pins.raw,
        _model_identity(phase.model),_readiness_identity(phase.readiness),scientific.public.blobs,
        _model_identity(scientific.model),_readiness_identity(scientific.readiness),scientific.request_bytes,
        scientific.request_manifest_bytes,scientific.statement_bytes,scientific.proof_bytes,scientific.expected_pins.raw)


@dataclass(frozen=True)
class ServerPacket:
    public:PublicContract
    raw:bytes
    expected_sha256:str
    phase:ServerPhaseAdmission
    scientific_proofs:tuple
    dependency_policy:bytes
    role_policy:bytes
    resource_policy:bytes
    resource_review:bytes
    def check(self):
        if type(self)is not ServerPacket or type(self.phase)is not ServerPhaseAdmission:raise TypeError('exact retained public server packet required')
        if type(self.scientific_proofs)is not tuple or len(self.scientific_proofs)!=len(PROOF_LINKS):raise ValueError('immutable complete scientific proof tuple required')
        if any(type(row)is not tuple or len(row)!=2 or type(row[0])is not str or type(row[1])is not bytes for row in self.scientific_proofs)or len({row[0]for row in self.scientific_proofs})!=len(PROOF_LINKS):raise ValueError('retained scientific proof bytes differ')
        self.phase.check()
        rebuilt=authenticate_packet(self.public,self.raw,expected_sha256=self.expected_sha256,scientific_proofs=dict(self.scientific_proofs))
        if _phase_identity(self.phase)!=_phase_identity(rebuilt.phase):raise ValueError('retained server admission differs from its exact public packet')
        for name in POLICIES:
            if type(getattr(self,name))is not bytes or getattr(self,name)!=getattr(rebuilt,name):raise ValueError('retained server policy differs')
        return self
    @property
    def execution_authorized(self):return False


def authenticate_packet(public,raw,*,expected_sha256,scientific_proofs):
    inspection=inspect_packet(public,raw,expected_sha256=expected_sha256)
    obj(scientific_proofs,PROOF_LINKS,'complete externally read scientific proofs')
    for descriptor in inspection.proof_descriptors:
        proof=scientific_proofs[descriptor.name]
        if type(proof)is not bytes or not 0<len(proof)<=descriptor.limit or sha(proof)!=descriptor.sha256:raise ValueError('scientific proof bytes differ from independently inspected descriptor')
    retained=tuple((name,bytes(scientific_proofs[name]))for name in sorted(PROOF_LINKS))
    packet=raw_object(raw,PACKET_LIMIT,'server authority packet')
    model,ready=_base_admissions(public,packet,ExpectedServerPhasePins(inspection.phase_pins_bytes).check())
    request=_bytes(packet['request'],'server request',1<<20);upload=_bytes(packet['request_manifest'],'request upload',1<<20)
    scientific=authenticate_scientific(public,model,ready,request,upload,inspection.scientific_statement_bytes,
        proofs=dict(retained),expected_pins=ExpectedScientificPins(inspection.scientific_expected_pins_bytes))
    phase=authenticate_server_phase(public,model,ready,request,upload,_bytes(packet['phase_grant'],'server phase grant',1<<16),
        expected_pins=ExpectedServerPhasePins(inspection.phase_pins_bytes),scientific=scientific)
    return ServerPacket(public,raw,expected_sha256,phase,retained,**{name:getattr(inspection,name)for name in POLICIES})


def create_actual_server_packet(*args,**kwargs):
    raise PermissionError('pure public packet admission cannot authorize proof reads, backend construction or execution')
