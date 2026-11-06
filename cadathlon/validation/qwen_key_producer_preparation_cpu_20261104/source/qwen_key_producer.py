"""Qwen166 producer composition, recording seam only; actual role denied.

Pre-generation intent never invents the random context/manifest identities.
Returned descriptors are internal consistency evidence, not external admission.
"""
from dataclasses import dataclass
from pathlib import Path
import json,math,re
from cleanup import cleanup_signals
from cleanup_origin import protect_body
from qwen_finalizer import Finalizer
from qwen_file_contract import PublicContract,canonical,same,sha,hash_value
from qwen_key_contract import (PublicProfile,ExpectedIds,KEYGEN,bindings,public_body,
    authenticate_public,authenticate_private)

_QUARANTINE=[]
NONCE=re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
CORE_MANIFEST_SHA256='11353173bbf8e2796813d0feb6f69fac308775c7b13cc164900170f32bc62088'

def create_actual_key_producer(*args,**kwargs):
    raise PermissionError('Qwen key generation has no independently admitted actual entry')

def _accessor(value):return(getattr(value,'__self__',None),getattr(value,'__func__',value))
def _same_accessor(a,b):return all(x is y for x,y in zip(_accessor(a),_accessor(b)))
def _path(value):
    if type(value)is not str or not value or '\x00'in value:raise ValueError('exact absolute output path required')
    p=Path(value)
    if not p.is_absolute()or str(p)!=value or '..'in p.parts or p==p.parent:raise ValueError('canonical fresh child output path required')
    return p

@dataclass(frozen=True)
class GenerationIntent:
    public:PublicContract
    raw:bytes
    def check(self):
        if type(self)is not GenerationIntent or type(self.public)is not PublicContract or type(self.raw)is not bytes or not 0<len(self.raw)<=65536:
            raise TypeError('exact immutable generation intent required')
        values=self.public.check();v=json.loads(self.raw)
        keys={'schema','nonce','model_id','generation_source_manifest_sha256','core_source_manifest_sha256',
            'request_producer_source_manifest_sha256','response_consumer_source_manifest_sha256',
            'keygen_source_sha256','public_root','private_root','bindings','parameters','resource_policy',
            'actual_execution_enabled','execution_grant_sha256','scientific_freeze_sha256','heldout_terminal_sha256'}
        if type(v)is not dict or set(v)!=keys or canonical(v)!=self.raw:raise ValueError('canonical exact generation intent schema required')
        if v['schema']!='qwen-key-generation-intent-v1'or type(v['nonce'])is not str or not NONCE.fullmatch(v['nonce']):raise ValueError('generation schema/nonce differs')
        for n in ('model_id','generation_source_manifest_sha256','core_source_manifest_sha256','request_producer_source_manifest_sha256','response_consumer_source_manifest_sha256','keygen_source_sha256'):hash_value(v[n])
        if v['keygen_source_sha256']!=KEYGEN or v['core_source_manifest_sha256']!=CORE_MANIFEST_SHA256:raise ValueError('fixed keygen/core source identity differs')
        if v['actual_execution_enabled']is not False or any(v[n]is not None for n in ('execution_grant_sha256','scientific_freeze_sha256','heldout_terminal_sha256')):
            raise PermissionError('this source successor is recording-only, not an actual grant consumer')
        profile=PublicProfile.from_bytes(values['inputs/profile_audit.json'],values['inputs/actual_key_plan.json'])
        same(v['parameters'],self.public.profile(),'generation/full public physical profile differs')
        same(v['parameters'],profile.parameters,'generation/key physical profile differs')
        same(v['bindings'],bindings(),'generation candidate/source bindings differ')
        same(v['resource_policy'],resource_policy(profile),'fixed logical generation storage/descriptor policy differs')
        a=_path(v['public_root']);b=_path(v['private_root']);source=Path(__file__).resolve().parent
        if a==b or a in b.parents or b in a.parents or any(p==source or source in p.parents for p in (a,b)):
            raise ValueError('fresh public/private outputs must be disjoint from each other and source')
        return v,profile
    @classmethod
    def build(cls,public,*,nonce,model_id,generation_sources,request_producer_sources,response_consumer_sources,public_root,private_root):
        if type(public)is not PublicContract:raise TypeError('exact full public contract required')
        values=public.check();profile=PublicProfile.from_bytes(values['inputs/profile_audit.json'],values['inputs/actual_key_plan.json'])
        raw=canonical(dict(schema='qwen-key-generation-intent-v1',nonce=nonce,model_id=model_id,
            generation_source_manifest_sha256=generation_sources,core_source_manifest_sha256=CORE_MANIFEST_SHA256,
            request_producer_source_manifest_sha256=request_producer_sources,response_consumer_source_manifest_sha256=response_consumer_sources,
            keygen_source_sha256=KEYGEN,public_root=public_root,private_root=private_root,bindings=bindings(),parameters=profile.parameters,
            resource_policy=resource_policy(profile),actual_execution_enabled=False,execution_grant_sha256=None,scientific_freeze_sha256=None,heldout_terminal_sha256=None))
        result=cls(public,raw);result.check();return result

def resource_policy(profile):
    rows=profile.arrays;raw=sum(s.raw_bytes for s in rows);secret=profile.secret.raw_bytes
    return dict(schema='qwen-key-generation-logical-resources-v1',public_arrays=335,private_arrays=1,
        public_required_bytes=raw+len(rows)*4096+(1<<30),private_required_bytes=secret+4096+(1<<20),
        owned_descriptors=342,temporary_descriptor_allowance=8,required_descriptor_soft_limit=350,
        scope='logical output files and bounded writer temporaries only; backend/runtime descriptors and working memory excluded')

@dataclass(frozen=True)
class RecordingDependencies:
    """All callbacks are explicitly trusted synthetic seams, not capabilities."""
    source_check:object
    security_check:object
    capacity_check:object
    store_factory:object
    gen_pke_pair:object
    gen_evaluation_key:object
    gen_rotation_key:object
    shield:object=cleanup_signals
    synthetic_only:bool=True
    def check(self):
        if type(self)is not RecordingDependencies or self.synthetic_only is not True:raise PermissionError('only explicit recording dependencies permitted')
        for name in ('source_check','security_check','capacity_check','store_factory','gen_pke_pair','gen_evaluation_key','gen_rotation_key','shield'):
            if not callable(getattr(self,name)):raise TypeError('recording dependency must be callable')

@dataclass(frozen=True)
class GeneratedManifests:
    public_bytes:bytes
    private_bytes:bytes
    completion_bytes:bytes
    @property
    def execution_authorized(self):return False
    @property
    def terminal_ready(self):return False


def _switch_metadata(value,p,shape):
    if (type(value)is not dict or set(value)!={'a','b','modulus','P','shape'} or value['modulus']!={'Q':p['q_towers'],'P':p['p_towers']}
            or type(value['P'])is not list or any(type(x)is not int for x in value['P'])
            or type(value['modulus'])is not dict or set(value['modulus'])!={'Q','P'}
            or any(type(value['modulus'][n])is not list or any(type(x)is not int for x in value['modulus'][n]) for n in ('Q','P'))
            or value['P']!=p['p_towers'] or type(value['shape'])is not tuple or any(type(x)is not int for x in value['shape']) or value['shape']!=shape):
        raise ValueError('key-switch generator metadata differs')


def _generate_for_recording(intent,dependencies):
    """Exercise sequence/ownership with recording crypto; never import a backend.

    Real KeyStore can be exercised separately on small owned files. Full physical
    335-array cryptography, host capacity and security predicates remain explicit
    recording seams here; even a successful return is not ready for consumption.
    """
    if type(intent)is not GenerationIntent or type(dependencies)is not RecordingDependencies:raise TypeError('exact recording intent/dependencies required')
    intent_value,profile=intent.check();dependencies.check()
    captured=tuple(getattr(dependencies,n)for n in ('source_check','security_check','capacity_check','store_factory','gen_pke_pair','gen_evaluation_key','gen_rotation_key','shield'))
    source_check,security_check,capacity_check,factory,pke,evaluation,rotation,shield=captured
    original_intent=intent.raw;original_public=intent.public;original_blobs=intent.public.blobs;p=profile.parameters;parameter_bytes=canonical(p)
    owners=[];methods=[];retained=[];result=None;primary=None;cleanup_report={};public_raw=None;private_raw=None
    def guard():
        if intent.raw is not original_intent or intent.public is not original_public or intent.public.blobs is not original_blobs:raise ValueError('original generation intent binding changed')
        intent.check();dependencies.check();same(intent_value,json.loads(original_intent),'parsed generation intent changed');same(p,json.loads(parameter_bytes),'generation parameters changed')
        now=tuple(getattr(dependencies,n)for n in ('source_check','security_check','capacity_check','store_factory','gen_pke_pair','gen_evaluation_key','gen_rotation_key','shield'))
        if any(not _same_accessor(a,b)for a,b in zip(now,captured)):raise ValueError('recording generation dependency changed')
        for owner,row in zip(owners,methods):
            for name,method in row.items():
                if not _same_accessor(getattr(owner,name),method):raise ValueError('original generation store accessor changed')
    def source():
        guard();observed=source_check(intent);guard()
        same(observed,dict(source_manifest_sha256=intent_value['generation_source_manifest_sha256'],core_source_manifest_sha256=CORE_MANIFEST_SHA256,keygen_source_sha256=KEYGEN,source_unchanged=True,synthetic_only=True),'recording source prerequisite differs')
    def acquire(root,specs,private):
        owner=factory(root,specs,private=private,shield=shield)
        # Retain before touching any returned accessor; partial acquisition is
        # quarantined if the trusted factory returns a malformed owner.
        owners.append(owner);row={};methods.append(row)
        row['close']=owner.close
        for name in ('write_array','verify','write_manifest','identity'):row[name]=getattr(owner,name)
        if any(not callable(v)for v in row.values()):raise TypeError('store owner methods required')
        guard();return row
    def write(row,name,value,spec):
        raw=row['write_array'](name,value);guard();spec.validate(raw);return raw
    try:
        source()
        security=security_check(p);guard()
        same(security,dict(qp_secure=True,p_coverage=True,sigma=p['sigma'],secret_distribution=p['secret_distribution'],synthetic_only=True),'recording security prerequisite differs')
        capacity=capacity_check(intent_value);guard()
        same(capacity,dict(public_root=intent_value['public_root'],private_root=intent_value['private_root'],fresh_disjoint_roots=True,storage_capacity=True,descriptor_capacity=True,synthetic_only=True),'recording destination/capacity prerequisite differs')
        pub=acquire(intent_value['public_root'],profile.arrays,False);priv=acquire(intent_value['private_root'],(profile.secret,),True)
        source()  # Includes post-acquisition binding checks, before first RNG seam.
        specs={s.name:s for s in profile.arrays}
        pair=pke(p['q_towers'],[],p['degree'],noise_std=p['sigma']);retained.append(pair);guard()
        if type(pair)is not dict or set(pair)!={'public_key','secret_key'}:raise ValueError('exact PKE pair required')
        secret=pair['secret_key'];retained.append(secret)
        secret_record=write(priv,'secret.npy',secret,profile.secret)
        arrays={'public_key.npy':write(pub,'public_key.npy',pair['public_key'],specs['public_key.npy'])}
        del retained[0];del pair
        ek=evaluation(secret,q=p['q_towers'],P=p['p_towers'],dnum=p['dnum'],noise_std=p['sigma']);retained.append(ek);guard()
        _switch_metadata(ek,p,specs['relin_a.npy'].shape)
        for part in ('a','b'):
            name='relin_'+part+'.npy';arrays[name]=write(pub,name,ek[part],specs[name])
        retained.pop();del ek
        for index in profile.rotations:
            generated=rotation(secret,p['q_towers'],p['p_towers'],index,dnum=p['dnum'],noise_std=p['sigma']);retained.append(generated);guard()
            if type(generated)is not dict or set(generated)!={index}or any(type(k)is not int for k in generated):raise ValueError('exact one-index rotation result required')
            entry=generated[index];_switch_metadata(entry,p,specs[f'rot_{index}_a.npy'].shape)
            for part in ('a','b'):
                name=f'rot_{index}_{part}.npy';arrays[name]=write(pub,name,entry[part],specs[name])
            retained.pop();del entry,generated
        same(pub['verify'](),arrays,'stored public inventory differs');guard()
        same(priv['verify'](),{'secret.npy':secret_record},'stored private inventory differs');guard()
        body=public_body(profile,arrays,KEYGEN);public_manifest=body|dict(context_id=sha(canonical(body)));public_raw=canonical(public_manifest)
        private_manifest=dict(schema='cross-qwen-full-private-keys-v1',context_id=public_manifest['context_id'],public_manifest_sha256=sha(public_raw),bindings=bindings(),secret=secret_record);private_raw=canonical(private_manifest)
        # Internal consistency only. These self-derived expectations are never
        # returned as external admission or a FilesAdmission authority object.
        expected=ExpectedIds(public_manifest['context_id'],sha(public_raw),KEYGEN,intent_value['model_id'],intent_value['request_producer_source_manifest_sha256'],intent_value['response_consumer_source_manifest_sha256'],sha(private_raw))
        bound=authenticate_public(public_raw,profile,expected,role='client');authenticate_private(private_raw,bound)
        written=priv['write_manifest'](private_raw)
        if type(written)is not bytes or written!=private_raw:raise ValueError('private manifest return differs')
        guard()
        written=pub['write_manifest'](public_raw)
        if type(written)is not bytes or written!=public_raw:raise ValueError('public manifest return differs')
        guard();same(pub['verify'](),arrays,'final public readback differs');guard();same(priv['verify'](),{'secret.npy':secret_record},'final private readback differs');guard()
        roots=[pub['identity'](),priv['identity']()];guard()
        for row,path,private in zip(roots,(intent_value['public_root'],intent_value['private_root']),(False,True)):
            if (type(row)is not dict or set(row)!={'root','device','inode','private'}or type(row['root'])is not str or row['root']!=path or
                    type(row['device'])is not int or row['device']<0 or type(row['inode'])is not int or row['inode']<=0 or row['private']is not private):
                raise ValueError('typed retained root identity differs')
        roots=json.loads(canonical(roots));source()
        result=dict(schema='qwen-key-generation-recording-completion-v1',scope='trusted recording seams only; not generated real keys or independently ready',intent_sha256=sha(original_intent),generation_source_manifest_sha256=intent_value['generation_source_manifest_sha256'],core_source_manifest_sha256=CORE_MANIFEST_SHA256,keygen_source_sha256=KEYGEN,bindings=bindings(),model_id=intent_value['model_id'],request_producer_source_manifest_sha256=intent_value['request_producer_source_manifest_sha256'],response_consumer_source_manifest_sha256=intent_value['response_consumer_source_manifest_sha256'],context_id=public_manifest['context_id'],public_manifest_sha256=sha(public_raw),private_manifest_sha256=sha(private_raw),rotations=166,public_arrays=335,private_arrays=1,roots=roots,cleanup_completed=False,source_postcheck_completed=False,actual_execution_enabled=False,independent_outer_receipt_sha256=None,terminal_ready=False)
    except BaseException as error:primary=protect_body(error)
    finally:
        # These original objects/callables live independently of public mirrors.
        original_owners=tuple(owners);original_methods=tuple(dict(row)for row in methods);original_retained=tuple(retained)
        final=Finalizer(cleanup_report,primary)
        def retire():
            final.attempt('generation final binding',guard)
            for row in reversed(original_methods):
                close=row.get('close')
                if close is not None:final.attempt('original key store close',lambda close=close:close(primary=primary))
                final.attempt('generation post-close binding',guard)
            final.attempt('generation terminal source postcheck',source)
        final.shielded(shield,retire)
        chosen=final.selected()
        if chosen is not None:
            _QUARANTINE.append((intent,dependencies,captured,original_public,original_blobs,original_owners,original_methods,original_retained,public_raw,private_raw,result,cleanup_report,chosen));raise chosen
    if result is None:raise RuntimeError('recording generation did not return a complete result')
    result['cleanup_completed']=True;result['source_postcheck_completed']=True
    retained.clear()
    return GeneratedManifests(public_raw,private_raw,canonical(result))
