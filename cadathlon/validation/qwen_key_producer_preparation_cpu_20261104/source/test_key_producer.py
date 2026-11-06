"""Full166/335 metadata/order with trusted recording crypto and stores only."""
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
import json,hashlib
import pytest
import qwen_key_producer as producer
import qwen_key_store as store
from qwen_key_contract import KEYGEN,canonical,sha,authenticate_public,authenticate_private,ExpectedIds
from cleanup_origin import is_cleanup

class Token:
    """No real physical335 key arrays may be materialized by this fixture."""
    def __init__(self,name):self.name=name
    def __array__(self,*args,**kwargs):raise AssertionError('recording crypto token was materialized')

class RecordingStore:
    def __init__(self,env,root,specs,private):
        self.env=env;self.root=root;self.specs=specs;self.private=private;self.records={};self.values=[]
        self.manifest=None;self.closed=False;self.close_error=None;self.on_write=None
    def write_array(self,name,value):
        assert not self.closed and name==self.specs[len(self.records)].name
        spec=self.specs[len(self.records)];self.values.append(value)
        row=dict(sha256=hashlib.sha256(name.encode()).hexdigest(),dtype='uint32',shape=list(spec.shape),bytes=spec.raw_bytes+128)
        self.records[name]=row;self.env.log.append(('write',self.private,name))
        if self.on_write is not None:self.on_write(self,name)
        return json.loads(canonical(row))
    def verify(self):
        assert not self.closed and len(self.records)==len(self.specs)
        self.env.log.append(('verify',self.private));return json.loads(canonical(self.records))
    def write_manifest(self,raw):
        assert not self.closed and len(self.records)==len(self.specs)and type(raw)is bytes
        self.env.log.append(('manifest',self.private));self.manifest=raw;return raw
    def identity(self):return dict(root=self.root,device=1,inode=101 if self.private else 100,private=self.private)
    def close(self,primary=None):
        assert not self.closed;self.closed=True;self.env.log.append(('close',self.private))
        if self.close_error is not None:raise self.close_error
        if primary is not None:raise primary


def environment(public,tmp_path):
    intent=producer.GenerationIntent.build(public,nonce='recording-keys',model_id='1'*64,generation_sources='2'*64,
        request_producer_sources='3'*64,response_consumer_sources='4'*64,public_root=str(tmp_path/'public'),private_root=str(tmp_path/'private'))
    iv,profile=intent.check();p=profile.parameters
    env=NS(intent=intent,profile=profile,params=p,log=[],stores=[],secret=[[17,23,29]],secret_uses=[],rotation_calls=[],
        fail_prerequisite=None,fail_private_factory=None,rotation_error=None,on_capacity=None,on_new_store=None)
    def source(got):
        assert got is intent;env.log.append(('source',))
        return dict(source_manifest_sha256=iv['generation_source_manifest_sha256'],core_source_manifest_sha256=producer.CORE_MANIFEST_SHA256,
            keygen_source_sha256=KEYGEN,source_unchanged=env.fail_prerequisite!='source',synthetic_only=True)
    def security(params):
        assert params==p;env.log.append(('security',))
        return dict(qp_secure=env.fail_prerequisite!='security',p_coverage=True,sigma=p['sigma'],secret_distribution=p['secret_distribution'],synthetic_only=True)
    def capacity(value):
        env.log.append(('capacity',))
        if env.on_capacity:env.on_capacity(value)
        return dict(public_root=value['public_root'],private_root=value['private_root'],fresh_disjoint_roots=True,
            storage_capacity=env.fail_prerequisite!='capacity',descriptor_capacity=True,synthetic_only=True)
    def factory(root,specs,*,private,shield):
        env.log.append(('factory',private));assert shield is nullcontext
        if private and env.fail_private_factory:raise env.fail_private_factory
        owner=RecordingStore(env,root,specs,private);env.stores.append(owner)
        if env.on_new_store:env.on_new_store(owner)
        return owner
    def pke(q,aux,n,*,noise_std):
        assert q==p['q_towers']and aux==[]and n==131072 and noise_std==p['sigma']
        env.log.append(('pke',));return dict(public_key=Token('public'),secret_key=env.secret)
    def switched(label):
        return dict(a=Token(label+'a'),b=Token(label+'b'),modulus={'Q':p['q_towers'],'P':p['p_towers']},P=p['p_towers'],shape=(8,114,131072))
    def evaluation(secret,*,q,P,dnum,noise_std):
        assert secret is env.secret and q==p['q_towers']and P==p['p_towers']and dnum==8 and noise_std==p['sigma']
        env.secret_uses.append(secret);env.log.append(('evaluation',));return switched('relin')
    def rotation(secret,q,P,index,*,dnum,noise_std):
        assert secret is env.secret and q==p['q_towers']and P==p['p_towers']and dnum==8 and noise_std==p['sigma']
        env.secret_uses.append(secret);env.rotation_calls.append(index);env.log.append(('rotation',index))
        if env.rotation_error:raise env.rotation_error
        return {index:switched('rot'+str(index))}
    env.dependencies=producer.RecordingDependencies(source,security,capacity,factory,pke,evaluation,rotation,nullcontext)
    env.run=lambda:producer._generate_for_recording(intent,env.dependencies)
    return env


def test_complete166_literal_order335_descriptors_original_secret_and_separate_identity(public,tmp_path):
    env=environment(public,tmp_path);result=env.run();done=json.loads(result.completion_bytes)
    assert env.rotation_calls==list(env.profile.rotations)and len(env.rotation_calls)==166
    assert 65535 in env.rotation_calls and 262143 in env.rotation_calls
    assert len(env.secret_uses)==167 and all(v is env.secret for v in env.secret_uses)
    assert env.stores[1].values==[env.secret]and len(env.stores[0].values)==335
    assert [n for n in env.stores[0].records]==[s.name for s in env.profile.arrays]
    assert all(s.closed for s in env.stores)
    pke=env.log.index(('pke',));assert env.log.index(('security',))<env.log.index(('capacity',))<pke
    assert len([x for x in env.log[:pke]if x==('source',)])==2
    assert [x for x in env.log if x[0]=='manifest']==[('manifest',True),('manifest',False)]
    assert [x for x in env.log if x[0]=='close']==[('close',True),('close',False)]and env.log[-1]==('source',)
    assert done['cleanup_completed']is True and done['source_postcheck_completed']is True
    assert done['actual_execution_enabled']is False and done['terminal_ready']is False and done['independent_outer_receipt_sha256']is None
    assert result.execution_authorized is False and result.terminal_ready is False
    manifest=json.loads(result.public_bytes);assert len(manifest['arrays'])==335
    expected=ExpectedIds(manifest['context_id'],sha(result.public_bytes),KEYGEN,'1'*64,'3'*64,'4'*64,sha(result.private_bytes))
    binding=authenticate_public(result.public_bytes,env.profile,expected,role='client');authenticate_private(result.private_bytes,binding)
    assert done['context_id']==manifest['context_id']and done['public_manifest_sha256']==sha(result.public_bytes)
    assert not (tmp_path/'public').exists()and not (tmp_path/'private').exists(), 'full335 stores remain recording-only'


@pytest.mark.parametrize('which',['source','security','capacity'],ids=['source','security','capacity'])
def test_prerequisite_rejection_precedes_store_and_rng(public,tmp_path,which):
    env=environment(public,tmp_path);env.fail_prerequisite=which
    with pytest.raises(ValueError):env.run()
    assert not env.stores and not any(x[0]in ('factory','pke','evaluation','rotation')for x in env.log)


@pytest.mark.parametrize('which',['keygen','candidate'],ids=['keygen','candidate'])
def test_wrong_authenticated_intent_rejects_before_any_dependency(public,tmp_path,which):
    env=environment(public,tmp_path);value=json.loads(env.intent.raw)
    if which=='keygen':value['keygen_source_sha256']='0'*64
    else:value['bindings']['candidate_sha256']='0'*64
    bad=producer.GenerationIntent(public,canonical(value))
    with pytest.raises(ValueError):producer._generate_for_recording(bad,env.dependencies)
    assert env.log==[]


def test_later_factory_failure_retires_earlier_original_store(public,tmp_path):
    env=environment(public,tmp_path);error=ValueError('controlled private factory');env.fail_private_factory=error
    with pytest.raises(ValueError)as caught:env.run()
    assert caught.value is error and len(env.stores)==1 and env.stores[0].closed
    assert ('close',False)in env.log and ('pke',)not in env.log
    assert producer._QUARANTINE[-1][-1]is error


def test_partial_rotation_failure_has_no_manifest_and_keeps_original_secret(public,tmp_path):
    env=environment(public,tmp_path);error=ValueError('controlled rotation');env.rotation_error=error
    with pytest.raises(ValueError)as caught:env.run()
    assert caught.value is error and len(env.rotation_calls)==1
    assert all(s.closed and s.manifest is None for s in env.stores)
    retained=producer._QUARANTINE[-1][7]
    assert any(v is env.secret for v in retained)


@pytest.mark.parametrize('cancel_type',[KeyboardInterrupt,SystemExit],ids=['keyboardinterrupt','systemexit'])
def test_cleanup_only_cancellation_outranks_ordinary_close(public,tmp_path,cancel_type):
    env=environment(public,tmp_path);ordinary=OSError('controlled private close');cancel=cancel_type('controlled public close')
    def setup(owner):owner.close_error=ordinary if owner.private else cancel
    env.on_new_store=setup
    with pytest.raises(cancel_type)as caught:env.run()
    assert caught.value is cancel and is_cleanup(cancel)
    assert all(s.closed and s.manifest is not None for s in env.stores)
    assert producer._QUARANTINE[-1][-1]is cancel


def test_genuine_rng_body_primary_survives_cleanup_cancellation(public,tmp_path):
    env=environment(public,tmp_path);primary=ValueError('controlled RNG body');cancel=KeyboardInterrupt('controlled cleanup')
    env.rotation_error=primary
    def setup(owner):
        if not owner.private:owner.close_error=cancel
    env.on_new_store=setup
    with pytest.raises(ValueError)as caught:env.run()
    assert caught.value is primary and all(s.closed for s in env.stores)
    assert producer._QUARANTINE[-1][-1]is primary


def test_store_accessor_mutation_rejects_and_closes_captured_original(public,tmp_path):
    env=environment(public,tmp_path);replacement_calls=[]
    def setup(owner):
        if not owner.private:
            def mutate(_owner,name):
                if name=='public_key.npy':env.stores[1].close=lambda primary=None:replacement_calls.append(primary)
            owner.on_write=mutate
    env.on_new_store=setup
    with pytest.raises(ValueError,match='accessor'):env.run()
    assert replacement_calls==[]and all(s.closed for s in env.stores)
    assert ('close',True)in env.log and ('close',False)in env.log


def test_parsed_intent_mutation_rejects_before_root_factory(public,tmp_path):
    env=environment(public,tmp_path)
    def mutate(value):value['public_root']=str(tmp_path/'replacement')
    env.on_capacity=mutate
    with pytest.raises(ValueError,match='parsed generation intent'):env.run()
    assert not env.stores and ('pke',)not in env.log


@pytest.mark.parametrize('entry',[producer.create_actual_key_producer,store.create_actual_key_store],ids=['producer','store'])
def test_actual_entries_deny_without_argument_access(entry):
    class Bomb:
        def __getattribute__(self,name):raise AssertionError('actual argument accessed')
    with pytest.raises(PermissionError):entry(Bomb(),intent=Bomb())
