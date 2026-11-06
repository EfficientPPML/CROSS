"""Optional control wiring and lifecycle only; no encrypted-model claim."""
from contextlib import nullcontext
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import pytest

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
from support import load_module
import encrypted_reference as reference
from encrypted_polynomial import EncryptedPolynomialEvaluator
from encrypted_packed_transformer import PackedTransformer, PackedPairLayout
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_bounded_arithmetic import BoundedArithmetic
from encrypted_packed_qwen import CompiledArithmetic, PackedQwen1
from encrypted_refresh import SecureRefresh, BoundedSecureRefresh

CONSTRUCTORS={
 'arithmetic':lambda context,**kw:CompiledArithmetic(context,**kw),
 'bounded_arithmetic':lambda context,**kw:BoundedArithmetic(context,**kw),
 'packed':lambda context,**kw:PackedTransformer(context,PackedPairLayout(1,1,32,64),**kw),
 'bounded_packed':lambda context,**kw:BoundedPackedTransformer(context,PackedPairLayout(1,1,32,64),None,**kw),
 'single_token':lambda context,**kw:PackedQwen1(context,**kw),
 'refresh':lambda context,**kw:SecureRefresh(context,period=4,**kw),
 'bounded_refresh':lambda context,**kw:BoundedSecureRefresh(context,period=4,**kw),
}

@pytest.mark.parametrize('kind',CONSTRUCTORS)
@pytest.mark.parametrize('bad',[None,1,'yes'])
def test_strict_flag_reaches_validator_before_context_use(kind,bad):
 with pytest.raises(TypeError,match='explicit boolean'):
  CONSTRUCTORS[kind](SimpleNamespace(num_slots=64),low_memory=True,preserve_dispatch_cache=bad)

@pytest.mark.parametrize('kind',CONSTRUCTORS)
def test_opt_in_requires_low_memory_at_every_entry(kind):
 with pytest.raises(ValueError,match='low_memory'):
  CONSTRUCTORS[kind](SimpleNamespace(num_slots=64),low_memory=False,preserve_dispatch_cache=True)

@pytest.mark.parametrize('kind',CONSTRUCTORS)
@pytest.mark.parametrize('selected',[False,True])
def test_constructor_forwards_default_or_explicit_policy(monkeypatch,kind,selected):
 observed=[]
 class Boundary(Exception):pass
 def validate(self,context,**kwargs):
  observed.append(kwargs);raise Boundary
 monkeypatch.setattr(EncryptedPolynomialEvaluator,'__init__',validate)
 options={'preserve_dispatch_cache':True} if selected else {}
 with pytest.raises(Boundary):CONSTRUCTORS[kind](SimpleNamespace(num_slots=64),low_memory=True,**options)
 assert observed[0]['preserve_dispatch_cache'] is selected

@pytest.mark.parametrize('bad',[None,1,'yes'])
def test_session_rejects_bad_flag_before_any_key_generation(monkeypatch,bad):
 from jaxite_word import key_gen
 def forbidden(*a,**k):pytest.fail('key generation reached with invalid option')
 monkeypatch.setattr(key_gen,'gen_pke_pair',forbidden)
 with pytest.raises(TypeError,match='preserve_dispatch_cache'):
  reference.create_toy_session('qwen',{}, {},batch=1,sequence_length=1,devices=[],preserve_dispatch_cache=bad)

@pytest.mark.parametrize('model,number,packed,packed_pairs',[
 ('bert',1,False,True),('qwen',2,False,True),('qwen',1,True,False),('qwen',1,False,False)])
@pytest.mark.parametrize('selected',[False,True])
def test_session_wires_ops_refresh_and_cleans_owned_provider(monkeypatch,model,number,packed,packed_pairs,selected):
 """Fake keys/contexts isolate factory control flow; the public ring remains real."""
 import jax
 from jaxite_word import key_gen,ckks_ctx
 import jaxite_word
 import encrypted_ntt, encrypted_refresh
 folder=ROOT/'cadathlon/agent_pset'/('p2_bert' if model=='bert' else 'p4_qwen')
 case=json.loads((folder/f'test/test{number}.in').read_text())
 weights,embeddings,*_=load_module(folder/'src/reference.py').make_case(case)
 if packed_pairs:
  import encrypted_packed_transformer as packed_module
  actual_plan=packed_module.plan_packed_transformer(model,case['config'],weights,
      batch=embeddings.shape[0],sequence=embeddings.shape[1])
  monkeypatch.setattr(packed_module,'plan_packed_transformer',lambda *a,**kw:actual_plan)
 events=[];observed={}
 class Context:
  def __init__(self,params):
   self.parameters=params;self.degree=params['degree'];self.num_slots=params['num_slots']
   self._param_cache=SimpleNamespace(ntt_q_max=object())
  def program_initialization(self,*args,**kwargs):pass
 monkeypatch.setattr(jaxite_word,'CKKSContext',Context)
 monkeypatch.setattr(jax,'default_device',lambda device:nullcontext())
 monkeypatch.setattr(jax.numpy,'asarray',lambda *a,**k:SimpleNamespace(transpose=lambda *x:object()))
 monkeypatch.setattr(key_gen,'gen_pke_pair',lambda *args,**kw:{'secret_key':object(),'public_key':object()})
 monkeypatch.setattr(key_gen,'gen_evaluation_key',lambda *a,**kw:{'a':[0],'b':[0]})
 owner=SimpleNamespace(keys={},close=lambda:events.append('keys'))
 monkeypatch.setattr(encrypted_refresh.ClientRotationKeyStore,'generate',lambda *a,**kw:owner)
 provider=SimpleNamespace(close=lambda:events.append('ntt'))
 monkeypatch.setattr(encrypted_ntt,'enable_scoped_ntt_subsets',lambda context:provider)
 def refresh(context,**kwargs):
  observed['refresh']=kwargs;return SimpleNamespace(ctx=context)
 monkeypatch.setattr(encrypted_refresh,'BoundedSecureRefresh',refresh)
 def graph(context,*args,**kwargs):
  observed['graph']=kwargs;return SimpleNamespace(arithmetic=SimpleNamespace(ctx=context))
 for module,name in [('encrypted_packed_transformer','PackedTransformer'),('encrypted_bounded_transformer','BoundedPackedTransformer'),('encrypted_packed_qwen','PackedQwen1')]:
  monkeypatch.setattr(importlib.import_module(module),name,graph)
 def evaluator(context,**kwargs):
  observed['evaluator']=kwargs;return SimpleNamespace(ctx=context)
 monkeypatch.setattr(reference,'EncryptedPolynomialEvaluator',evaluator)
 monkeypatch.setattr(reference,'EncryptedTensorOps',lambda evaluator,layout:object())
 monkeypatch.setattr(ckks_ctx,'evict_context_caches',lambda context:events.append('context'))
 monkeypatch.setattr(jax,'clear_caches',lambda:events.append('jax'))
 monkeypatch.setattr(reference.gc,'collect',lambda:events.append('gc'))
 monkeypatch.delenv('CROSS_SKIP_TOPLEVEL_ROTKEYS',raising=False)
 options={'preserve_dispatch_cache':True,'ntt_subsets':True} if selected else {}
 session=reference.create_toy_session(model,case['config'],weights,batch=embeddings.shape[0],sequence_length=embeddings.shape[1],devices=[SimpleNamespace(memory_stats=lambda:{})],packed=packed,packed_pairs=packed_pairs,bounded_refresh=packed_pairs,**options)
 assert session.preserve_dispatch_cache is selected
 for kwargs in observed.values():assert kwargs['preserve_dispatch_cache'] is selected
 assert ('refresh' in observed)==(model=='qwen' and number==2)
 assert session.ntt_subsets is (provider if selected else None)
 session.close();session.close()
 assert events==(['ntt'] if selected else [])+['context','context','jax','gc','keys']

@pytest.mark.parametrize('model,problem',[('bert','p2_bert'),('qwen','p4_qwen')])
def test_optimized_adapter_explicitly_opts_in_and_default_adapter_does_not(monkeypatch,model,problem):
 captured=[]
 def create(*args,**kwargs):captured.append((args,kwargs));return object()
 monkeypatch.setattr(reference,'create_toy_session',create)
 folder=ROOT/'cadathlon/agent_pset'/problem/'src'
 for name in (f'{model}_bounded_reference.py',f'{model}_optimized_reference.py'):
  module=load_module(folder/name)
  assert module.IMPLEMENTATION_STATUS=='experimental-unvalidated'
  module.create_session({}, {},batch=1,sequence_length=2,devices=[])
 old,new=[row[1] for row in captured]
 assert 'preserve_dispatch_cache' not in old and 'ntt_subsets' not in old
 assert new['preserve_dispatch_cache'] is True and new['ntt_subsets'] is True
 assert new['packed_pairs'] is True and new['bounded_refresh'] is True
