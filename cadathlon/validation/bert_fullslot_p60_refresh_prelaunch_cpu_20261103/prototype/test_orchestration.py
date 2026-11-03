"""Fake ciphertext lifecycle proof only: no cryptographic arithmetic or accuracy claim."""
from pathlib import Path
import sys,types,tempfile
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).parent))
import proof

@pytest.mark.parametrize('bad_second_square',(False,True))
def test_four_server_outputs_before_decode_and_independent_second_gate(monkeypatch,bad_second_square):
    import jax
    import jaxite_word
    from jaxite_word import key_gen,ckks_ctx
    import encrypted_refresh,encrypted_ntt,support
    events=[];params=encrypted_refresh.RefreshProfile(131072,49,60,61,8).parameters()
    q=params['q_towers'];scale=params['scaling_factor']
    class CT:
        def __init__(self,values,level,actual_scale=scale):
            self.polynomial=np.array(values,dtype=np.float64,copy=True)
            self.degree=131072;self.batch=1;self.num_elements=2;self.num_moduli=2*(level+1)
            self.moduli=q[:self.num_moduli];self.degree_layout=(256,512);self.precision=32
            self._ckks_scale=actual_scale;self._ckks_nsd=1
            self.ntt_ctx=types.SimpleNamespace(ff_ctx=types.SimpleNamespace())
    class Context:
        def __init__(self,parameters):
            self.parameters=parameters;self.secret_key=parameters.get('secret_key');self.public_key=parameters.get('public_key')
            self._param_cache=types.SimpleNamespace(secret_key=self.secret_key,
                level_for_num_q=lambda count:count//2-1,q_moduli_at_level=lambda level:q[:2*(level+1)])
            self.he_level_reduce=types.SimpleNamespace()
            class Levels:
                def __getitem__(self,pair):return types.SimpleNamespace(level_reduce=lambda value:CT(value.polynomial,pair[1]))
            self.he_level_reduce=Levels()
        def program_initialization(self,*args,**kwargs):
            events.append('initialize')
            # Class-wide guards must already cover every codec at initialization.
            for name in proof.CODEC_METHODS:
                with pytest.raises(AssertionError):getattr(self,name)(None)
            for module in {sys.modules[n] for n in ('key_gen','jaxite_word.key_gen')}:
                for name in proof.KEYGEN_METHODS:
                    with pytest.raises(AssertionError):getattr(module,name)()
        def encrypt_slots(self,values):events.append('encrypt');return CT(values,49)
        def decrypt_slots(self,value):
            assert events.count('refresh')==2 and events.count('square')==2
            events.append('decode');return value.polynomial.copy()
    for name in set(proof.CODEC_METHODS)-{'encrypt_slots','decrypt_slots'}:
        setattr(Context,name,lambda self,*args:None)
    class Store:
        required_bytes=1;keys={}
        @classmethod
        def generate(cls,*args,**kwargs):events.append('rotation_keys');return cls()
        def close(self):events.append('store_close')
    class Refresh:
        def __init__(self,server,**kwargs):
            self.ctx=server;self.refresh_count=0;self.plan={'period':65536};self.compiled_primitives=()
            self.engine=types.SimpleNamespace(_sparse=False,_gap=1,_stage_checkpoint=lambda *args:None)
            def multiply(a,b):
                events.append('square');values=a.polynomial*b.polynomial
                if bad_second_square and events.count('square')==2:values=values.copy();values[-1]+=.002
                return CT(values,24,a._ckks_scale*b._ckks_scale/(q[50]*q[51]))
            self.evaluator=types.SimpleNamespace(multiply=multiply,_clear_working_caches=lambda:events.append('cache_clear'))
        def refresh(self,value,*,public_bound):
            assert public_bound==(16.,128.)[self.refresh_count]
            assert events.count('encrypt')==2 and 'decode' not in events
            self.refresh_count+=1;events.append('refresh')
            for _ in range(2):
                self.engine._stage_checkpoint('approxmod-re');self.engine._stage_checkpoint('approxmod-im')
            return CT(value.polynomial,25)
    monkeypatch.setattr(jaxite_word,'CKKSContext',Context)
    monkeypatch.setattr(key_gen,'gen_pke_pair',lambda *a,**k:{'secret_key':np.zeros(1,dtype=np.uint32),'public_key':object()})
    monkeypatch.setattr(key_gen,'gen_evaluation_key',lambda *a,**k:{'a':np.zeros((1,1,1),dtype=np.uint32),'b':np.zeros((1,1,1),dtype=np.uint32)})
    monkeypatch.setattr(encrypted_refresh,'ClientRotationKeyStore',Store)
    monkeypatch.setattr(encrypted_refresh,'BoundedSecureRefresh',Refresh)
    monkeypatch.setattr(encrypted_ntt,'enable_scoped_ntt_subsets',lambda server:types.SimpleNamespace(close=lambda:events.append('owner_close')))
    monkeypatch.setattr(support,'require_secure_execution',lambda *args:{'fake_test_only':True})
    monkeypatch.setattr(ckks_ctx,'evict_context_caches',lambda ctx:events.append('context_release'))
    monkeypatch.setattr(jax,'clear_caches',lambda:events.append('jax_clear'))
    monkeypatch.setattr(proof.gc,'collect',lambda:events.append('gc'))
    monkeypatch.setattr(proof.os,'sched_getaffinity',lambda pid:set(proof.CPU_AFFINITY))
    report={}
    with tempfile.TemporaryDirectory(prefix='p60-fake-lifecycle-',dir='/dev/shm') as directory:
        monkeypatch.setenv('CROSS_PUBLIC_KEY_DIRECTORY',directory);monkeypatch.setenv('BOOTSTRAP_DISABLE_HOISTED_LT','0')
        proof.run(report)
    assert report['cleanup_completed'] and report['all_four_outputs_completed_before_first_client_decode']
    assert report['numerical_passed'] is (not bad_second_square)
    assert report['probes']['B16']['passed'] and report['probes']['B128']['passed'] is (not bad_second_square)
    assert events.count('decode')==4 and events.count('context_release')==2 and events.count('store_close')==1
    assert events.index('decode')>max(i for i,e in enumerate(events) if e in ('refresh','square'))
    assert all(row['retained_outputs_unchanged'] and row['input_payload_and_metadata_unchanged'] for row in report['probes'].values())
