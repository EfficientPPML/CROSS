"""Candidate integration planning only: all client key creation is forbidden."""
import os
import json,sys
from pathlib import Path
from types import SimpleNamespace
import pytest
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE),str(HERE.parent),str(ROOT),str(ROOT/'cadathlon/common')]
from support import load_module
candidate=load_module(ROOT/'cadathlon/common/encrypted_reference.py')
from jaxite_word import key_gen

class ReachedKeyBoundary(Exception):pass

@pytest.mark.parametrize('model,number', [('bert',n) for n in range(3,7)]+[('qwen',n) for n in range(2,7)])
def test_deep_opt_in_is_explicit_and_packed_only(monkeypatch,model,number):
    problem='p2_bert' if model=='bert' else 'p4_qwen'
    folder=ROOT/'cadathlon/agent_pset'/problem
    case=json.loads((folder/f'test/test{number}.in').read_text())
    weights,embeddings,*_=load_module(folder/'src/reference.py').make_case(case)
    calls=[]
    def forbidden(*args,**kwargs):
        calls.append((args,kwargs))
        raise ReachedKeyBoundary('no actual key generation in planning test')
    monkeypatch.setattr(key_gen,'gen_pke_pair',forbidden)
    monkeypatch.delenv('CROSS_SKIP_TOPLEVEL_ROTKEYS',raising=False)
    args=dict(batch=embeddings.shape[0],sequence_length=embeddings.shape[1],devices=(SimpleNamespace(memory_stats=lambda:{}),))
    with pytest.raises(ValueError,match='requires bounded homomorphic refresh'):
        candidate.create_toy_session(model,case['config'],weights,packed_pairs=True,**args)
    assert not calls
    with pytest.raises(ValueError,match='packed-pair layout'):
        candidate.create_toy_session(model,case['config'],weights,bounded_refresh=True,**args)
    assert not calls
    with pytest.raises(ReachedKeyBoundary):
        candidate.create_toy_session(model,case['config'],weights,packed_pairs=True,bounded_refresh=True,**args)
    assert len(calls)==1
