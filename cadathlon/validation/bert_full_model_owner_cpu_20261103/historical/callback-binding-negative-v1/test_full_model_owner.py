"""Joined full-width graph tests using ideal primitives, never HE evidence."""
from pathlib import Path
from collections import Counter
from types import SimpleNamespace
import copy,hashlib,json,math,time,os
import numpy as np
import pytest
from test_initial_ideal import ideal_primitives
from test_softmax_tile import refresh_and_tracked
from ideal_refresh import IdentityRefresh,install_public_bootstrap_plan
from ideal_primitives import Arithmetic,Value,IdealLinear,install_matrix_class,DELTA
from encrypted_bert_attention_tile import BertAttentionTile
from encrypted_bert_all_row_norm import BertRowLayerNorm
from encrypted_bert_layer_attention import BertLayerAttention
from encrypted_bert_all_ffn import BertFFNTile
from encrypted_bert_pooler_head import BertPoolerHead
from encrypted_bert_full_model import BertFullModel
from full_model_fixtures import load
import full_model_reference as reference
HERE=Path(__file__).resolve().parent;RAW=(HERE/'candidate_manifest.json').read_bytes();MANIFEST=json.loads(RAW)

@pytest.fixture(autouse=True)
def primitives(refresh_and_tracked,monkeypatch):
    install_matrix_class(monkeypatch);install_public_bootstrap_plan(monkeypatch)
    from test_initial_ideal import Value as BaseValue
    BaseValue.degree_layout=(256,512)

@pytest.fixture(scope='module')
def fixtures():
    weights,native,report=load();(HERE/'fixture_pins.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    return weights,native

def setup(fixtures,K=15,progress=None):
    weights,native=fixtures;ops=Arithmetic();refresh=IdentityRefresh(ops);linear=IdealLinear(ops);cache=ops.ctx._param_cache
    keys=set(BertAttentionTile.required_rotation_indices())|set(BertRowLayerNorm.required_rotation_indices())
    for a in weights.values():
        if a.ndim==2:keys.update(linear.required_rotation_indices(a,linear.layout))
    for key in keys:
        if key not in cache.raw_rotation_keys:cache.raw_rotation_keys[key]={'a':object(),'b':object()};cache.coef_maps[key]=object()
    cache.eval_key_a_full=SimpleNamespace(shape=(8,131072,114),dtype=np.dtype('uint32'))
    cache.eval_key_b_full=SimpleNamespace(shape=(8,131072,114),dtype=np.dtype('uint32'))
    positions=tuple(map(int,native[f'K{K}_positions']))
    owner=BertFullModel(ops,linear,refresh,weights,RAW,original_positions=positions,context_id='a'*64,request_id='b'*32,progress=progress)
    packed=np.zeros((128,4096));packed[:,:768]=native[f'K{K}_embeddings']
    sources=tuple(Value(row.reshape(-1),49,DELTA) for row in packed.reshape(8,16,4096))
    return owner,ops,refresh,sources

@pytest.mark.parametrize('during_callback',[False,True])
def test_callback_factory_binding_is_fixed(fixtures,during_callback):
    box={}
    def progress(*args):box['owner']._callback_bound=lambda *args:None
    owner,ops,refresh,sources=setup(fixtures,progress=progress if during_callback else None);box['owner']=owner
    if during_callback:
        with pytest.raises(ValueError):owner._callback('probe',Value(np.zeros(65536)),{})
    else:
        owner._callback_bound=lambda *args:None
        with pytest.raises(ValueError):owner._guard()
    assert not ops.calls and not refresh.calls
