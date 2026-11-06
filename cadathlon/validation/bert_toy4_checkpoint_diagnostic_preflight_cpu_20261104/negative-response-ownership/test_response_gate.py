"""No private or ciphertext body access; detect first private-loader call order."""
from pathlib import Path
from types import SimpleNamespace
from contextlib import contextmanager
import sys
import numpy as np
import pytest
SOURCES=Path(__file__).resolve().parent/'sources';sys.path.insert(0,str(SOURCES))
import component_worker as w
import component_contract as contract
import component_transport as transport
import fixture_assets,scale_replay,key_contexts

def test_all96_current_ciphertexts_authenticated_before_private_open(monkeypatch,tmp_path):
 plan=contract.numerical_plan();request=contract.make_contract('a'*64,'b'*32);authenticated=[];events=[]
 artifacts=[dict(index=i,manifest_sha256='c'*64,metadata={k:r[k]for k in('index','kind','label','stack','level','scale','nsd')})for i,r in enumerate(plan['checkpoints'])]
 setup=dict(request_contract=request,input_digest='d'*64);server=dict(artifacts=artifacts,all96_server_outputs_verified=True,arithmetic={'refresh_scales':[plan['profile']['scale']]*5})
 monkeypatch.setattr(scale_replay,'client_replay',lambda scales:{'trace':plan['checkpoints']})
 monkeypatch.setattr(fixture_assets,'prepare_client_reference',lambda:({},[np.zeros(65536)]*96))
 def authenticate(*a,**kw):authenticated.append(kw['tile_index']);return np.zeros(1),1.,{},{}
 monkeypatch.setattr(transport,'validate_ciphertext',authenticate)
 @contextmanager
 def private(*a,**kw):
  events.append('private-loader-entered')
  assert authenticated==list(range(96)),'private loader entered before complete96 current response authentication'
  yield SimpleNamespace(public_manifest={'context_id':'a'*64},ctx=SimpleNamespace(decrypt_slots=lambda value:np.zeros(65536)))
 monkeypatch.setattr(key_contexts,'load_client',private)
 w.validate(dict(workspace=str(tmp_path)),{},lambda:None,dict(setup=setup,server=server))
