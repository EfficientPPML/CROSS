"""Public producer provenance and synthetic terminal dispatch; no payload reads."""
from pathlib import Path
import copy,json
import pytest
import producer_gate as g

@pytest.fixture
def public_files(monkeypatch):
 root=g.HERE;paths=['producer_bindings.json','producer_sources.json','component_protocol.py','producer_ready/report.json','producer_ready/sources.tar.gz']
 raw={n:(root/n).read_bytes()for n in paths};calls=[]
 def read(path,maximum=1<<20):
  name=str(Path(path).relative_to(root));calls.append(name)
  if name not in raw:pytest.fail('unexpected producer read: '+name)
  if len(raw[name])>maximum:raise ValueError('bounded read')
  return raw[name]
 monkeypatch.setattr(g,'_read',read)
 return raw,calls

def test_actual_public_ready_map_and_archive(public_files):
 r=g.inspect();assert len(r['producer_sources'])==163 and r['bindings']['server_report_sha256']is None
 assert r['ready_report_sha256']==g.FIXED['ready_report_sha256']

def test_pending_server_fails_before_evidence_archive_or_private(public_files,monkeypatch):
 monkeypatch.setattr(g.protocol,'terminal_gate',lambda *a,**k:pytest.fail('terminal opened while pending'))
 with pytest.raises(ValueError,match='pending'):g.authenticate_server('/never-read','a'*64,workspace='/never-read',workspace_identity=[1,2],request={})
 assert public_files[1]==['producer_bindings.json']

@pytest.mark.parametrize('field',list(g.FIXED))
def test_fixed_bindings_cannot_be_replaced(public_files,field):
 r,calls=public_files;v=json.loads(r['producer_bindings.json']);v[field]=True if field=='producer_source_count'else'changed';r['producer_bindings.json']=json.dumps(v).encode()
 with pytest.raises(ValueError):g.inspect()
 assert calls==['producer_bindings.json']

@pytest.mark.parametrize('name',['producer_sources.json','component_protocol.py','producer_ready/report.json','producer_ready/sources.tar.gz'])
def test_exact_public_identity_rejects_changed_bytes(public_files,name):
 public_files[0][name]+=b' '
 with pytest.raises(ValueError):g.inspect()

@pytest.mark.parametrize('server,review',[(None,'a'*64),('a'*64,None),('bad','b'*64),('a'*64,True)])
def test_server_pin_pair_schema(public_files,server,review):
 r=public_files[0];v=json.loads(r['producer_bindings.json']);v.update(server_report_sha256=server,server_root_review_sha256=review);r['producer_bindings.json']=json.dumps(v).encode()
 with pytest.raises(ValueError):g.bindings()

@pytest.mark.parametrize('kind',['fixture','server'])
def test_old163_map_is_passed_to_unchanged_terminal_gate(public_files,monkeypatch,kind):
 raw,calls=public_files;v=json.loads(raw['producer_bindings.json'])
 if kind=='server':v.update(server_report_sha256='a'*64,server_root_review_sha256='b'*64);raw['producer_bindings.json']=json.dumps(v).encode()
 expected=v[kind+'_report_sha256'];captured={};sentinel=object()
 def terminal(evidence,pin,**kwargs):captured.update(evidence=evidence,pin=pin,**kwargs);return sentinel
 monkeypatch.setattr(g.protocol,'terminal_gate',terminal)
 f=g.authenticate_fixture if kind=='fixture'else g.authenticate_server
 assert f('public-terminal',expected,workspace='owned-workspace',workspace_identity=[11,22],request={'synthetic':True})is sentinel
 assert captured['expected_sources']==json.loads(raw['producer_sources.json']) and len(captured['expected_sources'])==163
 assert captured['stage']==kind and captured['pin']==expected and captured['workspace_identity']==[11,22]
 assert 'consumer_sources'not in captured

@pytest.mark.parametrize('kind',['fixture','server'])
def test_report_supplied_hash_cannot_replace_external_binding(public_files,monkeypatch,kind):
 raw,calls=public_files;v=json.loads(raw['producer_bindings.json'])
 if kind=='server':v.update(server_report_sha256='a'*64,server_root_review_sha256='b'*64);raw['producer_bindings.json']=json.dumps(v).encode()
 monkeypatch.setattr(g.protocol,'terminal_gate',lambda *a,**k:pytest.fail('wrong terminal hash consumed'))
 f=g.authenticate_fixture if kind=='fixture'else g.authenticate_server
 with pytest.raises(ValueError,match='externally bound'):f('/absent','c'*64,workspace='/absent',workspace_identity=[1,2],request={})
 assert calls==['producer_bindings.json']

def test_returned_source_map_is_not_reused_mutably(public_files):
 first=g.inspect();first['producer_sources'].clear();assert len(g.inspect()['producer_sources'])==163
