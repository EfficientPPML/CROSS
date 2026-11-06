import io,json,os
import numpy as np
import pytest
import qwen_file_lifecycle as l
import qwen_public_parameters as p
from qwen_file_contract import *

def request_files(tmp_path,public,length):
    identity=public.identity(context_id='1'*64,model_id='2'*64,producer_sources='3'*64,consumer_sources='4'*64)
    c=request(public,identity,'synthetic_upload',[2*i for i in range(length)])
    binding=ciphertext_binding(public,c,'request',0);root=tmp_path/'upload';root.mkdir()
    stream=io.BytesIO();np.savez(stream,payload=np.zeros(tuple(binding['shape']),dtype=np.uint32),
        moduli=np.asarray(binding['moduli'],dtype=np.int64),degree_layout=np.asarray([256,512],dtype=np.int64),
        precision=np.asarray(32,dtype=np.int64),scale=np.asarray(DELTA,dtype=np.float64),nsd=np.asarray(1,dtype=np.int64))
    raw=stream.getvalue();(root/'tile00.npz').write_bytes(raw);count=c['input']['input_tile_count']
    for i in range(1,count):os.link(root/'tile00.npz',root/f'tile{i:02}.npz')
    artifacts=[artifact(public,c,'request',i,payload_sha256=sha(raw))for i in range(count)]
    manifest=canonical(request_manifest(public,c,artifacts));(root/'request.json').write_bytes(manifest)
    return dict(public=public,contract=c,manifest_bytes=manifest,root=root,request_manifest_sha256=sha(manifest))

@pytest.mark.parametrize('length',[1,17])
def test_real_synthetic_fullshape_fresh_input_files_preserve_level_and_scale(tmp_path,public,length):
    args=request_files(tmp_path,public,length);owner=l.RequestFiles(**args)
    assert len(owner.records)==(length+15)//16
    for record in owner.records:
        b=json.loads(record.binding)
        assert b['level']==49 and b['scale']==DELTA and b['shape']==[1,2,256,512,100]
        assert record.array().shape==(1,2,256,512,100)and not record.array().flags.writeable
    # No metadata relabel or crypto reduction is performed by file admission.
    assert json.loads(owner.contract)['owner_input_level']==25
    owner.consumer_entered();owner.consumers_retired();owner.close()
    assert all(f.fd is None for f in owner.files)

def test_partial_input_inventory_rejected_before_payload(tmp_path,public,monkeypatch):
    args=request_files(tmp_path,public,17);(args['root']/'tile01.npz').unlink();opened=[]
    monkeypatch.setattr(l,'_HeldFile',lambda *a,**k:opened.append(a))
    with pytest.raises(ValueError,match='inventory'):l.RequestFiles(**args)
    assert opened==[]

def test_untrusted_upload_manifest_pin_rejected_before_directory(tmp_path,public,monkeypatch):
    args=request_files(tmp_path,public,1);args['request_manifest_sha256']='f'*64
    monkeypatch.setattr(l,'HeldDirectory',lambda *a:(_ for _ in ()).throw(AssertionError('directory opened')))
    with pytest.raises(ValueError,match='externally bound'):l.RequestFiles(**args)

def test_entered_server_failure_quarantines_request_backing(tmp_path,public):
    args=request_files(tmp_path,public,1);owner=l.RequestFiles(**args);owner.consumer_entered();body=RuntimeError('primitive failed before return')
    with pytest.raises(RuntimeError)as caught:owner.close(primary=body)
    assert caught.value is body and owner.failed is body and all(f.fd is not None for f in owner.files)
    owner.failed=None;owner.retired=True;owner.close() # synthetic OS-retirement emulation only

def test_source_bound_catalog_factory_rejects_before_body_directory(monkeypatch):
    monkeypatch.setattr(p,'PublicParameterFiles',lambda *a,**k:(_ for _ in ()).throw(AssertionError('body reader opened')))
    with pytest.raises(TypeError):p.open_source_bound_parameters(object(),qkv_root='/unused',decoder_root='/unused',gamma_file='/unused')

def test_source_bound_catalog_factory_uses_all_original_descriptors(public,monkeypatch):
    observed=[]
    def capture(catalog,**paths):observed.append((catalog,paths));return 'synthetic reader'
    monkeypatch.setattr(p,'PublicParameterFiles',capture)
    assert p.open_source_bound_parameters(public,qkv_root='/unused-q',decoder_root='/unused-d',gamma_file='/unused-g')=='synthetic reader'
    catalog=observed[0][0]
    assert len(json.loads(catalog.matrices))==198 and len(json.loads(catalog.gammas))==113
    assert json.loads(catalog.provenance)['historical_qkv_identity']['candidate_sha256'].startswith('7f6478')
