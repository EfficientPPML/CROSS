"""Artifact/schema tests; payloads are zero test arrays, never encrypted evidence."""
import copy,hashlib,importlib.util,io,json,os,shutil,sys,types,zipfile
from pathlib import Path
import numpy as np
import pytest
import transport as t
import transport_contract as c


def public():
    return dict(context_id='a'*64,model_id='b'*64,checkpoint_sha256=c.CHECKPOINT_SHA256,
        config_sha256=c.CONFIG_SHA256,candidate_sha256=c.CANDIDATE_SHA256,
        profile_sha256=c.PROFILE_SHA256,parameters=c.parameters())

def metadata(positions=(0,2,7,127)):
    k=len(positions)
    return dict(schema='bert-client-row-packing-v1',input_tile_count=8,original_positions=list(positions),
        original_attention_mask=[int(i in positions) for i in range(128)],compact_key_mask=[int(i<k) for i in range(128)],
        key_count=k,query_tiles=(k+15)//16,valid_rows=[max(0,min(16,k-16*i)) for i in range(8)],
        packing=dict(slots=65536,row_pitch=4096,rows_per_tile=16,active_features=768,row_order='increasing original positions',
            positions_applied='before compaction',feature_padding='zero',unused_rows='zero',input_dtype='float64'))

def contract():return c.make_contract(public(),'opaque_sample_X',metadata())

def manifest(binding):
    return t._manifest_fields(binding)|dict(files={'bundle.json':'d'*64,'ct_0.npz':'e'*64})


def test_all128_public_counts():
    for k in range(1,129):
        v=c.make_contract(public(),'opaque',metadata(tuple(range(k))))
        assert c.validate_contract(v)==v
        assert sum(v['input']['valid_rows'])==k
    assert c.parameters()['degree']==131072 and len(c.parameters()['q_towers'])==100 and len(c.parameters()['p_towers'])==14


def test_real_client_public_metadata_agrees_without_embeddings():
    path=Path('/tmp/bert-full-model-file-stages-20261103/client_inputs.py')
    spec=importlib.util.spec_from_file_location('transport_review_client_inputs',path);module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    try:
        spec.loader.exec_module(module)
        record=module.ValidatedInput(789,(101,)+tuple([0]*127),tuple([0]*128),tuple(metadata()['original_attention_mask']))
        prepared=module.pack_embeddings(record,np.zeros((128,768),dtype=np.float64))
        assert prepared.public_metadata()==metadata()
        assert c.make_contract(public(),'opaque',prepared.public_metadata())['input']==metadata()
        assert '789' not in c.canonical(c.make_contract(public(),'opaque',prepared.public_metadata())).decode()
    finally:sys.modules.pop(spec.name,None)


@pytest.mark.parametrize('field',['context_id','model_id','checkpoint_sha256','config_sha256','candidate_sha256','profile_sha256','parameters','extra'])
def test_public_identity_fail_closed(field):
    p=public()
    if field in ('context_id','model_id'):p[field]='A'*64
    elif field=='parameters':p[field]['dnum']=8.
    else:p[field]='f'*64
    with pytest.raises(ValueError):c.make_contract(p,'opaque',metadata())


@pytest.mark.parametrize('kind',['positions_float','positions_bool','missing_cls','order','original_mask','compact_mask','valid_rows','packing','private_id','token_ids'])
def test_public_packing_rejects_private_or_unbound_fields(kind):
    m=metadata()
    if kind=='positions_float':m['original_positions'][0]=0.
    elif kind=='positions_bool':m['original_positions'][0]=False
    elif kind=='missing_cls':m['original_positions']=m['original_positions'][1:]
    elif kind=='order':m['original_positions'].reverse()
    elif kind=='original_mask':m['original_attention_mask'][1]=1
    elif kind=='compact_mask':m['compact_key_mask'][127]=1
    elif kind=='valid_rows':m['valid_rows'][0]=16
    elif kind=='packing':m['packing']['positions_applied']='after compaction'
    else:m['target_idx' if kind=='private_id' else 'input_ids']=[1]
    with pytest.raises(ValueError):c.make_contract(public(),'opaque',m)


@pytest.mark.parametrize('nonce',[None,True,1,'','a/b','x'*129])
def test_nonce(nonce):
    with pytest.raises(ValueError):c.make_contract(public(),nonce,metadata())


def test_all_ordered_requests_bind_digest_and_response_plan():
    v=contract();ms=[manifest(c.ciphertext_contract(v,'request',i,None,None)) for i in range(8)]
    d=t.request_digest(ms,v)
    for alter in (ms[:-1],ms+[ms[0]],list(reversed(ms)),[ms[0]]*8):
        with pytest.raises(ValueError):t.request_digest(alter,v)
    changed=copy.deepcopy(ms);changed[7]['files']['ct_0.npz']='f'*64
    assert t.request_digest(changed,v)!=d
    different=copy.deepcopy(v);different['request_id']='other'
    with pytest.raises(ValueError):t.request_digest(ms,different)
    scale=float(c.parameters()['scaling_factor']*1.21)
    b=c.ciphertext_contract(v,'response',0,scale,d)
    assert b['level']==2 and len(b['moduli'])==6 and b['scale']==scale
    assert b['shape']==(1,2,256,512,6)
    assert t._validate_manifest(manifest(b),b)
    with pytest.raises(ValueError):t._validate_manifest(manifest(b),c.ciphertext_contract(v,'response',0,np.nextafter(scale,np.inf).item(),d))


@pytest.mark.parametrize('role,index,scale,digest',[
    ('request',True,None,None),('request',8,None,None),('response',1,1.,'a'*64),
    ('response',0,None,'a'*64),('response',0,True,'a'*64),('response',0,1,'a'*64),
    ('response',0,np.float64(1),'a'*64),('response',0,float('nan'),'a'*64),
    ('response',0,0.,'a'*64),('response',0,1.,None),('request',0,None,'a'*64),('request',0,1.,None)])
def test_role_metadata_rejection(role,index,scale,digest):
    with pytest.raises(ValueError):c.ciphertext_contract(contract(),role,index,scale,digest)


class FakePolynomial:
    def __init__(self,binding):
        self.shape=binding['shape'];self.degree_layout=tuple(binding['layout']);self.precision=32
        self._ckks_nsd=1;self._ckks_scale=binding['scale'];self.moduli=binding['moduli']
    def validate(self):pass
    def get_moduli(self):return self.moduli
    def to_array(self):return np.broadcast_to(np.array(0,dtype=np.uint32),self.shape)


@pytest.fixture
def fake_backend(monkeypatch):
    monkeypatch.setitem(sys.modules,'jaxite_word',types.SimpleNamespace(Polynomial=FakePolynomial))


@pytest.fixture(scope='module')
def request_cache(tmp_path_factory):
    v=contract();binding=c.ciphertext_contract(v,'request',0,None,None)
    path=tmp_path_factory.mktemp('full_transport')/'request'
    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules,'jaxite_word',types.SimpleNamespace(Polynomial=FakePolynomial))
        m=t.save_ciphertext(FakePolynomial(binding),path,contract=v,role='request',tile_index=0)
    return path,m,v


def test_real_full_shape_uncompressed_request_checked(request_cache):
    path,m,v=request_cache
    payload,scale,b,checked=t.validate_ciphertext(path,contract=v,role='request',tile_index=0)
    assert payload.shape==(1,2,256,512,100) and payload.dtype==np.uint32 and not payload.any()
    assert checked==m and payload.nbytes==104857600 and scale==c.parameters()['scaling_factor']


def test_response_actual_noncanonical_scale(tmp_path,fake_backend):
    v=contract();scale=float(c.parameters()['scaling_factor']*1.21);d='a'*64
    b=c.ciphertext_contract(v,'response',0,scale,d);p=tmp_path/'response'
    m=t.save_ciphertext(FakePolynomial(b),p,contract=v,role='response',tile_index=0,expected_scale=scale,input_digest=d)
    payload,actual,_,checked=t.validate_ciphertext(p,contract=v,role='response',tile_index=0,expected_scale=scale,input_digest=d)
    assert payload.nbytes==6291456 and actual==scale and checked==m
    for changed in (dict(expected_scale=c.parameters()['scaling_factor'],input_digest=d),dict(expected_scale=scale,input_digest='0'*64)):
        with pytest.raises(ValueError):t.validate_ciphertext(p,contract=v,role='response',tile_index=0,**changed)


def clone(cache,tmp_path):
    source,m,v=cache;p=tmp_path/'copy';p.mkdir()
    for f in source.iterdir():
        if f.name=='transport.json':shutil.copyfile(f,p/f.name)
        else:os.link(f,p/f.name)
    return p,copy.deepcopy(m),v


@pytest.mark.parametrize('kind',['index','scale','level','nsd_extra','ctx','nonce','contract','missing','extra','duplicate_json','bundle_plaintext','symlink','fifo'])
def test_artifact_rejections_before_backend(request_cache,tmp_path,kind):
    p,m,v=clone(request_cache,tmp_path)
    if kind=='missing':(p/'bundle.json').unlink()
    elif kind=='extra':(p/'secret.npy').write_bytes(b'not allowed')
    elif kind=='duplicate_json':(p/'transport.json').write_bytes(b'{"format":"bad",'+t._canonical(m)[1:])
    elif kind in ('symlink','fifo'):
        (p/'ct_0.npz').unlink()
        if kind=='symlink':(p/'ct_0.npz').symlink_to(request_cache[0]/'ct_0.npz')
        else:os.mkfifo(p/'ct_0.npz')
    elif kind=='bundle_plaintext':
        (p/'bundle.json').unlink();raw=b'{"shape":{"__int__":123},"count":1}'
        (p/'bundle.json').write_bytes(raw);m['files']['bundle.json']=hashlib.sha256(raw).hexdigest();(p/'transport.json').write_bytes(t._canonical(m))
    else:
        key,value={'index':('tile_index',1),'scale':('scale',int(m['scale'])),'level':('level',49.),'nsd_extra':('nsd',1),
            'ctx':('context_id','f'*64),'nonce':('request_id','other'),'contract':('request_contract_sha256','0'*64)}[kind]
        m[key]=value;(p/'transport.json').write_bytes(t._canonical(m))
    with pytest.raises((ValueError,OSError)):t.load_ciphertext(p,contract=v,role='request',tile_index=0)
    assert 'jax' not in sys.modules


def small_binding():return dict(shape=(1,2,2,2,2),moduli=[17,19],layout=[2,2],scale=1.)

def payload_arrays():
    return dict(payload=np.zeros((1,2,2,2,2),dtype=np.uint32),moduli=np.array([17,19],dtype=np.int64),degree_layout=np.array([2,2],dtype=np.int64),precision=np.int64(32),scale=np.float64(1),nsd=np.int64(1))

def npz(arrays,compressed=False):
    stream=io.BytesIO();(np.savez_compressed if compressed else np.savez)(stream,**arrays);return stream.getvalue()


@pytest.mark.parametrize('kind',['payload_dtype','payload_shape','residue','negative','scale','nsd','moduli','fortran','extra','compressed','truncated','oversized_header'])
def test_strict_payload_reader(kind):
    a=payload_arrays();raw=None
    if kind=='payload_dtype':a['payload']=a['payload'].astype(np.uint64)
    elif kind=='payload_shape':a['payload']=a['payload'].ravel()
    elif kind=='residue':a['payload'].flat[0]=17
    elif kind=='negative':a['payload']=a['payload'].astype(np.int32);a['payload'].flat[0]=-1
    elif kind=='scale':a['scale']=np.float64(np.nextafter(1.,2.))
    elif kind=='nsd':a['nsd']=np.float64(1.)
    elif kind=='moduli':a['moduli']=a['moduli'][::-1]
    elif kind=='fortran':a['payload']=np.asfortranarray(a['payload'])
    elif kind=='extra':a['private_target']=np.int64(123)
    elif kind=='oversized_header':
        stream=io.BytesIO()
        with zipfile.ZipFile(io.BytesIO(npz(a))) as old,zipfile.ZipFile(stream,'w') as new:
            for entry in old.infolist():
                data=old.read(entry)
                if entry.filename=='payload.npy':
                    buf=io.BytesIO();np.lib.format.write_array_header_1_0(buf,dict(descr='<u4',fortran_order=False,shape=(2**60,)))
                    data=buf.getvalue()
                new.writestr(entry.filename,data)
        raw=stream.getvalue()
    if raw is None:raw=npz(a,kind=='compressed')
    if kind=='truncated':raw=raw[:-1]
    with pytest.raises((ValueError,zipfile.BadZipFile,EOFError)):t._payload(raw,small_binding())


def test_npy_claim_rejected_before_numpy_allocation(monkeypatch):
    raw=npz(payload_arrays());calls=[]
    def forbidden(*a,**kw):calls.append(True);raise AssertionError('np.load called')
    monkeypatch.setattr(np,'load',forbidden)
    wrong=small_binding();wrong['shape']=(1,2,4,4,2)
    with pytest.raises(ValueError):t._payload(raw,wrong)
    assert not calls


def test_writer_replacement_and_primary_cancellation(tmp_path,fake_backend,monkeypatch):
    import bundle_io
    v=contract();b=c.ciphertext_contract(v,'request',0,None,None);ct=FakePolynomial(b);p=tmp_path/'tile';moved=tmp_path/'moved'
    def replacement(*args):
        p.rename(moved);p.mkdir();(p/'sentinel').write_text('replacement')
        raise KeyboardInterrupt('primary')
    monkeypatch.setattr(bundle_io,'save_bundle',replacement)
    with pytest.raises(KeyboardInterrupt) as error:t.save_ciphertext(ct,p,contract=v,role='request',tile_index=0)
    assert (p/'sentinel').exists() and moved.exists()
    assert any('ownership changed' in x for x in error.value.__notes__)


def test_cleanup_error_never_masks_primary(tmp_path,fake_backend,monkeypatch):
    import bundle_io
    def failure(*args):raise KeyboardInterrupt('primary')
    def cleanup(*args):raise OSError('secondary')
    monkeypatch.setattr(bundle_io,'save_bundle',failure);monkeypatch.setattr(t.shutil,'rmtree',cleanup)
    v=contract();ct=FakePolynomial(c.ciphertext_contract(v,'request',0,None,None))
    with pytest.raises(KeyboardInterrupt) as e:t.save_ciphertext(ct,tmp_path/'tile',contract=v,role='request',tile_index=0)
    assert any('secondary' in note for note in e.value.__notes__)


def test_existing_directory_untouched(tmp_path,fake_backend):
    p=tmp_path/'tile';p.mkdir();(p/'sentinel').write_text('keep');v=contract()
    with pytest.raises(FileExistsError):t.save_ciphertext(FakePolynomial(c.ciphertext_contract(v,'request',0,None,None)),p,contract=v,role='request',tile_index=0)
    assert (p/'sentinel').read_text()=='keep'


def test_imports_no_backend():assert 'jax' not in sys.modules and 'torch' not in sys.modules
