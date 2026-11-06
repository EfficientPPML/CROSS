import io,json,os,zipfile
from pathlib import Path
import numpy as np
import pytest
import qwen_public_parameters as p
from qwen_file_contract import canonical,sha

def npy(value):
    out=io.BytesIO();np.save(out,value,allow_pickle=False);return out.getvalue()

def fixture(tmp_path,*,value=None,gamma_count=113):
    roots={}
    for name in ('qkv','decoder','gamma'):
        root=tmp_path/name;root.mkdir();roots[name]=root
    if value is None:value=np.arange(12,dtype='<f8').reshape(3,4)
    raw=npy(value);(roots['qkv']/'matrix.npy').write_bytes(raw)
    row=dict(name='synthetic.q',file='matrix.npy',shape=list(value.shape),dtype='<f8',sha256=sha(value.tobytes()),file_sha256=sha(raw),
        source_dtype='BF16',source_raw_sha256='a'*64,promotion_exact=True)
    matrices={'synthetic.q':dict(family='qkv',descriptor=row),'synthetic.alias':dict(family='qkv',descriptor=row)}
    gammas={};arrays={}
    for i in range(gamma_count):
        value=np.array([i+1.,i+2.],dtype='<f8');name=f'synthetic.gamma.{i}';arrays[name]=value
        gammas[name]=dict(shape=[2],dtype='float64',sha256=sha(value.tobytes()))
    out=io.BytesIO();np.savez(out,**arrays);gamma_raw=out.getvalue();gamma=roots['gamma']/'gamma.npz';gamma.write_bytes(gamma_raw)
    catalog=p.ParameterCatalog(canonical(matrices),canonical(gammas),sha(gamma_raw),canonical(dict(scope='synthetic descriptors only')))
    return catalog,dict(qkv_root=roots['qkv'],decoder_root=roots['decoder'],gamma_file=gamma),row

def test_real_npy_and_all113_npz_members_alias_and_retirement(tmp_path):
    catalog,paths,_=fixture(tmp_path);owner=p.PublicParameterFiles(catalog,**paths)
    value=owner.matrix('synthetic.q');assert type(value)is np.ndarray and not value.flags.writeable
    assert owner.matrix('synthetic.alias')is value
    assert np.array_equal(owner.gamma('synthetic.gamma.112'),[113.,114.])
    assert len(owner.gamma_values)==113 and all(not a.flags.writeable for a in owner.gamma_values.values())
    owner.consumer_entered();owner.consumers_retired();owner.readback();owner.close()
    assert owner.closed and all(x.fd is None for x in owner.roots.values())
    with pytest.raises(ValueError):owner.matrix('synthetic.q')

@pytest.mark.parametrize('mode',['replacement','truncate','payload','symlink'])
def test_held_inode_and_bytes_changes_detected(tmp_path,mode):
    catalog,paths,row=fixture(tmp_path);owner=p.PublicParameterFiles(catalog,**paths);owner.matrix('synthetic.q')
    file=paths['qkv_root']/row['file']
    if mode=='replacement':
        file.rename(file.with_suffix('.old'));file.write_bytes(file.with_suffix('.old').read_bytes())
    elif mode=='truncate':
        with file.open('r+b')as f:f.truncate(32)
    elif mode=='payload':
        with file.open('r+b')as f:f.seek(-1,2);f.write(b'\xff')
    else:
        file.rename(file.with_suffix('.old'));file.symlink_to(file.with_suffix('.old').name)
    body=RuntimeError('original consumer error')
    with pytest.raises(ValueError):owner.readback()
    with pytest.raises(RuntimeError)as caught:owner.close(primary=body)
    assert caught.value is body and owner.closed

@pytest.mark.parametrize('mode',['fortran','endian','wrong_shape','extent','raw_hash','nonfinite'])
def test_headers_extent_and_raw_hash_before_exposure(tmp_path,mode):
    value=np.arange(12,dtype='<f8').reshape(3,4)
    if mode=='fortran':value=np.asfortranarray(value)
    if mode=='endian':value=value.astype('>f8')
    if mode=='nonfinite':value[0,0]=np.inf
    catalog,paths,row=fixture(tmp_path,value=value)
    rows=json.loads(catalog.matrices);file=paths['qkv_root']/row['file']
    if mode=='wrong_shape':
        b=io.BytesIO();np.lib.format.write_array_header_1_0(b,dict(shape=(1<<40,),fortran_order=False,descr='<f8'));file.write_bytes(b.getvalue())
    if mode=='extent':file.write_bytes(file.read_bytes()+b'padding')
    if mode in ('wrong_shape','extent'):
        for r in rows.values():r['descriptor']['file_sha256']=sha(file.read_bytes())
    if mode=='raw_hash':
        for r in rows.values():r['descriptor']['sha256']='b'*64
    catalog=p.ParameterCatalog(canonical(rows),catalog.gammas,catalog.gamma_container_sha256,catalog.provenance)
    owner=p.PublicParameterFiles(catalog,**paths)
    with pytest.raises(ValueError):owner.matrix('synthetic.q')
    owner.close()

@pytest.mark.parametrize('mode',['extra','missing','duplicate','oversized_header'])
def test_bounded_complete_npz_before_any_gamma_exposure(tmp_path,mode):
    catalog,paths,_=fixture(tmp_path);raw=paths['gamma_file'].read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw))as z:members={e.filename:z.read(e)for e in z.infolist()}
    if mode=='missing':members.pop(next(iter(members)))
    if mode=='extra':members['extra.npy']=npy(np.ones(1))
    if mode=='oversized_header':
        b=io.BytesIO();np.lib.format.write_array_header_1_0(b,dict(shape=(1<<40,),fortran_order=False,descr='<f8'));members['synthetic.gamma.0.npy']=b.getvalue()
    out=io.BytesIO()
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_STORED)as z:
        for name,value in members.items():z.writestr(name,value)
        if mode=='duplicate':z.writestr('synthetic.gamma.0.npy',members['synthetic.gamma.0.npy'])
    paths['gamma_file'].write_bytes(out.getvalue());catalog=p.ParameterCatalog(catalog.matrices,catalog.gammas,sha(out.getvalue()),catalog.provenance)
    owner=p.PublicParameterFiles(catalog,**paths)
    with pytest.raises(ValueError):owner.gamma('synthetic.gamma.0')
    assert owner.gamma_values=={}
    owner.close()

def test_fifo_rejected_without_read(tmp_path):
    catalog,paths,row=fixture(tmp_path);file=paths['qkv_root']/row['file'];file.unlink();os.mkfifo(file)
    owner=p.PublicParameterFiles(catalog,**paths)
    with pytest.raises(ValueError):owner.matrix('synthetic.q')
    owner.close()

def test_entered_failure_retains_backing_until_process_retirement(tmp_path):
    catalog,paths,_=fixture(tmp_path);owner=p.PublicParameterFiles(catalog,**paths);value=owner.matrix('synthetic.q');owner.consumer_entered()
    body=KeyboardInterrupt('genuine body')
    with pytest.raises(KeyboardInterrupt)as caught:owner.close(primary=body)
    assert caught.value is body and owner.failed is body and not owner.closed
    assert owner.matrices[('qkv','matrix.npy')].file.fd is not None and value[0,1]==1
    # Synthetic fixture retirement only: emulate dedicated-process OS release.
    owner.failed=None;owner.retired=True;owner.close()

def test_uncertain_file_close_detaches_and_never_retries(tmp_path,monkeypatch):
    catalog,paths,_=fixture(tmp_path);owner=p.PublicParameterFiles(catalog,**paths);owner.matrix('synthetic.q')
    held=owner.matrices[('qkv','matrix.npy')].file;fd=held.fd;original=p.os.close;calls=[]
    def close(value):
        if value==fd:calls.append(value);raise OSError('uncertain close')
        return original(value)
    with monkeypatch.context()as m:
        m.setattr(p.os,'close',close)
        with pytest.raises(OSError):owner.close()
        with pytest.raises(OSError):held.close()
    assert calls==[fd] and held.fd is None
    original(fd) # known synthetic injected failure did not perform the syscall

def test_setup_directory_failure_retires_already_opened_roots(tmp_path,monkeypatch):
    catalog,paths,_=fixture(tmp_path);seen=[];original=p.HeldDirectory
    def create(path):
        if len(seen)==1:raise RuntimeError('second directory failed')
        result=original(path);seen.append(result);return result
    monkeypatch.setattr(p,'HeldDirectory',create)
    with pytest.raises(RuntimeError,match='second directory'):p.PublicParameterFiles(catalog,**paths)
    assert seen[0].fd is None

@pytest.mark.parametrize('version,width',[(b'\x01\x00',2),(b'\x02\x00',4)])
def test_header_length_is_bounded_before_large_read(version,width):
    class Recording(io.BytesIO):
        def __init__(self,raw):super().__init__(raw);self.requests=[]
        def read(self,count=-1):self.requests.append(count);return super().read(count)
    length=32768 if width==2 else 1<<30
    reader=Recording(b'\x93NUMPY'+version+length.to_bytes(width,'little'))
    with pytest.raises(ValueError):p._header(reader,length+12,[1])
    assert max(reader.requests)<=p.HEADER_LIMIT
