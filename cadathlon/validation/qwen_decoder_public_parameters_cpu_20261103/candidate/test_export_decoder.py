"""Small synthetic format/lifecycle cases; never opens the real checkpoint."""
from pathlib import Path
from contextlib import contextmanager
import ast,hashlib,io,json,os,struct,subprocess,sys
import numpy as np
import pytest
import export_decoder as e

def fixture(bits=None,shape=(2,2),names=('a',)):
    if bits is None:bits=(0x3f80,0xc020,0x0001,0x8000)
    raw=np.asarray(bits,dtype='<u2').tobytes();head={};body=b''
    for name in names:
        head[name]=dict(dtype='BF16',shape=list(shape),data_offsets=[len(body),len(body)+len(raw)]);body+=raw
    header=json.dumps(head).encode();blob=struct.pack('<Q',len(header))+header+body
    return io.BytesIO(blob),len(blob)

def test_complete_inventory():
    rows=e.inventory();assert len(rows)==114
    for layer in range(28):
        assert rows[f'model.layers.{layer}.self_attn.o_proj.weight']==(1024,2048)
        assert rows[f'model.layers.{layer}.mlp.down_proj.weight']==(1024,3072)
        assert rows[f'model.layers.{layer}.mlp.gate_proj.weight']==rows[f'model.layers.{layer}.mlp.up_proj.weight']==(3072,1024)
    assert rows['lm_head.weight']==rows['model.embed_tokens.weight']==(151936,1024)

def test_roundtrip_all_bf16_bits_and_signs(tmp_path):
    # Every finite BF16 bit pattern, including signed zero/subnormals.
    bits=np.arange(65536,dtype='<u2');bits=bits[(bits&0x7f80)!=0x7f80]
    shape=(len(bits),1);s,size=fixture(bits,shape);h,base=e.header(s)
    row=e.export_matrix(s,h,base,size,'a',shape,tmp_path/'matrix.npy')
    actual=np.load(tmp_path/'matrix.npy',allow_pickle=False)
    expected=(bits.astype('<u4')<<16).view('<f4').astype('<f8').reshape(shape)
    assert actual.tobytes()==expected.tobytes() and row['sha256']==e.digest(expected.tobytes())
    assert row['source_raw_sha256']==e.digest(bits.tobytes())

def test_chunk_bound_and_last_partial(tmp_path):
    shape=(e.CHUNK_ROWS+1,2);bits=np.full(np.prod(shape),0x3f80,dtype='<u2')
    s,size=fixture(bits,shape);h,base=e.header(s);rows=list(e.blocks(s,base,shape))
    assert [len(x[0]) for x in rows]==[e.CHUNK_ROWS*4,4]
    row=e.export_matrix(s,h,base,size,'a',shape,tmp_path/'chunk.npy')
    assert e.verify_matrix(tmp_path/'chunk.npy',row)

@pytest.mark.parametrize('change',[
    lambda d:d.update(dtype='F32'),lambda d:d.update(shape=[True,4]),
    lambda d:d.update(shape=[2,3]),lambda d:d.update(shape=(2,2)),
    lambda d:d.update(data_offsets=[False,8]),lambda d:d.update(data_offsets=[-1,7]),
    lambda d:d.update(data_offsets=[0,9]),lambda d:d.update(data_offsets=[1,9]),
    lambda d:d.update(extra=1),lambda d:d.update(data_offsets=[0,2**63])])
def test_descriptor_rejects_bad_range_or_types(change):
    s,size=fixture();h,base=e.header(s);change(h['a'])
    with pytest.raises(ValueError):e.descriptor(h,'a',(2,2),base,size)

@pytest.mark.parametrize('raw',[b'',struct.pack('<Q',2**40),struct.pack('<Q',100)+b'{}',
    struct.pack('<Q',13)+b'{"x":1,"x":2}',struct.pack('<Q',2)+b'[]'])
def test_header_rejects_truncation_duplicate_or_type(raw):
    with pytest.raises(ValueError):e.header(io.BytesIO(raw))

@pytest.mark.parametrize('bits',[(0x7f80,0,0,0),(0xff80,0,0,0),(0x7fc0,0,0,0)])
def test_nonfinite_export_rejected_and_partial_retained(tmp_path,bits):
    s,size=fixture(bits);h,base=e.header(s);path=tmp_path/'partial.npy'
    with pytest.raises(ValueError,match='nonfinite'):e.export_matrix(s,h,base,size,'a',(2,2),path)
    assert path.is_file()

def test_truncated_body_rejected(tmp_path):
    s,size=fixture();h,base=e.header(s);s.truncate(size-1)
    with pytest.raises(ValueError,match='truncated'):e.export_matrix(s,h,base,size,'a',(2,2),tmp_path/'partial.npy')

def test_tied_alias_checks_actual_bytes_and_retains_descriptor(tmp_path):
    s,size=fixture(names=('head','embed'));h,base=e.header(s)
    row=e.export_matrix(s,h,base,size,'head',(2,2),tmp_path/'head.npy')
    alias=e.tied_alias(s,h,base,size,source_name='head',target_name='embed',shape=(2,2),record=row)
    assert alias['storage_alias_of']=='head' and alias['name']=='embed'
    assert alias['file']==row['file'] and alias['exact_bf16_alias_verified']is True
    assert len(list(tmp_path.iterdir()))==1

@pytest.mark.parametrize('mode',['different','truncated','wrong_export_digest'])
def test_tied_alias_rejects_unproven_equality(tmp_path,mode):
    s,size=fixture(names=('head','embed'));h,base=e.header(s)
    row=e.export_matrix(s,h,base,size,'head',(2,2),tmp_path/'head.npy')
    if mode=='different':s.seek(base+h['embed']['data_offsets'][0]);s.write(b'\x00\x40')
    elif mode=='truncated':s.truncate(size-1)
    else:row['sha256']='0'*64
    with pytest.raises(ValueError,match='tied'):e.tied_alias(s,h,base,size,source_name='head',target_name='embed',shape=(2,2),record=row)

@pytest.mark.parametrize('mode',['trailing','truncated','payload','descriptor'])
def test_saved_matrix_corruption_rejected(tmp_path,mode):
    s,size=fixture();h,base=e.header(s);path=tmp_path/'value.npy';row=e.export_matrix(s,h,base,size,'a',(2,2),path)
    raw=path.read_bytes()
    if mode=='trailing':path.write_bytes(raw+b'x')
    elif mode=='truncated':path.write_bytes(raw[:-1])
    elif mode=='payload':path.write_bytes(raw[:-8]+b'\0'*8)
    else:row['shape']=[1,4]
    with pytest.raises(ValueError):e.verify_matrix(path,row)

def test_write_all_handles_partial_writes():
    class Short(io.BytesIO):
        def write(self,raw):return super().write(raw[:2])
    out=Short();e.write_all(out,b'abcdefg');assert out.getvalue()==b'abcdefg'

@pytest.mark.parametrize('result',[None,0,-1,True,100])
def test_invalid_short_write_fails(result):
    class Bad:
        def write(self,raw):return result
    with pytest.raises(OSError):e.write_all(Bad(),b'abc')

def test_fdopen_failure_closes_exact_owned_descriptor(tmp_path,monkeypatch):
    p=tmp_path/'file';p.write_bytes(b'x');opened=[];closed=[];realopen=os.open;realclose=os.close
    def acquire(*a,**kw):
        fd=realopen(*a,**kw);opened.append(fd);return fd
    def close(fd):closed.append(fd);return realclose(fd)
    def failed(*a,**kw):raise RuntimeError('handoff')
    monkeypatch.setattr(e.os,'open',acquire);monkeypatch.setattr(e.os,'close',close);monkeypatch.setattr(e.os,'fdopen',failed)
    with pytest.raises(RuntimeError,match='handoff'):
        with e.owned_stream(p):pass
    assert closed==opened and len(opened)==1
    with pytest.raises(OSError):os.fstat(opened[0])

@pytest.mark.parametrize('body_failure',[False,True])
@pytest.mark.parametrize('close_cancel',[False,True])
def test_close_priority_and_attempt_all(tmp_path,monkeypatch,body_failure,close_cancel):
    p=tmp_path/'file';p.write_bytes(b'x');events=[];realclose=os.close;body=LookupError('body');ordinary=OSError('stream');cancel=KeyboardInterrupt('descriptor')
    class Fake:
        def close(self):events.append('stream');raise ordinary
    monkeypatch.setattr(e.os,'fdopen',lambda *a,**k:Fake())
    def close(fd):
        events.append('descriptor');realclose(fd)
        if close_cancel:raise cancel
    monkeypatch.setattr(e.os,'close',close)
    with pytest.raises(BaseException) as caught:
        with e.owned_stream(p):
            if body_failure:raise body
    assert events==['stream','descriptor']
    assert caught.value is (body if body_failure else cancel if close_cancel else ordinary)

@pytest.mark.parametrize('where',['enter','exit'])
def test_signal_cleanup_scope_failure_still_closes_fd(tmp_path,monkeypatch,where):
    p=tmp_path/'file';p.write_bytes(b'x');owned=[];realclose=os.close;failure=RuntimeError('mask')
    @contextmanager
    def broken():
        if where=='enter':raise failure
        yield
        raise failure
    def close(fd):owned.append(fd);realclose(fd)
    monkeypatch.setattr(e,'cleanup_signals',broken);monkeypatch.setattr(e.os,'close',close)
    with pytest.raises(RuntimeError) as caught:
        with e.owned_stream(p):pass
    assert caught.value is failure and len(owned)==1
    with pytest.raises(OSError):os.fstat(owned[0])

def test_symlink_and_existing_output_rejected(tmp_path):
    p=tmp_path/'value';p.write_bytes(b'x');link=tmp_path/'link';link.symlink_to(p)
    with pytest.raises(OSError):
        with e.owned_stream(link):pass
    with pytest.raises(FileExistsError):
        with e.owned_stream(p,write=True):pass
    assert p.read_bytes()==b'x'

def synthetic_full_assets(tmp_path,monkeypatch,*,mismatch=False):
    root=tmp_path/'source';root.mkdir();(root/'inputs').mkdir();assets=tmp_path/'assets';assets.mkdir()
    config=e.canonical({'tie_word_embeddings':True});candidate=b'{"synthetic":true}\n'
    (root/'inputs/config.json').write_bytes(config);(root/'inputs/candidate_manifest.json').write_bytes(candidate);(assets/'config.json').write_bytes(config)
    rows={n:(2,2) for n in e.inventory()};s,size=fixture(names=tuple(rows));raw=bytearray(s.getvalue())
    if mismatch:raw[-1]^=1
    (assets/'model.safetensors').write_bytes(raw)
    monkeypatch.setattr(e,'HERE',root);monkeypatch.setattr(e,'ASSETS',assets)
    monkeypatch.setattr(e,'CONFIG',e.digest(config));monkeypatch.setattr(e,'CANDIDATE',e.digest(candidate));monkeypatch.setattr(e,'CHECKPOINT',e.digest(raw));monkeypatch.setattr(e,'inventory',lambda:rows)
    return root,assets

def test_complete_synthetic_export_and_all_named_readbacks(tmp_path,monkeypatch):
    synthetic_full_assets(tmp_path,monkeypatch);out=tmp_path/'result';body=tmp_path/'bodies';r=e.run(out,body)
    assert r['passed']is True and r['operation_complete']is True and r['termination_scope_completed']is False
    assert len(r['matrices'])==114 and len(list(body.iterdir()))==113
    for row in r['matrices'].values():assert e.verify_matrix(body/row['file'],row)

def test_failed_tied_export_retains_partial_and_reports_failure(tmp_path,monkeypatch):
    synthetic_full_assets(tmp_path,monkeypatch,mismatch=True);out=tmp_path/'result';body=tmp_path/'bodies'
    with pytest.raises(ValueError,match='tied'):e.run(out,body)
    r=json.loads((out/'report.json').read_bytes());assert r['passed']is False and r['operation_complete']is False
    assert len(list(body.iterdir()))==113 and not (out/'parameter_manifest.json').exists()

def test_import_has_no_backend_model_modules():
    subprocess.run([sys.executable,'-c',"import export_decoder as e;e.backend_gate()"],cwd=e.HERE,check=True)

@pytest.mark.parametrize('entrypoint',['export_decoder','collect'])
@pytest.mark.parametrize('restoration_failure',[False,True])
def test_exact_cli_directory_report_and_outer_restoration(tmp_path,monkeypatch,entrypoint,restoration_failure):
    import importlib,finish as f
    module=importlib.import_module(entrypoint);original=f.finish;failure=RuntimeError('outer restoration')
    @contextmanager
    def scope():
        yield
        if restoration_failure:raise failure
    def wrapped(path,call,write,read):return original(path,call,write,read,termination=scope)
    monkeypatch.setattr(f,'finish',wrapped)
    out=tmp_path/'evidence';out.mkdir();report=out/'report.json';seen=[]
    def stub(*args):
        seen.append(args);r=dict(passed=True,operation_complete=True,termination_scope_completed=False,elapsed_s=0.)
        e.write_json(report,r);return r
    ns=dict(vars(module));ns.update(__name__='__main__',run=stub)
    tree=ast.parse(Path(module.__file__).read_bytes());guards=[n for n in tree.body if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)]
    guard=next(n for n in guards if isinstance(n.test.left,ast.Name) and n.test.left.id=='__name__')
    argv=[entrypoint,'--output',str(out)]+(['--body',str(tmp_path/'bodies')] if entrypoint=='export_decoder' else [])
    monkeypatch.setattr(sys,'argv',argv)
    code=compile(ast.Module(body=[guard],type_ignores=[]),str(module.__file__),'exec')
    if restoration_failure:
        with pytest.raises(RuntimeError) as caught:exec(code,ns)
        assert caught.value is failure
    elif entrypoint=='collect':
        with pytest.raises(SystemExit) as caught:exec(code,ns)
        assert caught.value.code==0
    else:exec(code,ns)
    result=json.loads(report.read_bytes());assert len(seen)==1 and seen[0][0]==out
    assert result['passed']is (not restoration_failure) and result['termination_scope_completed']is (not restoration_failure)
    if restoration_failure:assert result['terminal_scope_completed']is False

def test_finish_preserves_body_primary_over_republication_cancel(tmp_path):
    import finish as f
    primary=ValueError('body');secondary=KeyboardInterrupt('write');path=tmp_path/'report.json'
    def call():raise primary
    def write(*a):raise secondary
    with pytest.raises(ValueError) as caught:f.finish(path,call,write,lambda p:b'{}')
    assert caught.value is primary and any('KeyboardInterrupt' in n for n in primary.__notes__)

def test_finish_rejects_nonreport_result(tmp_path):
    import finish as f
    path=tmp_path/'report.json'
    with pytest.raises(ValueError,match='terminal report'):f.finish(path,lambda:[],e.write_json,e.public_report)
    result=json.loads(path.read_bytes());assert result['passed']is False and result['termination_scope_completed']is False
