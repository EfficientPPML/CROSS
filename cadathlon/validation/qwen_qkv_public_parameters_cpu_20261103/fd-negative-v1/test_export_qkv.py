"""Small public-format tests only; never opens the real checkpoint."""
import io,json,struct,sys,subprocess
import numpy as np
import pytest
import export_qkv as e

def fixture(bits=(0x3f80,0xc020,0x0001,0x8000)):
    raw=np.asarray(bits,dtype='<u2').tobytes();head={'q':dict(dtype='BF16',shape=[2,2],data_offsets=[0,len(raw)])}
    h=json.dumps(head).encode();blob=struct.pack('<Q',len(h))+h+raw
    return io.BytesIO(blob),len(blob)
def test_inventory_exact84_fullwidth_names():
    rows=e.inventory();assert len(rows)==84
    for layer in range(28):
        assert rows[f'model.layers.{layer}.self_attn.q_proj.weight']==(2048,1024)
        assert rows[f'model.layers.{layer}.self_attn.k_proj.weight']==rows[f'model.layers.{layer}.self_attn.v_proj.weight']==(1024,1024)
def test_exact_bf16_promotion_and_raw_digest():
    stream,size=fixture();header,start=e.header(stream);value,record=e.tensor(stream,header,start,'q',(2,2),size)
    np.testing.assert_array_equal(value,np.array([[1.,-2.5],[float.fromhex('0x1p-133'),-0.]],dtype='float64'))
    assert value.dtype==np.dtype('float64') and record['promotion_exact']is True
    assert record['sha256']==e.byte_sha(value.tobytes())
@pytest.mark.parametrize('change',[lambda x:x.update(dtype='F32'),lambda x:x.update(shape=[True,4]),
    lambda x:x.update(data_offsets=[-1,7]),lambda x:x.update(data_offsets=[0,9]),lambda x:x.update(data_offsets=[False,8])])
def test_bad_tensor_header_rejected(change):
    stream,size=fixture();header,start=e.header(stream);change(header['q'])
    with pytest.raises(ValueError):e.tensor(stream,header,start,'q',(2,2),size)
@pytest.mark.parametrize('raw',[b'',struct.pack('<Q',2**40),struct.pack('<Q',100)+b'{}',struct.pack('<Q',13)+b'{"x":1,"x":2}'])
def test_bad_or_duplicate_top_header_rejected(raw):
    with pytest.raises(ValueError):e.header(io.BytesIO(raw))
def test_nonfinite_tensor_rejected():
    stream,size=fixture((0x7f80,0,0,0));header,start=e.header(stream)
    with pytest.raises(ValueError,match='nonfinite'):e.tensor(stream,header,start,'q',(2,2),size)
def test_no_backend_model_import():
    code="import sys;import export_qkv;assert not any(n in sys.modules for n in ('torch','transformers','jax','jaxlib','safetensors'))"
    subprocess.run([sys.executable,'-c',code],cwd=e.ROOT,check=True,timeout=20)
