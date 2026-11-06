import contextlib,json
import numpy as np
import pytest
from qwen_file_contract import validate_request
from qwen_file_lifecycle import prepare_client_embeddings,ResponseFiles
from test_file_lifecycle import files

@pytest.mark.parametrize('count',[1,17,32])
def test_original_client_placement_binds_same_public_positions(public,count):
    identity=public.identity(context_id='1'*64,model_id='2'*64,producer_sources='3'*64,consumer_sources='4'*64)
    positions=tuple(3*i+7 for i in range(count));embeddings=np.arange(count*1024,dtype=np.float64).reshape(1,count,1024)
    result=prepare_client_embeddings(public,identity,'original_rows',embeddings,positions)
    c=json.loads(result.contract_bytes);validate_request(public,c)
    assert c['input']['original_positions']==list(positions)and c['input']['input_tile_count']==len(result.tiles)
    original=embeddings.copy();embeddings[:]=-77
    for token in range(count):
        tile=result.tiles[token//16];row=token%16
        assert np.array_equal(tile[row*4096:row*4096+1024],original[0,token])
        assert np.count_nonzero(tile[row*4096+1024:(row+1)*4096])==0
    assert all(not x.flags.writeable for x in result.tiles)

def test_reused_decoder_scratch_is_copied_before_next_call(tmp_path,public,request_case):
    args=files(tmp_path,public,request_case);owner=ResponseFiles(**args);scratch=np.empty(65536,dtype=np.float64)
    @contextlib.contextmanager
    def private():yield object()
    def decode(ctx,payload,binding):
        scratch[:]=123456789.;start=binding['logical']['start'];stop=binding['logical']['stop']
        scratch[:stop-start]=np.arange(start,stop,dtype=np.float64)
        return scratch
    actual=owner.run_client(private,decode);scratch[:]=-99
    assert np.array_equal(actual,np.arange(151936,dtype=np.float64).reshape(1,-1))
    assert not actual.flags.writeable
    with pytest.raises(ValueError):actual.setflags(write=True)
    owner.close()
