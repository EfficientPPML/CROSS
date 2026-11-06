import io,struct,sys
sys.path.insert(0,'/tmp/qwen-organizer-file-adapter-20261104')
import pytest
from qwen_public_parameters import _header,HEADER_LIMIT

def test_header_length_is_bounded_before_any_large_read():
    class Recording(io.BytesIO):
        def __init__(self,raw):super().__init__(raw);self.requests=[]
        def read(self,count=-1):self.requests.append(count);return super().read(count)
    reader=Recording(b'\x93NUMPY\x02\x00'+struct.pack('<I',1<<30))
    with pytest.raises(ValueError):_header(reader,(1<<30)+12,[1])
    assert max(reader.requests)<=HEADER_LIMIT,reader.requests
