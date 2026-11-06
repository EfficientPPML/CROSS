"""No actual public checkpoint reads: injected pre-handoff FD failure."""
import hashlib,os
from pathlib import Path
import pytest
import export_qkv as x

def test_fdopen_failure_releases_owned_descriptor(tmp_path,monkeypatch):
    root=tmp_path/'root';root.mkdir();(root/'inputs').mkdir()
    for name in ('export_qkv.py','test_export_qkv.py','inputs/config.json','inputs/candidate_manifest.json'):
        (root/name).write_text('synthetic public source')
    model=tmp_path/'model';model.mkdir();(model/'config.json').write_text('{}');(model/'model.safetensors').write_bytes(b'synthetic')
    real_sha=x.sha
    monkeypatch.setattr(x,'ROOT',root);monkeypatch.setattr(x,'MODEL',model)
    monkeypatch.setattr(x,'sha',lambda p:x.CONFIG if Path(p).name=='config.json' else x.CANDIDATE if Path(p).name=='candidate_manifest.json' else real_sha(p))
    owned=[];real_open=os.open
    def opened(*args,**kwargs):
        fd=real_open(*args,**kwargs);owned.append(fd);return fd
    monkeypatch.setattr(x.os,'open',opened)
    def failed(fd,*args,**kwargs):raise RuntimeError('fdopen before ownership handoff')
    monkeypatch.setattr(x.os,'fdopen',failed)
    try:
        with pytest.raises(RuntimeError,match='fdopen before ownership handoff'):x.collect()
        assert len(owned)==1
        with pytest.raises(OSError):os.fstat(owned[0])
    finally:
        for fd in owned:
            try:os.close(fd)
            except OSError:pass
