"""New owned small NPY files/direct hooks; never retained parameter bodies."""
from pathlib import Path
import json,os,hashlib
import numpy as np
import pytest
import qwen_public_parameters as parameters
from reused_client_io import HeldDirectory
from model_admission import ExpectedStoreIdentity
from qwen_file_contract import canonical
from conftest import synthetic_stores


def state_record(path):
    s=path.stat();return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,mode=s.st_mode&0o7777)


def small_file(tmp_path):
    root=tmp_path/'small';root.mkdir(mode=0o700)
    value=np.arange(6,dtype='<f8').reshape(2,3);path=root/'small.npy';np.save(path,value,allow_pickle=False)
    raw=path.read_bytes();record=dict(family='qkv',name=path.name,state=state_record(path),file_sha256=hashlib.sha256(raw).hexdigest(),raw_sha256=hashlib.sha256(value.tobytes()).hexdigest())
    desc=dict(file=path.name,shape=[2,3],file_sha256=record['file_sha256'],sha256=record['raw_sha256'])
    return root,path,record,desc


@pytest.mark.parametrize('field',['device','inode','mode'],ids=['device','inode','mode'])
def test_expected_root_mismatch_rejects_before_any_parameter_file_open(public,aggregate,tmp_path,monkeypatch,field):
    expected=synthetic_stores(aggregate,tmp_path,real_roots=True);v=json.loads(expected.raw)
    v['roots']['qkv'][field]+=1;bad=ExpectedStoreIdentity(aggregate,canonical(v));owners=[];hashes=[]
    real_directory=parameters.HeldDirectory
    def directory(path):
        owner=real_directory(path);owners.append(owner);return owner
    def forbidden_hash(*args):hashes.append(args);raise AssertionError('body hash ran before root admission')
    monkeypatch.setattr(parameters,'HeldDirectory',directory);monkeypatch.setattr(parameters,'_hash_fd',forbidden_hash)
    with pytest.raises(ValueError,match='expected held root'):parameters.open_identity_bound_parameters(public,bad)
    assert hashes==[]and len(owners)==1 and all(o.fd is None for o in owners)
    assert all(list((tmp_path/family).iterdir())==[]for family in ('qkv','decoder','gamma'))


@pytest.mark.parametrize('field',['device','inode','size','mtime_ns','mode'],ids=['device','inode','size','mtime','mode'])
def test_expected_file_mismatch_rejects_before_first_hash_header_or_mapping(tmp_path,monkeypatch,field):
    root,path,record,desc=small_file(tmp_path);record['state'][field]+=1;hashes=[];headers=[];maps=[]
    directory=HeldDirectory(root)
    def forbidden_hash(*args):hashes.append(args);raise AssertionError('body hash before expected inode')
    def forbidden_header(*args):headers.append(args);raise AssertionError('header before expected inode')
    def forbidden_map(*args,**kwargs):maps.append(args);raise AssertionError('mapping before expected inode')
    monkeypatch.setattr(parameters,'_hash_fd',forbidden_hash);monkeypatch.setattr(parameters,'_header',forbidden_header)
    # Change only this module's mapping namespace, not process-wide mmap.
    from types import SimpleNamespace
    monkeypatch.setattr(parameters,'mmap',SimpleNamespace(mmap=forbidden_map,ACCESS_READ=1))
    try:
        with pytest.raises(ValueError,match='expected held file'):parameters._MatrixLease(directory,desc,record)
        assert hashes==headers==maps==[]
    finally:directory.close()
    assert path.exists()


def test_matching_small_npy_identity_then_original_readback_and_retirement(tmp_path):
    root,path,record,desc=small_file(tmp_path);directory=HeldDirectory(root);lease=None
    try:
        lease=parameters._MatrixLease(directory,desc,record)
        assert lease.value.shape==(2,3)and lease.value.dtype==np.dtype('<f8')
        assert lease.value.tolist()==[[0.,1.,2.],[3.,4.,5.]]
        lease.file.readback()
    finally:
        if lease is not None:lease.close()
        directory.close()
    assert lease.mapping is None and lease.file.fd is None and path.read_bytes()
    assert Path(np.__file__).resolve().parent==Path('/home/jianming_gatech/miniconda3/envs/jaxite/lib/python3.13/site-packages/numpy')


def test_metadata_changed_after_open_rejects_before_second_hash(tmp_path,monkeypatch):
    root,path,record,desc=small_file(tmp_path);directory=HeldDirectory(root);file=None
    try:
        file=parameters._HeldFile(directory,path.name,4096,record['file_sha256'],record)
        old=path.stat();os.utime(path,ns=(old.st_atime_ns,old.st_mtime_ns+1000000));calls=[]
        def forbidden(*args):calls.append(args);raise AssertionError('replaced metadata rehashed')
        monkeypatch.setattr(parameters,'_hash_fd',forbidden)
        with pytest.raises(ValueError,match='changed'):file.readback()
        assert calls==[]
    finally:
        if file is not None:file.close()
        directory.close()
