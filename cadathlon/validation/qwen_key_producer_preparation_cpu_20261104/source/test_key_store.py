"""Focused unbuffered sink and new owned-small-file checks; no real key bodies."""
from types import SimpleNamespace as NS
from pathlib import Path
import hashlib,io,os,sys
import numpy as np
import pytest
import qwen_key_store as store
from qwen_key_contract import ArraySpec
from cleanup_origin import is_cleanup
from contextlib import nullcontext


def replace_os(monkeypatch,**overrides):
    values=dict(vars(os));values.update(overrides);monkeypatch.setattr(store,'os',NS(**values))


def small(tmp_path,shape=(2,3,7),private=False):
    value=(np.arange(np.prod(shape),dtype=np.uint32)%17).reshape(shape)
    spec=ArraySpec('secret.npy'if private else'public_key.npy',shape,(17,19,23))
    owner=store.KeyStore(tmp_path/'keys',(spec,),private=private,shield=nullcontext)
    return owner,spec,value


@pytest.mark.parametrize('shape,private', [((3,7),True),((2,3,7),False)],ids=['rank2-private','rank3-public'])
def test_exact_numpy_wire_bytes_and_original_bounded_readback(tmp_path,shape,private):
    owner,spec,value=small(tmp_path,shape,private)
    try:
        control=io.BytesIO();np.save(control,value,allow_pickle=False)
        record=owner.write_array(spec.name,value)
        actual=(owner.root/spec.name).read_bytes()
        assert actual==control.getvalue()
        assert record==dict(sha256=hashlib.sha256(actual).hexdigest(),dtype='uint32',shape=list(shape),bytes=len(actual))
        assert owner.verify()=={spec.name:record}
        assert owner.files[0].fd is not None and os.fstat(owner.files[0].fd).st_ino==owner.files[0].inode[1]
        manifest=b'{}\n';assert owner.write_manifest(manifest)==manifest
        assert set(p.name for p in owner.root.iterdir())=={spec.name,owner.manifest_name}
        assert owner.root.stat().st_mode&0o777==0o700
        assert all(p.stat().st_mode&0o777==0o600 for p in owner.root.iterdir())
        assert Path(np.__file__).resolve().parent==Path('/home/jianming_gatech/miniconda3/envs/jaxite/lib/python3.13/site-packages/numpy')
    finally:owner.close(primary=sys.exception())
    assert owner.closed and owner.parent.fd is None and owner.directory.fd is None


def test_write_all_positive_partial_progress_has_bounded_calls(monkeypatch):
    original=bytes(range(251))*(2*store.CHUNK//251+1);writes=[];calls=[]
    def write(fd,view):
        assert fd==7;calls.append(len(view));n=min(127,len(view));writes.append(bytes(view[:n]));return n
    replace_os(monkeypatch,write=write)
    owner=NS(fd=7);sink=store._NpySink(owner,len(original)+4096)
    assert sink.write(original)==len(original)
    assert b''.join(writes)==original and max(calls)<=store.CHUNK and sink._written==len(original)
    assert not any(hasattr(sink,n)for n in ('close','flush','fileno','__del__'))
    sink.invalidate();assert owner.fd==7
    with pytest.raises(ValueError,match='retired'):sink.write(b'x')


@pytest.mark.parametrize('kind',['zero','negative','boolean','too_large'],ids=['zero','negative','boolean','too-large'])
def test_invalid_progress_does_not_retry(monkeypatch,kind):
    calls=[]
    def write(fd,view):
        calls.append(bytes(view));return {'zero':0,'negative':-1,'boolean':True,'too_large':len(view)+1}[kind]
    replace_os(monkeypatch,write=write);sink=store._NpySink(NS(fd=7),100)
    with pytest.raises(OSError,match='progress'):sink.write(b'abc')
    assert calls==[b'abc']and sink._written==0
    sink.invalidate()


def test_cumulative_cap_rejects_before_new_write_and_empty_is_noop(monkeypatch):
    calls=[]
    def write(fd,view):calls.append(bytes(view));return len(view)
    replace_os(monkeypatch,write=write);sink=store._NpySink(NS(fd=7),5)
    assert sink.write(b'')==0 and calls==[]
    assert sink.write(b'abc')==3
    with pytest.raises(ValueError,match='bounded'):sink.write(b'def')
    assert calls==[b'abc']and sink._written==3
    assert sink.write(b'de')==2 and calls==[b'abc',b'de']
    sink.invalidate()


@pytest.mark.parametrize('error_type',[OSError,KeyboardInterrupt],ids=['oserror','keyboardinterrupt'])
def test_raised_write_preserves_identity_without_uncertain_replay(monkeypatch,error_type):
    error=error_type('controlled write failure');calls=[]
    def write(fd,view):
        calls.append(bytes(view))
        if len(calls)==1:return 2
        raise error
    replace_os(monkeypatch,write=write);sink=store._NpySink(NS(fd=7),100)
    with pytest.raises(error_type)as caught:sink.write(b'abcdef')
    assert caught.value is error and calls==[b'abcdef',b'cdef']and sink._written==2
    sink.invalidate()


def test_retained_sink_cannot_write_after_owned_numeric_fd_reuse(tmp_path):
    first=tmp_path/'first';second=tmp_path/'second';first.write_bytes(b'first');second.write_bytes(b'sentinel')
    owner=store._Descriptor('test original');owner.fd=os.open(first,os.O_RDWR);old_fd=owner.fd
    sink=store._NpySink(owner,100);replacement=None;opened=None
    try:
        # Detach while old_fd is still exclusively owned, then atomically
        # recycle that held descriptor. Never dup2 over an unowned/free slot.
        owner.fd=None
        opened=os.open(second,os.O_RDWR)
        assert opened!=old_fd
        replacement=os.dup2(opened,old_fd)
        with pytest.raises(ValueError,match='retired'):sink.write(b'corrupt')
        assert os.pread(replacement,8,0)==b'sentinel'
        sink.invalidate()
        assert os.pread(replacement,8,0)==b'sentinel'
    finally:
        if owner.fd is not None:owner.close()
        elif replacement is not None:os.close(replacement)
        else:os.close(old_fd)
        if opened is not None and opened!=replacement:os.close(opened)


def test_sink_has_no_automatic_close_or_buffer_authority(tmp_path):
    path=tmp_path/'file';fd=os.open(path,os.O_RDWR|os.O_CREAT|os.O_EXCL,0o600);owner=NS(fd=fd)
    try:
        sink=store._NpySink(owner,20);assert sink.write(b'abc')==3
        assert not isinstance(sink,io.IOBase)
        assert not any(n in store._NpySink.__dict__ for n in ('close','flush','fileno','__del__'))
        sink.invalidate();del sink
        assert os.fstat(fd).st_size==3 and os.pread(fd,3,0)==b'abc'
    finally:owner.fd=None;os.close(fd)


@pytest.mark.parametrize('primary_type',[ValueError,KeyboardInterrupt],ids=['body-valueerror','body-keyboardinterrupt'])
def test_save_body_wins_cleanup_cancellation_and_all_original_fds_retire(tmp_path,monkeypatch,primary_type):
    owner,spec,value=small(tmp_path);primary=primary_type('controlled save body');later=SystemExit('controlled cleanup cancellation')
    captured=[];closed=[];real_close=os.close;original_fds=(owner.parent.fd,owner.directory.fd)
    def save(sink,array,allow_pickle):
        assert array.dtype==np.dtype('uint32')and allow_pickle is False
        captured.append(sink);sink.write(b'partial');raise primary
    def close(fd):
        closed.append(fd);real_close(fd)
        if len(closed)==1:raise later
    fake_np=dict(vars(np));fake_np['save']=save;monkeypatch.setattr(store,'np',NS(**fake_np));replace_os(monkeypatch,close=close)
    with pytest.raises(primary_type)as caught:owner.write_array(spec.name,value)
    assert caught.value is primary and owner.failed is primary and owner.closed
    assert len(closed)==3 and len(set(closed))==3 and all(fd in closed for fd in original_fds)
    assert not (owner.root/owner.manifest_name).exists()
    assert captured[0]._owner is None and captured[0]._fd is None
    assert all(f.fd is None for f in owner.files)
    assert store._QUARANTINE and owner.retained


def test_uncertain_raw_close_is_not_retried_after_successful_npy_write(tmp_path,monkeypatch):
    owner,spec,value=small(tmp_path);owner.write_array(spec.name,value);owner.write_manifest(b'{}\n')
    real_close=os.close;calls=[];error=OSError('close returned uncertain');target=owner.files[-1].fd
    def close(fd):
        calls.append(fd);real_close(fd)
        if fd==target:raise error
    replace_os(monkeypatch,close=close)
    with pytest.raises(OSError)as caught:owner.close()
    assert caught.value is error and is_cleanup(error)and owner.closed
    before=tuple(calls)
    with pytest.raises(OSError)as again:owner.close()
    assert again.value is error and tuple(calls)==before and calls.count(target)==1
    assert all(f.fd is None for f in owner.files)
