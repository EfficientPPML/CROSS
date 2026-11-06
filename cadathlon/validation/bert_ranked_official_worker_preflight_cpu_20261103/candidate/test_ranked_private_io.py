from pathlib import Path
import os,stat,sys
import pytest
import ranked_private_io as io


@pytest.mark.parametrize('cancel',[KeyboardInterrupt,SystemExit])
def test_file_close_error_then_directory_cancel_keeps_cancellation(tmp_path,monkeypatch,cancel):
    p=tmp_path/'private';p.write_bytes(b'synthetic');p.chmod(0o600)
    owner=io.PrivateReader();parent=owner._open_dir(tmp_path);real=io.os.close;ordinary=OSError('first close');cancellation=cancel('later close');calls=[]
    def close(fd):
        calls.append(fd);real(fd)
        if fd==parent:raise cancellation
        raise ordinary
    monkeypatch.setattr(io.os,'close',close)
    with pytest.raises(cancel) as got:
        try:owner._read(p,parent)
        finally:owner.close(sys.exception())
    assert got.value is cancellation and not owner.closed and len(calls)==2
    owner.close();assert len(calls)==2 # Unknown/reused descriptor numbers are never retried.


def test_read_body_primary_survives_both_cleanup_errors(tmp_path,monkeypatch):
    p=tmp_path/'private';p.write_bytes(b'synthetic');p.chmod(0o600)
    owner=io.PrivateReader();parent=owner._open_dir(tmp_path);real=io.os.close;body=ValueError('body');calls=[]
    def read(*args):raise body
    def close(fd):
        calls.append(fd);real(fd)
        if fd==parent:raise KeyboardInterrupt('cleanup')
        raise OSError('cleanup')
    monkeypatch.setattr(io.os,'read',read);monkeypatch.setattr(io.os,'close',close)
    with pytest.raises(ValueError) as got:
        try:owner._read(p,parent)
        finally:owner.close(sys.exception())
    assert got.value is body and len(calls)==2 and not owner.closed


def test_first_ordinary_cleanup_identity_is_retained(tmp_path,monkeypatch):
    p=tmp_path/'private';p.write_bytes(b'synthetic');p.chmod(0o600)
    owner=io.PrivateReader();parent=owner._open_dir(tmp_path);real=io.os.close;first=OSError('first')
    def close(fd):real(fd);raise first if fd!=parent else RuntimeError('second')
    monkeypatch.setattr(io.os,'close',close)
    with pytest.raises(OSError) as got:
        try:owner._read(p,parent)
        finally:owner.close(sys.exception())
    assert got.value is first
