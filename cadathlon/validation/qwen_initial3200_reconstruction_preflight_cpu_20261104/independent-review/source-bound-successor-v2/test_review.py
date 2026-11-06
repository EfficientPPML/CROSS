from pathlib import Path
import importlib.util,os
import pytest

spec=importlib.util.spec_from_file_location('reviewed_terminal',Path(__file__).with_name('reviewer.py'))
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def test_constructor_fstat_failure_closes_acquired_directory(tmp_path,monkeypatch):
    opened=[];closed=[];real_open=os.open;real_close=os.close;real_fstat=os.fstat
    def acquire(*args,**kwargs):
        fd=real_open(*args,**kwargs);opened.append(fd);return fd
    def status(fd):
        if fd in opened:raise OSError('initial fstat failure')
        return real_fstat(fd)
    def release(fd):closed.append(fd);return real_close(fd)
    monkeypatch.setattr(r.os,'open',acquire);monkeypatch.setattr(r.os,'fstat',status);monkeypatch.setattr(r.os,'close',release)
    try:
        with pytest.raises(OSError):r.Publication(tmp_path/'out')
        assert closed==opened,'Acquired directory descriptor was not retired'
    finally:
        for fd in opened:
            if fd not in closed:real_close(fd)


@pytest.mark.parametrize('kind',[KeyboardInterrupt,SystemExit])
def test_nested_cleanup_only_cancellation_priority(tmp_path,monkeypatch,kind):
    path=tmp_path/'data.json';path.write_text('{}')
    publication=r.Publication(tmp_path/'out');owner=publication.fd
    real_close=os.close;calls=[];cancel=kind('later directory close')
    def close(fd):
        calls.append(fd);real_close(fd)
        if fd==owner:raise cancel
        raise OSError('earlier file close only')
    monkeypatch.setattr(r.os,'close',close)
    primary=None
    with pytest.raises(kind) as caught:
        try:r.read(path)
        except BaseException as error:primary=error;raise
        finally:publication.close(primary)
    assert caught.value is cancel and len(calls)==2


@pytest.mark.parametrize('kind',[ValueError,OSError,KeyboardInterrupt,SystemExit])
def test_genuine_body_rethrow_remains_primary(tmp_path,monkeypatch,kind):
    publication=r.Publication(tmp_path/'out'); primary=kind('body'); real_close=os.close
    def close(fd):
        real_close(fd);raise primary
    monkeypatch.setattr(r.os,'close',close)
    with pytest.raises(kind) as caught:
        try:raise primary
        finally:publication.close(primary)
    assert caught.value is primary
    assert not r.cleanup_origin(primary)
