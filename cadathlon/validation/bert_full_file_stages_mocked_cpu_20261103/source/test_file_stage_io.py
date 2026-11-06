"""Real tiny filesystem transactions; no keys, ciphertexts or model execution."""
from pathlib import Path
import os
import stat
import pytest
import file_stage_io as stage
import client_keys


def test_new_directory_manifest_roundtrip_and_permissions(tmp_path):
    path=tmp_path/'upload'
    with stage.new_directory(path) as owner:
        stage.write_manifest(owner,'batch.json',dict(test=[1,2]))
    assert stage.read_manifest(path/'batch.json')==dict(test=[1,2])
    assert stat.S_IMODE(path.stat().st_mode)==0o700
    assert stat.S_IMODE((path/'batch.json').stat().st_mode)==0o600
    with pytest.raises(FileExistsError):
        with stage.new_directory(path):pass
    assert stage.read_manifest(path/'batch.json')==dict(test=[1,2])


@pytest.mark.parametrize('error',[RuntimeError('failed'),KeyboardInterrupt('cancel')])
def test_directory_primary_and_children_cleanup(tmp_path,error):
    path=tmp_path/'upload'
    with pytest.raises(type(error)) as caught:
        with stage.new_directory(path) as owner:
            stage.write_manifest(owner,'batch.json',{'test':1})
            raise error
    assert caught.value is error and not path.exists()


def test_directory_replacement_survives_cancellation(tmp_path):
    path=tmp_path/'upload';old=tmp_path/'old';error=KeyboardInterrupt('primary')
    with pytest.raises(KeyboardInterrupt) as caught:
        with stage.new_directory(path):
            path.rename(old);path.mkdir();(path/'sentinel').write_text('keep');raise error
    assert caught.value is error and (path/'sentinel').read_text()=='keep' and old.exists()
    assert any('replacement' in note for note in error.__notes__)


@pytest.mark.parametrize('kind',['duplicate','noncanonical','oversized','symlink','fifo'])
def test_metadata_rejection_without_blocking(tmp_path,kind):
    path=tmp_path/'batch.json'
    if kind=='duplicate':path.write_bytes(b'{"a":1,"a":2}\n')
    elif kind=='noncanonical':path.write_bytes(b'{"a": 1}\n')
    elif kind=='oversized':path.write_bytes(b' '*2048)
    elif kind=='symlink':
        (tmp_path/'target').write_text('{}\n');path.symlink_to(tmp_path/'target')
    else:os.mkfifo(path)
    with pytest.raises((ValueError,OSError)):stage.read_manifest(path,maximum=1024)


def test_new_file_commit_and_no_overwrite(tmp_path):
    path=tmp_path/'results.jsonl'
    with stage.new_file(path,b'{"x":1}\n'):pass
    assert path.read_bytes()==b'{"x":1}\n' and stat.S_IMODE(path.stat().st_mode)==0o600
    with pytest.raises(FileExistsError):
        with stage.new_file(path,b'replacement'):pass
    assert path.read_bytes()==b'{"x":1}\n'


@pytest.mark.parametrize('failure',['body','write','parent','file','cleanup'])
def test_new_file_failure_preserves_primary_and_unowned_files(tmp_path,monkeypatch,failure):
    parent=tmp_path/'parent';parent.mkdir();path=parent/'results'
    error=KeyboardInterrupt('primary');old=tmp_path/'old';moved=None
    if failure=='write':
        def bad(fd):raise error
        monkeypatch.setattr(stage.os,'fsync',bad)
    try:
        with pytest.raises((KeyboardInterrupt,ValueError)) as caught:
            with stage.new_file(path,b'original'):
                if failure=='parent':
                    parent.rename(old);parent.mkdir();(parent/'results').write_text('replacement')
                elif failure=='file':
                    moved=parent/'old-result';path.rename(moved);path.write_text('replacement')
                elif failure=='cleanup':
                    def bad(*a,**kw):raise OSError('unlink failure')
                    monkeypatch.setattr(stage.os,'unlink',bad)
                    raise error
                else:raise error
        if failure in ('parent','file'):
            assert path.read_text()=='replacement'
            if failure=='parent':assert not (old/'results').exists()
            else:assert moved.read_bytes()==b'original'
        else:
            assert caught.value is error
            assert path.exists() is (failure=='cleanup')
    finally:
        # pytest owns cleanup of the intentionally retained tiny fixtures.
        pass


def test_directory_cleanup_failure_does_not_hide_primary(tmp_path,monkeypatch):
    error=KeyboardInterrupt('primary')
    def bad(self):raise OSError('remove failed')
    monkeypatch.setattr(client_keys._OwnedDirectory,'remove',bad)
    with pytest.raises(KeyboardInterrupt) as caught:
        with stage.new_directory(tmp_path/'owned'):raise error
    assert caught.value is error and any('remove failed' in n for n in error.__notes__)
