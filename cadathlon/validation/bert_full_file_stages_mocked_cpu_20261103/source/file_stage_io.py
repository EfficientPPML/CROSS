"""Small bounded metadata I/O and fresh-directory transactions for BERT stages."""
from contextlib import contextmanager
import os
from pathlib import Path
import stat
import sys
from client_keys import _OwnedDirectory, _read, _write, _json
from transport_contract import canonical


def _close(fd):
    active = sys.exception()
    try: os.close(fd)
    except BaseException as error:
        if active is None: raise
        active.add_note('stage directory close: '+repr(error))


def read_manifest(path, *, maximum=16 << 20):
    path = Path(path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        identity = os.fstat(fd)
        raw = _read(fd, path.name, maximum)
        result = _json(raw)
        if raw != canonical(result):
            raise ValueError('canonical stage manifest required')
        current = path.parent.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino) != (identity.st_dev,identity.st_ino):
            raise ValueError('stage directory changed while reading')
        return result
    finally: _close(fd)


@contextmanager
def new_directory(path):
    owner = _OwnedDirectory(path)
    try:
        yield owner
        owner.check()
    except BaseException as primary:
        try: owner.remove()
        except BaseException as cleanup: primary.add_note('stage directory cleanup: '+repr(cleanup))
        raise
    finally:
        active = sys.exception()
        try: owner.close()
        except BaseException as error:
            if active is None: raise
            active.add_note('stage directory handle close: '+repr(error))


def write_manifest(owner, name, value):
    if type(name) is not str or Path(name).name != name or name in ('.','..'):
        raise ValueError('one metadata filename required')
    _write(owner, name, value)


@contextmanager
def new_file(path, raw):
    """Create0600 exclusively; remove our inode on a failed surrounding stage."""
    if type(raw) is not bytes:
        raise TypeError('file bytes required')
    path = Path(path); fd = None; identity = None
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    parent_identity = os.fstat(parent)
    def check_parent():
        current = path.parent.lstat()
        if not stat.S_ISDIR(current.st_mode) or (current.st_dev,current.st_ino) != (parent_identity.st_dev,parent_identity.st_ino):
            raise ValueError('stage output parent changed; replacement retained')
    try:
        check_parent()
        fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
        identity = os.fstat(fd)
        with os.fdopen(fd,'wb',closefd=False) as stream:
            stream.write(raw); stream.flush(); os.fsync(fd)
        check_parent()
        yield
        check_parent()
        current = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if (current.st_dev,current.st_ino) != (identity.st_dev,identity.st_ino):
            raise ValueError('stage output file changed; replacement retained')
        os.fsync(parent)
    except BaseException as primary:
        if identity is not None:
            try:
                current = os.stat(path.name,dir_fd=parent,follow_symlinks=False)
                if (current.st_dev,current.st_ino) != (identity.st_dev,identity.st_ino):
                    raise ValueError('stage output cleanup retained replacement file')
                os.unlink(path.name,dir_fd=parent)
            except BaseException as cleanup: primary.add_note('stage output cleanup: '+repr(cleanup))
        raise
    finally:
        active=sys.exception(); error=None
        for descriptor in (fd,parent):
            if descriptor is None: continue
            try: os.close(descriptor)
            except BaseException as cleanup:
                if active is not None: active.add_note('stage output handle close: '+repr(cleanup))
                elif error is None: error=cleanup
                else: error.add_note('additional stage output handle close: '+repr(cleanup))
        if active is None and error is not None: raise error
