"""Bounded descriptor reads and inode-bound, fresh-only output publication."""
import hashlib
import json
import os
import stat

MAX_INPUT = 32 * 1024 * 1024


from cleanup_origin import cleanup_origin, mark_cleanup, protect_body, choose


def finish(primary, operation):
    # Capture body provenance BEFORE the cleanup can rethrow the same object.
    body = None if primary is None or cleanup_origin(primary) else protect_body(primary)
    try:
        operation()
    except BaseException as error:
        chosen = choose(primary, error, 'cleanup failure', body_primary=body)
        if chosen is not primary:
            raise chosen


def identity(value):
    return value.st_dev, value.st_ino


def read(path, expected=None, limit=MAX_INPUT):
    if type(limit) is not int or limit <= 0:
        raise ValueError('positive exact input bound required')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC)
    primary = None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError('bounded nonempty regular input required')
        chunks = []
        remaining = limit + 1
        while remaining:
            block = os.read(fd, min(remaining, 1024 * 1024))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        data = b''.join(chunks)
        after = os.fstat(fd)
        observed = os.stat(path, follow_symlinks=False)
        stable = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
        if (stable != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or identity(observed) != identity(before) or not stat.S_ISREG(observed.st_mode)
                or len(data) != before.st_size or len(data) > limit):
            raise ValueError('input changed during authenticated read')
        digest = hashlib.sha256(data).hexdigest()
        if expected is not None and digest != expected:
            raise ValueError('input SHA256 differs')
        return data
    except BaseException as error:
        primary = protect_body(error)
        raise
    finally:
        finish(primary, lambda: os.close(fd))


def parse(data):
    def pairs(items):
        result = {}
        for name, value in items:
            if name in result:
                raise ValueError('duplicate JSON name')
            result[name] = value
        return result
    def reject(value):
        raise ValueError('nonfinite JSON value')
    return json.loads(data, object_pairs_hook=pairs, parse_constant=reject)


def json_bytes(value):
    return (json.dumps(value, indent=2, allow_nan=False, sort_keys=True) + '\n').encode()


class Output:
    """An owned directory is retained on every outcome; files are never replaced."""
    def __init__(self, path):
        self.path = os.path.abspath(os.fspath(path))
        self.fd = None
        os.mkdir(self.path, 0o700)
        primary = None
        try:
            self.fd = os.open(self.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
            value = os.fstat(self.fd)
            self.owner = identity(value)
            if (not stat.S_ISDIR(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o700
                    or value.st_uid != os.getuid()):
                raise ValueError('fresh owned0700 directory required')
            self.check()
        except BaseException as error:
            primary = protect_body(error)
            if self.fd is not None:
                fd, self.fd = self.fd, None
                finish(primary, lambda: os.close(fd))
            raise

    @classmethod
    def reopen(cls, path, owner):
        """Reopen only the exact retained directory, after body handles close."""
        value = cls.__new__(cls)
        value.path = os.path.abspath(os.fspath(path))
        value.owner = tuple(owner)
        value.fd = os.open(value.path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
        primary = None
        try:
            value.check()
            return value
        except BaseException as error:
            primary = protect_body(error)
            value.close(primary)
            raise

    def check(self):
        value = os.stat(self.path, follow_symlinks=False)
        opened = os.fstat(self.fd)
        if (identity(value) != self.owner or identity(opened) != self.owner
                or not stat.S_ISDIR(value.st_mode) or stat.S_IMODE(value.st_mode) != 0o700
                or value.st_uid != os.getuid()):
            raise ValueError('owned output directory identity differs')

    def write(self, name, data):
        if type(name) is not str or not name or '/' in name or name in ('.', '..'):
            raise ValueError('single fresh output filename required')
        if type(data) is not bytes:
            raise ValueError('exact output bytes required')
        self.check()
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
                     0o600, dir_fd=self.fd)
        primary = None
        try:
            opened = os.fstat(fd)
            if not stat.S_ISREG(opened.st_mode) or stat.S_IMODE(opened.st_mode) != 0o600:
                raise ValueError('private regular output required')
            view = memoryview(data)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError('output write made no progress')
                view = view[count:]
            os.fsync(fd)
            final = os.fstat(fd)
            named = os.stat(name, dir_fd=self.fd, follow_symlinks=False)
            self.check()
            if (identity(final) != identity(opened) or identity(named) != identity(opened)
                    or final.st_size != len(data) or not stat.S_ISREG(named.st_mode)):
                raise ValueError('owned output file identity differs')
        except BaseException as error:
            primary = protect_body(error)
            raise
        finally:
            finish(primary, lambda: os.close(fd))
        return {'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data),
                'identity': list(identity(opened)), 'mode': '0600'}

    def json(self, name, value):
        return self.write(name, json_bytes(value))

    def close(self, primary=None):
        if self.fd is not None:
            fd, self.fd = self.fd, None
            finish(primary, lambda: os.close(fd))
