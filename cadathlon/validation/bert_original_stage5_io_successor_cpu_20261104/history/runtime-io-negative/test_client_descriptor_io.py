"""Actual descriptor-helper AST with newly owned tiny files; no key bodies."""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import ast
import errno
import hashlib
import json
import os
import re
import stat
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent / 'runtime/stage'


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':')) + '\n').encode()


def load(fake, np_module=np):
    base = dict(os=fake, sys=sys, stat=stat, Path=Path, hashlib=hashlib,
                contextmanager=contextmanager, np=np_module, math=__import__('math'),
                _IO_FAILURE_QUARANTINE=[], canonical=canonical, _json=json.loads)

    def compile_names(file, names, namespace):
        tree = ast.parse((ROOT / file).read_bytes())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ROOT / file), 'exec'), namespace)

    keys = dict(base)
    compile_names('client_keys.py', {'_CloseState', '_stream', '_sha_fd', '_read'}, keys)
    key_module = SimpleNamespace(**keys, HEX=re.compile('[0-9a-f]{64}\\Z'))
    ctx = dict(base, keys=key_module)
    compile_names('key_contexts.py', {'_file_state', '_map_array'}, ctx)
    files = dict(base, _read=keys['_read'])
    for name in ('_CloseState', '_stream'):
        if name in keys:
            files[name] = keys[name]
    compile_names('file_stage_io.py', {'_close', 'read_manifest', 'new_file'}, files)
    return SimpleNamespace(keys=keys, ctx=ctx, files=files)


class IO:
    def __init__(self):
        self.os = SimpleNamespace(**{n: getattr(os, n) for n in dir(os)})
        self.opened = []
        self.closed = []
        self.duplicated = []
        self.os.open = self.open
        self.os.dup = self.dup
        self.os.close = self.close

    def open(self, *args, **kwargs):
        fd = os.open(*args, **kwargs)
        self.opened.append(fd)
        return fd

    def dup(self, fd):
        result = os.dup(fd)
        self.duplicated.append(result)
        return result

    def close(self, fd):
        self.closed.append(fd)
        return os.close(fd)

    def remaining(self):
        result = []
        for fd in set(self.opened + self.duplicated):
            try:
                os.fstat(fd)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                result.append(fd)
        return result

    def retire_review_fds(self):
        for fd in self.remaining():
            os.close(fd)


@pytest.fixture
def files(tmp_path):
    path = tmp_path / 'metadata.json'
    path.write_bytes(canonical({'synthetic': True}))
    array = tmp_path / 'tiny.npy'
    np.save(array, np.array([[1, 2], [3, 4]], dtype=np.uint32), allow_pickle=False)
    raw = array.read_bytes()
    descriptor = dict(dtype='uint32', shape=[2, 2], bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    parent = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        yield SimpleNamespace(root=tmp_path, path=path, array=array, parent=parent,
                              descriptor=descriptor, owner=SimpleNamespace(fd=parent, private=False, check=lambda: None))
    finally:
        os.close(parent)


@pytest.mark.parametrize('helper', ['read', 'sha', 'map'])
def test_fdopen_construction_cannot_leak_descriptor(files, helper):
    io = IO()
    primary = ValueError('synthetic fdopen construction failure')
    io.os.fdopen = lambda *a, **k: (_ for _ in ()).throw(primary)
    api = load(io.os)
    borrowed = None
    try:
        try:
            if helper == 'read':
                result = api.keys['_read'](files.parent, files.path.name, 1024)
                assert result == files.path.read_bytes()
            elif helper == 'sha':
                borrowed = os.open(files.path, os.O_RDONLY)
                result = api.keys['_sha_fd'](borrowed)
                assert result == hashlib.sha256(files.path.read_bytes()).hexdigest()
            else:
                result, _ = api.ctx['_map_array'](files.owner, files.array.name, files.descriptor, (2, 2), (7, 11))
                result._mmap.close()
        except ValueError as error:
            assert error is primary
        assert not io.remaining(), 'constructor failure lost an owned/duplicated descriptor'
    finally:
        io.retire_review_fds()
        if borrowed is not None:
            os.close(borrowed)


@pytest.mark.parametrize('helper', ['read', 'manifest'])
@pytest.mark.parametrize('kind', [ValueError, KeyboardInterrupt, SystemExit])
def test_read_primary_survives_all_descriptor_closes(files, helper, kind):
    io = IO()
    primary = kind('synthetic metadata read primary')

    class Stream:
        def __init__(self, fd): self.fd = fd
        def __enter__(self): return self
        def read(self, maximum): raise primary
        def __exit__(self, *args): os.close(self.fd)

    io.os.fdopen = lambda fd, *a, **k: Stream(fd)
    io.os.read = lambda *a: (_ for _ in ()).throw(primary)

    def close(fd):
        io.close(fd)
        raise OSError('synthetic post-close failure')

    io.os.close = close
    api = load(io.os)
    try:
        with pytest.raises(BaseException) as got:
            if helper == 'read': api.keys['_read'](files.parent, files.path.name, 1024)
            else: api.files['read_manifest'](files.path)
        assert got.value is primary
        assert len(io.closed) == len(io.opened)
        assert not io.remaining()
    finally:
        io.retire_review_fds()


@pytest.mark.parametrize('kind', [KeyboardInterrupt, SystemExit])
def test_manifest_cleanup_only_cancellation_outranks_prior_close_error(files, kind):
    io = IO()
    ordinary, cancellation = OSError('file close'), kind('directory close')

    def close(fd):
        io.close(fd)
        raise ordinary if len(io.closed) == 1 else cancellation

    io.os.close = close
    api = load(io.os)
    try:
        with pytest.raises(BaseException) as got: api.files['read_manifest'](files.path)
        assert got.value is cancellation
        assert len(io.closed) == len(io.opened) == 2
        assert not io.remaining()
    finally:
        io.retire_review_fds()


class FaultyStream:
    def __init__(self, fd, mode, kwargs, close_error, write_error=None):
        self.stream = os.fdopen(fd, mode, **kwargs)
        self.close_error = close_error
        self.write_error = write_error
    def __getattr__(self, name): return getattr(self.stream, name)
    def __enter__(self): return self
    def __exit__(self, *args): self.close()
    def write(self, raw):
        if self.write_error is not None: raise self.write_error
        return self.stream.write(raw)
    def close(self):
        self.stream.close()
        raise self.close_error


@pytest.mark.parametrize('phase', ['header', 'map'])
@pytest.mark.parametrize('kind', [ValueError, KeyboardInterrupt, SystemExit])
def test_map_primary_survives_stream_and_descriptor_cleanup(files, phase, kind):
    io = IO()
    primary, secondary = kind('synthetic map body primary'), OSError('synthetic stream close')
    count = 0

    def fdopen(fd, mode, **kwargs):
        nonlocal count
        if sys._getframe(1).f_code.co_name == '_sha_fd':
            return os.fdopen(fd, mode, **kwargs)
        count += 1
        if count == (1 if phase == 'header' else 2):
            return FaultyStream(fd, mode, kwargs, secondary)
        return os.fdopen(fd, mode, **kwargs)

    io.os.fdopen = fdopen
    fmt = SimpleNamespace(**{n: getattr(np.lib.format, n) for n in dir(np.lib.format)})
    np_fake = SimpleNamespace(**{n: getattr(np, n) for n in dir(np)})
    np_fake.lib = SimpleNamespace(format=fmt)
    if phase == 'header': fmt.read_magic = lambda *a: (_ for _ in ()).throw(primary)
    else: np_fake.memmap = lambda *a, **k: (_ for _ in ()).throw(primary)
    api = load(io.os, np_fake)
    try:
        with pytest.raises(BaseException) as got:
            api.ctx['_map_array'](files.owner, files.array.name, files.descriptor, (2, 2), (7, 11))
        assert got.value is primary
        assert not io.remaining()
    finally:
        io.retire_review_fds()


@pytest.mark.parametrize('kind', [ValueError, KeyboardInterrupt, SystemExit])
def test_private_output_write_primary_survives_stream_cleanup(files, kind):
    io = IO()
    primary, secondary = kind('synthetic output write'), OSError('synthetic stream close')
    io.os.fdopen = lambda fd, mode, **kw: FaultyStream(fd, mode, kw, secondary, primary)
    api = load(io.os)
    output = files.root / 'new-private-result'
    try:
        with pytest.raises(BaseException) as got:
            with api.files['new_file'](output, b'synthetic'):
                pytest.fail('failed write cannot enter transaction')
        assert got.value is primary
        assert not io.remaining()
        assert not output.exists()
    finally:
        io.retire_review_fds()


@pytest.mark.parametrize('kind', [KeyboardInterrupt, SystemExit])
def test_private_output_cleanup_only_cancellation_priority(files, kind):
    io = IO()
    ordinary, cancellation = OSError('file close'), kind('parent close')

    def close(fd):
        io.close(fd)
        raise ordinary if len(io.closed) == 1 else cancellation

    io.os.close = close
    api = load(io.os)
    try:
        with pytest.raises(BaseException) as got:
            with api.files['new_file'](files.root / 'new-private-result', b'synthetic'): pass
        assert got.value is cancellation
        assert len(io.closed) == len(io.opened) == 2
        assert not io.remaining()
    finally:
        io.retire_review_fds()


def test_private_output_initial_fstat_failure_closes_parent(files):
    io = IO()
    primary = ValueError('synthetic initial parent fstat')
    io.os.fstat = lambda *a: (_ for _ in ()).throw(primary)
    api = load(io.os)
    try:
        with pytest.raises(ValueError) as got:
            with api.files['new_file'](files.root / 'new-private-result', b'synthetic'): pass
        assert got.value is primary
        assert not io.remaining(), 'initial parent snapshot failed before protected cleanup'
    finally:
        io.retire_review_fds()


def test_original_metadata_map_and_private_write_formats(files):
    io = IO()
    api = load(io.os)
    assert api.files['read_manifest'](files.path) == {'synthetic': True}
    borrowed = os.open(files.path, os.O_RDONLY)
    try: assert api.keys['_sha_fd'](borrowed) == hashlib.sha256(files.path.read_bytes()).hexdigest()
    finally: os.close(borrowed)
    mapped, state = api.ctx['_map_array'](files.owner, files.array.name, files.descriptor, (2, 2), (7, 11))
    try:
        np.testing.assert_array_equal(mapped, [[1, 2], [3, 4]])
        assert mapped.dtype == np.dtype('uint32') and mapped.flags.writeable is False
        assert state == api.ctx['_file_state'](files.array.stat())
    finally: mapped._mmap.close()
    output = files.root / 'new-private-result'
    with api.files['new_file'](output, b'synthetic-output'): pass
    assert output.read_bytes() == b'synthetic-output' and stat.S_IMODE(output.stat().st_mode) == 0o600
    assert not io.remaining()
