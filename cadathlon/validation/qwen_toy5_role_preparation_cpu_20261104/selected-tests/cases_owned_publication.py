"""Eleven owned-I/O boundary cases; no runtime imports during source authoring.

Only three selected cases import genuine NumPy. AssetBinding.check is the sole
synthetic asset-authority seam; file/hash/header/content/ownership code is real.
All fixture paths belong to the fresh child output directory.
"""
import errno
import gc
import hashlib
import io
import json
import os
import stat
import struct
import weakref
import zipfile
from contextlib import contextmanager

import case_fixtures as f


def _expect(call, kind, *, same=None, message=None):
    try:
        call()
    except kind as error:
        if same is not None:
            assert error is same, 'original exception identity changed'
        if message is not None:
            assert message in str(error), (message, str(error))
        return error
    raise AssertionError('expected ' + kind.__name__)


@contextmanager
def _patch(owner, name, replacement):
    original = getattr(owner, name)
    setattr(owner, name, replacement)
    try:
        yield original
    finally:
        setattr(owner, name, original)
        assert getattr(owner, name) is original


@contextmanager
def _managed(owner):
    from cleanup_origin import protect_body
    from qwen_finalizer import Finalizer
    primary = None
    try:
        yield owner
    except BaseException as error:
        primary = protect_body(error)
        raise
    finally:
        # A failed/uncertain close is one-use. Never retry it for test cleanup.
        if not owner.closed:
            final = Finalizer({}, primary)
            final.attempt('test owner original final close', lambda: owner.close(primary))
            selected = final.selected()
            if selected is not None and selected is not primary:
                raise selected


def _parent(name='owned_io'):
    path = f.OUT / name
    path.mkdir(mode=0o700)
    info = path.lstat()
    assert path.resolve() == path and stat.S_ISDIR(info.st_mode)
    assert stat.S_IMODE(info.st_mode) == 0o700 and info.st_uid == os.geteuid()
    return path


def _exporter():
    import q5_checkpoint_transport as transport
    parent = _parent()
    descriptor = transport.canonical(transport.directory(parent.lstat(), parent))
    result = transport.Exporter(object(), str(parent / 'exports'), descriptor,
                                {}, {'synthetic_generic_publication': True})
    return result


def _closed(fd):
    error = _expect(lambda: os.fstat(fd), OSError)
    assert error.errno == errno.EBADF


def _retire_failed(owner, primary):
    # Exporter records a body failure and may return from close; PublicAssets
    # re-raises it. In either case there must be no second retirement attempt.
    try:
        owner.close(primary)
    except BaseException as error:
        assert error is primary
    assert owner.closed is True


def test_bounded_write_and_writelines_reject_before_growth():
    from q5_owned_io import BoundedBuffer
    stream = BoundedBuffer(8)
    try:
        assert stream.write(b'ab') == 2
        before = (stream.tell(), stream.getvalue())
        _expect(lambda: stream.write(b'1234567'), ValueError, message='before write')
        assert (stream.tell(), stream.getvalue()) == before
        # The first line succeeds; rejection must preserve the state immediately
        # before the second line, not roll back the already completed line.
        _expect(lambda: stream.writelines([b'cd', b'12345']), ValueError,
                message='before write')
        assert stream.tell() == 4 and stream.getvalue() == b'abcd'
        assert stream.write(b'efgh') == 4 and stream.getvalue() == b'abcdefgh'
        f.event('bounded_write', bytes=8, rejected_write_unchanged=True,
                earlier_writelines_member_preserved=True)
    finally:
        stream.close()


def test_bounded_seek_and_truncate_preserve_extent_on_rejection():
    from q5_owned_io import BoundedBuffer
    stream = BoundedBuffer(8)
    try:
        stream.write(b'abcd')
        for call in (lambda: stream.seek(9), lambda: stream.seek(5, 1),
                     lambda: stream.seek(5, 2), lambda: stream.seek(-1),
                     lambda: stream.seek(True), lambda: stream.truncate(9),
                     lambda: stream.truncate(True)):
            before = (stream.tell(), stream.getvalue())
            _expect(call, ValueError)
            assert (stream.tell(), stream.getvalue()) == before
        assert stream.seek(1) == 1
        assert stream.seek(1, 1) == 2
        assert stream.seek(-1, 2) == 3
        assert stream.truncate(3) == 3 and stream.getvalue() == b'abc'
        f.event('bounded_position', rejected_operations=7, positive_extent=3)
    finally:
        stream.close()


def test_tiny_sixfield_npz_publish_matches_exact_bytes_after_buffer_close():
    import numpy as np
    import q5_owned_io as owned
    fields = dict(payload=np.arange(24, dtype=np.uint32).reshape(1, 2, 2, 3, 2),
                  moduli=np.array([17, 97], dtype=np.int64),
                  degree_layout=np.array([2, 3], dtype=np.int64),
                  precision=np.array(32, dtype=np.int64),
                  scale=np.array(8.0, dtype=np.float64),
                  nsd=np.array(1, dtype=np.int64))
    expected_stream = io.BytesIO()
    try:
        np.savez(expected_stream, **fields)
        expected = expected_stream.getvalue()
    finally:
        expected_stream.close()
    assert len(expected) <= 32768
    retained = []
    opened = []
    closes = []
    with _managed(_exporter()) as owner:
        original_open, original_close = os.open, os.close
        def write(stream):
            retained.append(stream)
            np.savez(stream, **fields)
        def open_file(path, flags, mode=0o777, *, dir_fd=None):
            if path == 'tiny.npz' and dir_fd == owner.root.fd:
                assert len(retained) == 1 and retained[0].closed
                fd = original_open(path, flags, mode, dir_fd=dir_fd)
                opened.append(fd)
                return fd
            return original_open(path, flags, mode, dir_fd=dir_fd)
        def close_file(fd):
            if fd in opened:
                closes.append(fd)
            return original_close(fd)
        with _patch(owned.os, 'open', open_file), _patch(owned.os, 'close', close_file):
            result = owner._write('tiny.npz', write, 32768)
        actual = (owner.path / 'tiny.npz').read_bytes()
        assert actual == expected and result['sha256'] == hashlib.sha256(expected).hexdigest()
        assert len(opened) == 1 and closes == opened and retained[0].closed
        _closed(opened[0])
        with zipfile.ZipFile(io.BytesIO(actual)) as archive:
            assert set(archive.namelist()) == {name + '.npy' for name in fields}
            for name, expected_array in fields.items():
                with archive.open(name + '.npy') as member:
                    observed = np.load(member, allow_pickle=False, max_header_size=512)
                    assert observed.dtype == expected_array.dtype
                    assert observed.shape == expected_array.shape
                    assert np.array_equal(observed, expected_array)
        f.event('tiny_generic_npz', bytes=len(actual), sha256=result['sha256'],
                encoding_closed_before_open=True, output_close_calls=1,
                physical_q5_artifact_admission=False)


def test_partial_positive_raw_writes_make_exact_forward_progress():
    import q5_owned_io as owned
    payload = b'partial-owned-write:' + bytes(range(32))
    pieces, output_fds, closes = [], [], []
    with _managed(_exporter()) as owner:
        original_write, original_close = os.write, os.close
        def short_write(fd, data):
            if not output_fds:
                output_fds.append(fd)
            assert fd == output_fds[0]
            piece = bytes(data[:3])
            count = original_write(fd, piece)
            assert count == len(piece)
            pieces.append(piece)
            return count
        def close_file(fd):
            if fd in output_fds:
                closes.append(fd)
            return original_close(fd)
        with _patch(owned.os, 'write', short_write), _patch(owned.os, 'close', close_file):
            result = owner._write('partial.json', lambda stream: stream.write(payload), 32768)
        assert len(pieces) > 1 and b''.join(pieces) == payload
        assert (owner.path / 'partial.json').read_bytes() == payload
        assert result['sha256'] == hashlib.sha256(payload).hexdigest()
        assert closes == output_fds
        _closed(output_fds[0])
        f.event('partial_writes', calls=len(pieces), bytes=len(payload),
                exact_forward_progress=True, original_close_calls=1)


def test_zero_raw_write_rejects_and_closes_original_once():
    import q5_owned_io as owned
    fds, closes, syncs = [], [], []
    with _managed(_exporter()) as owner:
        original_close, original_sync = os.close, os.fsync
        def zero_write(fd, data):
            fds.append(fd)
            assert len(data) > 0
            return 0
        def close_file(fd):
            if fd in fds:
                closes.append(fd)
            return original_close(fd)
        def sync_file(fd):
            syncs.append(fd)
            return original_sync(fd)
        with _patch(owned.os, 'write', zero_write), _patch(owned.os, 'close', close_file), _patch(owned.os, 'fsync', sync_file):
            error = _expect(lambda: owner._write('zero.json', lambda s: s.write(b'x'), 128),
                            OSError, message='invalid positive write progress')
        assert len(fds) == 1 and closes == fds and not syncs
        assert (owner.path / 'zero.json').stat().st_size == 0
        _closed(fds[0])
        assert owner.failure is error
        _retire_failed(owner, error)
        f.event('zero_write', output_close_calls=1, sync_calls=0, ready_result=False)


def test_write_failure_preserves_primary_while_attempting_close():
    import q5_owned_io as owned
    import q5_checkpoint_transport as transport
    body, cleanup = OSError('owned write body sentinel'), OSError('owned close secondary sentinel')
    fds, closes = [], []
    with _managed(_exporter()) as owner:
        original_close = os.close
        def fail_write(fd, data):
            fds.append(fd)
            raise body
        def fail_close(fd):
            if fd in fds:
                closes.append(fd)
                original_close(fd)
                raise cleanup
            return original_close(fd)
        with _patch(owned.os, 'write', fail_write), _patch(owned.os, 'close', fail_close):
            _expect(lambda: owner._write('error.json', lambda s: s.write(b'x'), 128),
                    OSError, same=body)
        assert len(fds) == 1 and closes == fds and owner.failure is body
        assert any(row[-1] is cleanup for row in transport._QUARANTINE)
        _closed(fds[0])
        _retire_failed(owner, body)
        f.event('write_and_close_failure', exact_body_primary=True,
                secondary_retained=True, original_close_calls=1, retried=False)


def test_late_output_close_failure_never_returns_success():
    import q5_owned_io as owned
    error = OSError('late owned close sentinel')
    fds, closes = [], []
    payload = b'complete-public-bytes-before-close'
    with _managed(_exporter()) as owner:
        original_open, original_close = os.open, os.close
        def open_file(path, flags, mode=0o777, *, dir_fd=None):
            fd = original_open(path, flags, mode, dir_fd=dir_fd)
            if path == 'late.json' and dir_fd == owner.root.fd:
                fds.append(fd)
            return fd
        def fail_close(fd):
            if fd in fds:
                closes.append(fd)
                original_close(fd)
                raise error
            return original_close(fd)
        with _patch(owned.os, 'open', open_file), _patch(owned.os, 'close', fail_close):
            _expect(lambda: owner._write('late.json', lambda s: s.write(payload), 32768),
                    OSError, same=error)
        assert (owner.path / 'late.json').read_bytes() == payload
        assert len(fds) == 1 and closes == fds and owner.failure is error
        _closed(fds[0])
        _retire_failed(owner, error)
        f.event('late_close_failure', bytes_present=True, ready_result=False,
                original_close_calls=1, retried=False)


def test_owned_root_replacement_is_rejected_without_deleting_replacement():
    import q5_checkpoint_transport as transport
    parent = _parent()
    path = parent / 'original'
    path.mkdir(mode=0o700)
    root = transport.HeldRoot(transport.canonical(transport.directory(path.lstat(), path)))
    original_fd = root.fd
    original_close = os.close
    closes = []
    primary = None
    def close_file(fd):
        if fd == original_fd:
            closes.append(fd)
        return original_close(fd)
    try:
        path.rename(parent / 'retained-original')
        path.mkdir(mode=0o700)
        sentinel = path / 'replacement.txt'
        sentinel.write_bytes(b'owned replacement must survive')
        state = transport.state(path.lstat())
        primary = _expect(root.check, ValueError, message='named original root differs')
        with _patch(transport.os, 'close', close_file):
            root.close(primary)
        assert closes == [original_fd]
        _closed(original_fd)
        assert transport.state(path.lstat()) == state
        assert sentinel.read_bytes() == b'owned replacement must survive'
        assert (parent / 'retained-original').is_dir()
        f.event('owned_root_replaced', rejected_before_publication=True,
                original_close_calls=1, replacement_preserved=True)
    finally:
        if root._original is not None:
            root.close(primary)


@contextmanager
def _asset_authority():
    import numpy as np
    import q5_fixture_assets as assets
    import q5_checkpoint_transport as transport
    root = _parent('owned_assets')
    literal = {'left': ((2,), (1.25, -2.5)),
               'right': ((2, 2), (0.0, 3.0, 4.0, -1.0))}
    records, contents = {}, {}
    for index, (name, (shape, values)) in enumerate(sorted(literal.items())):
        raw_content = struct.pack('=' + str(len(values)) + 'd', *values)
        array = np.frombuffer(raw_content, dtype=np.float64).reshape(shape)
        filename = 'weight_%02d.npy' % index
        path = root / filename
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            np.save(stream, array, allow_pickle=False)
        raw = path.read_bytes()
        assert len(raw) < 32768
        records[name] = dict(name=filename, sha256=hashlib.sha256(raw).hexdigest(),
                             bytes=len(raw), state=transport.state(path.lstat()),
                             content=dict(shape=list(shape), dtype='float64',
                                          sha256=hashlib.sha256(raw_content).hexdigest()))
        contents[name] = raw_content
    root_raw = transport.canonical(transport.directory(root.lstat(), root))
    manifest = transport.canonical(dict(schema='synthetic-two-array-asset-authority',
                                        root=json.loads(root_raw), arrays=records))
    binding = assets.AssetBinding(object(), manifest, hashlib.sha256(manifest).hexdigest(), root_raw)
    fixed = (binding.profile, binding.raw, binding.expected_sha256, binding.expected_root)
    checks = []
    def synthetic_check(instance):
        assert instance is binding
        observed = (instance.profile, instance.raw, instance.expected_sha256, instance.expected_root)
        assert all(a is b for a, b in zip(observed, fixed))
        assert hashlib.sha256(instance.raw).hexdigest() == fixed[2]
        checks.append(True)
        return json.loads(instance.raw)
    with _patch(assets.AssetBinding, 'check', synthetic_check):
        yield np, assets, binding, root, records, contents, checks
    f.event('asset_authority_seam', synthetic_arrays=2, original_seeded_catalog=False,
            original_check_restored=True, authority_checks=len(checks))


def test_real_public_assets_tiny_load_check_and_clean_retirement():
    with _asset_authority() as (np, assets, binding, root, records, contents, checks):
        with _managed(assets.PublicAssets(np, binding)) as owner:
            weights = owner.load()
            assert owner.check() is weights and owner.check() is weights
            assert set(weights) == set(contents)
            for name in weights:
                assert weights[name].tobytes(order='C') == contents[name]
                assert weights[name].flags.writeable is False
            saved = assets._OWNERS[id(owner)]
            file_fds = [entry[0] for entry in saved['files']]
            root_fd = owner.root.fd
            root_ref = weakref.ref(owner.root)
            array_refs = [weakref.ref(array) for array in weights.values()]
            original_close = os.close
            closes = []
            def close_file(fd):
                if fd in file_fds or fd == root_fd:
                    closes.append(fd)
                return original_close(fd)
            with _patch(assets.os, 'close', close_file):
                owner.close()
                owner.close()
                primary = ValueError('repeat asset close primary sentinel')
                _expect(lambda: owner.close(primary), ValueError, same=primary)
            assert closes == list(reversed(file_fds)) + [root_fd]
            assert id(owner) not in assets._OWNERS and saved == {}
            assert all(getattr(owner, name) is None for name in ('weights', 'np', 'binding', 'root'))
            for fd in file_fds + [root_fd]:
                _closed(fd)
            weights = None
            gc.collect()
            assert root_ref() is None and all(ref() is None for ref in array_refs)
            f.event('real_assets_clean', original_file_closes=len(file_fds),
                    original_root_closes=1, immutable_arrays=True,
                    registry_removed=True, mirrors_cleared=True, repeatclose_safe=True)


def test_real_public_assets_fifo_rejects_without_blocking_or_body_read():
    with _asset_authority() as (np, assets, binding, root, records, contents, checks):
        with _managed(assets.PublicAssets(np, binding)) as owner:
            name = records['left']['name']
            path = root / name
            path.unlink()
            os.mkfifo(path, 0o600)
            assert stat.S_ISFIFO(path.lstat().st_mode)
            fds, closes = [], []
            counts = dict(hash=0, pread=0, load=0)
            original_open, original_close, original_pread = os.open, os.close, os.pread
            original_hash, original_load = assets._hash, np.load
            root_fd = owner.root.fd
            def open_fifo(candidate, flags, mode=0o777, *, dir_fd=None):
                if candidate == name and dir_fd == root_fd:
                    # AssertionError is deliberately not the expected ValueError.
                    assert flags & os.O_NONBLOCK, 'missing O_NONBLOCK before real FIFO open'
                    fd = original_open(candidate, flags, mode, dir_fd=dir_fd)
                    fds.append(fd)
                    return fd
                return original_open(candidate, flags, mode, dir_fd=dir_fd)
            def close_file(fd):
                if fd in fds or fd == root_fd:
                    closes.append(fd)
                return original_close(fd)
            def read_file(fd, size, offset):
                if fd in fds:
                    counts['pread'] += 1
                    raise AssertionError('FIFO body read forbidden')
                return original_pread(fd, size, offset)
            def hash_file(fd, size):
                counts['hash'] += 1
                return original_hash(fd, size)
            def load_file(*args, **kwargs):
                counts['load'] += 1
                return original_load(*args, **kwargs)
            with _patch(assets.os, 'open', open_fifo), _patch(assets.os, 'close', close_file), _patch(assets.os, 'pread', read_file), _patch(assets, '_hash', hash_file), _patch(np, 'load', load_file):
                error = _expect(owner.load, ValueError,
                                message='original owned public weight identity differs')
                _expect(lambda: owner.close(error), ValueError, same=error)
            assert len(fds) == 1 and closes == fds + [root_fd]
            assert counts == dict(hash=0, pread=0, load=0)
            assert assets._OWNERS[id(owner)]['failure'] is error and owner.closed
            assert any(row[0] is owner for row in assets._QUARANTINE)
            for fd in fds + [root_fd]:
                _closed(fd)
            f.event('real_assets_fifo_rejected', fixture_path=str(path.relative_to(f.OUT)),
                    fixture_kind='owned_fifo_metadata_only', original_file_closes=1,
                    original_root_closes=1, body_reads=counts, nonblocking_flag=True,
                    failed_original_retained=True, retried=False)


def test_real_exporter_clean_registry_mirrors_and_repeatclose():
    import q5_checkpoint_transport as transport
    with _managed(_exporter()) as owner:
        result = owner._write('generic.json', lambda stream: stream.write(b'{"synthetic":true}\n'), 128)
        assert result['sha256'] == hashlib.sha256(b'{"synthetic":true}\n').hexdigest()
        original_fds = [owner.root.fd, owner.parent.fd]
        refs = [weakref.ref(owner.root), weakref.ref(owner.parent)]
        saved = transport._PUBLISH_OWNERS[id(owner)]
        original_close = os.close
        closes = []
        def close_file(fd):
            if fd in original_fds:
                closes.append(fd)
            return original_close(fd)
        with _patch(transport.os, 'close', close_file):
            owner.close()
            owner.close()
            primary = ValueError('repeat exporter close primary sentinel')
            _expect(lambda: owner.close(primary), ValueError, same=primary)
        assert closes == original_fds
        assert id(owner) not in transport._PUBLISH_OWNERS and saved == {}
        assert all(getattr(owner, name) is None for name in
                   ('parent', 'root', 'np', 'path', 'input_bytes', 'bindings_bytes', 'records', '_fixed'))
        for fd in original_fds:
            _closed(fd)
        gc.collect()
        assert all(ref() is None for ref in refs)
        f.event('real_exporter_clean', original_root_closes=1, original_parent_closes=1,
                registry_removed=True, mirrors_cleared=True, repeatclose_safe=True,
                complete47_checkpoint_claim=False)
