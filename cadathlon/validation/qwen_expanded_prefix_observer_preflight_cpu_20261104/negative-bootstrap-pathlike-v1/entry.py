"""Isolated stdlib bootstrap; authenticates exact source bytes before local imports."""
import argparse
import hashlib
import importlib.abc
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAX_SOURCE = 8 * 1024 * 1024
FORBIDDEN = frozenset(('jax', 'jaxlib', 'jaxite_word', 'tensorflow'))
BACKENDS = frozenset(('torch', 'transformers', 'safetensors', 'tokenizers'))


def bounded(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC)
    primary = None
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError('bounded regular bootstrap source required')
        chunks = []
        size = 0
        while size <= limit:
            block = os.read(fd, min(65536, limit + 1 - size))
            if not block:
                break
            chunks.append(block)
            size += len(block)
        after = os.fstat(fd)
        named = os.stat(path, follow_symlinks=False)
        fields = lambda x: (x.st_dev, x.st_ino, x.st_size, x.st_mtime_ns, x.st_ctime_ns)
        if (fields(before) != fields(after) or named.st_dev != before.st_dev
                or named.st_ino != before.st_ino or not stat.S_ISREG(named.st_mode)
                or size != before.st_size or size > limit):
            raise ValueError('bootstrap source changed')
        return b''.join(chunks)
    except BaseException as error:
        primary = error
        if getattr(error, '_qwen_observer_cleanup_origin', False) is not True:
            error._qwen_observer_body_origin = True
        raise
    finally:
        try:
            os.close(fd)
        except BaseException as error:
            if getattr(error, '_qwen_observer_body_origin', False) is not True:
                error._qwen_observer_cleanup_origin = True
            if primary is None:
                raise
            if (getattr(primary, '_qwen_observer_body_origin', False) is not True and
                    getattr(primary, '_qwen_observer_cleanup_origin', False) is True and
                    isinstance(primary, Exception) and not isinstance(error, Exception)):
                raise error from primary
            primary.add_note('bootstrap close failure: ' + type(error).__name__)


def parse(data):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate bootstrap JSON key')
            value[key] = item
        return value
    return json.loads(data, object_pairs_hook=pairs)


def digest(value):
    return type(value) is str and len(value) == 64 and all(x in '0123456789abcdef' for x in value)


def authenticate(manifest, expected_digest):
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise ValueError('fresh Python -I -B entry required')
    if not digest(expected_digest):
        raise ValueError('external source manifest digest required')
    raw = bounded(manifest, 1024 * 1024)
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError('external source manifest SHA256 differs')
    expected = parse(raw)
    if type(expected) is not dict or not 8 <= len(expected) <= 256:
        raise ValueError('bounded exact source inventory required')
    for name, value in expected.items():
        if (type(name) is not str or not name or name.startswith('/') or '\\' in name
                or any(part in ('', '.', '..') for part in name.split('/')) or not digest(value)):
            raise ValueError('canonical source mapping required')
    required = {'entry.py', 'freeze_io.py', 'observer_process.py', 'input_gate.py', 'cleanup_origin.py',
                'model_runtime.py', 'native_observer.py', 'ranges.py', 'workload.py', 'reference.py', 'arithmetic.py',
                'execution_authorization.json', 'external_inputs.json', 'public/protocol.json',
                'public/model_pins.json', 'public/candidate3_manifest.json', 'backend_sources.json'}
    if not required <= expected.keys():
        raise ValueError('complete executable source closure required')
    actual = set()
    def walk(directory, prefix=''):
        with os.scandir(directory) as entries:
            for item in entries:
                name = prefix + item.name
                if item.is_symlink():
                    raise ValueError('symlink in source closure')
                if item.is_dir(follow_symlinks=False):
                    if not any(n.startswith(name + '/') for n in expected):
                        raise ValueError('unexpected directory in source closure')
                    walk(item.path, name + '/')
                elif item.is_file(follow_symlinks=False):
                    actual.add(name)
                else:
                    raise ValueError('nonregular source closure member')
    walk(ROOT)
    if actual != set(expected):
        raise ValueError('unexpected or missing source closure member')
    blobs = {name: bounded(ROOT / name, MAX_SOURCE) for name in expected}
    if sum(map(len, blobs.values())) > 32 * 1024 * 1024:
        raise ValueError('total source closure bound exceeded')
    if {name: hashlib.sha256(data).hexdigest() for name, data in blobs.items()} != expected:
        raise ValueError('source closure SHA256 differs')
    local_names = {Path(name).stem for name in expected if name.endswith('.py') and '/' not in name}
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] in FORBIDDEN | BACKENDS:
            raise ValueError('preloaded forbidden model backend')
        values = [getattr(module, '__file__', None), getattr(getattr(module, '__spec__', None), 'origin', None)]
        for value in values:
            if type(value) is not str or value in ('built-in', 'frozen'):
                continue
            path = Path(value).resolve()
            local = path == ROOT or ROOT in path.parents
            if (local and not (name == '__main__' and path == ROOT / 'entry.py')) or (name in local_names and name != '__main__'):
                raise ValueError('preloaded local helper or unknown local alias')
            if local:
                origins[name] = str(path.relative_to(ROOT))
    return raw, expected, blobs, origins


class FrozenLoader(importlib.abc.Loader):
    def __init__(self, name, data):
        self.name, self.data = name, data
    def create_module(self, spec):
        return None
    def exec_module(self, module):
        module.__file__ = str(ROOT / (self.name + '.py'))
        exec(compile(self.data, module.__file__, 'exec'), module.__dict__)


class FrozenImports(importlib.abc.MetaPathFinder):
    def __init__(self, blobs, allow_model=False):
        self.blobs = blobs
        self.allow_model = allow_model
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in FORBIDDEN or (not self.allow_model and fullname.split('.')[0] in BACKENDS):
            raise ImportError('backend import forbidden in this observer phase')
        key = fullname + '.py'
        if '.' not in fullname and key in self.blobs:
            return importlib.util.spec_from_loader(fullname, FrozenLoader(fullname, self.blobs[key]),
                                                  origin=str(ROOT / key))
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--manifest-sha256', required=True)
    parser.add_argument('--mode', choices=('collect-preflight', 'test-worker', 'collect-observer', 'observer-worker'), required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--private-output')
    parser.add_argument('--grant')
    parser.add_argument('--grant-sha256')
    args = parser.parse_args()
    raw, expected, blobs, origins = authenticate(args.manifest, args.manifest_sha256)
    sys.meta_path.insert(0, FrozenImports(blobs, allow_model=False))
    import observer_process
    return observer_process.dispatch(args, raw, expected, blobs, origins)


if __name__ == '__main__':
    raise SystemExit(main())
