"""Resolve the P2 shared helpers in a checkout, release or staged benchmark."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys


def common_directory(source_file):
    # The caller anchors resolution: Python may have cached p2_runtime from a
    # different checkout before loading an explicitly selected staged adapter.
    source = Path(source_file).resolve().parent
    manifest_path = source.parent / 'cross_staging_manifest.json'
    staged = source / '_cross_common'
    if manifest_path.is_file() or staged.exists():
        if not manifest_path.is_file():
            raise FileNotFoundError('Staged P2 helpers require cross_staging_manifest.json; restage the benchmark')
        manifest = json.loads(manifest_path.read_text())
        expected = manifest.get('common_sha256')
        if (manifest.get('common_directory') != 'submissions/_cross_common'
                or not isinstance(expected, dict) or not expected
                or any(Path(name).name != name or not name.endswith('.py') for name in expected)):
            raise ValueError('Missing or invalid staged P2 helper provenance; restage the benchmark')
        actual = {path.name for path in staged.glob('*.py')}
        if actual != set(expected):
            raise ValueError('Staged P2 helper inventory differs from its manifest; restage the benchmark')
        for name, wanted in expected.items():
            path = staged / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != wanted:
                raise ValueError(f'Staged P2 helper changed: {name}; restage the benchmark')
        return staged
    common = source.parents[2] / 'common'
    if not (common / 'support.py').is_file():
        raise FileNotFoundError('P2 common helpers missing from checkout/release; use benchmark-stage for file-based runs')
    return common


def install_common_path(source_file):
    common = common_directory(source_file)
    for path in common.glob('*.py'):
        loaded = sys.modules.get(path.stem)
        origin = getattr(loaded, '__file__', None)
        if origin is not None and Path(origin).resolve() != path.resolve():
            raise ValueError(f'Shared helper {path.stem} already loaded from another runtime; '
                             'use the matching staged runner in a fresh process')
    sys.path.insert(0, str(common))
    return common
