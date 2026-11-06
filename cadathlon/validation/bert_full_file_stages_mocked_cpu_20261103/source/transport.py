"""Strict isolated full-model tile transport; no secret/client data loader.

The parent key loader authenticates ``public`` against its full public key
manifest. This module binds ciphertext artifacts to that already-validated
identity, an opaque request nonce, and (for responses) the input manifest hash.
It reuses P1's writer only inside a newly created owned directory. Reads rebuild
from checked bytes directly, never the permissive bundle_io.load_bundle path.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import zipfile

import numpy as np

FORMAT = 'cross-bert-full-model-tile-v1'
FILES = {'bundle.json', 'ct_0.npz', 'transport.json'}
NPZ_FIELDS = {'payload', 'moduli', 'degree_layout', 'precision', 'scale', 'nsd'}
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
REQUEST_ID = re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
MAX_JSON_BYTES = 16384
from transport_contract import canonical as _canonical, ciphertext_contract


def manifest_digest(manifest):
    return hashlib.sha256(_canonical(manifest)).hexdigest()


def _object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON field: '+key)
        value[key] = item
    return value


def _json(raw):
    def invalid(value):
        raise ValueError('nonfinite JSON value: '+value)
    return json.loads(raw, object_pairs_hook=_object, parse_constant=invalid)


def _read_regular(path, maximum, *, directory_fd=None):
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory_fd)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
            raise ValueError('unexpected ciphertext artifact type or size')
        with os.fdopen(descriptor, 'rb', closefd=False) as stream: raw = stream.read(maximum+1)
        if len(raw) > maximum: raise ValueError('ciphertext artifact exceeds size bound')
        return raw
    finally: os.close(descriptor)


def _open_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try: return fd, os.fstat(fd)
    except BaseException as primary:
        try: os.close(fd)
        except BaseException as cleanup: primary.add_note('Directory close also failed: '+str(cleanup))
        raise


def _owned(path, fd, identity):
    held = os.fstat(fd); current = path.lstat()
    pair = (identity.st_dev, identity.st_ino)
    if ((held.st_dev,held.st_ino) != pair or (current.st_dev,current.st_ino) != pair
            or not stat.S_ISDIR(current.st_mode)):
        raise RuntimeError('ciphertext directory ownership changed; replacement retained')


def _close(fd):
    primary = sys.exception()
    try: os.close(fd)
    except BaseException as cleanup:
        if primary is None: raise
        primary.add_note('Directory handle close also failed: '+str(cleanup))


def _payload(raw, contract):
    maximum = math.prod(contract['shape'])*4 + 65536
    if len(raw) > maximum:
        raise ValueError('ciphertext container exceeds public size bound')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (len(entries) != len(NPZ_FIELDS)
                or {entry.filename for entry in entries} != {name+'.npy' for name in NPZ_FIELDS}
                or any(entry.compress_type != zipfile.ZIP_STORED for entry in entries)
                or sum(entry.file_size for entry in entries) > maximum):
            raise ValueError('unexpected ciphertext container inventory or compression')
        # Validate NPY headers before np.load can allocate a claimed shape.
        expected = {
            'payload': (np.dtype('uint32'), contract['shape']),
            'moduli': (np.dtype('int64'), (len(contract['moduli']),)),
            'degree_layout': (np.dtype('int64'), (2,)),
            'precision': (np.dtype('int64'), ()), 'nsd': (np.dtype('int64'), ()),
            'scale': (np.dtype('float64'), ()),
        }
        for entry in entries:
            name = entry.filename[:-4]
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                if version not in ((1,0),(2,0)):raise ValueError('unsupported NPY format')
                reader = np.lib.format.read_array_header_1_0 if version == (1,0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream,max_header_size=512)
                wanted_dtype,wanted_shape=expected[name]
                if dtype!=wanted_dtype or shape!=wanted_shape or fortran:
                    raise ValueError('NPY dtype/shape/order differs from role contract')
                if entry.file_size!=stream.tell()+math.prod(shape)*dtype.itemsize:
                    raise ValueError('NPY byte count differs from validated header')
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {name: np.asarray(data[name]) for name in NPZ_FIELDS}
    payload = arrays['payload']
    if payload.dtype != np.dtype('uint32') or payload.shape != contract['shape']:
        raise ValueError('ciphertext payload dtype or shape differs from public contract')
    for name, expected in (('moduli', contract['moduli']), ('degree_layout', contract['layout'])):
        value = arrays[name]
        if value.dtype != np.dtype('int64') or value.ndim != 1 or value.tolist() != expected:
            raise ValueError('ciphertext '+name+' differs from public contract')
    for name, expected in (('precision', 32), ('nsd', 1)):
        value = arrays[name]
        if value.dtype != np.dtype('int64') or value.shape != () or int(value) != expected:
            raise ValueError('ciphertext '+name+' differs from public contract')
    scale = arrays['scale']
    if (scale.dtype != np.dtype('float64') or scale.shape != () or not math.isfinite(float(scale))
            or float(scale) <= 0 or float(scale) != contract['scale']):
        raise ValueError('ciphertext scale differs from public contract')
    # Check before conversion/Polynomial construction; a cast must never hide
    # negative, oversized or non-integral serialized coefficients.
    for limb, modulus in enumerate(contract['moduli']):
        if np.any(payload[..., limb] >= modulus):
            raise ValueError('ciphertext residue is outside its declared modulus')
    return payload, float(scale)


def _manifest_fields(binding):
    return dict(format=FORMAT, **{k:v for k,v in binding.items() if k not in ('shape','moduli','layout')})


def _validate_manifest(manifest, binding):
    expected = _manifest_fields(binding)
    if type(manifest) is not dict or set(manifest) != set(expected)|{'files'}:
        raise ValueError('transport manifest inventory differs')
    if _canonical({k:manifest[k] for k in expected}) != _canonical(expected):
        raise ValueError('ciphertext identity, role, order or actual metadata binding differs')
    files = manifest['files']
    if (type(files) is not dict or set(files) != {'bundle.json','ct_0.npz'}
            or any(type(v) is not str or not SHA256.fullmatch(v) for v in files.values())):
        raise ValueError('ciphertext artifact digest inventory differs')
    return manifest


def request_digest(manifests, contract):
    """Bind all eight ordered, validated tile manifests; payload validation is separate.

    Callers must pass results of validate/save, and retain the original digest on
    the client. A self-declared manifest digest does not authenticate a payload.
    """
    if type(manifests) is not list or len(manifests) != 8:
        raise ValueError('exactly eight ordered request manifests required')
    checked=[]
    for i,manifest in enumerate(manifests):
        binding=ciphertext_contract(contract,'request',i,None,None)
        checked.append(manifest_digest(_validate_manifest(manifest,binding)))
    return manifest_digest(dict(request_contract_sha256=binding['request_contract_sha256'],tiles=checked))


def _bundle(raw):
    bundle=_json(raw)
    if (type(bundle) is not dict or set(bundle) != {'shape','count'}
            or type(bundle['count']) is not int or bundle['count'] != 1
            or type(bundle['shape']) is not dict or set(bundle['shape']) != {'__ct__'}
            or type(bundle['shape']['__ct__']) is not int or bundle['shape']['__ct__'] != 0):
        raise ValueError('one ciphertext without plaintext bookkeeping required')


def validate_ciphertext(directory, *, contract, role, tile_index, expected_scale=None, input_digest=None):
    """Check bound bytes before any Polynomial/JAX construction."""
    binding=ciphertext_contract(contract,role,tile_index,expected_scale,input_digest)
    directory=Path(directory);fd,identity=_open_directory(directory)
    try:
        if set(os.listdir(fd)) != FILES: raise ValueError('exact ciphertext file inventory required')
        manifest_raw=_read_regular('transport.json',MAX_JSON_BYTES,directory_fd=fd)
        manifest=_validate_manifest(_json(manifest_raw),binding)
        if manifest_raw != _canonical(manifest): raise ValueError('canonical manifest required')
        bundle_raw=_read_regular('bundle.json',MAX_JSON_BYTES,directory_fd=fd)
        payload_raw=_read_regular('ct_0.npz',math.prod(binding['shape'])*4+65536,directory_fd=fd)
        for name,raw in (('bundle.json',bundle_raw),('ct_0.npz',payload_raw)):
            if hashlib.sha256(raw).hexdigest() != manifest['files'][name]:
                raise ValueError('ciphertext artifact digest differs')
        _bundle(bundle_raw);payload,scale=_payload(payload_raw,binding)
        _owned(directory,fd,identity)
        return payload,scale,binding,manifest
    finally:_close(fd)


def save_ciphertext(ct, directory, *, contract, role, tile_index, expected_scale=None, input_digest=None):
    """Write one newly owned tile directory, publishing its manifest last."""
    binding=ciphertext_contract(contract,role,tile_index,expected_scale,input_digest)
    from jaxite_word import Polynomial
    import bundle_io
    if not isinstance(ct,Polynomial): raise TypeError('one CROSS Polynomial required')
    ct.validate()
    if (tuple(ct.shape)!=binding['shape'] or list(ct.get_moduli())!=binding['moduli']
            or tuple(ct.degree_layout)!=tuple(binding['layout'])
            or any(type(x) is not int for x in ct.degree_layout)
            or type(ct.precision) is not int or ct.precision!=32
            or type(getattr(ct,'_ckks_nsd',None)) is not int or ct._ckks_nsd!=1):
        raise ValueError('ciphertext role metadata differs')
    scale=getattr(ct,'_ckks_scale',None)
    if type(scale) is not float or not math.isfinite(scale) or scale!=binding['scale']:
        raise ValueError('ciphertext actual scale differs from checked public plan')
    directory=Path(directory);directory.mkdir(mode=0o700,parents=False,exist_ok=False)
    fd,identity=_open_directory(directory)
    try:
        bundle_io.save_bundle(ct,directory)
        _owned(directory,fd,identity)
        if set(os.listdir(fd)) != {'bundle.json','ct_0.npz'}:raise ValueError('writer artifact inventory differs')
        bundle_raw=_read_regular('bundle.json',MAX_JSON_BYTES,directory_fd=fd)
        payload_raw=_read_regular('ct_0.npz',math.prod(binding['shape'])*4+65536,directory_fd=fd)
        _bundle(bundle_raw);_payload(payload_raw,binding)
        manifest=_manifest_fields(binding)|dict(files={
            'bundle.json':hashlib.sha256(bundle_raw).hexdigest(),'ct_0.npz':hashlib.sha256(payload_raw).hexdigest()})
        for name in ('bundle.json','ct_0.npz'):
            part=os.open(name,os.O_RDONLY|os.O_NOFOLLOW,dir_fd=fd)
            try:os.fsync(part)
            finally:os.close(part)
        _owned(directory,fd,identity)
        temporary=os.open('.transport.tmp',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600,dir_fd=fd)
        with os.fdopen(temporary,'wb') as stream:
            stream.write(_canonical(manifest));stream.flush();os.fsync(stream.fileno())
        os.replace('.transport.tmp','transport.json',src_dir_fd=fd,dst_dir_fd=fd);os.fsync(fd)
        validate_ciphertext(directory,contract=contract,role=role,tile_index=tile_index,
                            expected_scale=expected_scale,input_digest=input_digest)
        _owned(directory,fd,identity)
        return manifest
    except BaseException as primary:
        try:
            _owned(directory,fd,identity);shutil.rmtree(directory)
        except FileNotFoundError:
            if os.fstat(fd).st_nlink != 0:primary.add_note('Original tile directory moved; retained')
        except BaseException as cleanup:primary.add_note('Owned transport cleanup also failed: '+str(cleanup))
        raise
    finally:_close(fd)


def load_ciphertext(directory, *, contract, role, tile_index, expected_scale=None, input_digest=None):
    payload,scale,binding,manifest=validate_ciphertext(directory,contract=contract,role=role,tile_index=tile_index,
                                                     expected_scale=expected_scale,input_digest=input_digest)
    import jax.numpy as jnp
    from jaxite_word import Polynomial
    ct=Polynomial.from_array(jnp.asarray(payload,dtype=jnp.uint32),
        dict(batch=1,num_elements=2,degree=131072,num_moduli=len(binding['moduli']),precision=32,degree_layout=tuple(binding['layout'])),
        dict(moduli=binding['moduli']))
    ct._ckks_scale=scale;ct._ckks_nsd=1;ct.validate()
    return ct,manifest
