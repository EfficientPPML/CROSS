"""Exact reviewed default125 _payload body; pure NumPy container checks only."""
import io,math,zipfile
import numpy as np
NPZ_FIELDS = {'payload', 'moduli', 'degree_layout', 'precision', 'scale', 'nsd'}

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
