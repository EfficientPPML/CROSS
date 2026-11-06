"""Pure host placement/readout for the reviewed batch-one Qwen file contract.

No codecs, model operations, numerical tolerances, file I/O or authorization.
Returned arrays are backed by new immutable bytes, never by input storage.
Original position metadata remains separate; physical rows use token order.
"""
import numpy as np

SLOTS = 65536
ROWS = 16
ROW_WIDTH = 4096
HIDDEN = 1024
VOCABULARY = 151936
BLOCKS = 38


def _array(value, shape, label):
    if type(value) is not np.ndarray or value.dtype != np.dtype('float64'):
        raise TypeError(label + ' must be an exact float64 ndarray')
    if value.shape != shape:
        raise ValueError(label + ' shape differs')
    if not np.isfinite(value).all():
        raise ValueError(label + ' must be finite in every physical slot')


def _immutable(value):
    # The immutable bytes owner is independent of all caller arrays. OWNDATA
    # is false for this view, but setflags(write=True) cannot expose mutation.
    return np.frombuffer(value.tobytes(order='C'), dtype=np.float64).reshape(value.shape)


def place_embeddings(embeddings, positions):
    """Map (1,S,1024) into one/two distinct 16x4096 row tiles.

    Positions are the original strictly increasing public positions, not slot
    offsets. No features/rows are repeated, normalized, summed or averaged.
    """
    if (type(positions) is not tuple or not 1 <= len(positions) <= 32 or
            any(type(p) is not int or not 0 <= p < 40960 for p in positions) or
            tuple(sorted(set(positions))) != positions):
        raise ValueError('one to32 increasing original positions required')
    count = len(positions)
    _array(embeddings, (1, count, HIDDEN), 'embeddings')
    tiles = []
    for first in range(0, count, ROWS):
        valid = min(ROWS, count - first)
        tile = np.zeros((ROWS, ROW_WIDTH), dtype=np.float64)
        tile[:valid, :HIDDEN] = embeddings[0, first:first + valid]
        tiles.append(_immutable(tile.reshape(SLOTS)))
    return tuple(tiles)


def read_vocabulary(decoded_blocks, descriptors):
    """Read all ordered row-zero vocabulary slices from decoder slot vectors.

    ``CKKSContext.decrypt_slots`` produces one float64 slot vector. Callers
    must explicitly select their batch-one vector from a lower-level batched
    decoder result. Descriptors are the authenticated file contract converted
    to an exact tuple, with index/start/stop/physical_row/level integer fields.
    This helper checks layout only; source/request/scale authentication stays
    with the caller. Every physical slot must be finite, including padding.
    """
    if type(decoded_blocks) is not tuple or len(decoded_blocks) != BLOCKS:
        raise ValueError('exact ordered38 decoded block tuple required')
    if type(descriptors) is not tuple or len(descriptors) != BLOCKS:
        raise ValueError('exact ordered38 descriptor tuple required')
    # Validate the complete inventory before constructing any output.
    for index, (values, descriptor) in enumerate(zip(decoded_blocks, descriptors)):
        expected = dict(index=index, start=index * ROW_WIDTH,
                        stop=min((index + 1) * ROW_WIDTH, VOCABULARY),
                        physical_row=0, level=2)
        if (type(descriptor) is not dict or set(descriptor) != set(expected) or
                any(type(descriptor[name]) is not int or descriptor[name] != value
                    for name, value in expected.items())):
            raise ValueError('ordered vocabulary descriptor differs')
        _array(values, (SLOTS,), 'decoded vocabulary block')
    result = np.empty((1, VOCABULARY), dtype=np.float64)
    for index, values in enumerate(decoded_blocks):
        start = index * ROW_WIDTH
        stop = min(start + ROW_WIDTH, VOCABULARY)
        result[0, start:stop] = values[:stop - start]
    return _immutable(result)
