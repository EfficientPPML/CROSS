"""Isolated scheduling contracts; execute only after CPU calibration release."""
from pathlib import Path
from types import SimpleNamespace
import gc
import sys
import weakref

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'cadathlon/common')]
import numpy as np
import pytest
from encrypted_packed_transformer import (pending_limit, output_payload_bytes, giant_chunks,
    PackedTransformer as ChunkedPackedTransformer, PackedPairLayout)


@pytest.mark.parametrize('size,groups,budget,maximum,expected', [
    (512*1024**2, 16, 256*1024**2, 8, 1),
    (256*1024**2, 16, 256*1024**2, 8, 2),
    (85*1024**2, 16, 256*1024**2, 8, 4),
    (16*1024**2, 16, 256*1024**2, 8, 8),
    (16*1024**2, 3, 256*1024**2, 8, 3),
    (16, 3, 0, 8, 1), (16, 3, 500, 1, 1), (None, 3, 500, 8, 1),
])
def test_payload_cap(size, groups, budget, maximum, expected):
    count = pending_limit(size, groups, extra_payload_bytes=budget, max_pending=maximum)
    assert count == expected
    if size is not None:
        assert (count-1)*size <= budget


@pytest.mark.parametrize('size,groups,budget,maximum', [
    (0, 3, 5, 2), (True, 3, 5, 2), (1, 0, 5, 2), (1, True, 5, 2),
    (1, 3, -1, 2), (1, 3, True, 2), (1, 3, 5, 0), (1, 3, 5, True),
])
def test_invalid_payload_caps(size, groups, budget, maximum):
    with pytest.raises(ValueError):
        pending_limit(size, groups, extra_payload_bytes=budget, max_pending=maximum)


@pytest.mark.parametrize('batch,level', [(1, 1), (2, 25), (1, 49)])
def test_output_bytes_include_batch_components_degree_remaining_limbs_and_dtype(batch, level):
    value = SimpleNamespace(polynomial=SimpleNamespace(shape=(batch, 2, 256, 512, 2*(level+1)), dtype='uint32'))
    graph = SimpleNamespace(ctx=SimpleNamespace(_param_cache=SimpleNamespace(num_q_at_level=lambda n: 2*(n+1))),
                            arithmetic=SimpleNamespace(_level=lambda _: level))
    assert output_payload_bytes(graph, value) == batch*2*131072*(2*level)*4


@pytest.mark.parametrize('limit', [1, 2, 3, 8])
def test_queue_peak_and_consumed_reference_retirement(limit):
    alive = weakref.WeakSet()
    made, consumed, peak = [], [], 0
    class Token: pass
    def make(giant, indices):
        nonlocal peak
        token = Token()
        token._public_absolute_bound = giant + 1
        alive.add(token)
        peak = max(peak, len(alive))
        made.append(giant)
        return token
    for giant, token in giant_chunks([(n, [n]) for n in range(19)], make, limit):
        assert token._public_absolute_bound == giant + 1
        consumed.append(giant)
        del token
    assert made == consumed == list(range(19))
    assert peak <= limit
    assert not alive


def test_queue_releases_unconsumed_items_on_failure_and_generator_close():
    alive = weakref.WeakSet()
    class Token: pass
    def make(giant, indices):
        if giant == 2:
            raise RuntimeError('inner failure')
        item = Token(); alive.add(item); return item
    with pytest.raises(RuntimeError, match='inner failure'):
        next(giant_chunks([(n, [n]) for n in range(4)], make, 4))
    gc.collect()
    assert not alive
    iterator = giant_chunks([(0, []), (1, [])], make, 2)
    giant, token = next(iterator)
    del token
    iterator.close()
    gc.collect()
    assert not alive
