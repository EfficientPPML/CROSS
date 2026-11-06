"""Mock-only review of lazy formatted-key readiness on failed server stages."""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from test_key_contexts import mocked, case, server
import key_contexts as loaders

def test_lazy_formatted_transfer_failure_cleanup_observation(case, monkeypatch):
    events=[];original=loaders._map_array
    def mapped(*args):
        value,state=original(*args);backing=value._mmap
        value._mmap=SimpleNamespace(close=lambda:(events.append('map-close'),backing.close()))
        return value,state
    monkeypatch.setattr(loaders,'_map_array',mapped)
    class Pending:
        def block_until_ready(self):events.append('lazy-key-ready')
    with pytest.raises(RuntimeError,match='stage failure after lazy key conversion'):
        with server(case) as handle:
            # Matches get_rot_key's ownership location before rotation's
            #successful result barrier and formatted-key eviction.
            handle.ctx._param_cache._formatted_rotation_keys={1:(Pending(),Pending())}
            raise RuntimeError('stage failure after lazy key conversion')
    result={'scope':'Mock lifecycle observation only; no JAX/HE execution or demonstrated DMA corruption.',
            'lazy_key_barrier_called':'lazy-key-ready' in events,
            'mapped_backings_closed':events.count('map-close'),
            'events':events}
    Path('/tmp/review_lazy_key_cleanup.json').write_text(json.dumps(result,indent=2)+'\n')
    assert not result['lazy_key_barrier_called'] and result['mapped_backings_closed']>0
