"""Independent mocked review: lazy keys must remain backed after stage failure."""
from types import SimpleNamespace
import json
from pathlib import Path
import pytest
from test_key_contexts import mocked, case, server
import key_contexts as loaders

@pytest.mark.parametrize('discarded',[False,True])
def test_lazy_formatted_failure_retains_backing_and_blocks_reuse(case,monkeypatch,discarded):
    events=[];original=loaders._map_array;quarantine=[];backings=[]
    monkeypatch.setattr(loaders,'_FAILURE_QUARANTINE',quarantine)
    def mapped(*args):
        value,state=original(*args);backing=value._mmap;backings.append(backing)
        value._mmap=SimpleNamespace(close=lambda:(events.append('map-close'),backing.close()))
        return value,state
    monkeypatch.setattr(loaders,'_map_array',mapped)
    class Pending:
        def block_until_ready(self):events.append('lazy-key-ready')
    try:
        with pytest.raises(RuntimeError,match='stage failure after lazy key conversion') as caught:
            with server(case) as handle:
                handle.ctx._param_cache._formatted_rotation_keys={1:(Pending(),Pending())}
                if discarded:handle.ctx._param_cache._formatted_rotation_keys.clear()
                raise RuntimeError('stage failure after lazy key conversion')
        assert events.count('lazy-key-ready')==(0 if discarded else 2)
        assert 'map-close' not in events and len(quarantine)==1
        assert all(not b.closed for b in backings)
        assert any('cleanup incomplete' in note for note in caught.value.__notes__)
        with pytest.raises(RuntimeError,match='must exit'):
            with server(case):pass
        report={'scope':'Mock lifecycle only; no JAX/HE or DMA-corruption claim.',
                'discarded_controls':discarded,'reachable_barriers':events.count('lazy-key-ready'),
                'all_backings_retained':all(not b.closed for b in backings),'reuse_rejected':True,
                'primary_preserved':str(caught.value)=='stage failure after lazy key conversion'}
        Path(__file__).with_name('discarded.json' if discarded else 'reachable.json').write_text(json.dumps(report,indent=2)+'\n')
    finally:
        # These are tiny synchronous mocks. Production exposes no reclaim API.
        for backing in backings:backing.close()
        quarantine.clear()
