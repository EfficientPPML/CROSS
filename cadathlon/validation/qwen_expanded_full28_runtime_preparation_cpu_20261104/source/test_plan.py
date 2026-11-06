import json,pytest
from plan import derive
@pytest.fixture(scope='module')
def report():return derive()
@pytest.mark.parametrize('layer',range(28))
def test_all_original_tail_rules(report,layer):
    rows=[r for r in report['examples'] if r['layer']==layer];assert len(rows)==6;meta=report['layers'][layer]
    for r in rows:
        s={x['name']:x for x in r['stages']}
        assert [s[n]['level'] for n in ['raw_post','normalized_post','gate','up','silu','gated','down','final_residual']]==[25,25,24,24,6,5,4,4]
        assert s['o']['level']==r['attention_level']-1
        assert s['original_residual_mask']['level']==24
        assert s['attention_residual']['scale']==s['original_residual_mask']['scale']
        assert s['final_residual']['scale']==s['attention_residual']['scale']
        assert s['raw_post']['public_bound']==meta['raw_post_bound'] and s['normalized_post']['public_bound']==meta['normalized_post_bound']
    assert meta['matrix_f64_bytes']==88*1024**2

def test_examples_not_scale_relabel(report):
    assert len(report['examples'])==168 and sum(r['silu_events'] for r in report['layers'])==145
    assert len(json.loads(json.dumps(report,allow_nan=False))['examples'])==168
    stages={x['name']:x for x in next(r for r in report['examples'] if r['layer']==3)['stages']}
    assert stages['silu']['scale']!=stages['up']['scale']
