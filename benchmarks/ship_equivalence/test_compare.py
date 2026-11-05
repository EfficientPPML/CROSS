"""Ensure an unmatched or inaccurate run cannot become a performance table."""
import copy
import pytest
from compare import compare


def reports():
    manifest = dict(degree=32,q=[101,97],p=[89,83,79],scale=97.,dnum=2,
                    seed=42,magnitude=.125,error_gate=2**-10,weight=5,theta=4,gamma=8)
    template = dict(passed=True,stage='complete',manifest=manifest,
        backend='tpu',device_kind='test TPU',jax_version='test',numpy_version='test',
        scope='test full bootstrap',output_q=[101,97],output_scale=97.,output_level=1,
        output_nsd=1,slots=16,retained_multiplication_levels=1,
        errors=dict(ramp=.0001,random=.0002),post_square_errors=dict(ramp=.0001,random=.0002),
        jaxlib_version='test',python_version='test',host='same-host',visible_chips='0',
        harness_sha256='same-harness',libtpu_version='test',xla_flags='',stack_limit_bytes=65536,
        persistent_cache_enabled=False,cross_source=dict(commit='same',patch_sha256='same'),
        warm_wall_ms=[5.,6.,7.],device_profiles=[dict(device_active_ms=x) for x in (1.,2.,3.)])
    return [dict(copy.deepcopy(template),variant=v) for v in
            ('cross-native','cross-ship','jaxite-patched','jaxite-optimized')]


def test_complete_matched_comparison():
    result = compare(reports())
    assert len(result) == 4
    assert all(r['warm_wall_median_ms'] == 6. and r['device_active_median_ms'] == 2.
               for r in result)


@pytest.mark.parametrize('failure', ['q','scale','chip','source','error','samples','incomplete'])
def test_reject_incomparable_runs(failure):
    data=reports()
    bad=data[-1]
    if failure == 'q': bad['manifest']['q'][0]=103
    elif failure == 'scale': bad['output_scale']=98.
    elif failure == 'chip': bad['visible_chips']='1'
    elif failure == 'source': bad['cross_source']['commit']='different'
    elif failure == 'error': bad['post_square_errors']['ramp']=.01
    elif failure == 'samples': bad['warm_wall_ms']=[]
    else: bad['stage']='setup'
    with pytest.raises(ValueError):
        compare(data)
