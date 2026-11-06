"""Public bootstrap contracts plus an opt-in real secure refresh/continuation."""
import math
import os
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
from encrypted_refresh import BoundedSecureRefresh, ClientRotationKeyStore, RefreshProfile, refresh_plan


@pytest.mark.parametrize('profile', [RefreshProfile(), RefreshProfile(65536, 22),
                                     RefreshProfile(131072, 49, dnum=4)])
def test_refresh_profile_actual_ring_is_secure(profile):
    from jaxite_word import he_params
    params = profile.parameters()
    q, p = params['q_towers'], params['p_towers']
    target = he_params.SecurityTarget(128, 'classical')
    assert he_params.qp_is_secure(q, p, profile.degree, target)
    assert math.log2(math.prod(q+p)) <= he_params.security_ceiling(profile.degree, target)
    assert he_params.p_covers_q(q, p, profile.dnum)
    assert len(q) == 2 * (profile.depth + 1)
    assert params['num_slots'] == profile.degree // 2


def test_insecure_actual_refresh_chain_is_rejected():
    with pytest.raises(ValueError, match='128-bit security'):
        RefreshProfile(32768, 22).parameters()


def test_lower_key_memory_profile_preserves_usable_refresh_capacity():
    plan = refresh_plan(131072, 1024, (4, 4), max_level=49)
    assert plan['output_level'] == 25
    assert len(plan['rotations']) < 90


@pytest.mark.parametrize('period', [1 << exponent for exponent in range(11)])
def test_sparse_plan_includes_fold_reconstruction_and_conjugation_keys(period):
    plan = refresh_plan(131072, period, (2, 2), max_level=52)
    slots = max(4, period)
    assert plan['output_level'] == 32
    assert 2 * 131072 - 1 in plan['rotations']
    assert slots in plan['rotations']
    assert 131072 // 4 in plan['rotations']
    assert plan['bootstrap_slots'] == slots
    # Bounded refresh projects onto the public periodic layout. Every such
    # rotation must already have an offline key; the keyless server cannot
    # repair a missing key during evaluation.
    shift = period
    while shift < 131072 // 2:
        assert shift in plan['rotations']
        shift *= 2


@pytest.mark.parametrize('bound', [0., -1., float('inf'), float('nan'), True, '16'])
def test_bounded_refresh_rejects_invalid_public_domain_before_setup(bound):
    with pytest.raises((TypeError, ValueError), match='public magnitude bound'):
        BoundedSecureRefresh(None, period=4, public_bound=bound)


@pytest.mark.parametrize('precision', [0, 63, True, 1.5])
def test_meta_precision_rejected_before_context_setup(precision):
    with pytest.raises(ValueError, match='meta_precision'):
        BoundedSecureRefresh(None, period=4, meta_precision=precision)


@pytest.mark.parametrize('period,budget,depth', [(3, (2, 2), 52), (4, (1, 1), 52),
                                                (4, (3, 3), 52), (4, (2, 2), 19)])
def test_invalid_or_exhausted_refresh_plan_rejected(period, budget, depth):
    with pytest.raises(ValueError):
        refresh_plan(131072, period, budget, max_level=depth)


def test_client_public_key_store_uses_readonly_arrays_and_cleans_up(tmp_path):
    import numpy as np
    from jaxite_word import key_gen
    q, p, degree = [12289], [40961], 2048
    secret = key_gen.gen_pke_pair(q, [], degree)['secret_key']
    with ClientRotationKeyStore.generate(secret, q, p, degree=degree, dnum=1,
                                        rotations=[1, 2], directory=tmp_path) as store:
        assert set(store.keys) == {1, 2}
        assert not any('secret' in name for name in vars(store))
        for values in store.keys.values():
            for array in values.values():
                assert isinstance(array, np.memmap)
                assert array.shape == (1, 2, degree)
                assert array.dtype == np.uint32
                assert not array.flags.writeable
        assert len(list(Path(store._directory.name).glob('*.npy'))) == 4
    assert not list(tmp_path.iterdir())
    assert store.closed and not store.keys
    store.close()


def test_public_key_store_partial_failure_removes_files(monkeypatch, tmp_path):
    from jaxite_word import key_gen
    import numpy as np

    def key_with_wrong_shape(*args, **kwargs):
        index = args[3]
        return {index: {'a': np.zeros((1, 2, 2048)), 'b': np.zeros((1,))}}

    monkeypatch.setattr(key_gen, 'gen_rotation_key', key_with_wrong_shape)
    with pytest.raises(ValueError, match='shape'):
        ClientRotationKeyStore.generate(None, [12289], [40961], degree=2048,
                                        dnum=1, rotations=[1], directory=tmp_path)
    assert not list(tmp_path.iterdir())


def test_public_key_store_space_check_precedes_generation(monkeypatch, tmp_path):
    from types import SimpleNamespace
    import encrypted_refresh
    from jaxite_word import key_gen

    monkeypatch.setattr(encrypted_refresh.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    monkeypatch.setattr(key_gen, 'gen_rotation_key', lambda *a, **k: pytest.fail('must not generate'))
    with pytest.raises(ValueError, match='bytes'):
        ClientRotationKeyStore.generate(None, [12289], [40961], degree=2048,
                                        dnum=1, rotations=[1], directory=tmp_path)
    assert not list(tmp_path.iterdir())


@pytest.mark.slow
@pytest.mark.secure_profile
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('CROSS_TEST_SECURE_REFRESH') != '1', reason='opt-in large secure bootstrap')
def test_actual_secure_keyless_refresh_and_continuation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
    from check_secure_refresh import run
    report = run()
    assert report['passed'], report
    assert report['actual_output_level'] >= report['plan']['output_level']
    assert report['server_has_secret_key'] is False
    assert report['runtime_key_generation_forbidden'] is True


@pytest.mark.slow
@pytest.mark.secure_profile
@pytest.mark.integration
@pytest.mark.skipif(os.environ.get('CROSS_TEST_BOUNDED_REFRESH') != '1', reason='opt-in deep-profile bounded bootstrap')
def test_actual_deep_profile_bounded_refresh_and_continuation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'validation'))
    from check_secure_refresh import run
    report = run(degree=131072, depth=49, dnum=4, period=128, level_budget=(4, 4),
                 compiled=True, amplitude=14.2, public_bound=16., meta_precision=10,
                 disk_keys=True)
    assert report['passed'], report
    assert report['actual_output_level'] >= 25
    assert report['refresh_max_abs_error'] < 1e-3
    assert report['max_abs_error'] < 1e-3
    assert report['server_has_secret_key'] is False
    assert report['runtime_key_generation_forbidden'] is True
