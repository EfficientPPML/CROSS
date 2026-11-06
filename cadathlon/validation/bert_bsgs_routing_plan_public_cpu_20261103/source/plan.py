"""Public NumPy algebra only: no encrypted owner, backend or payload reads."""
import numpy as np
from encrypted_factored_score_routing import FactoredScoreLayout, SLOTS

def rotate(x, shift):
    return np.roll(x, -shift)

def stage(x, diagonals, *, stride, terms):
    """Output-coordinate diagonals: sum_g R_g(sum_b R_-g(d_g+b) R_b(x))."""
    masks = {shift % SLOTS: mask for shift, mask in diagonals}
    expected = {(stride*k) % SLOTS for k in range(terms)}
    if set(masks) != expected or len(masks) != terms:
        raise ValueError('exact public arithmetic progression required')
    babies = [rotate(x, stride*b) if b else x for b in range(4)]
    output = np.zeros_like(x)
    for first in range(0, terms, 4):
        giant = stride*first
        partial = np.zeros_like(x)
        for baby, value in enumerate(babies):
            partial += rotate(masks[(giant+stride*baby) % SLOTS], -giant)*value
        output += rotate(partial, giant) if giant else partial
    return output

def evaluate(x, tile, offset, *, inverse=False, gain=1.):
    if type(x) is not np.ndarray or x.dtype != np.dtype('float64') or x.shape != (SLOTS,) or not np.isfinite(x).all():
        raise ValueError('finite public float64 full-slot vector required')
    if type(inverse) is not bool or type(gain) is not float or gain not in (1., .125) or inverse and gain != 1.:
        raise ValueError('exact public routing direction/gain required')
    layout = FactoredScoreLayout(tile, offset)
    if inverse:
        local = rotate(x, layout.tile_shift) if layout.tile_shift else x
        heads = stage(local, layout.key_diagonals(inverse=True), stride=16, terms=16)
        return stage(heads, layout.head_diagonals(inverse=True), stride=-63, terms=12)
    heads = stage(x, layout.head_diagonals(gain=gain), stride=63, terms=12)
    local = stage(heads, layout.key_diagonals(), stride=-16, terms=16)
    return rotate(local, -layout.tile_shift) if layout.tile_shift else local

def key_inventory():
    steps = {s*stride*k % SLOTS for s in (-1,1) for stride,indices in ((63,(1,2,3,4,8)),(16,(1,2,3,4,8,12))) for k in indices}
    steps |= {s*256*k % SLOTS for s in (-1,1) for k in range(1,8)}
    original = set().union(*(set(FactoredScoreLayout(t,o).rotation_indices()) for t in range(8) for o in range(16)))
    assert steps <= original
    return dict(original=sorted(original), proposed=sorted(steps), subset=True)

def public_plan():
    maps = 12*8*128*2
    return dict(scope='Public slot algebra and operation counts only; no HE, speed or error claim.',
        per_map=dict(original_internal_rotations=26, proposed_internal_rotations=11,
                     plaintext_terms_before=28, plaintext_terms_after=28,
                     ciphertext_additions_before=26, ciphertext_additions_after=26,
                     rescale_calls_before=2, rescale_calls_after=7,
                     critical_rescale_depth_before=2, critical_rescale_depth_after=2),
        full_k128_routing_maps=maps, saved_internal_rotations=maps*15,
        extra_rescale_calls=maps*5, old_total_rotations_excluding_bootstrap=906829,
        proposed_total_rotations_excluding_bootstrap=906829-maps*15,
        key_inventory=key_inventory(), unchanged_outer_tile_rotation=True,
        requirement='Fresh actual HE numerical and performance proof required because operation/key-switch/rescale order changes.')
