"""Public NumPy semantics only: no key generation, encryption, or HE timing."""
from pathlib import Path
import hashlib, importlib.util, json, tarfile, time
import numpy as np

ROOT = Path('/home/jianming_gatech/CROSS')
LAB = Path(__file__).resolve().parent
SLOTS, ROWS, PITCH, KEYS, HEADS, PAD = 65536, 16, 4096, 128, 12, 16
SHIFTS = tuple(16*(1 << b) for b in range(7))


def canonical_mask(keys, valid_rows=16):
    keys = np.asarray(keys)
    if keys.dtype != np.bool_ or keys.shape not in ((KEYS,), (ROWS, KEYS)):
        raise ValueError('public key mask must be bool[128] or bool[16,128]')
    if type(valid_rows) is not int or not 1 <= valid_rows <= ROWS:
        raise ValueError('valid_rows outside1..16')
    keys = np.broadcast_to(keys, (ROWS, KEYS)).copy()
    keys[valid_rows:] = False
    count = keys.sum(axis=1)
    if np.any(count[:valid_rows] == 0):
        raise ValueError('each requested query must have at least one valid key')
    out = np.zeros((ROWS, PITCH))
    out[:, :KEYS*PAD] = np.repeat(keys[:, :, None], PAD, axis=2).reshape(ROWS, -1)
    out[:, :KEYS*PAD].reshape(ROWS, KEYS, PAD)[:, :, HEADS:] = 0.
    return out.reshape(-1), count


def broadcast_field(reduced, *, valid_rows, dummy=False):
    anchors = np.zeros((ROWS, PITCH))
    anchors[:valid_rows, :HEADS] = 1.
    bias = np.zeros_like(anchors)
    if dummy:
        # These anchors expand within their own2048-slot half. The public bias
        # can be folded into weighted_sum's existing anchor-mask/rescale stage.
        bias[:, :PAD] = 1.
        bias[:valid_rows, :HEADS] = 0.
        bias[:, KEYS*PAD:KEYS*PAD+PAD] = 1.
    out = reduced*anchors.reshape(-1) + bias.reshape(-1)
    for shift in SHIFTS:
        out = out + np.roll(out, shift)
    return out


def reduce_field(value, keys, *, valid_rows=16, mean=False, square=False,
                 input_invariant=False, dummy=False):
    value = np.asarray(value, dtype=np.float64)
    if value.shape != (SLOTS,) or not np.all(np.isfinite(value)):
        raise ValueError('finite full65536-slot input required')
    mask, counts = canonical_mask(keys, valid_rows)
    if mean and square:
        raise ValueError('this probe distinguishes mean and squared-sum operators')
    if mean and input_invariant:
        raise ValueError('mean path always supplies explicit key/feature weights')
    maxima = []
    result = value.copy()
    pt, ct = 0, 0
    if not input_invariant:
        weight = mask.copy()
        if mean:
            weight = (weight.reshape(ROWS, PITCH) /
                      np.maximum(counts, 1)[:, None]).reshape(-1)
        result = result*weight
        pt += 1
        maxima.append(float(np.max(np.abs(result))))
    if square:
        result = result*result
        ct += 1
        maxima.append(float(np.max(np.abs(result))))
    for shift in SHIFTS:
        result = result + np.roll(result, -shift)
        maxima.append(float(np.max(np.abs(result))))
    result = broadcast_field(result, valid_rows=valid_rows, dummy=dummy)
    pt += 1
    return result, {'rotations': 14, 'rotation_additions': 14,
                    'public_multiplications': pt, 'ciphertext_multiplications': ct,
                    'path_levels': pt+ct, 'public_bias_additions': int(dummy),
                    'maximum_reduction_intermediate': max(maxima)}


def oracle(value, keys, *, valid_rows=16, mean=False, square=False, dummy=False):
    # Independent dense gather: no rotate/add implementation reuse.
    key_matrix = np.broadcast_to(np.asarray(keys), (ROWS, KEYS))
    out = np.ones(SLOTS) if dummy else np.zeros(SLOTS)
    for row in range(valid_rows):
        chosen = np.flatnonzero(key_matrix[row])
        for head in range(HEADS):
            data = value[row*PITCH+chosen*PAD+head]
            expected = np.sum(data*data if square else data)
            if mean:
                expected /= len(chosen)
            out[row*PITCH+np.arange(KEYS)*PAD+head] = expected
    return out


def run():
    started = time.perf_counter()
    observed = [Path(__file__).resolve(),
        ROOT/'cadathlon/validation/bert_score_only_cpu_20261103/prototype/score_layout.py',
        ROOT/'cadathlon/validation/bert_factored_scores_cpu_20261103/score_layout.py',
        Path('/tmp/bert-qk-file-component-20261103/checkout/cadathlon/common/encrypted_score_routing.py')]
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    before = {str(p): digest(p) for p in observed}
    cases = []
    rng = np.random.default_rng(20261103)
    masks = {'all128': np.ones(KEYS, bool), 'first15': np.arange(KEYS)<15,
        'interleaved': np.arange(KEYS)%3 == 0, 'single_last127': np.arange(KEYS)==127,
        'boundary_keys': np.isin(np.arange(KEYS), [0,15,16,63,64,111,112,127]),
        'per_row': np.asarray([np.arange(KEYS)%(r+2)==r%(r+2) for r in range(ROWS)])}
    # Every physical row is distinct; the upper half and dummy heads deliberately
    # contain large sentinels. No32768-period duplication is used.
    values = rng.uniform(-8., 8., SLOTS)
    shaped = values.reshape(ROWS, PITCH)
    shaped[:, KEYS*PAD:] = 1e6 + np.arange(ROWS)[:,None]
    shaped[:, :KEYS*PAD].reshape(ROWS,KEYS,PAD)[:,:,HEADS:] = -2e6
    assert not np.array_equal(values[:SLOTS//2], values[SLOTS//2:])
    for mask_name, keys in masks.items():
        for valid_rows in (1, 7, 16):
            for operation in ('masked_mean', 'hostile_squared_sum', 'invariant_squared_sum'):
                mean = operation == 'masked_mean'
                square = not mean
                invariant = operation == 'invariant_squared_sum'
                mask, _ = canonical_mask(keys, valid_rows)
                if invariant:
                    # Public-generated positive probabilities for this layout
                    # diagnostic only, independently normalized per row/head.
                    probability = np.zeros(SLOTS)
                    km = np.broadcast_to(keys,(ROWS,KEYS))
                    for r in range(valid_rows):
                        for h in range(HEADS):
                            idx=r*PITCH+np.flatnonzero(km[r])*PAD+h
                            v=rng.uniform(.1,1.,len(idx))
                            probability[idx]=v/v.sum()
                    source=probability
                else:
                    source=values
                output, counts=reduce_field(source,keys,valid_rows=valid_rows,
                    mean=mean,square=square,input_invariant=invariant,dummy=square)
                expected=oracle(source,keys,valid_rows=valid_rows,
                    mean=mean,square=square,dummy=square)
                error=float(np.max(np.abs(output-expected)))
                assert error <= 1e-11, (mask_name,valid_rows,operation,error)
                if invariant:
                    assert counts['maximum_reduction_intermediate'] <= 1.+1e-14
                if mean:
                    assert counts['maximum_reduction_intermediate'] <= 8.+1e-14
                # Every inactive denominator slot equals1, ready for a domain
                # which contains1. An unfilled zero field is unsafe for inverse.
                active=np.zeros((ROWS,PITCH),bool)
                active[:valid_rows,:KEYS*PAD].reshape(valid_rows,KEYS,PAD)[:,:,:HEADS]=True
                if square:
                    assert np.all(output[~active.reshape(-1)] == 1.)
                else:
                    assert np.all(output[~active.reshape(-1)] == 0.)
                cases.append(dict(mask=mask_name,valid_rows=valid_rows,operation=operation,
                                  maximum_absolute_error=error,passed=True,**counts))
    # Impulses prove the extreme rows, key boundaries and last real head stay
    # in exactly their own row/head. Allkeys is essential for key127 coverage.
    for row,key,head in ((0,0,0),(0,127,11),(7,63,5),(8,64,6),(15,127,11)):
        impulse=np.zeros(SLOTS);impulse[row*PITCH+key*PAD+head]=1.
        result,_=reduce_field(impulse,masks['all128'])
        expected=np.zeros(SLOTS);expected[row*PITCH+np.arange(KEYS)*PAD+head]=1.
        assert np.array_equal(result,expected)
        cases.append(dict(operation='boundary_impulse',row=row,key=key,head=head,passed=True))
    # Demonstrate why the OLD replica mask is not the current physical mask.
    old_path=observed[1]
    spec=importlib.util.spec_from_file_location('historical_score_layout',old_path)
    old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    current_mask,_=canonical_mask(masks['first15'])
    old_mask=old.public_mask(masks['first15'],0,128)
    mismatch=int(np.sum(old_mask != current_mask.astype(bool)))
    assert mismatch > 0
    old_result=values.reshape(2,16,128,16)*old_mask.reshape(2,16,128,16)
    new_result=values*current_mask
    assert not np.array_equal(old_result.reshape(-1),new_result)
    failed_duplicate=False
    try: old.duplicate_period(values)
    except AssertionError: failed_duplicate=True
    assert failed_duplicate
    # A reshape alone does not change contiguous2048-slot chunk boundaries.
    # With the CORRECT factored mask, the old view's sum equals all32 chunks,
    # half of which are deliberately inactive. It is still only a simulator.
    chunks=(values*current_mask).reshape(2,16,128,16).sum(axis=2)
    proper=(values*current_mask).reshape(16,2,128,16).sum(axis=2)
    assert np.array_equal(chunks.reshape(32,16),proper.reshape(32,16))
    after={str(p):digest(p) for p in observed}
    unchanged=before==after
    report={'scope':'public NumPy rotation/mask layout proof only; no ciphertext/key/refresh/model claim',
        'layout':dict(slots=SLOTS,rows=ROWS,row_pitch=PITCH,keys=KEYS,heads=HEADS,
                      padded_heads=PAD,period=SLOTS,aliases=1),
        'cases':cases,'case_count':len(cases),'source_files_unchanged':unchanged,
        'source_sha256_before':before,'source_sha256_after':after,
        'rotations_positive_reduce':list(SHIFTS),
        'rotations_negative_broadcast':[(-s)%SLOTS for s in SHIFTS],
        'historical_original_mask_mismatched_slots':mismatch,
        'historical_original_duplicate_period_rejected_current_nonaliased_rows':failed_duplicate,
        'historical_reshape_alone_preserves2048_slot_chunk_partition':True,
        'dummy_fill':'public bias at active half-block anchors folded into existing PTCT+rescale anchor stage',
        'conditioning_caveat':'Probability partial-sum bounds are ideal normalized-value contracts; no CKKS noise margin or inverse-domain proof.',
        'passed':all(x['passed'] for x in cases) and unchanged,
        'elapsed_s':time.perf_counter()-started}
    (LAB/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    with tarfile.open(LAB/'sources.tar.gz','w:gz') as tar:
        for i,p in enumerate(observed): tar.add(p,arcname=f'{i}_{p.name}')
    with tarfile.open(LAB/'sources.tar.gz') as tar:
        assert [hashlib.sha256(tar.extractfile(x).read()).hexdigest() for x in tar.getmembers()] == list(before.values())
    print(json.dumps({'passed':report['passed'],'cases':len(cases),
        'maximum_absolute_error':max(c.get('maximum_absolute_error',0) for c in cases),
        'source_files_unchanged':unchanged,'elapsed_s':report['elapsed_s'],
        'historical_mask_mismatched_slots':mismatch}))
    return 0 if report['passed'] else 1

if __name__=='__main__': raise SystemExit(run())
