"""Private two-logit comparison; public return contains aggregates only."""
import math
import numpy as np

def vector(value):
    a=np.asarray(value)
    if a.shape!=(2,) or a.dtype.kind not in 'fiu' or not np.isfinite(a).all():
        raise ValueError('exact two finite real logits required')
    with np.errstate(over='ignore',invalid='ignore'):a=a.astype(np.float64)
    if not np.isfinite(a).all():raise ValueError('finite binary64 logits required')
    return a

def compare(decoded,reference):
    actual=vector(decoded);official=vector(reference['native_fp32_logits']);approx=vector(reference['approximate_fp64_logits'])
    with np.errstate(over='ignore',invalid='ignore'):
        native_error=np.abs(actual-official);approx_error=np.abs(actual-approx)
    finite=bool(np.isfinite(native_error).all() and np.isfinite(approx_error).all())
    same=bool((actual[1]>=actual[0])==(official[1]>=official[0]))
    if not finite:
        return dict(passed=False,original_P2_logit_gate=False,prediction_agreement=same,finite_differences=False,
            he_vs_official_max_abs=None,he_vs_frozen_approx_max_abs=None,he_vs_frozen_approx_nrmse=None)
    # RMS normalizer is diagnostic only; it never replaces the published gate.
    error_rms=math.hypot(*map(float,approx_error))/math.sqrt(2.)
    reference_rms=math.hypot(*map(float,approx))/math.sqrt(2.)
    nrmse=error_rms/max(reference_rms,1e-12)
    gate=bool(np.all(native_error<=.02+.01*np.abs(official)))
    return dict(passed=gate and same,original_P2_logit_gate=gate,prediction_agreement=same,finite_differences=True,
        he_vs_official_max_abs=float(native_error.max()),he_vs_frozen_approx_max_abs=float(approx_error.max()),
        he_vs_frozen_approx_nrmse=nrmse if math.isfinite(nrmse) else None,
        nrmse_definition='RMS(HE-frozen_approx)/max(RMS(frozen_approx),1e-12); diagnostic only')
