"""Fixed independent full-slot round inputs and client-only original-Newton oracles."""
import math
import numpy as np
SLOTS,ROWS,PITCH,KEYS,HEADS=65536,16,4096,128,12

def definitions():
    masks=[]
    for keys,valid,seed in ((128,16,2026110303),(15,9,2026110304)):
        positions=np.arange(128)
        mask=np.array([((positions+17*row+1)%128)<keys for row in range(16)])
        mask[valid:]=False
        masks.append(dict(name=f'K{keys}_rows{valid}',key_count=keys,valid_rows=valid,key_mask=mask,seed=seed))
    return masks

def checkpoint_names(updates):
    return ('masked_input','squared_numerator','denominator',
        *[f'y{i:02}' for i in range(1,updates+1)],*[f'e{i:02}' for i in range(1,updates+1)],'probabilities')

def make_probabilities(item):
    values=np.random.default_rng(item['seed']).uniform(-15.5,15.5,SLOTS)
    for row in range(item['valid_rows']):
        keys=np.flatnonzero(item['key_mask'][row]);K=len(keys)
        for head in range(HEADS):
            raw=1.+.013*np.cos(keys*.217+row*.191+head*.137)
            base=raw/raw.sum()
            if (row+head)%3:
                concentration=.73+.24*(12*row+head+1)/(12*item['valid_rows']+1)
                base*=(1-concentration);base[(7*row+3*head)%K]+=concentration
            values[row*PITCH+keys*16+head]=base
    assert values.shape==(SLOTS,) and np.isfinite(values).all() and np.max(np.abs(values))<16
    return values

def client_oracles(values,item,contract):
    """Direct selected-key dot products; original Newton and independent residual powers."""
    masked=np.zeros(SLOTS);den=np.ones(SLOTS);masses=[];variations=[]
    for row in range(item['valid_rows']):
        keys=np.flatnonzero(item['key_mask'][row])
        if len(keys)!=contract.key_count:raise ValueError('fixture K differs from frozen contract')
        for head in range(HEADS):
            indices=row*PITCH+keys*16+head;broadcast=row*PITCH+np.arange(KEYS)*16+head
            selected=np.asarray(values)[indices];mass=selected.sum()
            if not np.isfinite(selected).all() or selected.min()<0 or abs(mass-1)>1e-12:raise ValueError('fixed input is not a normalized positive probability vector')
            variance=np.dot(selected,selected)
            if not contract.domain[0]<=variance<=contract.domain[1]:raise ValueError('fixed squared-denominator domain violation')
            masked[indices]=selected;den[broadcast]=variance;masses.append(float(mass));variations.append(float(selected.max()-selected.min()))
    result={'masked_input':masked,'squared_numerator':masked*masked,'denominator':den}
    y=den*contract.first_weight+contract.first_bias
    # This is a client diagnostic of the actual rounded first-affine polynomial,
    # evaluated in extended precision; it is not used by the encrypted owner.
    extended_den=den.astype(np.longdouble)
    first_y=extended_den*np.longdouble(contract.first_weight)+np.longdouble(contract.first_bias)
    e1=1-extended_den*first_y
    for update in range(1,contract.updates+1):
        if update>1:y=y*(2-den*y)
        result[f'y{update:02}']=y.copy()
        result[f'e{update:02}']=np.asarray(e1**(2**(update-1)),dtype=np.float64)
    result['probabilities']=result['squared_numerator']*y
    result['true_quotient']=result['squared_numerator']/den
    details=dict(minimum_active_input_mass=min(masses),maximum_active_input_mass=max(masses),
        minimum_active_vector_spread=min(variations),maximum_active_vector_spread=max(variations),
        nearuniform_vectors=sum(v<.01 for v in variations),concentrated_vectors=sum(v>.7 for v in variations),
        denominator_minimum=float(den.min()),denominator_maximum=float(den.max()))
    if not details['nearuniform_vectors'] or not details['concentrated_vectors']:raise ValueError('both concentration regimes required')
    return result,details


def checkpoint_error(decoded,expected):
    """Fixed all-slot gate whose failure record always contains finite JSON values."""
    values=np.asarray(decoded);reference=np.asarray(expected)
    valid=(values.shape==(SLOTS,) and reference.shape==(SLOTS,) and
        values.dtype.kind in 'fiu' and reference.dtype.kind in 'fiu' and
        np.isfinite(values).all() and np.isfinite(reference).all())
    report=dict(passed=False,shape_and_finite=bool(valid),max_abs_error=None,
        error_arithmetic_finite=False,decoded_minimum=None,decoded_maximum=None)
    if not valid:return report
    with np.errstate(over='ignore',invalid='ignore'):
        values=values.astype(np.float64);reference=reference.astype(np.float64)
    if not (np.isfinite(values).all() and np.isfinite(reference).all()):return report
    with np.errstate(over='ignore',invalid='ignore'):
        error=float(np.max(np.abs(values-reference)))
    if math.isfinite(error):report.update(max_abs_error=error,error_arithmetic_finite=True,passed=error<=1e-6)
    report.update(decoded_minimum=float(values.min()),decoded_maximum=float(values.max()))
    return report


def diagnostics(decoded,item,input_mask):
    """Descriptive mass/padding diagnostics, fail closed if a sum overflows."""
    values=np.asarray(decoded);active=np.asarray(input_mask,dtype=bool)
    valid=(values.shape==(SLOTS,) and values.dtype.kind in 'fiu' and np.isfinite(values).all())
    result=dict(shape_and_finite=bool(valid),reduction_arithmetic_finite=False,
        maximum_active_mass_defect=None,minimum_probability=None,maximum_probability=None,
        maximum_padding_magnitude=None,active_head_mass_shape=[item['valid_rows'],HEADS])
    if not valid:return result
    with np.errstate(over='ignore',invalid='ignore'):values=values.astype(np.float64)
    if not np.isfinite(values).all():return result
    padding=values[~active]
    with np.errstate(over='ignore',invalid='ignore'):
        mass=values.reshape(ROWS,PITCH)[:,:KEYS*16].reshape(ROWS,KEYS,16).sum(axis=1)
        live=mass[:item['valid_rows'],:HEADS]
        defect=float(np.max(np.abs(live-1)))
    if math.isfinite(defect):result.update(maximum_active_mass_defect=defect,reduction_arithmetic_finite=True)
    result.update(minimum_probability=float(values[active].min()),maximum_probability=float(values[active].max()),
        maximum_padding_magnitude=float(np.max(np.abs(padding))) if padding.size else 0.)
    return result


def derived_checks(decoded,references,updates):
    """Client-only derived gates; overflow/shape failures stay strict-JSON-safe."""
    zero=np.zeros(SLOTS)
    def evaluate(operation,*arrays):
        values=[np.asarray(x) for x in arrays]
        if any(x.shape!=(SLOTS,) or x.dtype.kind not in 'fiu' or not np.isfinite(x).all() for x in values):
            return np.full(SLOTS,np.nan)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            values=[x.astype(np.float64) for x in values]
            return operation(*values)
    out={};den=decoded['denominator'];reference_den=references['denominator']
    for update in range(1,updates+1):
        y=decoded[f'y{update:02}'];e=decoded[f'e{update:02}']
        observed_residual=evaluate(lambda d,v:1-d*v,den,y)
        out[f'invariant{update:02}']=checkpoint_error(e,observed_residual)
        true_residual=evaluate(lambda d,v:1-d*v,reference_den,y)
        out[f'true_residual_vs_original{update:02}']=checkpoint_error(true_residual,references[f'e{update:02}'])
    final_y=decoded[f'y{updates:02}']
    out['final_true_inverse']=checkpoint_error(final_y,evaluate(lambda d:1/d,reference_den))
    out['final_true_relative_residual']=checkpoint_error(evaluate(lambda d,v:1-d*v,reference_den,final_y),zero)
    out['final_true_normalized_quotient']=checkpoint_error(decoded['probabilities'],references['true_quotient'])
    return out
