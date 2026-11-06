"""Independent NumPy-only row-packed attention semantics; no encrypted claim.

No labels, pretrained coefficients or analytic approximations are fitted. Exact
exp/division are only the plaintext oracle for validating the layout schedule.
"""
import hashlib,json,math,time
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
SLOTS=65536;F=4096;ROWS=16;TOKENS=128;HEADS=12;D=64;H=HEADS*D
ROTATIONS=set()

def rotate(values,shift):
    ROTATIONS.add(shift%SLOTS)
    return np.roll(values,-shift)

def pack(values):
    values=np.asarray(values)
    assert values.shape==(TOKENS,H)
    rows=np.zeros((TOKENS,F));rows[:,:H]=values
    return [x.reshape(-1).copy() for x in rows.reshape(8,ROWS,F)]

def unpack(tiles):
    return np.concatenate([x.reshape(ROWS,F)[:,:H] for x in tiles])

def key_row(tile,row):
    selected=rotate(tile,row*F)*np.tile(np.r_[np.ones(F),np.zeros(SLOTS-F)],1)
    for step in (F,2*F,4*F,8*F):selected=selected+rotate(selected,-step)
    return selected

def head_sum(values):
    summed=values.copy()
    for step in (1,2,4,8,16,32):summed=summed+rotate(summed,step)
    feature=np.arange(SLOTS)%F
    summed*=((feature%D)==0)&(feature<H)
    for step in (1,2,4,8,16,32):summed=summed+rotate(summed,-step)
    return summed

def hidden_sum(values):
    feature=np.arange(SLOTS)%F
    summed=values*(feature<H)
    for bit in range(12):summed=summed+rotate(summed,1<<bit)
    summed*=feature==0
    for bit in range(12):summed=summed+rotate(summed,-(1<<bit))
    return summed

def streaming(query,key,value,mask):
    mask=np.asarray(mask,dtype=bool)
    if mask.shape!=(TOKENS,) or not mask.any():raise ValueError('nonempty valid-key mask required')
    qs,ks,vs=map(pack,(query,key,value));valid=np.flatnonzero(mask);out=[]
    active=(np.arange(SLOTS)%F)<H
    for q in qs:
        mean=np.zeros(SLOTS)
        for index in valid:
            mean+=head_sum(q*key_row(ks[index//ROWS],int(index%ROWS)))/math.sqrt(D)
        mean/=len(valid)
        numerator=np.zeros(SLOTS);denominator=np.zeros(SLOTS)
        for index in valid:
            scores=head_sum(q*key_row(ks[index//ROWS],int(index%ROWS)))/math.sqrt(D)
            exponential=np.exp(scores-mean)*active
            numerator+=exponential*key_row(vs[index//ROWS],int(index%ROWS))
            denominator+=exponential
        # Inactive packed features are public padding. A ciphertext circuit
        # would need public in-domain denominator fillers before reciprocal.
        result=np.zeros(SLOTS)
        np.divide(numerator,denominator,out=result,where=active)
        out.append(result)
    return unpack(out)

def dense(query,key,value,mask):
    q=query.reshape(TOKENS,HEADS,D).transpose(1,0,2)
    k=key.reshape(TOKENS,HEADS,D).transpose(1,0,2)
    v=value.reshape(TOKENS,HEADS,D).transpose(1,0,2)
    scores=q@k.swapaxes(-1,-2)/math.sqrt(D)
    scores=np.where(np.asarray(mask)[None,None,:],scores,-np.inf)
    exp=np.exp(scores-scores.max(axis=-1,keepdims=True));weights=exp/exp.sum(axis=-1,keepdims=True)
    return (weights@v).transpose(1,0,2).reshape(TOKENS,H)

def main():
    started=time.monotonic();source=Path(__file__);before=hashlib.sha256(source.read_bytes()).hexdigest();checks=[]
    tile=np.arange(SLOTS,dtype=float).reshape(ROWS,F)
    for row in range(ROWS):
        actual=key_row(tile.reshape(-1),row).reshape(ROWS,F)
        np.testing.assert_array_equal(actual,np.broadcast_to(tile[row],actual.shape))
    checks.append(dict(name='every_local_key_row_broadcast',rows=ROWS,exact=True))
    values=np.arange(SLOTS,dtype=float).reshape(ROWS,F)
    expected=np.zeros_like(values)
    for row in range(ROWS):
        for head in range(HEADS):
            expected[row,head*D:(head+1)*D]=values[row,head*D:(head+1)*D].sum()
    np.testing.assert_array_equal(head_sum(values.reshape(-1)).reshape(ROWS,F),expected)
    checks.append(dict(name='head_local_reduction_with_nonzero_padding_sentinels',heads=HEADS,rows=ROWS,exact=True))
    expected=np.broadcast_to(values[:,:H].sum(axis=-1,keepdims=True),(ROWS,F))
    np.testing.assert_array_equal(hidden_sum(values.reshape(-1)).reshape(ROWS,F),expected)
    checks.append(dict(name='nonpower_hidden768_reduction_without_cross_row_wrap',exact=True))
    rng=np.random.default_rng(20261103)
    query,key,value=(rng.normal(0,.2,(TOKENS,H)) for _ in range(3))
    boundary=np.zeros(TOKENS,bool);boundary[[0,15,16,31,32,63,64,95,96,126,127]]=True
    masks=[('all_keys',np.ones(TOKENS,bool)),('tile_boundaries_and_padding',boundary),('single_last_key',np.arange(TOKENS)==127)]
    for name,mask in masks:
        actual=streaming(query,key,value,mask);expected=dense(query,key,value,mask)
        error=np.max(np.abs(actual-expected));np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)
        checks.append(dict(name=name,valid_keys=int(mask.sum()),all_query_max_abs_error=float(error),
            padded_query_max_abs_error=float(np.max(np.abs(actual[~mask]-expected[~mask]))) if (~mask).any() else None,
            padded_query_outputs_nonzero=bool(np.any(actual[~mask])) if (~mask).any() else None))
    try:streaming(query,key,value,np.zeros(TOKENS,bool))
    except ValueError:checks.append(dict(name='fully_masked_rejected',passed=True))
    else:raise AssertionError('fully masked keys accepted')
    report=dict(passed=True,scope=__doc__,slots=SLOTS,features=F,rows_per_tile=ROWS,tiles=8,tokens=TOKENS,
        heads=HEADS,head_dimension=D,checks=checks,observed_rotation_indices=sorted(ROTATIONS-{0}),
        source_sha256=before,source_unchanged=before==hashlib.sha256(source.read_bytes()).hexdigest(),seconds=time.monotonic()-started,
        limitations=['Exact exp/division are plaintext comparison operations, not an encrypted implementation.',
            'This tests direct-index token broadcasting; binary-composed alignment is inventoried separately.',
            'No pretrained block, polynomial bounds/depth, CKKS error or memory measurement is validated.'])
    assert report['source_unchanged']
    (OUT/'layout_checks.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
