import copy
import pytest
import ranked_readback as r


def fixture(count,correct=80,native=82):
    # All truths1, with specified native/decrypted correct counts.
    refs=[dict(target_idx=3*i,label=1,native_fp32_logits=[0.,1.] if i<native else [1.,0.]) for i in range(count)]
    rows=[dict(target_idx=3*i,he_logits=[0.,1.] if i<correct else [1.,0.]) for i in range(count)]
    return rows,refs


@pytest.mark.parametrize('n,native,minimum',[(100,82,80),(100,79,77),(100,84,82),(200,162,158),(200,172,168),(200,166,162),(408,345,337)])
def test_integer_ranked_boundaries(n,native,minimum):
    rows,refs=fixture(n,minimum,native);pred=[r.label(x['he_logits']) for x in rows]
    report=r.evaluate_records(rows,refs,count=n,predictions=pred)
    assert report['final_ranked_acceptance'] and report['prediction_agreement']<1.
    assert report['per_logit_tolerance_required_for_ranked_acceptance']is False
    rows[minimum-1]['he_logits']=[1.,0.]
    assert not r.evaluate_records(rows,refs,count=n)['ranked_labels_passed']


def test_ties_and_postprocess_are_required_but_decryption_receipt_is_not_accuracy():
    rows,refs=fixture(100,100,100);rows[0]['he_logits']=[7.,7.]
    assert r.label(rows[0]['he_logits'])==1
    report=r.evaluate_records(rows,refs,count=100)
    assert report['ranked_labels_passed'] and not report['final_ranked_acceptance']
    with pytest.raises(ValueError):r.evaluate_records(rows,refs,count=100,predictions=[0]+[1]*99)


@pytest.mark.parametrize('fault',['missing','extra','duplicate','swap','bool_id','float_id','nan','inf','complex','wrong_shape','bool_logit','huge_int'])
def test_last_or_ordered_private_failure(fault):
    rows,refs=fixture(408,345,345)
    if fault=='missing':rows.pop()
    elif fault=='extra':rows.append(copy.deepcopy(rows[0]))
    elif fault=='duplicate':rows[-1]=copy.deepcopy(rows[0])
    elif fault=='swap':rows[0],rows[-1]=rows[-1],rows[0]
    elif fault=='bool_id':rows[-1]['target_idx']=True
    elif fault=='float_id':rows[-1]['target_idx']=float(rows[-1]['target_idx'])
    else:rows[-1]['he_logits']={'nan':[float('nan'),0.],'inf':[0.,float('inf')],'complex':[1j,0.],
        'wrong_shape':[0.],'bool_logit':[False,1.],'huge_int':[10**10000,0.]}[fault]
    with pytest.raises(ValueError):r.evaluate_records(rows,refs,count=408)
