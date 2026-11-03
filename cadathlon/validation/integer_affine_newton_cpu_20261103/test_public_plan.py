import math
import pytest
from integer_affine import backward_scale_plan

Q=(101,103,107,109,113,127,131,137)


def test_real_initial_affine_and_two_product_metadata():
    delta=float(Q[-2]*Q[-1])
    p=backward_scale_plan(Q,input_scale=delta,first_output_level=2,updates=2,final_scale=delta)
    first=delta*p['initial_plaintext_scale']/(Q[6]*Q[7])
    product=delta*first/(Q[4]*Q[5])
    final=first*product/(Q[2]*Q[3])
    assert p['rows']==[dict(update=1,level=2,scale=first),dict(update=2,level=0,product_scale=product,scale=final)]
    assert abs(p['final_relative_difference'])<1e-14


@pytest.mark.parametrize('field,value',[
    ('updates',0),('updates',True),('first_output_level',1),
    ('first_output_level',3),('input_scale',0.),('input_scale',float('nan')),
    ('final_scale',float('inf')),('final_scale',True)])
def test_invalid_schedule_rejected(field,value):
    args=dict(input_scale=100.,first_output_level=2,updates=2,final_scale=100.)
    args[field]=value
    with pytest.raises(ValueError):backward_scale_plan(Q,**args)


def test_reject_invalid_composite_moduli():
    with pytest.raises(ValueError):
        backward_scale_plan(Q[:-1],input_scale=100.,first_output_level=2,updates=2,final_scale=100.)
