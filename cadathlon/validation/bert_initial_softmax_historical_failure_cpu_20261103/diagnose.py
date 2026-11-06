"""Replay only observed public scale metadata; no ciphertext/model imports."""
from pathlib import Path
import json,math
P=Path(__file__).resolve().parent

def derive():
    r=json.loads((P/'run-v1/report.json').read_bytes());product=r['phases'][-2];output=r['phases'][-1]
    assert product['label']=='all128_distinct16 initial softmax base_times_inverse' and product['level']==17
    assert output['label']=='all128_distinct16 initial softmax probabilities' and output['level']==16
    q=r['ring']['q_towers'];delta=r['ring']['scale'];source=product['scale'];level=product['level']
    divisor=int(q[2*level])*int(q[2*level+1])
    # Existing weighted_sum's public target, actual plaintext scale, then
    # CompiledPlaintextMultiply product and CompiledRescale division order.
    target_before=delta*divisor;encoded_scale=target_before/source
    product_scale=float(source)*float(encoded_scale);actual=product_scale/divisor
    assert actual==output['scale']==math.nextafter(delta,math.inf)
    return dict(scope='Public binary64 metadata reproduction only; no CKKS accuracy or numeric gate result.',
        source_level=level,output_level=output['level'],source_scale=source,canonical_scale=delta,
        prime_pair=q[2*level:2*level+2],integer_divisor=divisor,target_before_rescale=target_before,
        encoded_plaintext_scale=encoded_scale,product_scale=product_scale,
        replayed_output_scale=actual,observed_output_scale=output['scale'],difference=actual-delta,
        difference_in_ulps=(actual-delta)/math.ulp(delta),frozen_equality_assertion_false=actual!=delta)
if __name__=='__main__':
    d=derive();(P/'metadata_diagnosis.json').write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');print(d)
