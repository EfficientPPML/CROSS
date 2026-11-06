from pathlib import Path
import hashlib,json
from certificate import derive
P=Path(__file__).resolve().parent
def run():
    pins=json.loads((P/'input_pins.json').read_bytes())
    for name,row in pins.items():
        if hashlib.sha256((P/'inputs'/name).read_bytes()).hexdigest()!=row['sha256']:raise ValueError('fixed public input bytes changed')
    report=derive((P/'inputs/candidate_manifest.json').read_bytes(),json.loads((P/'inputs/continuation_plan.json').read_bytes()))
    (P/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(passed=report['passed'],softmax_layers=28,keys=32,rational_silu=22,tanh_doublings=145,
        largest_reciprocal_residual=report['uniform_mass_residual_upper']['upper_binary64'],base_tanh_max=max(x['base_abs_upper']['upper_binary64'] for x in report['rational_silu']),B4_is_not_universal=True)))
    return report
if __name__=='__main__':run()
