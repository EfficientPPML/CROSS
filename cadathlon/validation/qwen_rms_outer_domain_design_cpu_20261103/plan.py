"""Public architecture and ideal cost illustration; no tensor/model evaluation."""
from pathlib import Path
import hashlib,json
from bounds import ideal_newton_cost
P=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def make_plan():
    pins=json.loads((P/'model_pins.json').read_text());config=json.loads((P/'config.json').read_text())
    if sha(P/'config.json')!=pins['asset_sha256']['config.json']:raise ValueError('pinned architecture mismatch')
    if (config['num_hidden_layers'],config['hidden_size'],config['head_dim'],config['num_attention_heads'],config['num_key_value_heads'])!=(28,1024,128,16,8):raise ValueError('architecture changed')
    if config['attention_bias'] is not False or config['attention_dropout']!=0.:raise ValueError('bias-free deterministic native attention assumption changed')
    return dict(scope='Design/reference arithmetic only; exact-real native graph envelopes, not approximate/HE/native-float32 certificates.',
        model_revision=pins['revision'],config_sha256=sha(P/'config.json'),model_pin_sha256=sha(P/'model_pins.json'),
        rms_sites=dict(input=28,post_attention=28,q_heads=28,k_heads=28,final=1,total=113),
        epsilon_binary64_hex=float(config['rms_norm_eps']).hex(),
        q_head_variance_count=28*16,k_head_variance_count=28*8,
        extraction_not_performed=True,model_evaluated=False,coefficients_selected=False,validation_outputs_used=False,
        initializer='Full embedding table coordinate maxima, optionally exact full-row maximum for the first site.',
        ideal_newton_cost_illustrations=[ideal_newton_cost(10**n) for n in (2,4,6,12,18,24)],
        limitations=['No actual113-site outer widths have been computed by this design checkpoint.',
            'Illustrative ideal Newton counts select no site coefficients and exclude rounded seed/arithmetic/CKKS errors.',
            'Approximate attention needs a separate positivity or row-L1 certificate; exact native softmax convexity does not transfer automatically.',
            'Approximate norm/SiLU uniform envelopes must be certified on the new outer domains before propagating that graph.',
            'Future candidate selection must follow a new untouched input freeze and preserve original published regression obligations.'])
if __name__=='__main__':print(json.dumps(make_plan(),indent=2))
