"""Backend-free checks of the public interval/count and corpus contracts."""
from fractions import Fraction as F
from pathlib import Path
import json,unittest
from reciprocal_schedule import contract
HERE=Path(__file__).resolve().parent
class ProtocolTests(unittest.TestCase):
    def test_all_key_counts_outward_domain_and_fixed_coefficient_formula(self):
        for K in range(1,33):
            row=contract(K);a,b=row['domain'];z=row['initial_estimate']
            self.assertLessEqual(F.from_float(a),F(1,2*K));self.assertEqual(b,2)
            self.assertEqual(row['rounded_first_weight'],(-z)*z)
            self.assertEqual(row['rounded_first_bias'],2*z)
            self.assertLess(a,row['inactive_midpoint']);self.assertLess(row['inactive_midpoint'],b)
    def test_actual_rounded_affine_exact_residual_and_minimal_count(self):
        gate=F(1,10**12)
        for K in range(1,33):
            row=contract(K);a,b=map(F.from_float,row['domain']);w=F.from_float(row['rounded_first_weight']);bias=F.from_float(row['rounded_first_bias'])
            residual=lambda x:1-x*(w*x+bias)
            vertex=-bias/(2*w);points=[a,b]+([vertex] if a<=vertex<=b else [])
            bound=max(abs(residual(x)) for x in points)
            certificate=row['exact_first_absolute_residual_bound']
            self.assertEqual(bound,F(int(certificate['numerator']),int(certificate['denominator'])))
            n=1;previous=None
            while bound>gate:previous=bound;bound*=bound;n+=1
            self.assertEqual(n,row['iterations']);self.assertLessEqual(bound,gate)
            if previous is not None:self.assertGreater(previous,gate)
    def test_interior_points_covered_by_exact_residual_bound(self):
        for K in range(1,33):
            row=contract(K);a,b=map(F.from_float,row['domain']);w=F.from_float(row['rounded_first_weight']);bias=F.from_float(row['rounded_first_bias']);cert=row['exact_first_absolute_residual_bound'];bound=F(int(cert['numerator']),int(cert['denominator']))
            for j in range(65):
                x=a+(b-a)*F(j,64);self.assertLessEqual(abs(1-x*(w*x+bias)),bound)
    def test_saved_schedule_and_protocol_cover_every_required_key(self):
        saved=json.loads((HERE/'reciprocal_schedule.json').read_text());protocol=json.loads((HERE/'protocol.json').read_text())
        self.assertEqual(saved['round_contracts'],[contract(K) for K in range(1,33)])
        self.assertEqual(protocol['softmax']['reciprocal_keys'],list(range(1,33)))
        self.assertEqual(saved['initial_mean_exp_contract']['iterations'],6)
        self.assertEqual(saved['round_contracts'][-1]['iterations'],11)
    def test_frozen_input_cardinality_and_decode_membership(self):
        corpus=json.loads((HERE/'corpus.json').read_text());rows=json.loads((HERE/'tokenized_calibration_inputs.json').read_text());report=json.loads((HERE/'input_preflight.json').read_text())
        self.assertEqual(len(rows),288);self.assertEqual(len(corpus['calibration_prompts']),96)
        for S in (8,16,32):
            selected=[r for r in rows if r['sequence_length']==S];self.assertEqual(len(selected),96)
            self.assertEqual(len({tuple(r['token_ids']) for r in selected}),96)
            self.assertTrue(all(len(r['token_ids'])==S and r['raw_token_count']>=32 for r in selected))
        self.assertEqual(len(set(corpus['decode_prompt_ids'])),16)
        self.assertTrue(set(corpus['decode_prompt_ids']) <= {r['id'] for r in rows})
        self.assertEqual(report['validation_token_intersection'],0)
    def test_no_model_or_encrypted_measurement_claim(self):
        protocol=json.loads((HERE/'protocol.json').read_text());preflight=json.loads((HERE/'input_preflight.json').read_text())
        self.assertTrue(protocol['execution_authorized_in_this_artifact'])
        self.assertFalse(protocol['runtime']['model_forward_before_root_protocol_review'])
        self.assertTrue(preflight['no_model_instantiated']);self.assertTrue(preflight['no_jax_imported'])
        self.assertEqual(protocol['maximum_calibration_candidates'],3)
if __name__=='__main__':unittest.main(verbosity=2)
