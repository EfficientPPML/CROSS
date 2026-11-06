import unittest
import numpy as np
from ranked_metrics import accuracy_gate_counts,evaluate

class Ranked(unittest.TestCase):
    def test_exact_count_boundaries(self):
        self.assertTrue(accuracy_gate_counts(408,337,345)['ranked_accuracy_pass'])
        self.assertFalse(accuracy_gate_counts(408,336,345)['ranked_accuracy_pass'])
        self.assertTrue(accuracy_gate_counts(100,75,77)['ranked_accuracy_pass'])
        self.assertFalse(accuracy_gate_counts(100,74,74)['ranked_accuracy_pass'])
        self.assertTrue(accuracy_gate_counts(200,162,166)['ranked_accuracy_pass'])
        self.assertFalse(accuracy_gate_counts(200,161,166)['ranked_accuracy_pass'])

    def fixture(self):
        labels=np.arange(408)%2;prediction=labels.copy();prediction[::8]=1-prediction[::8]
        native=np.column_stack((1-prediction,prediction)).astype(float)
        return np.arange(408),labels,native,native.copy(),np.ones(408,dtype=bool)

    def test_prediction_and_logit_disagreement_do_not_replace_accuracy(self):
        args=list(self.fixture());args[3][0]=args[3][0,::-1];args[3][1]=args[3][1,::-1]
        result=evaluate(*args);large=next(r for r in result['subsets'] if r['size']==408)
        self.assertTrue(large['ranked_accuracy_pass']);self.assertEqual(large['approximate_correct'],large['native_correct'])
        self.assertFalse(large['diagnostics']['all_predictions_agree']);self.assertFalse(large['diagnostics']['local_toy_logit_tolerance_pass'])

    def test_incomplete_coverage_has_no_accuracy_claim(self):
        args=list(self.fixture());args[4][0]=False
        result=evaluate(*args);large=next(r for r in result['subsets'] if r['size']==408)
        self.assertFalse(large['ranked_accuracy_pass']);self.assertIsNone(large['approximate_accuracy'])
        self.assertNotIn('diagnostics',large)

    def test_identity_and_binary_label_contracts(self):
        args=list(self.fixture());args[0][0]=args[0][1]
        with self.assertRaises(ValueError):evaluate(*args)
        args=list(self.fixture());args[1][0]=2
        with self.assertRaises(ValueError):evaluate(*args)

if __name__=='__main__':unittest.main()
