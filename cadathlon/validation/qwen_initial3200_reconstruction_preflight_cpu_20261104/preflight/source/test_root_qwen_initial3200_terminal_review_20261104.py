"""Synthetic-only negative checks for independent reconstruction primitives."""
import copy, importlib.util, pathlib, unittest
from unittest.mock import patch
P=pathlib.Path(__file__).with_name('root_qwen_initial3200_terminal_review_20261104.py')
spec=importlib.util.spec_from_file_location('root_review',P);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

def stats(n,lo=1.,hi=2.):
    return dict(count=n,minimum=lo,maximum=hi,nonfinite=0,nonpositive=n if hi<=0 else 0)
def observed(s):
    ranges={k:stats(n)for k,n in r.counts(s).items()};rows={}
    for i in range(28):
        layer={}
        for k in range(1,s+1):
            layer[str(k)]=dict(query=k-1,key_count=k,shape=[1,16,k],k1_bypass=k==1,reciprocal_observations=0,raw_valid=stats(16*k),centered=stats(16*k,0.,0.))
        rows[str(i)]=layer
        ranges[f'{i}.softmax.centered']=stats(16*s*(s+1)//2,0.,0.)
    return dict(complete=True,invalid=False,sequence_length=s,norm_input_sites=113,silu_gate_sites=28,ranges=ranges,score_rows=rows)

class Reconstruction(unittest.TestCase):
    def test_fixed_workload_counts(self):
        es=r.entries();self.assertEqual(len(es),3200);self.assertEqual(sum(e['original416_member']for e in es),416)
        self.assertEqual(sum(e['length']for e in es),52928)
        self.assertEqual(sum(r.counts(e['length'])['0.softmax.raw_valid']for e in es),9524224)
        self.assertEqual(sum(r.counts(e['length'])['0.silu']for e in es),162594816)
    def test_all_lengths_observation(self):
        for s in range(1,33):r.observation(observed(s),s)
    def test_independent_aggregate_counts_and_k(self):
        a=r.empty_aggregate()
        for s in (1,2,2):r.add_observation(a,observed(s))
        self.assertEqual(a['contexts'],3);self.assertEqual(a['ranges']['0.q.inverse_sqrt']['count'],80)
        self.assertEqual(a['score_rows']['0']['1']['contexts'],3);self.assertEqual(a['score_rows']['0']['2']['contexts'],2)
        self.assertEqual(a['score_rows']['0']['2']['raw_valid']['count'],64)
    def test_missing_site(self):
        x=observed(1);del x['ranges']['final.inverse_sqrt']
        with self.assertRaises(ValueError):r.observation(x,1)
    def test_wrong_q_head_count(self):
        x=observed(2);x['ranges']['0.q.inverse_sqrt']['count']=16
        with self.assertRaises(ValueError):r.observation(x,2)
    def test_wrong_score_head_shape(self):
        x=observed(2);x['score_rows']['0']['2']['shape']=[1,8,2]
        with self.assertRaises(ValueError):r.observation(x,2)
    def test_missing_actual_k(self):
        x=observed(2);del x['score_rows']['0']['2']
        with self.assertRaises(ValueError):r.observation(x,2)
    def test_invented_reciprocal(self):
        x=observed(1);x['score_rows']['0']['1']['reciprocal_observations']=1
        with self.assertRaises(ValueError):r.observation(x,1)
    def test_k1_not_zero(self):
        x=observed(1);x['score_rows']['0']['1']['centered']=stats(16)
        with self.assertRaises(ValueError):r.observation(x,1)
    def test_flat_per_k_mismatch(self):
        x=observed(2);x['ranges']['0.softmax.raw_valid']['maximum']=3.
        with self.assertRaises(ValueError):r.observation(x,2)
    def test_nonfinite_positive_and_bool_rejected(self):
        for key,value in [('count',True),('nonfinite',1),('minimum',float('nan')),('minimum',0.),('nonpositive',1)]:
            x=stats(1);x[key]=value
            with self.assertRaises(ValueError):r.statistic(x,1,True)
    def test_no_selection_epsilon_policy_here(self):
        # Terminal authenticity records positive native float32 epsilon as-is;
        # the selector separately applies its stricter frozen binary64 policy.
        r.statistic(stats(1,9.999999974752427e-7,9.999999974752427e-7),1,True)
    def test_json_and_typed_binding(self):
        for raw in (b'{"a":1,"a":2}',b'{"a":NaN}'):
            with self.assertRaises(ValueError):r.parse(raw)
        with self.assertRaises(ValueError):r.same(True,1)
    def test_empty_and_outside_vocabulary(self):
        for values in ([],[True],[151936],[-1]):
            with self.assertRaises(ValueError):r.tokens(values,1)
    def test_output_cannot_mutate_inputs(self):
        protected=(r.LAB,r.ACTUAL,r.PRIVATE,pathlib.Path('/tmp/qwen-expanded-prefix-token-inputs-20261104-v1'))
        for path in (r.LAB/'new',r.ACTUAL/'new',r.PRIVATE/'new',protected[-1]/'new',pathlib.Path('/tmp'),pathlib.Path('relative')):
            with self.assertRaises(ValueError):r.validate_output(path,protected)
    def test_body_survives_cleanup_cancellation(self):
        body=ValueError('body');cleanup=KeyboardInterrupt('cleanup')
        with patch.object(r.os,'close',side_effect=cleanup):r.close(100,body)
        self.assertEqual(body.__notes__,['review cleanup close failure: KeyboardInterrupt'])
    def test_cleanup_alone_propagates(self):
        cleanup=KeyboardInterrupt('cleanup')
        with patch.object(r.os,'close',side_effect=cleanup):
            with self.assertRaises(KeyboardInterrupt)as caught:r.close(100,None)
        self.assertIs(caught.exception,cleanup)

if __name__=='__main__':unittest.main()
