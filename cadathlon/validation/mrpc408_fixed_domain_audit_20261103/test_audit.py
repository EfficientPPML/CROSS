import unittest
import math
import numpy as np
import audit

class Contracts(unittest.TestCase):
    def test_summary_and_aggregate_keep_failures(self):
        a=audit.summarize(np.array([-2.,0.,3.,np.nan]),[-1.,1.])
        b=audit.summarize(np.array([]),[-1.,1.])
        total=audit.merge([{'op':a},{'op':b}])['op']
        self.assertEqual((total['count'],total['below'],total['above'],total['nonfinite']),(4,1,1,1))
        self.assertEqual(total['examples_with_violations'],1)
        self.assertEqual((total['minimum'],total['maximum']),(-2.,3.))

    def test_exact_harness_selection(self):
        import random
        for size in (100,200,408):
            selected,seed=audit.subset_indices(408,size,42)
            self.assertEqual(seed,int(np.random.default_rng(42).integers(0,0x7fffffff)))
            expected=list(range(408)) if size==408 else sorted(random.Random(seed).sample(range(408),size))
            self.assertEqual(selected,expected)

    def test_hooks_preserve_native_logits_and_cover_padding(self):
        import torch
        from transformers import BertConfig,BertForSequenceClassification
        torch.set_num_threads(1);torch.manual_seed(7)
        config=BertConfig(vocab_size=32,hidden_size=8,num_hidden_layers=2,num_attention_heads=2,intermediate_size=12)
        config._attn_implementation='eager'
        model=BertForSequenceClassification(config).eval()
        manifest=dict(norms={f'{i}.{p}':{'domain':[.01,2.]} for i in range(2) for p in ('attention','mlp')},
            gelu={str(i):{'domain':[-2.,2.]} for i in range(2)},
            softmax={str(i):{'centered_domain':[-1.,1.],'squarings':2} for i in range(2)},tanh={'domain':[-2.,2.]})
        tokens=dict(input_ids=torch.arange(128).reshape(1,-1)%32,attention_mask=torch.tensor([[1]*7+[0]*121]),token_type_ids=torch.zeros((1,128),dtype=torch.long))
        with torch.no_grad():before=model(**tokens).logits.clone()
        obs=audit.Observer(model,manifest);obs.mask=tokens['attention_mask'][0].numpy().astype(bool)
        with torch.no_grad():after=model(**tokens).logits
        self.assertTrue(torch.equal(before,after));self.assertEqual(len(obs.rows),21);self.assertFalse(obs.pending)
        self.assertEqual(obs.rows['0.gelu.valid_queries']['count'],7*12)
        self.assertEqual(obs.rows['0.gelu.padded_queries']['count'],121*12)
        self.assertEqual(obs.rows['0.softmax.centered_scores.valid_queries']['count'],2*7*7)
        self.assertEqual(obs.rows['0.softmax.centered_scores.padded_queries']['count'],2*121*7)
        self.assertEqual(obs.rows['0.attention.inverse_sqrt.valid_queries']['count'],7)
        obs.close();self.assertFalse(obs.handles)

    def test_score_centering_masks_keys_before_mean(self):
        import torch
        from types import SimpleNamespace
        obj=object.__new__(audit.Observer);obj.pending={};obj.rows={};obj.mask=np.array([True,True]+[False]*126)
        obj.manifest={'softmax':{'0':{'centered_domain':[-1.,1.],'squarings':2}}}
        q=torch.ones((1,128,2));k=torch.full((1,128,2),1000.)
        k[0,0]=torch.tensor([1.,2.]);k[0,1]=torch.tensor([3.,4.])
        obj.pending[(0,'q')]=q;obj.pending[(0,'k')]=k
        obj.scores_hook(0)(SimpleNamespace(num_attention_heads=1,attention_head_size=2),None,None)
        row=obj.rows['0.softmax.centered_scores.valid_queries']
        expected=(float((k[0,1].sum()/math.sqrt(2)).item())-float((k[0,0].sum()/math.sqrt(2)).item()))/2
        self.assertAlmostEqual(row['minimum'],-expected);self.assertAlmostEqual(row['maximum'],expected)
        self.assertEqual(row['below'],2);self.assertEqual(row['above'],2)

if __name__=='__main__':unittest.main()
