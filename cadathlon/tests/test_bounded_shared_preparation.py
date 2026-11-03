"""Control-flow regression only: preparation must feed both shared branches.

The fake preparation returns a fresh wrapper, as genuine refresh does. These
checks exercise graph alias ownership; they do not emulate refresh noise or
claim encrypted numerical validity.
"""
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(Path(__file__).resolve().parent),str(ROOT),str(ROOT/'cadathlon/common')]
from encrypted_bounded_transformer import BoundedPackedTransformer
from encrypted_transformer import public_toy_approximations

class BranchesChecked(Exception):pass

class Arithmetic:
    def __init__(self):self.prepared=None;self.prepare_calls=0;self.mean_source=None
    def _ensure_capacity(self,values,required):
        assert len(values)==1 and required==1
        value=values[0]
        if value is not self.prepared:
            self.prepare_calls+=1
            self.prepared=SimpleNamespace(_public_absolute_bound=1.)
        return (self.prepared,)
    def add(self,left,right,subtract=False):
        assert subtract
        assert left is self.mean_source is self.prepared
        assert self.prepare_calls==1
        raise BranchesChecked
    @staticmethod
    def public_bound(value):return getattr(value,'_public_absolute_bound',1.)
    @staticmethod
    def declare_bound(value,bound,reason=''):value._public_absolute_bound=bound;return value
    @staticmethod
    def multiply(left,right):return SimpleNamespace()


def test_centered_normalization_reuses_prepared_input_for_mean_and_subtraction():
    graph=object.__new__(BoundedPackedTransformer);graph.features=4;graph.arithmetic=Arithmetic()
    def linear(value,*args,**kwargs):
        # Mean's own public linear primitive also performs capacity preparation.
        (prepared,)=graph.arithmetic._ensure_capacity((value,),1)
        graph.arithmetic.mean_source=prepared
        return SimpleNamespace()
    graph.linear=linear
    with pytest.raises(BranchesChecked):
        graph.normalize(SimpleNamespace(),np.ones(4),1e-5,public_toy_approximations('bert')['inverse_sqrt'],center=True)


def test_attention_reuses_prepared_scores_for_mean_and_centering():
    graph=object.__new__(BoundedPackedTransformer);graph.features=4
    graph.layout=SimpleNamespace(width=2);graph.arithmetic=Arithmetic()
    graph.head_mask=lambda *args,**kwargs:np.ones(8)
    graph.transpose_pairs=lambda value:value
    calls=[]
    def segmented(value,*args,**kwargs):
        calls.append(value)
        if len(calls)==1:return SimpleNamespace()
        (prepared,)=graph.arithmetic._ensure_capacity((value,),1)
        graph.arithmetic.mean_source=prepared
        return SimpleNamespace()
    graph.segmented_sum=segmented
    with pytest.raises(BranchesChecked):
        graph.attention(SimpleNamespace(),SimpleNamespace(),SimpleNamespace(),heads=1,head_dim=2,
            mask=np.ones(8),counts=np.full(8,2.),approximations=public_toy_approximations('bert'))
