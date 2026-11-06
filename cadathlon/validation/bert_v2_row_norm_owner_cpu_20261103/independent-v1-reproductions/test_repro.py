"""Read-only negative reproductions against the frozen source; no HE/backend."""
import sys
from pathlib import Path
import numpy as np
LAB=Path('/tmp/bert-v2-attention-row-norm-20261103')
sys.path.insert(0,str(LAB))
from test_v2_row_norm import backend_free,setup,fixture,Value,norm,Arithmetic,weights,RAW


def test_captured_checkpoint_mutation_is_not_rejected():
    seen={}
    def callback(label,value,metadata):
        name=label.rsplit(' ',1)[1];seen[name]=value
        if name=='output':seen['mean']._ckks_nsd=True
    ops,owner=setup(progress=callback)
    result=owner.normalize_tile(Value(fixture(owner.contract,2)),valid_rows=2,original_positions=(0,1),capture_checkpoints=True)
    assert result.checkpoints['mean']._ckks_nsd is True
    assert owner.last_run is not None


def test_callback_public_annotation_is_returned():
    def callback(label,value,metadata):value._public_abs_bound=0.
    ops,owner=setup(progress=callback)
    result=owner.normalize_tile(Value(fixture(owner.contract,2)),valid_rows=2,original_positions=(0,1),capture_checkpoints=True)
    assert result.output._public_abs_bound==0.
    assert all(value._public_abs_bound==0. for value in result.checkpoints.values())


def test_bound_aware_owner_accepted_before_loss_of_bound():
    class BoundAware(Arithmetic):
        def declare_bound(self,v,bound):v._public_abs_bound=bound;return v
        def rotate(self,v,shift):return self.declare_bound(super().rotate(v,shift),v._public_abs_bound)
        def add(self,a,b,*,subtract=False):
            bound=a._public_abs_bound+b._public_abs_bound
            return self.declare_bound(super().add(a,b,subtract=subtract),bound)
        def weighted_sum(self,values,weights,*,bias=0.):
            bound=sum(v._public_abs_bound*float(np.max(np.abs(w))) for v,w in zip(values,weights))+float(np.max(np.abs(bias)))
            return self.declare_bound(super().weighted_sum(values,weights,bias=bias),bound)
    ops=BoundAware();owner=norm.V2RowLayerNorm(ops,RAW,*weights('0.attention'))
    source=ops.declare_bound(Value(fixture(owner.contract,2)),100.)
    try:owner.normalize_tile(source,valid_rows=2,original_positions=(0,1))
    except AttributeError as exc:assert '_public_abs_bound' in str(exc)
    else:raise AssertionError('expected lost bound')
    assert ops.calls and not owner._active
