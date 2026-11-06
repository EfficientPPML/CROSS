"""Public NumPy storage contracts only; no backend/model/HE execution."""
import importlib.util
from pathlib import Path
import warnings
import numpy as np
import pytest
import encrypted_score_reduction as candidate
spec=importlib.util.spec_from_file_location('before_reduction',Path(__file__).parent/'before/encrypted_score_reduction.py')
before=importlib.util.module_from_spec(spec);spec.loader.exec_module(before)

NAMES=candidate._LAYOUT_ARRAYS

def owner(layout):
    result=object.__new__(candidate.ScoreReduction);result.layout=layout
    result._bind_layout_fingerprint();result._layout_identity=result._layout_fingerprint()
    return result

def unchanged(result):return result._layout_fingerprint()==result._layout_identity

@pytest.mark.parametrize('valid',range(1,17))
@pytest.mark.parametrize('mask_kind',['all','alternating','last','row_distinct'])
def test_exact_layout_bytes_and_full_digest_parity(valid,mask_kind):
    if mask_kind=='all':mask=np.ones(128,bool)
    elif mask_kind=='alternating':mask=np.arange(128)%2==0
    elif mask_kind=='last':mask=np.arange(128)==127
    else:mask=(np.arange(128)[None,:]+np.arange(16)[:,None])%7==0
    old=before.ScoreReductionLayout(mask,valid_rows=valid)
    new=candidate.ScoreReductionLayout(mask,valid_rows=valid)
    result=owner(new)
    assert result._immutable_layout_bindings is not None
    old_owner=object.__new__(before.ScoreReduction);old_owner.layout=old
    assert result._full_layout_digest()==old_owner._layout_fingerprint()
    for name in NAMES:
        a=getattr(new,name);b=getattr(old,name)
        assert type(a)is np.ndarray and type(a.base)is bytes and not a.flags.writeable
        assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
    assert unchanged(result)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('method',['setflags','item','view'])
def test_normal_numpy_writes_cannot_reenable_or_change_bytes(name,method):
    result=owner(candidate.ScoreReductionLayout(np.ones(128,bool)))
    a=getattr(result.layout,name)
    with pytest.raises((ValueError,TypeError)):
        if method=='setflags':a.setflags(write=True)
        elif method=='item':a.flat[0]=0
        else:a.view().setflags(write=True)
    assert unchanged(result)

@pytest.mark.parametrize('name',NAMES)
@pytest.mark.parametrize('replacement',['mutable_copy','readonly_copy','bytes_copy'])
def test_even_equal_byte_array_replacement_rejected(name,replacement):
    result=owner(candidate.ScoreReductionLayout(np.ones(128,bool)))
    old=getattr(result.layout,name);new=old.copy()
    if replacement=='readonly_copy':new.setflags(write=False)
    elif replacement=='bytes_copy':new=np.ndarray(old.shape,dtype=old.dtype,buffer=old.tobytes())
    setattr(result.layout,name,new)
    with pytest.raises(ValueError,match='immutable score mask'):result._layout_fingerprint()

@pytest.mark.parametrize('mutation',['shape','dtype','strides'])
def test_same_object_metadata_mutation_rejected(mutation):
    result=owner(candidate.ScoreReductionLayout(np.ones(128,bool)))
    a=result.layout.mean_weight
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        if mutation=='shape':a.shape=(16,4096)
        elif mutation=='dtype':a.dtype=np.uint64
        else:a.strides=(0,)
    with pytest.raises(ValueError,match='immutable score mask'):result._layout_fingerprint()

@pytest.mark.parametrize('rows',[True,1.,np.int64(1),2])
def test_rows_are_typed_and_construction_bound(rows):
    result=owner(candidate.ScoreReductionLayout(np.ones(128,bool),valid_rows=1))
    result.layout.valid_rows=rows
    assert not unchanged(result)

@pytest.mark.parametrize('name',NAMES)
def test_legacy_mutable_backing_uses_complete_hash_and_catches_hidden_write(name):
    layout=candidate.ScoreReductionLayout(np.ones(128,bool))
    mutable=getattr(layout,name).copy();view=mutable.view();view.setflags(write=False)
    setattr(layout,name,view);result=owner(layout)
    assert result._immutable_layout_bindings is None and unchanged(result)
    mutable.flat[0]=0 if mutable.flat[0] else 1
    assert not unchanged(result)

@pytest.mark.parametrize('kind',['subclass','readonly_memoryview','owned_readonly'])
def test_nonstandard_backing_falls_back_without_caching(kind):
    layout=candidate.ScoreReductionLayout(np.ones(128,bool));a=layout.mean_weight
    if kind=='subclass':
        class Array(np.ndarray):pass
        replacement=a.view(Array)
    elif kind=='readonly_memoryview':replacement=np.frombuffer(memoryview(a.tobytes()),dtype=a.dtype)
    else:replacement=a.copy();replacement.setflags(write=False)
    layout.mean_weight=replacement;result=owner(layout)
    assert result._immutable_layout_bindings is None and unchanged(result)


def test_immutable_view_backing_chain_and_shape_change_bound():
    layout=candidate.ScoreReductionLayout(np.ones(128,bool))
    base=np.frombuffer(layout.key_mask.tobytes(),dtype=np.bool_)
    layout.key_mask=base.reshape(16,128)
    result=owner(layout);assert result._immutable_layout_bindings is not None and unchanged(result)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning);base.shape=(2,1024)
    with pytest.raises(ValueError):result._layout_fingerprint()


def test_cache_never_rehashes_immutable_arrays_but_fallback_does(monkeypatch):
    result=owner(candidate.ScoreReductionLayout(np.ones(128,bool)))
    monkeypatch.setattr(result,'_full_layout_digest',lambda:(_ for _ in ()).throw(AssertionError('rehash')))
    assert unchanged(result)
    mutable=candidate.ScoreReductionLayout(np.ones(128,bool));mutable.mean_weight=mutable.mean_weight.copy()
    legacy=owner(mutable);calls=[];original=legacy._full_layout_digest
    monkeypatch.setattr(legacy,'_full_layout_digest',lambda:(calls.append(1),original())[1])
    assert unchanged(legacy) and unchanged(legacy) and len(calls)==2
