"""Concrete original-array/callback-binding regressions, with mocked device work."""
import pytest
from test_grouped_rotation import env,lease_env,setup,Array

def test_all_zero_callback_cannot_skip_original_input_barrier(env):
    owner,_,_,_,value=setup(env);original=value.polynomial
    def callback(*args):value.polynomial=Array(original.data)
    with pytest.raises(ValueError):owner.rotate_many(value,(0,),progress=callback)
    assert any(item is original for item in env.barriers)

def test_pending_original_input_is_quarantined_after_wrapper_substitution(env):
    owner,_,_,_,value=setup(env);original=value.polynomial
    env.fail=lambda item:item is original
    def callback(*args):value.polynomial=Array(original.data)
    with pytest.raises(ValueError):owner.rotate_many(value,(0,),progress=callback)
    assert owner._failed is not None and env.quarantine
    with pytest.raises(RuntimeError):owner.clear()

def test_original_result_gets_retirement_barrier_after_callback_substitution(env):
    owner,_,_,_,value=setup(env);record=[]
    def callback(n,index,result):
        record.append(result.polynomial);env.barriers.clear();result.polynomial=Array(result.polynomial.data)
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2),progress=callback)
    assert any(item is record[0] for item in env.barriers)

def test_binding_fingerprint_truncation_fails_before_next_completion(env):
    owner,_,_,_,value=setup(env)
    def callback(*args):owner._group.binding_states.clear()
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2),progress=callback)
    assert env.work==1

def test_removing_both_binding_lists_is_rejected(env):
    owner,_,_,_,value=setup(env)
    def callback(*args):
        owner._group.bindings.clear();owner._group.binding_states.clear()
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2),progress=callback)
    assert env.work==1

def test_original_control_arrays_retained_after_binding_replacement(env,monkeypatch):
    owner,_,_,_,value=setup(env);controls=[];original=owner._group_compiler.bind
    def bind(*args,**kwargs):
        binding=original(*args,**kwargs);array=Array([1]);binding.controls=(array,);controls.append(array);return binding
    monkeypatch.setattr(owner._group_compiler,'bind',bind)
    def callback(*args):
        env.barriers.clear()
        for binding in owner._group.bindings:binding.controls=()
    with pytest.raises(ValueError):owner.rotate_many(value,(1,2),progress=callback)
    assert controls and all(any(value is control for value in env.barriers) for control in controls)

def test_nonjax_payload_is_explicit_api_rejection_before_work(env):
    owner,_,_,_,value=setup(env);value.polynomial=value.polynomial.data
    with pytest.raises(TypeError,match='immutable JAX payload'):owner.rotate_many(value,(1,))
    assert env.work==env.decompositions==0
