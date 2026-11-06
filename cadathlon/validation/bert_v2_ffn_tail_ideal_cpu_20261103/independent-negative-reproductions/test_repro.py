"""Read-only copied-draft contract reproductions; no HE/backend."""
from pathlib import Path
from types import SimpleNamespace
import copy,json,sys
import numpy as np
import pytest
LAB=Path('/tmp/bert-v2-ffn-tail-20261103');HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(LAB))
from test_routing import backend_free,SLOTS
from test_gelu64 import Arithmetic,value,SCALE,MANIFEST
# Load exact copied reviewed source even if the owner fixes its draft meanwhile.
import importlib.util

def copied(name):
    spec=importlib.util.spec_from_file_location(name,HERE/'sources'/f'{name}.py')
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module);return module

gelu=copied('encrypted_bert_v2_gelu')
binding=copied('encrypted_fullslot_refresh_binding')


class FakeBounded:
    def __init__(self,ops):
        self.ctx=ops.ctx;self.ctx.max_level=49
        self.evaluator=SimpleNamespace(ctx=self.ctx,scale=ops.scale,refresher=None)
        self.plan=json.loads((LAB/'p60_profile_and_refresh.json').read_text())['refresh_plan']
        self.input_level=2;self.output_level=25;self.meta_precision=10;self.public_bound=16.
        view=copy.copy(self.ctx)
        self.engine=SimpleNamespace(ctx=view,_cache=self.ctx._param_cache,_degree=131072,_m=262144,
            _num_slots=65536,_gap=1,_sparse=False,_composite_degree=2,_level_budget=[4,4],_dc_subtract=False)
        self.compiled_primitives=[SimpleNamespace(ctx=self.ctx)]*5
        cache=self.ctx._param_cache
        cache.per_level_rotation_keys=False
        cache.raw_rotation_keys={k:{'a':object(),'b':object()} for k in self.plan['rotations']}
        cache.coef_maps={k:object() for k in self.plan['rotations']}
    def refresh(self,source,*,public_bound):return value(source.data,25,SCALE)


def setup(monkeypatch):
    monkeypatch.setitem(sys.modules,'encrypted_refresh',SimpleNamespace(BoundedSecureRefresh=FakeBounded))
    ops=Arithmetic();ref=FakeBounded(ops)
    return ops,ref


def test_empty_plan_bypasses_all_required_bootstrap_keys(monkeypatch):
    ops,ref=setup(monkeypatch)
    ref.plan['rotations']=[];ops.ctx._param_cache.raw_rotation_keys={};ops.ctx._param_cache.coef_maps={}
    owner=binding.FullSlotRefreshBinding(ops,ref)
    owner.check() # Construction accepted an unexecutable empty key inventory.


@pytest.mark.parametrize('flag',[0,0.,None])
def test_reusable_key_mode_truthiness_accepts_wrong_types(monkeypatch,flag):
    ops,ref=setup(monkeypatch);ops.ctx._param_cache.per_level_rotation_keys=flag
    binding.FullSlotRefreshBinding(ops,ref).check()


def test_missing_compiled_helpers_accepted(monkeypatch):
    ops,ref=setup(monkeypatch);ref.compiled_primitives=[]
    binding.FullSlotRefreshBinding(ops,ref).check()


def test_internal_evaluator_can_become_implicitly_refreshing(monkeypatch):
    ops,ref=setup(monkeypatch);owner=binding.FullSlotRefreshBinding(ops,ref)
    ref.evaluator.refresher=object()
    owner.check()


def test_gelu_callback_adds_unproved_return_bound():
    ops=Arithmetic()
    def progress(name,result,counts):result._public_abs_bound=0.
    owner=gelu.BertV2Gelu(ops,gelu.Gelu64Contract(MANIFEST),progress=progress)
    result=owner.evaluate_tile(value(np.zeros(SLOTS),24,SCALE))
    assert result._public_abs_bound==0. and owner.last_plan is not None


def test_gelu_does_not_reject_wrong_initial_layout_before_primitive():
    ops=Arithmetic();source=value(np.zeros(SLOTS),24,SCALE)
    source.degree_layout=(512,256)
    reached=[]
    def progress(*args):reached.append(len(ops.calls));raise KeyboardInterrupt('stop after first primitive')
    owner=gelu.BertV2Gelu(ops,gelu.Gelu64Contract(MANIFEST),progress=progress)
    with pytest.raises(KeyboardInterrupt):owner.evaluate_tile(source)
    assert reached==[1]
