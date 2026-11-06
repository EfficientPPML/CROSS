from pathlib import Path
import json,sys
import pytest
HERE=Path(__file__).resolve().parent;sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'runtime')]
import group_protocol as p

def test_actual_public_profile_inventory_storage_and_counts():
    plan=p.plan((HERE/'runtime/stage/server_inputs/p60_profile_and_refresh.json').read_bytes(),(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes())
    assert plan['actual_rotation_calls']==116 and plan['exact_pair_comparisons']==66 and plan['shared_decompositions']==6
    assert len(plan['required_rotation_indices'])==26
    assert {r['level']:r['digit_bytes'] for r in plan['rows']}=={49:478150656,25:138412032}
    assert max(r['result_bytes'] for r in plan['rows'])==1572864000
    assert json.loads(json.dumps(plan))==plan

def test_exact_parent_and_group_overlays_failclosed_pending():
    assert p.check_sources(allow_pending=True)['parent_source_count']==231
    with pytest.raises(RuntimeError,match='pending'):p.check_sources()

def test_nested_output_is_not_source(monkeypatch,tmp_path):
    for name in ('runtime','joined','base','server_inputs'):(tmp_path/name).mkdir()
    (tmp_path/'a.py').write_text('x=1\n');(tmp_path/'proof/snapshot').mkdir(parents=True)
    (tmp_path/'proof/snapshot/ignored.py').write_text('must not enter closure')
    monkeypatch.setattr(p,'HERE',tmp_path)
    assert set(p.sources())=={'a.py'}

def test_late_runtime_drift_rejected(monkeypatch,tmp_path):
    original=p.HERE;origin=json.loads((original/'group_origin.json').read_bytes())
    # Lightweight fake root uses a single parent member and exact reviewed new references.
    origin['parent_source_hashes']={'source.py':'0'*64}
    (tmp_path/'group_origin.json').write_text(json.dumps(origin))
    (tmp_path/'group_dependencies.json').write_bytes((original/'group_dependencies.json').read_bytes())
    (tmp_path/'source.py').write_text('changed')
    monkeypatch.setattr(p,'HERE',tmp_path)
    with pytest.raises(ValueError,match='source drift'):p.check_sources(allow_pending=True)


def test_public_plan_rejects_missing_cpu_before_import(monkeypatch):
    monkeypatch.delenv('JAX_PLATFORMS',raising=False)
    monkeypatch.setitem(sys.modules,'client_keys',None)
    with pytest.raises(ValueError,match='explicit CPU'):p.plan(b'',b'')
