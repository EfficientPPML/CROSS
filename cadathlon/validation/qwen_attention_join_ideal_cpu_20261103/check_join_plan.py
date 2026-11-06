"""Actual public QKV matrix inventories and all28 unchanged join contracts."""
from pathlib import Path
import hashlib,json,sys
import pytest
from join_ideal import install_matrix,weights
from test_join import make_contract
from encrypted_chained_streaming_linear import linear_rotation_plan
from encrypted_streaming_linear import RowTileLayout
P=Path(__file__).resolve().parent
def derive():
    patch=pytest.MonkeyPatch()
    try:
        install_matrix(patch)
        from encrypted_packed_qwen import MatrixDiagonals
        keys=set();layers=[]
        for layer in range(28):
            contract=make_contract(layer);raw,values=weights(layer);matrices={}
            for kind,value in values.items():
                source=MatrixDiagonals(value,65536,4096);plan=linear_rotation_plan(source,strategy='chained')
                keys.update(plan.required_keys)
                matrices[kind]=dict(shape=list(value.shape),active_diagonals=len(source.active),baby_size=source.n1,
                    required_keys=list(plan.required_keys),nonzero_rotation_calls=plan.nonzero_rotation_calls,
                    weighted_sums=len(plan.giant_indices),additions=max(0,len(plan.giant_indices)-1),
                    array_sha256=hashlib.sha256(value.astype('<f8').tobytes()).hexdigest())
            layers.append(dict(layer=layer,matrices=matrices,outer_events=[vars(e) for e in contract.outer_events]))
        assert len(layers)==28 and sum(len(r['matrices']) for r in layers)==84
        assert sum(e['stage'].endswith('_reserve') for row in layers for e in row['outer_events'])==9
        return dict(scope='Public genuine84-matrix plan; no model or encrypted execution.',layers=layers,
            linear_key_union=sorted(keys),actual_nonlinear_coefficients_unchanged=True,
            centered_domain_implications_proven=0,centered_domain_implications_unproved=868,
            output_levels=dict(single_token=22,nontrivial_causal=20),
            inherited_join_plan_sha256=hashlib.sha256((P/'inputs/join_plan.json').read_bytes()).hexdigest())
    finally:patch.undo()
if __name__=='__main__':
    result=derive();(P/'join_actual_inventory.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    assert not any(name in sys.modules for name in ('jax','jaxlib','torch','transformers','safetensors'))
    print(json.dumps(dict(layers=28,matrices=84,linear_key_union=result['linear_key_union'],reserve_events=9)))
