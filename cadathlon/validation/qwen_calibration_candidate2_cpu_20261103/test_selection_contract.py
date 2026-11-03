from pathlib import Path
from fractions import Fraction
import copy,json
import numpy as np
import pytest
import select_candidate2 as selection
from domain_rule import silu_domain
P=Path(__file__).resolve().parent


def test_actual28_certificates_derive_exact_public_domains():
    rows=json.loads((P/'norm_multipliers.json').read_text())
    assert set(rows)=={str(i) for i in range(28)}
    for row in rows.values():
        d=Fraction(int(row['delta_fraction']['numerator_hex'],16),int(row['delta_fraction']['denominator_hex'],16))
        assert 0<=d<1e-6 and row['certified']
        assert silu_domain(row['native_gate_upper'],d)['domain']==row['public_silu_domain']


@pytest.mark.parametrize('field',['norm','softmax'])
def test_frozen_rules_reject_mutually_consistent_runtime_modification(field):
    protocol=json.loads((P/'protocol.json').read_text());base=json.loads((P/'inputs/base_manifest.json').read_text())
    selection.require_fixed(base,protocol)
    if field=='norm':base['nonlinear']['0.post.inverse_sqrt']['evaluation_coefficients'][0]+=1e-5
    else:base['softmax']['0']['squarings']+=1
    with pytest.raises(ValueError,match='Frozen norm/softmax'):selection.require_fixed(base,protocol)


def test_actual_source_input_binding_and_substitution(tmp_path):
    (tmp_path/'x').write_text('original');protocol={'source_input_sha256':{'x':selection.sha(tmp_path/'x')}}
    selection.verify_inputs(tmp_path,protocol);(tmp_path/'x').write_text('replacement')
    with pytest.raises(ValueError,match='source/input mismatch'):selection.verify_inputs(tmp_path,protocol)


def test_wide_domain_keeps_fixed_dense_transition_grid():
    grids=selection.grids([-3601,3601])
    assert len(grids['full'])==len(grids['central'])==16385
    assert grids['central'][0]==-16 and grids['central'][-1]==16 and 0 in grids['central']
    assert np.array_equal(grids['endpoints_zero'],[-3601,0,3601])


def test_rational_intermediate_domain_failure_cannot_pass_final_error(monkeypatch):
    monkeypatch.setattr(selection.numeric,'polynomial',lambda x,*args:np.full_like(x,2.))
    candidate={'base':{},'doublings':1,'reciprocal':{}}
    actual,details=selection.rational_checked(np.array([-1.,0.,1.]),candidate)
    assert actual is None and not details[0]['inside_domain'] and details[0]['minimum']==5.
