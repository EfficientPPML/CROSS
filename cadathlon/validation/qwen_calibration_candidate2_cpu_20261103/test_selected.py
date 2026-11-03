from pathlib import Path
from fractions import Fraction as F
import hashlib,json
import numpy as np
import pytest
import arithmetic
import select_candidate2 as selection
P=Path(__file__).resolve().parent
M=json.loads((P/'candidate-v2/manifest.json').read_text())
PROTOCOL=json.loads((P/'protocol.json').read_text())


def test_all28_supported_and_fixed_norm_softmax_bytes_identical():
    assert selection.sha(P/'candidate-v2/manifest.json')=='0af45a8fc6d13b9d429ee1af2311b2cb6ac0aa4cde02eca56478472762482489'
    assert M['all_silu_sites_supported'] and len(M['nonlinear'])==141
    selection.require_fixed(M,PROTOCOL)
    old=json.loads((P/'inputs/base_manifest.json').read_text())
    assert selection.fixed_parts(M)==selection.fixed_parts(old)
    assert {k for k in M['nonlinear'] if k.endswith('.silu')}=={f'{i}.silu' for i in range(28)}


@pytest.mark.parametrize('layer',range(28))
def test_selected_rules_through_unchanged_actual_arithmetic(layer):
    label=f'{layer}.silu';rule=M['nonlinear'][label]
    assert rule['domain']==PROTOCOL['final_public_silu_domains'][str(layer)]
    for name,grid in selection.grids(rule['domain']).items():
        engine=arithmetic.QwenArithmetic(M)
        actual=engine.silu(arithmetic.Value(grid),rule,label)
        truth=grid*np.exp(-np.logaddexp(0,-grid))
        assert actual.depth==rule['depth']
        assert not engine.domain_failures() and np.isfinite(actual.data).all()
        assert np.max(np.abs(actual.data-truth))<=1e-4,(label,name)


def test_all_actual_certificate_display_endpoints_are_outward():
    rows=json.loads((P/'norm_multipliers.json').read_text())
    for row in rows.values():
        exact=row['delta_fraction'];delta=F(int(exact['numerator_hex'],16),int(exact['denominator_hex'],16))
        assert F(row['relative_upper_binary64'])>=delta


def test_selected_assessments_are_first_supported_prescribed_choices():
    assessments=json.loads((P/'candidate-v2/assessments.json').read_text())
    assert len(assessments)==28
    for label,assessment in assessments.items():
        assert assessment['supported'] and assessment['attempts'][-1]['passed']
        assert all(not row['passed'] for row in assessment['attempts'][:-1])
        assert all(row['points'] in (3,16385) for row in assessment['attempts'][-1]['grids'].values())
