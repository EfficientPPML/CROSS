"""Independent arithmetic and real-ring checks for the reduced CNN schedule."""
import json
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / 'non_agent_pset/p3_cnn/src'


@pytest.mark.parametrize('name', ['student1', 'student2'])
def test_compact_fixture_matches_direct_convolution_and_secure_plan(monkeypatch, name):
    monkeypatch.syspath_prepend(str(SRC))
    # Unique loading avoids the other problems' identically named reference modules.
    import importlib.util
    spec = importlib.util.spec_from_file_location('compact_reference', SRC / 'reference.py')
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    baseline_spec = importlib.util.spec_from_file_location('compact_baseline', SRC / 'baseline_student.py')
    baseline = importlib.util.module_from_spec(baseline_spec)
    baseline_spec.loader.exec_module(baseline)
    from jaxite_word import he_params

    path = SRC.parent / 'test' / f'{name}.in'
    model, shape, inputs, actual = reference.make_case(json.loads(path.read_text()))
    weights = model.conv.weight.detach().numpy()
    bias = model.conv.bias.detach().numpy()
    kernel = weights.shape[-1]
    padded = np.pad(inputs, ((0, 0), (0, 0), (kernel // 2, kernel // 2),
                            (kernel // 2, kernel // 2)))
    width = (shape[-1] + 3) // 4
    convolution = np.empty((1, weights.shape[0], width, width))
    for channel in range(weights.shape[0]):
        for row in range(width):
            for column in range(width):
                patch = padded[0, :, row * 4:row * 4 + kernel,
                               column * 4:column * 4 + kernel]
                convolution[0, channel, row, column] = np.sum(patch * weights[channel]) + bias[channel]
    expected = (convolution ** 2).reshape(1, -1) @ model.fc.weight.detach().numpy().T
    expected += model.fc.bias.detach().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-14, rtol=1e-14)
    np.testing.assert_allclose(actual, json.loads(path.with_suffix('.out').read_text())['logits'],
                               atol=1e-14, rtol=1e-14)

    packed = baseline.pack_model(model, shape)
    ring = packed.ring_config
    assert packed.depth == 3
    assert ring.degree == 8192
    assert he_params.qp_is_secure(ring.q_towers, ring.p_towers, ring.degree,
                                 he_params.SecurityTarget(128, he_params.CostModel.CLASSICAL))
