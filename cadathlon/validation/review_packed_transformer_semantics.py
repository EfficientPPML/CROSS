"""Reproduce all12 packed toy graphs over public NumPy/symbolic primitives.

No CKKS encryption, decryption, bootstrap, noise or scale-error simulation is
performed. The actual packed graph/BSGS composition and polynomial schedule
run against a validation-only arithmetic double. This is not HE evidence.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if os.environ['JAX_PLATFORMS'] != 'cpu':
    raise RuntimeError('Packed planning requires JAX_PLATFORMS=cpu')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
TEST = ROOT/'cadathlon/tests/test_packed_transformer.py'
paths = [Path(__file__), TEST, ROOT/'jaxite_word/bsgs.py']
paths += [ROOT/'cadathlon/common'/name for name in ('encrypted_packed_transformer.py',
    'encrypted_packed_qwen.py', 'encrypted_transformer.py', 'encrypted_polynomial.py', 'support.py')]
for problem in ('p2_bert', 'p4_qwen'):
    folder = ROOT/'cadathlon/agent_pset'/problem
    paths += [folder/'src/reference.py', *sorted((folder/'test').glob('test*.in')),
              *sorted((folder/'test').glob('test*.out'))]
def snapshot():
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).with_name('packed_transformer_semantics_review.json'))
    args = parser.parse_args()
    before = snapshot()
    spec = importlib.util.spec_from_file_location('packed_transformer_planning_validation', TEST)
    testing = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = testing
    spec.loader.exec_module(testing)
    testing.test_packed_pair_client_codec_preserves_nonpower_padding_and_feature_zeros()
    for model in ('bert', 'qwen'):
        for number in range(1, 7):
            testing.test_all_public_packed_graphs_match_feature_oracle_and_domain_contracts(model, number)
    report = {
        'scope': __doc__, 'encrypted_execution': False, 'homomorphic_refresh_executed': False,
        'public_codec_padding_check': True,
        'cases': testing.PACKED_CASE_OBSERVATIONS,
        'plaintext_numerical_gates_pass': True,
        'polynomial_domains_valid_for_observed_fixtures': True,
        'certified_arbitrary_input_domains': False,
        'depth_note': 'Initial symbolic level1000 permits complete graph analysis; no actual secure ring or refresh is simulated.',
        'comparison_note': 'Actual packed graph and sequential BSGS composition vs independent per-feature graph; each compared against original stored exact public fixtures.',
        'source_sha256': before, 'source_unchanged_at_recording': before == snapshot(),
        'versions': {'python': sys.version.split()[0], 'numpy': testing.np.__version__},
    }
    if not report['source_unchanged_at_recording']:
        raise RuntimeError('source changed during packed analysis; rerun against stable files')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({case: row['symbolic_packed_depth'] for case, row in report['cases'].items()}))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
