"""Reproduce plaintext layout/domain and conditional symbolic-depth review.

Run with JAX_PLATFORMS=cpu. No encrypted inference or bootstrap is executed.
The numeric backend exists only in this validation script. Symbolic refresh
returns exactly its advertised minimum capacity; counts are conditional, not
measured bootstrap counts or latency estimates.
"""
import argparse
from pathlib import Path
import sys
import json
import importlib.util
import hashlib
from dataclasses import dataclass
from types import SimpleNamespace
from collections import Counter
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'cadathlon/common'))
from encrypted_polynomial import EncryptedPolynomialEvaluator
from encrypted_transformer import PairLayout, EncryptedTensorOps, public_toy_approximations, plan_ciphertext_depth, bert_ciphertext_forward, qwen_ciphertext_forward
from support import metrics

@dataclass
class Value:
    level: int
    _ckks_scale: float = 1.0

class Sub:

    def sub(self, left, right):
        return Value(min(left.level, right.level))

class SubAccessor:

    def __getitem__(self, key):
        return Sub()

class ActualOperationDepth(EncryptedPolynomialEvaluator):
    """Run runtime polynomial/scheduling methods with symbolic primitives."""

    def __init__(self, slots, refresh_level=None):
        self.ctx = SimpleNamespace(num_slots=slots, he_sub=SubAccessor())
        self.refresher = None
        self.strategy = 'power'
        self.scale = 1.0
        self.counts = Counter()
        if refresh_level is not None:

            def refresh(value):
                self.counts['refreshes'] += 1
                return Value(refresh_level)
            self.refresher = SimpleNamespace(output_level=refresh_level, refresh=refresh)

    def _level(self, v):
        return v.level

    def _lower(self, v, level):
        return Value(level)

    def checkpoint(self, values, label):
        return values

    def weighted_sum(self, values, weights, *, bias=0.0):
        values = self._ensure_capacity(tuple(values), 1)
        weights = tuple(weights)
        assert len(values) == len(weights) and values
        self.counts['weighted_sum'] += 1
        self.counts['plaintext_multiply'] += len(values)
        self.counts['plaintext_add'] += int(np.any(np.asarray(bias)))
        self.counts['rescale'] += 1
        return Value(min((v.level for v in values)) - 1)

    def multiply(self, left, right):
        self.counts['ciphertext_multiply'] += 1
        left, right = self._ensure_capacity((left, right), 1)
        return Value(min(left.level, right.level) - 1)

    def add(self, left, right, subtract=False):
        self.counts['ciphertext_add_or_subtract'] += 1
        return Value(min(left.level, right.level))

    def rotate(self, value, shift):
        self.counts['nonzero_rotation'] += int(shift % self.ctx.num_slots != 0)
        return value

class PublicPowerArithmetic:
    """Plaintext test double confined to this validation program."""

    def __init__(self, slots):
        self.ctx = SimpleNamespace(num_slots=slots)
        self.ranges = {}

    def add(self, left, right, subtract=False):
        return left - right if subtract else left + right

    def rotate(self, value, shift):
        return np.roll(value, -shift)

    def weighted_sum(self, values, weights, bias=0.0):
        return sum((v * w for v, w in zip(values, weights))) + bias

    def multiply(self, left, right):
        return left * right

    def observe(self, approx, value):
        value = np.asarray(value)
        key = approx.name
        low, high = approx.domain
        r = self.ranges.setdefault(key, {'min': float('inf'), 'max': -float('inf'), 'domain': list(approx.domain), 'outside': 0})
        r['min'] = min(r['min'], float(np.min(value)))
        r['max'] = max(r['max'], float(np.max(value)))
        r['outside'] += int(np.count_nonzero((value < low) | (value > high) | ~np.isfinite(value)))

    def evaluate(self, value, approximation, output_weights=1.0):
        self.observe(approximation, value)
        lo, hi = approximation.domain
        coefficients = np.polynomial.chebyshev.cheb2poly(approximation.coefficients)
        if approximation.input_power_basis:
            t = value
            coefficients = coefficients / hi ** np.arange(len(coefficients))
        else:
            t = (2 * value - lo - hi) / (hi - lo)
        return np.polynomial.polynomial.polyval(t, coefficients) * output_weights

    def normalize(self, features, approximation, epsilon, center=False):
        x = np.asarray(features)
        if center:
            x = x - x.mean(axis=0, keepdims=True)
        var = np.mean(x * x, axis=0) + epsilon
        return list(x * self.evaluate(var, approximation))

def analyze():
    out = {'scope': 'Read-only plaintext algebra/domain analysis and symbolic execution of runtime polynomial/normalization methods. No encrypted pass or TPU performance evidence.', 'cases': {}}
    for model, problem in [('bert', 'p2_bert'), ('qwen', 'p4_qwen')]:
        folder = ROOT / 'cadathlon/agent_pset' / problem
        spec = importlib.util.spec_from_file_location('oracle_' + model, folder / 'src/reference.py')
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        for number in range(1, 7):
            case = json.loads((folder / f'test/test{number}.in').read_text())
            args = oracle.make_case(case)
            w, x, expected = (args[0], args[1], args[-1])
            golden = np.asarray(json.loads((folder / f'test/test{number}.out').read_text())['logits'])
            np.testing.assert_allclose(expected, golden, atol=1e-10, rtol=1e-10)
            b, s, h = x.shape
            slots = (1 << (b - 1).bit_length()) * (1 << (s - 1).bit_length()) ** 2
            approx = public_toy_approximations(model)
            layout = PairLayout(b, s, slots)
            symbolic = ActualOperationDepth(slots)
            numeric = PublicPowerArithmetic(slots)
            outputs = []
            for evaluator, values in [(symbolic, [Value(1000) for _ in range(h)]), (numeric, layout.pack_client_embeddings(x))]:
                ops = EncryptedTensorOps(evaluator, layout)
                outputs.append(bert_ciphertext_forward(ops, values, case['config'], w, args[2], approx) if model == 'bert' else qwen_ciphertext_forward(ops, values, case['config'], w, approx))
            actual = layout.unpack_client_features(outputs[1])[:, 0 if model == 'bert' else -1]
            err = metrics(actual, expected)
            planned = plan_ciphertext_depth(model, case['config'], w, batch=b, sequence=s, attention_mask=args[2] if model == 'bert' else None, approximations=approx)
            depth = 1000 - min((v.level for v in outputs[0]))
            assert depth == planned['depth'], (model, number, depth, planned)
            numerical = bool(np.all(np.abs(actual - expected) <= 0.02 + 0.01 * np.abs(expected)) and np.array_equal(actual[:, 1] >= actual[:, 0], expected[:, 1] >= expected[:, 0])) if model == 'bert' else err['nrmse'] <= 0.1 and err['max_abs_error'] <= 0.2
            data = {'max_abs_error': err['max_abs_error'], 'nrmse': err['nrmse'], 'plaintext_numeric_gate': numerical, 'runtime_polynomial_symbolic_depth': depth, 'static_depth': planned['depth'], 'selected_operation_counts_without_refresh': dict(symbolic.counts), 'ranges_including_padding_and_masked_slots': numeric.ranges}
            out['cases'][f'{model}/test{number}.in'] = data
            print(model, number, 'depth', depth, 'counts', dict(symbolic.counts), 'numeric', numerical, 'error', err['max_abs_error'], 'outside', sum((r['outside'] for r in numeric.ranges.values())))
    from encrypted_refresh import refresh_plan
    for model, problem in [('bert', 'p2_bert'), ('qwen', 'p4_qwen')]:
        folder = ROOT / 'cadathlon/agent_pset' / problem
        spec = importlib.util.spec_from_file_location('refresh_oracle_' + model, folder / 'src/reference.py')
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        for number in range(1, 7):
            key = f'{model}/test{number}.in'
            row = out['cases'][key]
            if row['static_depth'] <= 52:
                continue
            case = json.loads((folder / f'test/test{number}.in').read_text())
            args = oracle.make_case(case)
            w, x = (args[0], args[1])
            b, s, h = x.shape
            slots = (1 << (b - 1).bit_length()) * (1 << (s - 1).bit_length()) ** 2
            layout = PairLayout(b, s, slots)
            approx = public_toy_approximations(model)
            budget = min(4, max(2, (layout.period - 1).bit_length()))
            plan = refresh_plan(131072, layout.period, (budget, budget), max_level=49)
            e = ActualOperationDepth(slots, refresh_level=plan['output_level'])
            ops = EncryptedTensorOps(e, layout)
            features = [Value(49) for _ in range(h)]
            results = bert_ciphertext_forward(ops, features, case['config'], w, args[2], approx) if model == 'bert' else qwen_ciphertext_forward(ops, features, case['config'], w, approx)
            row['symbolic_refresh_schedule'] = {'output_capacity': plan['output_level'], 'refresh_calls': e.counts['refreshes'], 'final_level': min((v.level for v in results)), 'note': 'Uses current scheduler with ideal refresh objects. No bootstrap executed and no latency claim.'}
            print('refresh estimate', model, number, row['symbolic_refresh_schedule'])
    return out
INTERPRETATION = {'accuracy_scope': 'All six public fixtures for each model, using the selected NumPy power basis '
                   'arithmetic. Checked cleartext oracle outputs against the stored .out fixtures. '
                   'No CKKS noise, scale error, bootstrap error or TPU execution was simulated.',
 'certified_domain_bounds': False,
 'depth_scope': 'Actual runtime polynomial evaluate() and normalize() methods executed with '
                'symbolic level objects; ciphertext multiply and weighted sum each modeled as one '
                'level. This checks critical-path accounting, not cryptographic execution.',
 'domain_observation_scope': 'Minimum complete PairLayout period including padded batch/query/key '
                             'coordinates and polynomial arguments in slots later masked out. Full '
                             'CKKS slot vectors repeat that period.',
 'encrypted_execution': False,
 'generalization': 'Observed fixture domains do not certify arbitrary inputs, weights, seeds or '
                   'pretrained checkpoints.',
 'operation_count_scope': 'Selected facade/arithmetic counts without refresh. '
                          'Ciphertext_add_or_subtract counts only explicit tensor calls; it omits '
                          'additions internal to weighted sums and normalization. Rescale counts '
                          'are weighted-sum rescales; ciphertext multiplication includes its own '
                          'rescale.',
 'refresh_count_scope': 'Conditional symbolic schedule only: initial chain level49, ideal '
                        'bootstrap always returning exactly refresh_plan.output_level (25 or29), '
                        'current per-operation capacity checks and reference sharing. Actual '
                        'bootstrap may return more capacity, and measured refresh counts/times can '
                        'differ. No refresh executed.',
 'refresh_error_scope': 'Ideal refresh preserved values exactly. These counts establish neither '
                        'repeated-refresh numerical stability nor practical performance.'}


def source_snapshot():
    paths = [Path(__file__).resolve(), ROOT / 'jaxite_word/bootstrapping.py']
    paths += [ROOT / 'cadathlon/common' / name for name in
              ('encrypted_transformer.py', 'encrypted_polynomial.py', 'encrypted_refresh.py',
               'encrypted_reference.py')]
    for folder in ('p2_bert', 'p4_qwen'):
        problem = ROOT / 'cadathlon/agent_pset' / folder
        paths.append(problem / 'src/reference.py')
        paths += sorted((problem / 'test').glob('test*.in'))
        paths += sorted((problem / 'test').glob('test*.out'))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'cadathlon/validation/transformer_semantics_review.json')
    args = parser.parse_args()
    before = source_snapshot()
    report = analyze()
    report['interpretation'] = INTERPRETATION
    script = Path(__file__).resolve()
    report['analysis_program_sha256'] = hashlib.sha256(script.read_bytes()).hexdigest()
    report['analysis_program'] = str(script.relative_to(ROOT))
    report['source_sha256'] = before
    report['source_unchanged_at_recording'] = before == source_snapshot()
    if not report['source_unchanged_at_recording']:
        raise RuntimeError('runtime source changed during analysis; repeat against frozen sources')
    report['plaintext_numerical_gates_pass'] = all(case['plaintext_numeric_gate'] for case in report['cases'].values())
    report['within_observed_domains'] = all(interval['outside'] == 0 for case in report['cases'].values()
                                          for interval in case['ranges_including_padding_and_masked_slots'].values())
    report['versions'] = {'python': sys.version.split()[0], 'numpy': np.__version__}
    report['conditional_refresh_profile'] = {'degree': 131072, 'initial_level': 49,
                                             'refresh_for_static_depth_above': 52,
                                             'output_capacity': 'exactly the advertised minimum'}
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return 0 if report['plaintext_numerical_gates_pass'] and report['within_observed_domains'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
