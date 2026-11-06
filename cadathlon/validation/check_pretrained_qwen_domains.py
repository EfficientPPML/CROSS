"""Offline full Qwen checkpoint semantics and empirical polynomial-domain audit.

Three fixed public prompts exercise every layer and all final logits. This is
not encrypted inference, generation, accuracy evaluation or certified bounds.
Assets must already be present; only pinned safetensors are loaded. Install
transformers==4.51.3 and safetensors in a separate environment.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'common'))
from support import load_module

EXPECTED_ASSETS = {
    'config.json': '660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd',
    'generation_config.json': '2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2',
    'model.safetensors': 'f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
}
PROMPTS = ('The sky is', 'A small robot', 'One plus one equals')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    hashes = {name: digest(args.snapshot / name) for name in EXPECTED_ASSETS}
    if hashes != EXPECTED_ASSETS:
        raise ValueError('Local assets do not match the pinned full Qwen snapshot')

    import torch
    from transformers import AutoTokenizer, Qwen3ForCausalLM

    torch.set_num_threads(4)
    reference = load_module(ROOT / 'agent_pset/p4_qwen/src/reference.py')
    planning = load_module(ROOT / 'validation/plan_transformer_polynomials.py')
    sources = (Path(__file__), ROOT / 'agent_pset/p4_qwen/src/reference.py',
               ROOT / 'agent_pset/p4_qwen/test/model_source.json',
               ROOT / 'validation/plan_transformer_polynomials.py',
               ROOT / 'common/encrypted_polynomial.py', ROOT / 'common/encrypted_transformer.py')
    before = {str(path.relative_to(ROOT.parent)): digest(path) for path in sources}
    provenance = json.loads((ROOT / 'agent_pset/p4_qwen/test/model_source.json').read_text())
    started = time.perf_counter()
    model, loading = Qwen3ForCausalLM.from_pretrained(
        args.snapshot, local_files_only=True, use_safetensors=True,
        torch_dtype=torch.float64, attn_implementation='eager', output_loading_info=True)
    if any(loading.get(name) for name in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Checkpoint tensor mismatch: {loading}')
    model.eval()
    config = model.config.to_dict()
    if (config['num_hidden_layers'], config['hidden_size'], config['vocab_size']) != (28, 1024, 151936):
        raise ValueError('Full pretrained Qwen architecture differs from the benchmark')
    tokenizer = AutoTokenizer.from_pretrained(args.snapshot, local_files_only=True)
    weights = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    plan = planning.Plan('qwen', exact=True)
    examples = []
    for prompt in PROMPTS:
        tokens = tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
        with torch.no_grad():
            expected = model(**tokens, use_cache=False).logits[:, -1].numpy()
        embeddings = weights['model.embed_tokens.weight'][tokens['input_ids'].numpy()]
        actual = planning.qwen_forward(config, weights, embeddings, plan)
        oracle = reference.forward(config, weights, embeddings)
        error = float(np.max(np.abs(actual - expected)))
        nrmse = float(np.sqrt(np.mean((actual - expected)**2)) / np.sqrt(np.mean(expected**2)))
        oracle_error = float(np.max(np.abs(actual - oracle)))
        top1 = bool(np.array_equal(np.argmax(actual, -1), np.argmax(expected, -1)))
        # Upstream Qwen explicitly uses float32 for some norm/RoPE internals
        # even with double model weights; this is not a bit-exact comparison.
        passed = error <= 1e-3 and nrmse <= 1e-5 and oracle_error <= 1e-8 and top1
        examples.append(dict(prompt=prompt, token_ids=tokens['input_ids'].tolist(),
            sequence_length=embeddings.shape[1], all_final_logits=actual.shape[1],
            max_abs_error_vs_transformers=error, nrmse_vs_transformers=nrmse,
            max_abs_error_vs_numpy_oracle=oracle_error, top1_agreement=top1,
            numerical_gates_passed=passed))
        print(json.dumps(examples[-1]), flush=True)
    after = {str(path.relative_to(ROOT.parent)): digest(path) for path in sources}
    report = dict(scope=__doc__, model_id=provenance['repo_id'], revision=provenance['revision'],
        asset_sha256=hashes, layers=28, examples=examples, loading_info=loading,
        weights_loaded_without_missing_or_mismatched_tensors=True,
        semantic_match=all(example['numerical_gates_passed'] for example in examples),
        semantic_gates=dict(transformers_max_abs=1e-3, transformers_nrmse=1e-5,
                            numpy_oracle_max_abs=1e-8, exact_top1=True),
        toy_contracts=plan.contracts, observed_ranges=plan.ranges,
        observed_toy_domain_violations=plan.violations,
        toy_domains_cover_observed_intermediates=not plan.violations,
        approximate_full_forward_evaluated=False, encrypted_inference_evaluated=False,
        limitation='Observed examples are not certified bounds. Toy-polynomial gates cannot establish pretrained Qwen readiness.',
        elapsed_s=time.perf_counter()-started,
        versions={name: importlib.metadata.version(name) for name in
                  ('numpy', 'torch', 'transformers', 'safetensors', 'tokenizers')},
        source_hashes_before=before, source_hashes_after=after, source_files_unchanged=before == after)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: report[name] for name in ('semantic_match', 'toy_domains_cover_observed_intermediates',
        'observed_toy_domain_violations', 'source_files_unchanged', 'elapsed_s')}, indent=2))
    return 0 if report['semantic_match'] and before == after else 1


if __name__ == '__main__':
    raise SystemExit(main())
