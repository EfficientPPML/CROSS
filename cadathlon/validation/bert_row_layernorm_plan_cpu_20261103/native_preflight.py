"""Native-only fixed layer0 LayerNorm inputs; no encrypted or model-quality claim."""
from pathlib import Path
import hashlib
import importlib.metadata
import io
import json
import os
import sys
import tarfile
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/home/jianming_gatech/CROSS')
sys.path.insert(0, str(ROOT/'cadathlon/common'))
from encrypted_polynomial import PolynomialApproximation
from support import load_module


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def describe(array):
    array = np.asarray(array, dtype='<f8')
    return dict(shape=list(array.shape), minimum=float(array.min()), maximum=float(array.max()),
                sha256=hashlib.sha256(array.tobytes()).hexdigest())


def main():
    started = time.perf_counter()
    plan = json.loads((HERE/'report.json').read_text())
    source_paths = [Path(__file__), HERE/'report.json',
        ROOT/'cadathlon/validation/check_pretrained_bert_domains.py',
        ROOT/'cadathlon/common/encrypted_polynomial.py', ROOT/'cadathlon/common/support.py']
    source_data = {str(path): path.read_bytes() for path in source_paths}
    before = {name: hashlib.sha256(data).hexdigest() for name, data in source_data.items()}
    with tarfile.open(HERE/'native_sources.tar.gz', 'w:gz') as archive:
        for name, data in source_data.items():
            info = tarfile.TarInfo(name.lstrip('/'))
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    audit = load_module(ROOT/'cadathlon/validation/check_pretrained_bert_domains.py')
    snapshot = Path('/home/jianming_gatech/cross_validation_assets/bert')/plan['model_revision']
    assets = {name: sha(snapshot/name) for name in audit.EXPECTED_ASSETS}
    assert assets == audit.EXPECTED_ASSETS
    import torch
    from transformers import AutoTokenizer, BertForSequenceClassification
    torch.set_num_threads(1)
    network, loading = BertForSequenceClassification.from_pretrained(snapshot, local_files_only=True,
        use_safetensors=True, torch_dtype=torch.float64, attn_implementation='eager', output_loading_info=True)
    assert not loading['missing_keys'] and not loading.get('mismatched_keys') and not loading.get('error_msgs')
    assert set(loading['unexpected_keys']) <= {'bert.embeddings.position_ids'}
    network.eval()
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokens = tokenizer(*audit.EXAMPLES[0], max_length=128, padding='max_length', truncation=True, return_tensors='pt')
    token_bytes = json.dumps({name: value.tolist() for name, value in tokens.items()}, sort_keys=True, separators=(',', ':')).encode()
    layer = network.bert.encoder.layer[0]
    with torch.no_grad():
        embedding = network.bert.embeddings(input_ids=tokens['input_ids'], token_type_ids=tokens['token_type_ids'])
        mask = tokens['attention_mask'].bool()
        additive = torch.zeros(mask[:, None, None, :].shape, dtype=torch.float64).masked_fill(
            ~mask[:, None, None, :], torch.finfo(torch.float64).min)
        context = layer.attention.self(embedding, attention_mask=additive)[0]
        attention_input = layer.attention.output.dense(context) + embedding
        attention_output = layer.attention.output.LayerNorm(attention_input)
        assert torch.equal(attention_output, layer.attention(embedding, attention_mask=additive)[0])
        mlp_input = layer.output.dense(layer.intermediate(attention_output)) + attention_output
        mlp_output = layer.output.LayerNorm(mlp_input)
    records = []
    for phase, raw_input, native_output, module in [
        ('attention', attention_input, attention_output, layer.attention.output.LayerNorm),
        ('mlp', mlp_input, mlp_output, layer.output.LayerNorm)]:
        rows = raw_input[0, :2].numpy().copy()
        native = native_output[0, :2].numpy().copy()
        assert rows.shape == (2, 768) and not np.array_equal(rows[0], rows[1])
        rule = next(case for case in plan['cases'] if case['phase'] == phase and case['valid_rows'] == 2)
        contract = PolynomialApproximation('inverse_sqrt', tuple(rule['rule']['domain']), tuple(rule['coefficients']))
        gamma = module.weight.detach().numpy()
        beta = module.bias.detach().numpy()
        for name, value in [('gamma', gamma), ('beta', beta)]:
            assert hashlib.sha256(value.astype('<f4').tobytes()).hexdigest() == plan['layernorm_tensor_sha256'][phase][name]
        mean = rows.mean(axis=1, keepdims=True)
        centered = rows - mean
        variance = (centered * centered).mean(axis=1, keepdims=True) + module.eps
        exact = centered / np.sqrt(variance) * gamma + beta
        approximate = centered * contract.evaluate_public(variance) * gamma + beta
        native_error = float(np.max(np.abs(exact-native)))
        assert native_error < 1e-12
        records.append(dict(phase=phase, fixed_row_positions=[0,1], input=describe(rows),
            mean=describe(mean), variance=describe(variance), output=describe(native),
            fixed_inverse_domain=list(contract.domain), domain_violations=0,
            exact_numpy_vs_native_max_abs=native_error,
            polynomial_vs_native_max_abs=float(np.max(np.abs(approximate-native))),
            polynomial_vs_native_nrmse=float(np.sqrt(np.mean((approximate-native)**2))/max(.01,np.sqrt(np.mean(native**2)))),
            epsilon=module.eps, gamma=describe(gamma), beta=describe(beta)))
    after = {str(path): sha(path) for path in source_paths}
    assert before == after
    report = dict(scope=__doc__, passed=True, source_hashes_before=before, source_hashes_after=after,
        source_files_unchanged=True, assets=assets, pair=list(audit.EXAMPLES[0]),
        tokenization_sha256=hashlib.sha256(token_bytes).hexdigest(),
        native_preprocessing_boundary='Computes actual attention/MLP to obtain two fixed LayerNorm inputs; this exceeds embedding-only P2 client preprocessing and is only an isolated component fixture.',
        records=records, future_he_absolute_gate=1e-6,
        native_model_quality_gate_evaluated=False, cpu_affinity=sorted(os.sched_getaffinity(0)),
        elapsed_s=time.perf_counter()-started,
        versions={name:importlib.metadata.version(name) for name in ('numpy','torch','transformers','tokenizers','safetensors')})
    (HERE/'native_preflight.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'records':records,'elapsed_s':report['elapsed_s']}, indent=2))


if __name__ == '__main__':
    main()
