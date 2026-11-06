"""Optional local readback: print only aggregate checks, never private values."""
from pathlib import Path
import collections, hashlib, json, random, stat

ROOT = Path(__file__).resolve().parent
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    r = json.loads((ROOT / 'report.json').read_bytes())
    assert sha(ROOT / 'report.json') == 'ad1458f5183c6ccf2631c2a8771d0da92196f5f9c690016e2a0b12fa7b88f4d7'
    assert r['passed'] is True and r['terminal_scope_completed'] is True
    workspace = Path(r['private_workspace'])
    s = workspace.stat()
    assert {'device': s.st_dev, 'inode': s.st_ino} == r['private_workspace_identity']
    assert not workspace.is_symlink() and stat.S_IMODE(s.st_mode) == 0o700
    assets = r['asset_file_sha256_before']
    assert {p: sha(Path(p)) for p in assets} == assets
    dataset = next(Path(p) for p in assets if p.endswith('/validation.parquet'))
    model = next(Path(p).parent for p in assets if p.endswith('/tokenizer.json'))
    rows = pq.read_table(dataset).to_pylist()
    assert len(rows) == 408
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    assert type(tokenizer).__name__ == 'BertTokenizerFast' and tokenizer.is_fast
    expected_children = {f"size{c['size']}-seed{c['base_seed']}" for c in r['cases']}
    assert {p.name for p in workspace.iterdir()} == expected_children
    full_inputs, total = [], 0
    for c in r['cases']:
        child = workspace / f"size{c['size']}-seed{c['base_seed']}"
        paths = list(child.rglob('*'))
        assert all(not p.is_symlink() for p in (child, *paths))
        assert all(stat.S_IMODE(p.stat().st_mode) == (0o700 if p.is_dir() else 0o600) for p in (child, *paths))
        assert {p.relative_to(child).as_posix(): sha(p) for p in paths if p.is_file()} == c['input_file_sha256']
        assert len(c['input_file_sha256']) == 6
        seed = int(np.random.Generator(np.random.PCG64(c['base_seed'])).integers(0, 2147483647))
        order = list(range(408)) if c['count'] == 408 else sorted(random.Random(seed).sample(range(408), c['count']))
        selected = [rows[i] for i in order]
        selection = json.loads((child / 'private_selection.json').read_bytes())
        assert selection == dict(schema='private-bert-ranked-selection-v1', size=c['size'], count=c['count'], base_seed=c['base_seed'], generator_seed=seed, row_ordinals=order, target_ids=[x['idx'] for x in selected], labels=[x['label'] for x in selected])
        assert hashlib.sha256(json.dumps(order).encode()).hexdigest() == c['private_row_order_sha256']
        assert hashlib.sha256(json.dumps(selection['target_ids']).encode()).hexdigest() == c['private_target_order_sha256']
        def sentence_bytes(items):
            return ''.join(json.dumps(dict(target_idx=x['idx'], sentence1=x['sentence1'], sentence2=x['sentence2'])) + '\n' for x in items).encode()
        def label_bytes(items):
            return ''.join(str(x['label']) + '\n' for x in items).encode()
        for suffix, expected in (('dataset_samples.jsonl', sentence_bytes(rows)), ('dataset_labels.txt', label_bytes(rows)), ('test_input.txt', sentence_bytes(selected)), ('test_labels.txt', label_bytes(selected))):
            path = next(p for p in paths if p.name == suffix)
            assert path.read_bytes() == expected
        tokens = []
        for row in selected:
            item = tokenizer.encode_plus(row['sentence1'], row['sentence2'], max_length=128, padding='max_length', truncation=True)
            token = dict(target_idx=row['idx'], **{k:item[k] for k in ('input_ids', 'token_type_ids', 'attention_mask')})
            assert all(len(token[k]) == 128 and all(type(v) is int for v in token[k]) for k in ('input_ids', 'token_type_ids', 'attention_mask'))
            assert all(0 <= v < tokenizer.vocab_size for v in token['input_ids'])
            assert all(v in (0, 1) for k in ('token_type_ids', 'attention_mask') for v in token[k])
            tokens.append(token)
        preprocessed = Path(c['preprocessed_input_path'])
        assert preprocessed.is_relative_to(child) and preprocessed.name == 'client_preprocessed_input'
        payload = ''.join(json.dumps(x) + '\n' for x in tokens).encode()
        assert preprocessed.read_bytes() == payload
        histogram = {str(k):v for k,v in collections.Counter(sum(x['attention_mask']) for x in tokens).items()}
        assert histogram == c['valid_token_count_histogram']
        assert len(tokens) == c['record_count'] == c['count']
        if c['count'] == 408:
            full_inputs.append(hashlib.sha256(payload).hexdigest())
        total += len(tokens)
    assert len(full_inputs) == 3 and len(set(full_inputs)) == 1 and total == 2124
    assert {'device': workspace.stat().st_dev, 'inode': workspace.stat().st_ino} == r['private_workspace_identity']
    print(json.dumps(dict(passed=True, cases=9, selected_records=total, private_files=54, full408_inputs_identical=True, selection_and_tokenization_byte_parity=True, private_permissions_verified=True, model_forward=False, encrypted_execution=False), sort_keys=True))

if __name__ == '__main__':
    main()
