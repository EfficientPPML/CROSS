"""Exact upstream size0 input preparation; tokenizer only, no model execution.

The selected upstream function bodies are compiled unchanged. Only the dataset
loader and tokenizer locator are replaced with authenticated local assets.
"""
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
import ast
import hashlib
import importlib.metadata
import inspect
import json
import os
import random
import signal
import stat
import sys
import tarfile
import tempfile
import time

from owned_directory import open_owned_directory, check_owned_directory, remove_owned_directory

ROOT = Path(__file__).resolve().parent
MODEL = Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
DATA = Path('/home/jianming_gatech/cross_validation_assets/mrpc/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/validation.parquet')
DATA_SHA = '33c007dbf5bfa8463d87a13e6226df8c0fcf2596c2cd39d0f3bb79754e00f50f'
ASSETS = {
    'config.json': 'ec2b6cd6ce741dda6c972aea96f9d1a449cbb4b5349e688c5370a9b2f6872d30',
    'model.safetensors': 'e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034',
    'tokenizer.json': '5b3360be30cdf39081153cf9d30bfb0503c011462fc9f77f097127c83ef0f4d9',
    'tokenizer_config.json': '0f6d13e6f4da6f9e24f22ada6bc3be571123d858d7c0c05a8a7cd55a9c23c2e8',
    'vocab.txt': 'eeaa9875b23b04b4c54ef759d03db9d1ba1554838f8fb26c5d96fa551df93d02',
}


def sha(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('regular input/source file required')
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('xb') as stream:
        stream.write(raw)
    tmp.replace(path)


def compile_function(source, name, namespace):
    tree = ast.parse(source.read_text(), filename=str(source))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name]
    if len(nodes) != 1:
        raise ValueError('unique upstream function required')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
    return namespace[name]


def upstream_seed(seed, source=None):
    """Execute the exact two seed assignments, asserting their loop placement."""
    import numpy as np
    source = source or ROOT / 'upstream/harness/run_submission.py'
    tree = ast.parse(source.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    loops = [n for n in main.body if isinstance(n, ast.For) and ast.unparse(n.target) == 'run']
    if len(loops) != 1:
        raise ValueError('upstream repeated-run loop differs')
    blocks = [n for n in loops[0].body if isinstance(n, ast.If) and ast.unparse(n.test) == 'seed is not None']
    if len(blocks) != 1:
        raise ValueError('upstream seed block differs')
    nodes = blocks[0].body[:2]
    expected = ('rng = np.random.default_rng(seed)', 'genqry_seed = rng.integers(0, 2147483647)')
    if tuple(ast.unparse(n) for n in nodes) != expected:
        raise ValueError('upstream seed derivation changed')
    ns = {'np': np, 'seed': seed}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    return int(ns['genqry_seed'])


def instance_params(workspace):
    ns = {'__name__': '_isolated_upstream_params'}
    path = ROOT / 'upstream/submissions/params.py'
    exec(compile(path.read_bytes(), str(path), 'exec'), ns)
    return ns['InstanceParams'](0, rootdir=workspace, dataset='mrpc')


def export(rows, samples, labels, count, seed):
    loader_calls = []
    def local_loader(data_dir):
        loader_calls.append(data_dir)
        return None, rows, None
    ns = {'DATA_DIR': 'authenticated-local-parquet', 'load_and_preprocess_data': local_loader,
          'json': json, 'random': random}
    fn = compile_function(ROOT / 'upstream/harness/mrpc/mrpc.py', 'export_test_sentence_pairs', ns)
    fn(samples_file=samples, labels_file=labels, num_samples=count, seed=seed)
    if len(loader_calls) != 1:
        raise ValueError('upstream exporter dataset reads changed')


def preprocess(workspace, tokenizer):
    calls = []
    class LocalTokenizer:
        @staticmethod
        def from_pretrained(name):
            if name != 'google-bert/bert-base-cased-finetuned-mrpc':
                raise ValueError('upstream tokenizer model identity changed')
            calls.append(name)
            return tokenizer
    ns = {'json': json, 'sys': SimpleNamespace(argv=['client_preprocess_input.py', '0'], stderr=sys.stderr),
          'InstanceParams': lambda size, dataset: instance_params(workspace) if size == 0 and dataset == 'mrpc' else None,
          'AutoTokenizer': LocalTokenizer, 'MODEL_ID': 'google-bert/bert-base-cased-finetuned-mrpc',
          'MAX_LENGTH': 128, 'print': lambda *args, **kwargs: None}
    compile_function(ROOT / 'upstream/submissions/client_preprocess_input.py', 'main', ns)()
    if len(calls) != 1:
        raise ValueError('exactly one tokenizer lookup required')


@contextmanager
def no_model_execution():
    import torch
    from transformers import PreTrainedModel
    old_init = PreTrainedModel.__init__
    old_call = torch.nn.Module.__call__
    def forbidden(*args, **kwargs):
        raise RuntimeError('model construction/forward forbidden in tokenizer-only fixture')
    PreTrainedModel.__init__ = forbidden
    torch.nn.Module.__call__ = forbidden
    try:
        yield
    finally:
        PreTrainedModel.__init__ = old_init
        torch.nn.Module.__call__ = old_call


def validate_tokens(record):
    if set(record) != {'target_idx', 'input_ids', 'token_type_ids', 'attention_mask'}:
        raise ValueError('upstream preprocessing schema differs')
    if type(record['target_idx']) is not int or record['target_idx'] < 0:
        raise ValueError('target ID must be a nonnegative integer')
    for name, upper in (('input_ids', 28996), ('token_type_ids', 2), ('attention_mask', 2)):
        a = record[name]
        if type(a) is not list or len(a) != 128 or any(type(x) is not int or not 0 <= x < upper for x in a):
            raise ValueError('invalid fixed128 token field')
    if record['input_ids'][0] != 101 or record['attention_mask'][0] != 1:
        raise ValueError('original CLS must be present')


def prepare(workspace):
    import numpy as np
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    if sha(DATA) != DATA_SHA or {k: sha(MODEL/k) for k in ASSETS} != ASSETS:
        raise ValueError('pinned dataset/model/tokenizer asset mismatch')
    table = pq.read_table(DATA)
    if table.num_rows != 408 or set(table.column_names) != {'sentence1', 'sentence2', 'label', 'idx'}:
        raise ValueError('pinned MRPC validation schema differs')
    rows = table.to_pylist()
    if any(type(r['idx']) is not int or type(r['label']) is not int or r['label'] not in (0, 1)
           or type(r['sentence1']) is not str or type(r['sentence2']) is not str for r in rows):
        raise ValueError('MRPC row types differ')
    if len({r['idx'] for r in rows}) != 408:
        raise ValueError('MRPC target IDs are not unique')
    params = instance_params(workspace)
    if params.get_batch_size() != 1:
        raise ValueError('upstream size0 must select one record')
    params.dataset_intermediate_dir().mkdir(parents=True)
    export(rows, params.datadir()/'dataset_samples.jsonl', params.datadir()/'dataset_labels.txt', -1, None)
    seed = upstream_seed(42)
    export(rows, params.get_test_input_file(), params.get_ground_truth_labels_file(), 1, seed)
    # Independent selection checks preserve MRPC row ordinal versus target_idx.
    independent_seed = int(np.random.Generator(np.random.PCG64(42)).integers(0, 2147483647))
    chosen = sorted(random.Random(independent_seed).sample(tuple(range(408)), 1))
    selected = [rows[i] for i in chosen]
    expected_raw = ''.join(json.dumps({k: r[k] for k in ('target_idx', 'sentence1', 'sentence2')})+'\n'
                           for r in ({'target_idx':r['idx'], **r} for r in selected))
    expected_labels = ''.join(str(r['label'])+'\n' for r in selected)
    if seed != independent_seed or params.get_test_input_file().read_text() != expected_raw or params.get_ground_truth_labels_file().read_text() != expected_labels:
        raise ValueError('upstream selection/order/target identity mismatch')
    with no_model_execution():
        tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
        if type(tokenizer).__name__ != 'BertTokenizerFast' or not tokenizer.is_fast:
            raise ValueError('pinned tokenizer implementation differs')
        preprocess(workspace, tokenizer)
        expected = []
        for r in selected:
            encoded = tokenizer.encode_plus(r['sentence1'], r['sentence2'], max_length=128,
                                            padding='max_length', truncation=True)
            expected.append({'target_idx': r['idx'], **{k: encoded[k] for k in ('input_ids', 'token_type_ids', 'attention_mask')}})
    output = params.io_intermediate_dir()/'client_preprocessed_input'
    if output.read_text() != ''.join(json.dumps(r)+'\n' for r in expected):
        raise ValueError('upstream preprocessing differs from independent tokenization')
    for row in expected:
        validate_tokens(row)
    private_selection = {'schema': 'private-bert-size0-selection-v1', 'size': 0, 'base_seed': 42,
                         'generator_seed': seed, 'row_ordinals': chosen,
                         'target_ids': [r['idx'] for r in selected], 'labels': [r['label'] for r in selected]}
    write_json(workspace/'private_selection.json', private_selection)
    files = sorted(p for p in workspace.rglob('*') if p.is_file())
    for path in files:
        path.chmod(0o600)
    return {'generator_seed': seed, 'record_count': 1, 'dataset_record_count': 408,
            'preprocessed_input_path': str(output), 'input_file_sha256': {str(p.relative_to(workspace)): sha(p) for p in files},
            'private_target_order_sha256': hashlib.sha256(json.dumps(private_selection['target_ids']).encode()).hexdigest(),
            'token_lengths': [len(r['input_ids']) for r in expected],
            'valid_token_counts': [sum(r['attention_mask']) for r in expected],
            'selection_byte_parity': True, 'tokenization_byte_parity': True,
            'model_construction_and_forward_guards': True}


def source_paths():
    paths = {str(p.relative_to(ROOT)): p for p in ROOT.rglob('*') if p.is_file()
             and ('__pycache__' not in p.parts) and (p.suffix == '.py' or p.name in ('model_manifest.json', 'dataset_manifest.json', 'upstream_origin.json'))}
    # Library entry points plus their native binary hashes are an explicit
    # implementation inventory, not a claim to hermetic dependency closure.
    import numpy, pyarrow.parquet, transformers.tokenization_utils_base
    import transformers.tokenization_utils_fast, transformers.tokenization_utils
    import transformers.models.auto.tokenization_auto, transformers.models.bert.tokenization_bert
    import transformers.models.bert.tokenization_bert_fast
    modules = [random, numpy, pyarrow.parquet, transformers.tokenization_utils_base,
               transformers.tokenization_utils_fast, transformers.tokenization_utils,
               transformers.models.auto.tokenization_auto, transformers.models.bert.tokenization_bert,
               transformers.models.bert.tokenization_bert_fast]
    for module in modules:
        paths['libraries/'+module.__name__.replace('.', '/')+'.py'] = Path(module.__file__)
    return paths


def archive_sources(paths, target):
    with tarfile.open(target, 'w:gz') as archive:
        for name, path in sorted(paths.items()):
            archive.add(path, arcname=name, recursive=False)
    expected = {k: sha(p) for k, p in paths.items()}
    with tarfile.open(target, 'r:gz') as archive:
        members = archive.getmembers()
        if len(members) != len(expected) or {m.name for m in members} != set(expected):
            raise ValueError('source archive inventory mismatch')
        for member in members:
            if not member.isfile() or hashlib.sha256(archive.extractfile(member).read()).hexdigest() != expected[member.name]:
                raise ValueError('source archive bytes mismatch')
    return expected


def collect(output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = {'schema': 'bert-upstream-size0-input-fixture-v1', 'passed': False, 'size': 0, 'base_seed': 42,
              'scope': 'client-only dataset selection and tokenizer; no embeddings, encoder, inference, keys or HE',
              'private_workspace_retained': False, 'cleanup_errors': []}
    paths = {}; descriptor = None; workspace = None; identity = None
    started = time.monotonic(); previous_umask = os.umask(0o077)
    write_json(output/'report.json', report)
    try:
        paths = source_paths()
        report['source_hashes_before'] = archive_sources(paths, output/'sources.tar.gz')
        report['source_archive_sha256'] = sha(output/'sources.tar.gz')
        origin = json.loads((ROOT/'upstream_origin.json').read_text())
        report['upstream_original_sha256_before'] = {k: sha(v['original']) for k, v in origin.items()}
        if report['upstream_original_sha256_before'] != {k: v['sha256'] for k, v in origin.items()}:
            raise ValueError('upstream originals differ from frozen copy')
        report['asset_file_sha256_before'] = {str(MODEL/k): sha(MODEL/k) for k in ASSETS} | {str(DATA): sha(DATA)}
        workspace = Path(tempfile.mkdtemp(prefix='bert-size0-input-client-'))
        descriptor, identity = open_owned_directory(workspace)
        report['private_workspace'] = str(workspace); report['private_workspace_identity'] = identity
        report.update(prepare(workspace))
        check_owned_directory(workspace, descriptor, identity)
        report['private_workspace_permissions'] = oct(stat.S_IMODE(workspace.stat().st_mode))
        report['private_file_permissions_all_0600'] = all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in workspace.rglob('*') if p.is_file())
        report['versions'] = {k: importlib.metadata.version(k) for k in ('numpy', 'pyarrow', 'transformers', 'tokenizers', 'torch')}
        report['python'] = sys.version
        report['cpu_affinity'] = sorted(os.sched_getaffinity(0))
        report['passed'] = report['private_file_permissions_all_0600']
    except BaseException as error:
        report['error'] = {'type': type(error).__name__, 'message': str(error)}
        raise
    finally:
        primary = sys.exception(); errors = []
        def attempt(name, action):
            try: return action()
            except BaseException as error:
                errors.append(error); report['cleanup_errors'].append(name+': '+repr(error)); report['passed'] = False
                if primary is not None: primary.add_note(name+': '+repr(error))
        # Defer repeated cancellation through cleanup/publication; restore after.
        old = {s: signal.signal(s, signal.SIG_IGN) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            if paths:
                after = attempt('source final hashes', lambda: {k: sha(p) for k, p in paths.items()})
                report['source_hashes_after'] = after
                report['sources_unchanged'] = after == report.get('source_hashes_before') and after is not None
                assets = attempt('asset final hashes', lambda: {str(MODEL/k): sha(MODEL/k) for k in ASSETS} | {str(DATA): sha(DATA)})
                report['asset_file_sha256_after'] = assets
                report['assets_unchanged'] = assets == report.get('asset_file_sha256_before') and assets is not None
                report['passed'] = report['passed'] and report['sources_unchanged'] and report['assets_unchanged']
            if workspace is not None and not report['passed']:
                report['private_workspace_removed_on_failure'] = attempt('private cleanup', lambda: remove_owned_directory(workspace, descriptor, identity)) is True
            if descriptor is not None:
                attempt('owned handle close', lambda: os.close(descriptor))
            report['private_workspace_retained'] = bool(report['passed'])
            report['elapsed_s'] = time.monotonic()-started
            attempt('report publication', lambda: write_json(output/'report.json', report))
        finally:
            os.umask(previous_umask)
            for sig, handler in old.items(): signal.signal(sig, handler)
        if primary is None and errors: raise errors[0]
    if not report['passed']: raise RuntimeError('fixture evidence failed')
    return report


if __name__ == '__main__':
    def stop(signum, frame): raise SystemExit(128+signum)
    for sig in (signal.SIGTERM, signal.SIGHUP): signal.signal(sig, stop)
    result = collect(sys.argv[1])
    print(json.dumps({k: result[k] for k in ('passed', 'size', 'base_seed', 'generator_seed', 'record_count', 'preprocessed_input_path', 'source_archive_sha256')}))
