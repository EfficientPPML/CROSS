"""Synthetic public payload and private receipt readbacks; no retained data.

The functions under test are imported directly. Codec/model modules are tiny
explicit doubles; receipt observations use the copied production definitions
on newly owned, deliberately non-ciphertext files only.
"""
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import copy
import hashlib
import json
import os
import stat
import sys

import numpy as np
import pytest

import separate_public_readback as public
import separate_client_readback as private

HERE = Path(__file__).resolve().parent


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       allow_nan=False) + '\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def module(monkeypatch, name, **attributes):
    value = ModuleType(name)
    value.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, value)
    return value


@pytest.fixture
def public_case(tmp_path, monkeypatch):
    events = []
    paths = dict(upload=tmp_path / 'upload', download=tmp_path / 'response',
                 server_report=tmp_path / 'server_report')
    request = dict(request_id='a' * 32, contract={'request_id': 'a' * 32})
    batch = dict(public={'context_id': 'b' * 64}, requests=[request])
    manifest = dict(synthetic='transport-manifest')
    response = dict(responses=[dict(request_id=request['request_id'], scale=1.25,
                                    response_manifest_sha256=digest(manifest))])
    requests = dict(public_batch_sha256=digest(batch), input_digest='c' * 64)
    server = dict(response_batch_sha256=digest(response),
                  response_manifest_sha256=digest(manifest))
    state = SimpleNamespace(manifest=manifest, binding={'level': 2},
                            array=np.zeros((1, 2, 2), dtype=np.uint32), error=None)

    def forbidden(*args, **kwargs):
        raise AssertionError('private input/reference/key/decode accessed')

    def read_manifest(path):
        events.append(('public-manifest', Path(path)))
        assert Path(path) in (paths['upload'] / 'batch.json',
                              paths['download'] / 'batch.json')
        return batch if Path(path).parent == paths['upload'] else response

    scale_checker = object()

    def check_public(value, *, public, count):
        assert value is batch and public is batch['public'] and count == 1
        events.append(('public-contract',))
        return value

    def check_response(value, original, *, check_scale):
        assert original is batch and check_scale is scale_checker
        events.append(('response-contract',))
        return value

    def validate_ciphertext(path, **kwargs):
        events.append(('public-payload', Path(path), kwargs))
        assert Path(path) == paths['download'] / request['request_id']
        assert kwargs == dict(contract=request['contract'], role='response',
                              tile_index=0, expected_scale=response['responses'][0]['scale'],
                              input_digest=requests['input_digest'])
        if state.error is not None:
            raise state.error
        return state.array, response['responses'][0]['scale'], state.binding, state.manifest

    module(monkeypatch, 'file_stage_io', read_manifest=read_manifest)
    module(monkeypatch, 'file_stages', _inputs=lambda: ({}, b'candidate'),
           _scale_checker=lambda *_: scale_checker, _records=forbidden,
           client_decrypt_decode=forbidden)
    module(monkeypatch, 'file_stage_contract', validate_public_batch=check_public,
           validate_response_batch=check_response)
    module(monkeypatch, 'transport_contract', digest=digest)
    module(monkeypatch, 'transport', validate_ciphertext=validate_ciphertext,
           manifest_digest=digest, load_ciphertext=forbidden)
    module(monkeypatch, 'key_contexts', load_client=forbidden, load_server=forbidden)
    module(monkeypatch, 'reference_gate', load_private=forbidden)
    return SimpleNamespace(paths=paths, requests=requests, server=server, batch=batch,
                           response=response, state=state, events=events)


def run_public(case):
    return public.verify(case.paths, server=case.server, requests=case.requests)


def test_actual_public_readback_authenticates_bytes_without_private_access(public_case):
    c = public_case
    out = run_public(c)
    assert out['passed'] is True and out['private_data_read'] is False
    assert out['payload_bytes'] == c.state.array.nbytes
    assert out['shape'] == list(c.state.array.shape) and out['level'] == 2
    assert out['response_manifest_sha256'] == c.server['response_manifest_sha256']
    assert [e[0] for e in c.events] == ['public-manifest', 'public-contract',
                                       'public-manifest', 'response-contract', 'public-payload']


@pytest.mark.parametrize('field', ['request', 'response'])
def test_public_batch_digest_rejects_before_payload(public_case, field):
    c = public_case
    (c.requests if field == 'request' else c.server)[
        'public_batch_sha256' if field == 'request' else 'response_batch_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='batch changed'):
        run_public(c)
    assert all(e[0] != 'public-payload' for e in c.events)


@pytest.mark.parametrize('kind', ['row-manifest', 'server-manifest', 'payload-manifest', 'level'])
def test_public_payload_manifest_and_level_binding(public_case, kind):
    c = public_case
    if kind == 'row-manifest':
        c.response['responses'][0]['response_manifest_sha256'] = '0' * 64
        c.server['response_batch_sha256'] = digest(c.response)
    elif kind == 'server-manifest':
        c.server['response_manifest_sha256'] = '0' * 64
    elif kind == 'payload-manifest':
        c.state.manifest = dict(synthetic='replacement')
    else:
        c.state.binding = {'level': 3}
    with pytest.raises(ValueError):
        run_public(c)
    assert sum(e[0] == 'public-payload' for e in c.events) == 1


@pytest.mark.parametrize('kind', [ValueError, KeyboardInterrupt, SystemExit])
def test_public_payload_failure_identity_preserved(public_case, kind):
    c = public_case
    c.state.error = kind('synthetic payload rejection')
    with pytest.raises(kind) as got:
        run_public(c)
    assert got.value is c.state.error


def lifecycle_double(monkeypatch):
    """Actual receipt-path/observation AST, synthetic local file bodies only."""
    names = {'_paths', '_identity', '_file_state', '_tree', '_observe', 'completion_path'}
    tree = ast.parse((HERE / 'runtime/stage/request_lifecycle.py').read_bytes())
    selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in selected} == names

    def read_at(fd, name, maximum):
        child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=fd)
        try:
            raw = os.read(child, maximum + 1)
            assert len(raw) <= maximum
            return raw
        finally:
            os.close(child)

    m = module(monkeypatch, 'request_lifecycle', Path=Path, stat=stat, os=os,
               hashlib=hashlib, _read=read_at)
    # Execute the actual shared cleanup-origin class without importing runtime.
    keys_tree = ast.parse((HERE / 'runtime/stage/client_keys.py').read_bytes())
    close_state = [n for n in keys_tree.body if isinstance(n, ast.ClassDef) and n.name == '_CloseState']
    m.sys = sys; m._IO_FAILURE_QUARANTINE = []
    exec(compile(ast.Module(body=close_state, type_ignores=[]), '<actual cleanup state>', 'exec'), m.__dict__)
    exec(compile(ast.Module(body=selected, type_ignores=[]),
                 '<copied production receipt observations>', 'exec'), m.__dict__)
    module(monkeypatch, 'transport_contract', digest=digest)
    return m


@pytest.fixture
def private_case(tmp_path, monkeypatch):
    lifecycle = lifecycle_double(monkeypatch)
    nonce = 'a' * 32
    owned = tmp_path / 'owned-result'
    owned.mkdir(mode=0o700)
    original = tmp_path / 'original-client-private'
    original.mkdir(mode=0o700)
    paths = dict(client_private=original, upload=tmp_path / 'upload',
                 download=tmp_path / 'separate-download', server_report=tmp_path / 'server_report',
                 decrypted_results=owned / 'decrypted_results')
    batch = dict(public={'context_id': 'b' * 64}, requests=[dict(contract={'request_id': nonce})])

    def put(path, raw):
        path.write_bytes(raw)
        path.chmod(0o600)

    put(original / 'requests.json', b'{"synthetic_original_request":true}\n')
    put(paths['server_report'], b'{"synthetic_public_server":true}\n')
    for kind in ('upload', 'download'):
        root = paths[kind]
        root.mkdir(mode=0o700)
        put(root / 'batch.json', canonical(batch))
        sample = root / nonce
        sample.mkdir(mode=0o700)
        leaves = [sample]
        if kind == 'upload':
            leaves = [sample / ('tile' + str(i)) for i in range(8)]
            for leaf in leaves:
                leaf.mkdir(mode=0o700)
        for leaf in leaves:
            for name in ('bundle.json', 'ct_0.npz', 'transport.json'):
                put(leaf / name, b'synthetic non-ciphertext fixture\n')
    ref = dict(target_idx=17, native_fp32_logits=[.25, .75], approximate_fp64_logits=[.2501, .7499])
    receipt_path = owned / 'completed.json'
    c = SimpleNamespace(paths=paths, owned=owned, original=original, batch=batch,
                        reference=ref, lifecycle=lifecycle, put=put, receipt_path=receipt_path)

    def publish(logits=(.25, .75), target=17):
        put(paths['decrypted_results'], canonical(dict(target_idx=target, he_logits=list(logits))))
        objects = {kind: lifecycle._observe(path, kind, [nonce])
                   for kind, path in lifecycle._paths(paths).items()}
        receipt = dict(schema='cross-bert-client-completion-v1', context_id=batch['public']['context_id'],
                       public_batch_sha256=digest(batch), nonces=[nonce], objects=objects)
        put(receipt_path, canonical(receipt))
        c.receipt = receipt

    c.publish = publish
    publish()
    c.original_raw = (original / 'requests.json').read_bytes()
    c.original_state = (original / 'requests.json').stat()
    return c


def run_private(c):
    return private.validate(c.paths, request_batch=c.batch, reference=c.reference,
                            completion_directory=c.owned)


def test_actual_private_readback_uses_isolated_receipt_and_preserves_original(private_case):
    c = private_case
    out = run_private(c)
    assert out['metrics']['passed'] is True
    assert out['private_receipt_validated'] and out['isolated_completion_receipt']
    assert out['retirement_performed'] is False and out['result_permissions_0600']
    assert out['decoded_result_sha256'] == hashlib.sha256(c.paths['decrypted_results'].read_bytes()).hexdigest()
    assert out['completed_receipt_sha256'] == hashlib.sha256(c.receipt_path.read_bytes()).hexdigest()
    assert (c.original / 'requests.json').read_bytes() == c.original_raw
    after = (c.original / 'requests.json').stat()
    assert tuple(getattr(after, n) for n in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')) == tuple(
        getattr(c.original_state, n) for n in ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns'))
    assert not (c.original / 'completed.json').exists()
    assert not {'he_logits', 'native_fp32_logits', 'approximate_fp64_logits', 'target_idx'} & set(out)


@pytest.mark.parametrize('kind', ['context', 'nonce', 'batch', 'extra', 'objects'])
def test_isolated_receipt_binding_rejects_changes(private_case, kind):
    c = private_case
    receipt = copy.deepcopy(c.receipt)
    if kind == 'context':
        receipt['context_id'] = '0' * 64
    elif kind == 'nonce':
        receipt['nonces'] = ['0' * 32]
    elif kind == 'batch':
        receipt['public_batch_sha256'] = '0' * 64
    elif kind == 'extra':
        receipt['unbound'] = True
    else:
        receipt['objects']['client_requests']['sha256'] = '0' * 64
    c.put(c.receipt_path, canonical(receipt))
    with pytest.raises(ValueError, match='receipt differs'):
        run_private(c)


@pytest.mark.parametrize('kind', ['request', 'upload-child', 'response-child', 'server-report'])
def test_receipt_detects_retained_object_change(private_case, kind):
    c = private_case
    nonce = c.batch['requests'][0]['contract']['request_id']
    changed = {'request': c.original / 'requests.json',
               'upload-child': c.paths['upload'] / nonce / 'tile0/ct_0.npz',
               'response-child': c.paths['download'] / nonce / 'ct_0.npz',
               'server-report': c.paths['server_report']}[kind]
    c.put(changed, b'new synthetic replacement with different size\n')
    with pytest.raises(ValueError, match='receipt differs'):
        run_private(c)


def test_original_ledger_cannot_substitute_for_isolated_receipt(private_case):
    c = private_case
    (c.original / 'completed.json').write_bytes(c.receipt_path.read_bytes())
    c.receipt_path.unlink()
    with pytest.raises(FileNotFoundError):
        run_private(c)
    assert (c.original / 'completed.json').exists()


@pytest.mark.parametrize('kind', ['result-mode', 'receipt-mode', 'target', 'target-bool', 'extra-row'])
def test_private_result_permissions_and_target_order(private_case, kind):
    c = private_case
    if kind == 'result-mode':
        c.paths['decrypted_results'].chmod(0o644)
    elif kind == 'receipt-mode':
        c.receipt_path.chmod(0o644)
    elif kind == 'target':
        c.publish(target=18)
    elif kind == 'target-bool':
        c.publish(target=True)
    else:
        raw=c.paths['decrypted_results'].read_bytes()
        c.put(c.paths['decrypted_results'], raw + raw)
    with pytest.raises(ValueError):
        run_private(c)


@pytest.mark.parametrize('kind', ['native-gate', 'prediction', 'approx-diagnostic-only'])
def test_original_numeric_failure_preserved_with_valid_receipt(private_case, kind):
    c = private_case
    if kind == 'native-gate':
        c.publish(logits=(.5, .75))
    elif kind == 'prediction':
        c.reference['native_fp32_logits'] = [.25, .26]
        c.publish(logits=(.27, .25))
    else:
        c.reference['approximate_fp64_logits'] = [50., -70.]
    out = run_private(c)
    assert out['private_receipt_validated'] and out['isolated_completion_receipt']
    assert out['metrics']['passed'] is (kind == 'approx-diagnostic-only')
    if kind == 'native-gate':
        assert out['metrics']['original_P2_logit_gate'] is False
        assert out['metrics']['prediction_agreement'] is True
    elif kind == 'prediction':
        assert out['metrics']['original_P2_logit_gate'] is True
        assert out['metrics']['prediction_agreement'] is False
    else:
        assert out['metrics']['he_vs_frozen_approx_max_abs'] > 50.
    assert c.receipt_path.exists() and c.paths['decrypted_results'].exists()
    assert (c.original / 'requests.json').read_bytes() == c.original_raw
