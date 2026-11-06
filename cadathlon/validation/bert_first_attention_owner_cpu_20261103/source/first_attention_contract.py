"""Pinned public first-attention schema; no codec, filesystem transport, or HE.

The schema declares the client's original-position/compaction obligation. It cannot
prove that encrypted slots obey that obligation. Output scales are supplied from
an independently checked public operation plan, never relabeled to canonical.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent
INPUT_SHA256 = {
    'candidate_manifest.json': '4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9',
    'p60_profile_and_refresh.json': 'fe2408637f9d6de19ac6aef9f6f749aaef9dd56ccbb37b5334293cd223ede035',
    'verified_model_manifest.json': 'b7a39739f0eb11092de7a78e9d0bc3ae18603338aa402e85dd4dd55086d24c79',
    'matrix_inventory.json': 'd087eb087bf521faaf01f62719c9f35e6afce2889f8057f688c320eebdc30292',
}
TENSOR_SHA256 = {
    'query': {'weight': '1a8914c482024a643b1503123f2c1bac81ce565d0ab7c98f06d8951d8b277f47', 'bias': '85b53d121a7751dbd243a12ce73c6be48132471667f817fde6020e1baaaed9c4'},
    'key': {'weight': '854e6ccfbdd78eb9108127baaf3374ec8a69120994aae532b1985cab67103981', 'bias': 'aee8e9d6fb2d9f305221ee344c00e6a026bac065717dfd5191d86bcf8b0cdc4f'},
    'value': {'weight': 'd21b08742d7d298ef2139215a704546011010b5ed40fd315ae430af9b50eb174', 'bias': 'dab39da7473f97300c2380a0cff1028e91062f1da40efa95fc3c419cfa19cff9'},
    'output': {'weight': 'e00c6a53896c0ed3d2a4fe73edd1313952d23e80bfc88e40402b2b9fd2df3e70', 'bias': 'b80a500392b9bff59848d76a2d89d77a43cb132747beb292261f7b07bad800f3'},
}
PREFIX = {'query': 'attention.self.query', 'key': 'attention.self.key', 'value': 'attention.self.value', 'output': 'attention.output.dense'}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _pinned(name):
    raw = (ROOT / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != INPUT_SHA256[name]:
        raise ValueError('changed pinned input: ' + name)
    return json.loads(raw)


def parameters():
    """Return a new canonical public context dictionary, without initialization."""
    p = _pinned('p60_profile_and_refresh.json')['profile']
    return dict(degree=p['degree'], num_slots=p['slots'], q_towers=p['q'],
                p_towers=p['p'], composite_degree=2, degree_layout=[256, 512],
                precision=32, max_level=p['depth'], scaling_factor=p['canonical_scale'],
                dnum=p['dnum'], batch=1, sigma=3.190000057220458984375,
                secret_distribution='uniform_ternary')


def _hex(value, length, name):
    if type(value) is not str or re.fullmatch('[0-9a-f]{%d}' % length, value) is None:
        raise ValueError(name + ' must be lowercase hex')
    return value


def make_contract(original_positions, context_id, request_id):
    if type(original_positions) is not tuple or not 1 <= len(original_positions) <= 128:
        raise ValueError('original_positions must be a nonempty tuple of at most 128 indices')
    if any(type(i) is not int or not 0 <= i < 128 for i in original_positions):
        raise ValueError('positions must be exact Python integers in 0..127')
    if any(a >= b for a, b in zip(original_positions, original_positions[1:])):
        raise ValueError('positions must be strictly increasing')
    _hex(context_id, 64, 'context_id'); _hex(request_id, 32, 'request_id')
    model = _pinned('verified_model_manifest.json')
    _pinned('candidate_manifest.json'); _pinned('matrix_inventory.json')
    k = len(original_positions); t = (k + 15) // 16
    return dict(schema='bert-first-attention-contract-v1', context_id=context_id,
                request_id=request_id, key_count=k, query_tiles=t,
                original_positions=list(original_positions),
                output_valid_rows=[min(16, k - 16*i) for i in range(t)],
                input_tile_count=8, input_level=49, output_level=9,
                profile=parameters(), source_input_sha256=dict(INPUT_SHA256),
                model=dict(model_id=model['model_id'], revision=model['revision'],
                           checkpoint_sha256=model['files']['model.safetensors']['sha256'],
                           tensors={role: {kind: dict(name='bert.encoder.layer.0.' + PREFIX[role] + '.' + kind,
                                                       shape=[768, 768] if kind == 'weight' else [768],
                                                       dtype='<f4', sha256=sha)
                                           for kind, sha in pins.items()}
                                    for role, pins in TENSOR_SHA256.items()}),
                packing=dict(slots=65536, row_pitch=4096, rows_per_tile=16,
                             active_features=768, score_heads=12, score_head_dim=64,
                             learned_position_order='apply-original-positions-before-compaction',
                             compact_rows='selected-original-positions-in-increasing-order',
                             tail_rows='zero-pad-compacted-input-to-128-rows',
                             feature_padding='zero', output_rows='compacted-order'),
                public_key_mask=[i < k for i in range(128)])


def validate_contract(value):
    if type(value) is not dict or type(value.get('original_positions')) is not list:
        raise ValueError('contract must be a canonical mapping')
    expected = make_contract(tuple(value['original_positions']), value.get('context_id'), value.get('request_id'))
    if canonical(value) != canonical(expected):
        raise ValueError('contract differs from pinned schema, profile, model or packing')
    return expected


def validate_projection_arrays(model):
    """Return immutable bytes-backed float64 arrays with exact pinned F32 values."""
    if type(model) is not dict or set(model) != set(TENSOR_SHA256):
        raise ValueError('projection role inventory mismatch')
    result = {}
    for role, pins in TENSOR_SHA256.items():
        item = model[role]
        if type(item) is not dict or set(item) != {'weight', 'bias'}:
            raise ValueError('projection tensor inventory mismatch')
        result[role] = {}
        for kind, sha in pins.items():
            a = np.asarray(item[kind])
            shape = (768, 768) if kind == 'weight' else (768,)
            if a.shape != shape or a.dtype.kind != 'f' or a.dtype.itemsize not in (4, 8) or not np.isfinite(a).all():
                raise ValueError('projection must be finite F32 or exact F64 promotion with pinned shape')
            b = a.astype('<f4')
            if not np.array_equal(a.astype('<f8'), b.astype('<f8')) or hashlib.sha256(b.tobytes(order='C')).hexdigest() != sha:
                raise ValueError('projection differs from pinned tensor bytes')
            result[role][kind] = np.frombuffer(b.astype('<f8').tobytes(), dtype='<f8').reshape(shape)
    return result


def _scale(value):
    if type(value) is not float or not math.isfinite(value) or value <= 0:
        raise ValueError('scale must be an actual finite positive binary64 float')
    return value


def make_descriptor(contract, *, role, tile_index, payload_sha256, scale=None, input_digest=None):
    """Metadata envelope only; payload_sha256 must come from trusted serialization."""
    c = validate_contract(contract); p = c['profile']
    if role not in ('attention_input', 'attention_output'):
        raise ValueError('unknown role')
    count = 8 if role == 'attention_input' else c['query_tiles']
    if type(tile_index) is not int or not 0 <= tile_index < count:
        raise ValueError('tile index out of range')
    _hex(payload_sha256, 64, 'payload_sha256')
    level = c['input_level'] if role == 'attention_input' else c['output_level']
    if role == 'attention_input':
        if input_digest is not None or (scale is not None and (type(scale) is not float or scale != p['scaling_factor'])):
            raise ValueError('input scale or response binding is invalid')
        scale = p['scaling_factor']
    else:
        _scale(scale); _hex(input_digest, 64, 'input_digest')
    return dict(schema='bert-first-attention-descriptor-v1', context_id=c['context_id'],
                request_id=c['request_id'], contract_sha256=digest(c), role=role,
                tile_index=tile_index, tile_count=count,
                valid_rows=max(0, min(16, c['key_count'] - 16*tile_index)),
                level=level, moduli=p['q_towers'][:2*(level+1)], scale=scale, nsd=1,
                degree=p['degree'], degree_layout=p['degree_layout'], precision=p['precision'],
                batch=p['batch'], ciphertext_components=2, payload_sha256=payload_sha256,
                input_digest=input_digest)


def validate_descriptors(values, contract, *, role, expected_scales=None, input_digest=None):
    c = validate_contract(contract)
    count = 8 if role == 'attention_input' else c['query_tiles'] if role == 'attention_output' else 0
    if not count or type(values) is not list or len(values) != count:
        raise ValueError('missing, duplicate or extra descriptor inventory')
    if role == 'attention_output':
        if type(expected_scales) not in (list, tuple) or len(expected_scales) != count:
            raise ValueError('responses require the actual public scale plan')
        for s in expected_scales: _scale(s)
    elif expected_scales is not None or input_digest is not None:
        raise ValueError('request inventory has response-only fields')
    result = []
    for i, value in enumerate(values):
        if type(value) is not dict:
            raise ValueError('descriptor must be a mapping')
        expected = make_descriptor(c, role=role, tile_index=i, payload_sha256=value.get('payload_sha256'),
                                   scale=None if role == 'attention_input' else expected_scales[i],
                                   input_digest=input_digest)
        if canonical(value) != canonical(expected):
            raise ValueError('descriptor metadata, order or request binding mismatch')
        result.append(expected)
    return result


def request_digest(values, contract):
    return digest(validate_descriptors(values, contract, role='attention_input'))
