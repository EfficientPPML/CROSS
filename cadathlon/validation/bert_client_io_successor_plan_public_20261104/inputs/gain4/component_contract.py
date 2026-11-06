"""Strict public roles for the isolated fixed-gain4 normalization proof, never full BERT."""
from pathlib import Path
import hashlib
import json
import math
import re

ROOT = Path(__file__).resolve().parent
PLAN_SHA = '55c53a7249766061e8b99bb3c9536fba62f75e15e40136faf485b4c7c2a0a18c'
PROFILE_SHA = 'fe2408637f9d6de19ac6aef9f6f749aaef9dd56ccbb37b5334293cd223ede035'
KEY_PLAN_SHA = 'bdc527179b9d1082d3dc0d90f1203523089aa8a24831941ad42a2513d20f7e04'
OWNER_SHA = '4abedac20a9c6974a7a7214f613d1a788bc6405fa7652612e82a8d831efb70eb'
CONTEXT_ID = 'ba28d6a81704eba60c51ad59fbfee3901516b783a1144d973e8199a014d87fb9'
SHA = re.compile(r'[0-9a-f]{64}\Z')
NONCE = re.compile(r'[0-9a-f]{32}\Z')


def canonical(value):
    def check(item):
        if item is None or type(item) in (str, int, bool, float):
            return
        if type(item) is list:
            for child in item:check(child)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for child in item.values():check(child)
            return
        raise ValueError('exact JSON types required')
    check(value)
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def numerical_plan():
    raw = (ROOT/'proof_plan.json').read_bytes()
    if hashlib.sha256(raw).hexdigest() != PLAN_SHA:
        raise ValueError('exact reviewed public numerical plan required')
    return json.loads(raw)


def make_contract(context_id, nonce):
    # The worker separately authenticates this identity against the actual
    # completed external key proof and loaded public context. This is not a
    # substitute for that gate, nor permission to read any key material.
    if type(context_id) is not str or context_id != CONTEXT_ID:
        raise ValueError('fixed authenticated existing-key context required')
    if type(nonce) is not str or not NONCE.fullmatch(nonce):
        raise ValueError('new opaque sixteen-byte attempt nonce required')
    plan = numerical_plan()
    return dict(schema='cross-normalized-gain4-component-request-v1', context_id=context_id,
        nonce=nonce, profile_sha256=PROFILE_SHA, key_plan_sha256=KEY_PLAN_SHA,
        owner_sha256=OWNER_SHA, numerical_plan_sha256=PLAN_SHA,
        input_patterns=[r['input_sha256']for r in plan['cases']],
        output_count=54, absolute_gate=1e-6)


def validate_contract(value):
    if type(value) is not dict:
        raise ValueError('exact component request contract required')
    expected = make_contract(value.get('context_id'), value.get('nonce'))
    if canonical(value) != canonical(expected):
        raise ValueError('component input/owner/plan contract differs')
    return expected


def ciphertext_contract(value, role, tile_index, expected_scale, input_digest):
    contract = validate_contract(value)
    plan = numerical_plan()
    if type(role) is not str or role not in ('request', 'response'):
        raise ValueError('exact component ciphertext role required')
    count = 2 if role == 'request' else 54
    if type(tile_index) is not int or not 0 <= tile_index < count:
        raise ValueError('exact ordered component tile index required')
    if role == 'request':
        if input_digest is not None:
            raise ValueError('fixture inputs cannot carry a response binding')
        scale = plan['profile']['canonical_scale']
        if expected_scale is not None and (type(expected_scale) is not float or expected_scale != scale):
            raise ValueError('fresh fixture must have actual canonical scale')
        level = 49
        case = None
        direction = 'fixture'
        label = plan['cases'][tile_index]['name']
    else:
        if type(input_digest) is not str or not SHA.fullmatch(input_digest):
            raise ValueError('authenticated complete two-fixture digest required')
        checkpoint=plan['checkpoints'][tile_index]
        group=plan['cases'][checkpoint['case_index']]
        level, scale = checkpoint['level'], checkpoint['scale']
        if (type(expected_scale) is not float or not math.isfinite(expected_scale) or
                expected_scale != scale):
            raise ValueError('actual result scale differs from fixed production recurrence')
        case, direction, label = group['name'], 'normalization', checkpoint['label']
    q = plan['profile']['q'][:2*(level+1)]
    return dict(context_id=contract['context_id'], request_id=contract['nonce'],
        request_contract_sha256=digest(contract), numerical_plan_sha256=PLAN_SHA,
        role=role, tile_index=tile_index, tile_count=count, input_digest=input_digest,
        case=case, direction=direction, label=label, level=level, scale=scale,
        shape=(1, 2, 256, 512, len(q)), moduli=q, layout=[256, 512])
