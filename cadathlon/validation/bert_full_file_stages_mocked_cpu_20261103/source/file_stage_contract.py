"""Ordered batch bindings for the five BERT file stages, without a backend.

These digests bind authenticated artifacts. Their origin is authenticated by
the key/model loaders and the client-retained request record, not by hashing.
Client target IDs and token arrays are excluded from all public schemas.
"""
import hashlib
from transport_contract import canonical, digest, public_identity, validate_contract, _sha

MAX_BATCH = 408


def _count(value):
    if type(value) is not int or not 1 <= value <= MAX_BATCH:
        raise ValueError('batch count must be an exact integer in1..408')
    return value


def _same(value, expected, message):
    if canonical(value) != canonical(expected):
        raise ValueError(message)
    return expected


def public_batch(public, requests, *, count):
    public = public_identity(public); count = _count(count)
    if type(requests) is not list or len(requests) != count:
        raise ValueError('one ordered request per sample required')
    checked = []; nonces = set()
    for row in requests:
        if type(row) is not dict or set(row) != {'contract', 'input_digest'}:
            raise ValueError('exact public request schema required')
        contract = validate_contract(row['contract'])
        _same(contract['public'], public, 'request public identity differs')
        if contract['request_id'] in nonces:
            raise ValueError('duplicate opaque request nonce')
        nonces.add(contract['request_id'])
        checked.append(dict(contract=contract, input_digest=_sha(row['input_digest'])))
    return dict(schema='cross-bert-full-upload-batch-v1', public=public,
                sample_count=count, requests=checked)


def validate_public_batch(value, *, public, count):
    if type(value) is not dict:
        raise ValueError('public batch must be an object')
    expected = public_batch(public, value.get('requests'), count=count)
    return _same(value, expected, 'public batch inventory/order/identity differs')


def private_requests(batch, targets, input_sha256):
    if type(batch) is not dict:
        raise ValueError('validated public batch required')
    batch = validate_public_batch(batch, public=batch.get('public'), count=batch.get('sample_count'))
    count = batch['sample_count']
    if (type(targets) is not list or len(targets) != count
            or any(type(i) is not int or i < 0 for i in targets)
            or len(set(targets)) != count):
        raise ValueError('one unique private target ID per input required')
    return dict(schema='cross-bert-full-client-requests-v1',
                context_id=batch['public']['context_id'], model_id=batch['public']['model_id'],
                public_batch_sha256=digest(batch), input_file_sha256=_sha(input_sha256),
                targets=list(targets), requests=batch['requests'])


def validate_private_requests(value, batch, targets, input_sha256):
    return _same(value, private_requests(batch, targets, input_sha256),
                 'client-retained input identity/order/request binding differs')


def response_batch(batch, responses):
    if type(batch) is not dict:
        raise ValueError('validated public batch required')
    batch = validate_public_batch(batch, public=batch.get('public'), count=batch.get('sample_count'))
    if type(responses) is not list or len(responses) != batch['sample_count']:
        raise ValueError('one ordered response per request required')
    checked = []
    for request, response in zip(batch['requests'], responses):
        if type(response) is not dict:
            raise ValueError('response binding must be an object')
        # A separately checked public replay must accept scale before the
        #payload loader uses it. Merely copying a server scale is insufficient.
        scale = response.get('scale')
        import math
        if type(scale) is not float or not math.isfinite(scale) or scale <= 0:
            raise ValueError('actual binary64 response scale required')
        expected = dict(request_id=request['contract']['request_id'],
                        input_digest=request['input_digest'], scale=scale,
                        response_manifest_sha256=_sha(response.get('response_manifest_sha256')))
        checked.append(_same(response, expected, 'response identity/order/inventory differs'))
    return dict(schema='cross-bert-full-download-batch-v1',
                public_batch_sha256=digest(batch), responses=checked)


def validate_response_batch(value, batch, *, check_scale):
    if type(value) is not dict or not callable(check_scale):
        raise ValueError('response object and independent public scale checker required')
    expected = response_batch(batch, value.get('responses'))
    _same(value, expected, 'response batch differs from original client request')
    for row in expected['responses']:
        if check_scale(row['scale']) is not True:
            raise ValueError('response scale differs from independent public replay')
    return expected


def file_digest(raw):
    if type(raw) is not bytes:
        raise ValueError('exact input bytes required')
    return hashlib.sha256(raw).hexdigest()
