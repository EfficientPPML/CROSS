"""Fixed original toy4 roles; no key, input, model or backend loading.

Response scales are checked against the caller's independent recurrence. The
client must replay all96 from the five authenticated refresh metadata returns;
the request manifest alone does not certify server arithmetic or accuracy.
"""
from pathlib import Path
import hashlib,json,math,re

ROOT=Path(__file__).resolve().parent
PLAN_SHA='bb0d15f96fc2b01dd611a526f9aa9274965fdefe9b350e2663593a57283268c7'
PROFILE_SHA='f86b3b56bed72aeabec50b81dababeb62ba1627cf6764efcec189c930f708f18'
ORIGINAL_SHA='bc0ad1b2a3796fe4ecb5944d68f3ca65561ad5740357470a1e634cc243ec387b'
SHA=re.compile(r'[0-9a-f]{64}\Z');NONCE=re.compile(r'[0-9a-f]{32}\Z')

def canonical(value):
    def check(x):
        if x is None or type(x)in(str,int,bool,float):return
        if type(x)is list:
            for child in x:check(child)
            return
        if type(x)is dict and all(type(k)is str for k in x):
            for child in x.values():check(child)
            return
        raise ValueError('exact JSON types required')
    check(value)
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def numerical_plan():
    raw=(ROOT/'public_plan/plan.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PLAN_SHA:raise ValueError('exact222-source public plan required')
    profile=(ROOT/'profile.json').read_bytes()
    if hashlib.sha256(profile).hexdigest()!=PROFILE_SHA:raise ValueError('exact original p56 profile required')
    plan=json.loads(raw)
    if canonical(plan['profile'])!=profile:raise ValueError('profile and public plan disagree')
    return plan

def make_contract(context_id,nonce):
    # Externally authenticate context_id against freshly completed key evidence
    #before opening any ciphertext. This does not trust a self-declared key ID.
    if type(context_id)is not str or not SHA.fullmatch(context_id):raise ValueError('authenticated context digest required')
    if type(nonce)is not str or not NONCE.fullmatch(nonce):raise ValueError('fresh opaque sixteen-byte nonce required')
    plan=numerical_plan()
    return dict(schema='cross-toy4-checkpoint-request-v1',context_id=context_id,nonce=nonce,
        original_source_archive_sha256=ORIGINAL_SHA,profile_sha256=PROFILE_SHA,numerical_plan_sha256=PLAN_SHA,
        case_sha256=plan['case_sha256'],input_pattern_sha256=plan['checkpoints'][0]['expected_array_sha256'],
        request_count=1,output_count=96,absolute_gate=1e-6,original_decoder_stddev_limit=.03125)

def validate_contract(value):
    if type(value)is not dict:raise ValueError('exact diagnostic request required')
    expected=make_contract(value.get('context_id'),value.get('nonce'))
    if canonical(value)!=canonical(expected):raise ValueError('diagnostic request binding differs')
    return expected

def ciphertext_contract(value,role,tile_index,expected_scale,input_digest):
    contract=validate_contract(value);plan=numerical_plan()
    if type(role)is not str or role not in('request','response'):raise ValueError('exact ciphertext role required')
    count=1 if role=='request' else 96
    if type(tile_index)is not int or not 0<=tile_index<count:raise ValueError('exact ordered checkpoint index required')
    row=plan['checkpoints'][0 if role=='request' else tile_index]
    if role=='request':
        if input_digest is not None:raise ValueError('input cannot carry response digest')
        scale=plan['profile']['scale']
        if expected_scale is not None and(type(expected_scale)is not float or expected_scale!=scale):raise ValueError('fresh canonical scale required')
    else:
        if type(input_digest)is not str or not SHA.fullmatch(input_digest):raise ValueError('authenticated input digest required')
        if type(expected_scale)is not float or not math.isfinite(expected_scale)or expected_scale<=0:raise ValueError('independently replayed positive finite scale required')
        scale=expected_scale
    level=row['level'];q=plan['profile']['q_towers'][:2*(level+1)]
    return dict(context_id=contract['context_id'],request_id=contract['nonce'],request_contract_sha256=digest(contract),
        numerical_plan_sha256=PLAN_SHA,role=role,tile_index=tile_index,tile_count=count,input_digest=input_digest,
        case='bert/test4',direction=row['kind'],label=row['label'],level=level,scale=scale,
        shape=(1,2,256,512,len(q)),moduli=q,layout=[256,512])
