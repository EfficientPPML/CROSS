"""Public full-model request contract; no key reader, client IDs or backend.

The caller authenticates public context/model identities against the validated
public key and model manifests. Digests here bind artifacts, not their origin.
"""
from pathlib import Path
import hashlib,json,math,re
ROOT=Path(__file__).resolve().parent
PROFILE_SHA256='fe2408637f9d6de19ac6aef9f6f749aaef9dd56ccbb37b5334293cd223ede035'
CANDIDATE_SHA256='4e2277beea7189aa1d77b168c500c8a175f6c630afa520fcc6b38cb5a179cff9'
CHECKPOINT_SHA256='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
CONFIG_SHA256='ec2b6cd6ce741dda6c972aea96f9d1a449cbb4b5349e688c5370a9b2f6872d30'
SHA=re.compile(r'[0-9a-f]{64}\Z')
NONCE=re.compile(r'[A-Za-z0-9_-]{1,128}\Z')

def canonical(value):
    def check(v):
        if v is None or type(v) in (str,int,bool,float):return
        if type(v)is list:
            for x in v:check(x)
            return
        if type(v)is dict and all(type(k)is str for k in v):
            for x in v.values():check(x)
            return
        raise ValueError('exact JSON types required')
    check(value)
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def digest(value):return hashlib.sha256(canonical(value)).hexdigest()

def parameters():
    raw=(ROOT/'p60_profile_and_refresh.json').read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROFILE_SHA256:raise ValueError('fixed common profile changed')
    p=json.loads(raw)['profile']
    return dict(degree=p['degree'],num_slots=p['slots'],q_towers=p['q'],p_towers=p['p'],
        composite_degree=2,degree_layout=[256,512],precision=32,max_level=p['depth'],
        scaling_factor=p['canonical_scale'],dnum=p['dnum'],batch=1,
        sigma=3.190000057220458984375,secret_distribution='uniform_ternary')

def _sha(value):
    if type(value)is not str or not SHA.fullmatch(value):raise ValueError('lowercase SHA256 identity required')
    return value

def public_identity(public):
    if type(public)is not dict:raise ValueError('externally authenticated public identities required')
    expected=dict(context_id=_sha(public.get('context_id')),model_id=_sha(public.get('model_id')),
        checkpoint_sha256=CHECKPOINT_SHA256,config_sha256=CONFIG_SHA256,
        candidate_sha256=CANDIDATE_SHA256,profile_sha256=PROFILE_SHA256,parameters=parameters())
    if canonical(public)!=canonical(expected):raise ValueError('fixed public model/circuit/profile identity differs')
    return expected

def packing_metadata(value):
    if type(value)is not dict or type(value.get('original_positions'))is not list:
        raise ValueError('exact client public_metadata schema required')
    positions=value['original_positions']
    if not 1<=len(positions)<=128 or any(type(x)is not int or not 0<=x<128 for x in positions):
        raise ValueError('valid original positions required')
    if positions[0]!=0 or any(a>=b for a,b in zip(positions,positions[1:])):
        raise ValueError('valid original CLS and increasing positions required')
    k=len(positions)
    expected=dict(schema='bert-client-row-packing-v1',input_tile_count=8,
        original_positions=list(positions),original_attention_mask=[int(i in positions) for i in range(128)],
        compact_key_mask=[int(i<k) for i in range(128)],key_count=k,query_tiles=(k+15)//16,
        valid_rows=[max(0,min(16,k-16*i)) for i in range(8)],
        packing=dict(slots=65536,row_pitch=4096,rows_per_tile=16,active_features=768,
            row_order='increasing original positions',positions_applied='before compaction',
            feature_padding='zero',unused_rows='zero',input_dtype='float64'))
    if canonical(value)!=canonical(expected):raise ValueError('public mask/position/packing binding differs')
    return expected

def make_contract(public,request_id,metadata):
    if type(request_id)is not str or not NONCE.fullmatch(request_id):raise ValueError('opaque request nonce required')
    return dict(schema='cross-bert-full-model-request-v1',public=public_identity(public),request_id=request_id,
        input=packing_metadata(metadata),input_level=49,response_level=2,
        response=dict(ciphertext_count=1,logit_slots=[0,1],class_order=[0,1],remaining_slots='not returned as logits'))

def validate_contract(value):
    if type(value)is not dict:raise ValueError('full-model request contract required')
    expected=make_contract(value.get('public'),value.get('request_id'),value.get('input'))
    if canonical(value)!=canonical(expected):raise ValueError('request schema or response layout differs')
    return expected

def ciphertext_contract(value,role,tile_index,expected_scale,input_digest):
    c=validate_contract(value);p=c['public']['parameters']
    if type(role)is not str or role not in ('request','response'):raise ValueError('unknown ciphertext role')
    count=8 if role=='request' else 1
    if type(tile_index)is not int or not 0<=tile_index<count:raise ValueError('ciphertext tile index differs')
    if role=='request':
        if input_digest is not None:raise ValueError('request cannot have a response binding')
        if expected_scale is not None and (type(expected_scale)is not float or expected_scale!=p['scaling_factor']):
            raise ValueError('fresh input requires canonical scale')
        expected_scale=p['scaling_factor'];level=49
    else:
        _sha(input_digest);level=2
        if type(expected_scale)is not float or not math.isfinite(expected_scale) or expected_scale<=0:
            raise ValueError('response requires actual finite positive binary64 scale plan')
    q=p['q_towers'][:2*(level+1)]
    return dict(context_id=c['public']['context_id'],model_id=c['public']['model_id'],
        request_id=c['request_id'],request_contract_sha256=digest(c),role=role,tile_index=tile_index,tile_count=count,
        input_digest=input_digest,level=level,scale=expected_scale,
        shape=(1,2,256,512,len(q)),moduli=q,layout=[256,512])
