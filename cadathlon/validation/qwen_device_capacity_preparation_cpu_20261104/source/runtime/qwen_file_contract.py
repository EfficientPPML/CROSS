"""Pure typed Qwen file-boundary contracts; no body I/O or execution authority."""
from dataclasses import dataclass
import hashlib,json,math,re
from qwen_public_parse import parse
from public_source_pins import PUBLIC_PINS,PARENT_MANIFEST_SHA256

SHA=re.compile(r'[0-9a-f]{64}\Z')
NONCE=re.compile(r'[A-Za-z0-9_-]{1,128}\Z')
CANDIDATE='344954b3873a8847ececba320646f52612ca7277412c7de81a7313894e21799a'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
CHECKPOINT='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
Q_SHA='fc2b841869ef2b2b556209323ed033c5a177344d1de395fc94e8a5865df39a49'
P_SHA='c03a649ab504238a4f24d57d7fe8e49f2a445c5d842579494cd243a7043d1d49'
DELTA=1.1467305003699405e18

def canonical(value):
    def check(v):
        if v is None or type(v)in(str,int,bool,float):return
        if type(v)is list:
            for x in v:check(x)
            return
        if type(v)is dict and all(type(k)is str for k in v):
            for x in v.values():check(x)
            return
        raise TypeError('exact JSON scalar/list/dict types required')
    check(value)
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def sha(raw):return hashlib.sha256(raw).hexdigest()
def digest(value):return sha(canonical(value))
def hash_value(value):
    if type(value)is not str or not SHA.fullmatch(value):raise ValueError('exact lowercase SHA256 required')
    return value
def same(actual,expected,label):
    if canonical(actual)!=canonical(expected):raise ValueError(label)
    return expected
def scale(value):
    if type(value)is not float or not math.isfinite(value)or value<=0:raise ValueError('actual finite positive binary64 scale required')
    return value

@dataclass(frozen=True)
class PublicContract:
    """Exact source-bound metadata, distinct from external role admission."""
    blobs:tuple
    def __post_init__(self):self.check()
    @classmethod
    def from_blobs(cls,blobs):
        if type(blobs)is not dict or set(blobs)!=set(PUBLIC_PINS):raise ValueError('exact public source inventory required')
        return cls(tuple(sorted(blobs.items())))
    def check(self):
        if type(self.blobs)is not tuple or len(self.blobs)!=len(PUBLIC_PINS):raise ValueError('public source state changed')
        values={}
        for pair in self.blobs:
            if type(pair)is not tuple or len(pair)!=2:raise ValueError('public source pair differs')
            name,raw=pair
            if type(name)is not str or name not in PUBLIC_PINS or name in values or type(raw)is not bytes or sha(raw)!=PUBLIC_PINS[name]:raise ValueError('public source binding differs')
            values[name]=raw
        if set(values)!=set(PUBLIC_PINS):raise ValueError('public source inventory differs')
        return values
    def json(self,name):return parse(self.check()[name])
    def profile(self):
        rows=self.json('inputs/profile_audit.json')['profiles']
        selected=[r for r in rows if r['name']=='depth49_p60_d8']
        if len(selected)!=1:raise ValueError('one fixed p60/d8 profile required')
        p=selected[0]
        for name,count,pin in (('q',100,Q_SHA),('p',14,P_SHA)):
            if type(p[name])is not list or len(p[name])!=count or any(type(v)is not int for v in p[name])or sha(json.dumps(p[name]).encode())!=pin:raise ValueError('ordered common Q/P differs')
        for name,value in (('degree',131072),('slots',65536),('depth',49),('dnum',8),('canonical_scale',DELTA)):
            same(p[name],value,'fixed public profile differs: '+name)
        return dict(degree=131072,num_slots=65536,q_towers=p['q'],p_towers=p['p'],composite_degree=2,
            degree_layout=[256,512],precision=32,max_level=49,scaling_factor=DELTA,dnum=8,batch=1,
            sigma=3.190000057220458984375,secret_distribution='uniform_ternary')
    def identity(self,*,context_id,model_id,producer_sources,consumer_sources):
        return dict(schema='qwen-public-file-identity-v1',context_id=hash_value(context_id),model_id=hash_value(model_id),
            producer_source_manifest_sha256=hash_value(producer_sources),consumer_source_manifest_sha256=hash_value(consumer_sources),
            full_owner_source_manifest_sha256=PARENT_MANIFEST_SHA256,candidate_sha256=CANDIDATE,
            checkpoint_sha256=CHECKPOINT,config_sha256=CONFIG,public_source_sha256=dict(PUBLIC_PINS),parameters=self.profile())
    def check_identity(self,value):
        if type(value)is not dict:raise ValueError('public identity dictionary required')
        expected=self.identity(context_id=value.get('context_id'),model_id=value.get('model_id'),
            producer_sources=value.get('producer_source_manifest_sha256'),consumer_sources=value.get('consumer_source_manifest_sha256'))
        return same(value,expected,'public producer/consumer/context/model identity differs')

def packing(positions):
    if type(positions)is not list or not 1<=len(positions)<=32 or any(type(x)is not int or not 0<=x<40960 for x in positions)or positions!=sorted(set(positions)):
        raise ValueError('one to32 increasing original positions required')
    count=(len(positions)+15)//16
    return dict(schema='qwen-original-embedding-row-packing-v1',batch=1,sequence_length=len(positions),
        original_positions=list(positions),input_tile_count=count,valid_rows=[min(16,len(positions)-16*i)for i in range(count)],
        packing=dict(slots=65536,row_pitch=4096,rows_per_tile=16,active_features=1024,feature_padding='zero',unused_rows='zero',
            row_order='increasing original positions',row_replication=False,input_dtype='float64',input_semantics='original client embeddings'))

def vocabulary():
    return [dict(index=i,start=4096*i,stop=min(4096*(i+1),151936),physical_row=0,level=2)for i in range(38)]

def request(public_contract,public,request_id,positions):
    if type(request_id)is not str or not NONCE.fullmatch(request_id):raise ValueError('opaque request nonce required')
    return dict(schema='cross-qwen-full28-request-v1',public=public_contract.check_identity(public),request_id=request_id,
        input=packing(positions),transport_input_level=49,owner_input_level=25,
        preparation=dict(operation='genuine_prefix_level_reduce',input_level=49,output_level=25,scale_preserved=True),
        response=dict(ciphertext_count=38,vocabulary_size=151936,last_original_position=positions[-1],blocks=vocabulary()))

def validate_request(public_contract,value):
    if type(value)is not dict or type(value.get('input'))is not dict:raise ValueError('Qwen request dictionary required')
    expected=request(public_contract,value.get('public'),value.get('request_id'),value['input'].get('original_positions'))
    return same(value,expected,'Qwen request shape/packing/role schema differs')

def ciphertext_binding(public_contract,contract,role,index,*,actual_scale=None,input_digest=None):
    c=validate_request(public_contract,contract);p=c['public']['parameters']
    if type(role)is not str or role not in ('request','response'):raise ValueError('exact ciphertext role required')
    count=c['input']['input_tile_count']if role=='request'else 38
    if type(index)is not int or not 0<=index<count:raise ValueError('ordered ciphertext index differs')
    if role=='request':
        if input_digest is not None:raise ValueError('input cannot bind another request digest')
        if actual_scale is not None:same(actual_scale,DELTA,'fresh input scale differs')
        actual_scale=DELTA;level=49;logical=dict(valid_rows=c['input']['valid_rows'][index],features=1024)
    else:
        hash_value(input_digest);scale(actual_scale);level=2;logical=vocabulary()[index]
    q=p['q_towers'][:2*(level+1)]
    return dict(format='cross-qwen-full28-tile-v1',context_id=c['public']['context_id'],model_id=c['public']['model_id'],
        producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],consumer_source_manifest_sha256=c['public']['consumer_source_manifest_sha256'],
        request_id=c['request_id'],request_contract_sha256=digest(c),role=role,index=index,count=count,input_digest=input_digest,
        logical=logical,level=level,scale=actual_scale,nsd=1,precision=32,payload_dtype='uint32',payload_order='C',
        shape=[1,2,256,512,len(q)],moduli=q,degree_layout=[256,512])

def artifact(public_contract,contract,role,index,*,payload_sha256,actual_scale=None,input_digest=None):
    return dict(binding=ciphertext_binding(public_contract,contract,role,index,actual_scale=actual_scale,input_digest=input_digest),
        payload_sha256=hash_value(payload_sha256))

def checked_artifact(public_contract,contract,value,role,index,*,input_digest=None):
    if type(value)is not dict or type(value.get('binding'))is not dict:raise ValueError('exact artifact dictionary required')
    expected=artifact(public_contract,contract,role,index,payload_sha256=value.get('payload_sha256'),
        actual_scale=value['binding'].get('scale'),input_digest=input_digest)
    return same(value,expected,'artifact role/order/actual metadata differs')

def request_digest(public_contract,contract,artifacts):
    c=validate_request(public_contract,contract)
    if type(artifacts)is not list or len(artifacts)!=c['input']['input_tile_count']:raise ValueError('complete ordered input inventory required')
    rows=[checked_artifact(public_contract,c,row,'request',i)for i,row in enumerate(artifacts)]
    return digest(dict(contract_sha256=digest(c),artifacts=rows))

def request_manifest(public_contract,contract,artifacts):
    c=validate_request(public_contract,contract)
    input_hash=request_digest(public_contract,c,artifacts)
    return dict(schema='cross-qwen-full28-upload-v1',contract=c,artifacts=artifacts,input_digest=input_hash)

def validate_request_manifest(public_contract,contract,value):
    if type(value)is not dict:raise ValueError('request manifest dictionary required')
    return same(value,request_manifest(public_contract,contract,value.get('artifacts')),'uploaded request contract/order/digest differs')

def response(public_contract,contract,input_digest,artifacts):
    c=validate_request(public_contract,contract);hash_value(input_digest)
    if type(artifacts)is not list or len(artifacts)!=38:raise ValueError('all38 ordered output artifacts required')
    rows=[checked_artifact(public_contract,c,row,'response',i,input_digest=input_digest)for i,row in enumerate(artifacts)]
    return dict(schema='cross-qwen-full28-response-v1',request_id=c['request_id'],request_contract_sha256=digest(c),
        input_digest=input_digest,last_original_position=c['input']['original_positions'][-1],artifacts=rows)

def validate_response(public_contract,contract,input_digest,value):
    if type(value)is not dict:raise ValueError('response dictionary required')
    return same(value,response(public_contract,contract,input_digest,value.get('artifacts')),'response/source/request binding differs')
