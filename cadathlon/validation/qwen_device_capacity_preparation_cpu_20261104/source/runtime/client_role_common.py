"""Pure shared validation; no filesystem, NumPy, backend or execution authority."""
from pathlib import PurePosixPath
from qwen_file_contract import canonical,sha,hash_value,same,PublicContract
from qwen_public_parse import parse
from model_admission import require_admitted
from keygen_authority import GeneratedKeyReadiness,MODEL_FIELDS,READY_PIN_FIELDS

ROLES=('request_producer','response_consumer')
KEY_FIELDS=(READY_PIN_FIELDS-{'schema'})|{'expected_pins_sha256'}
POLICY_FIELDS={'dependency_policy_sha256','role_policy_sha256','resource_policy_sha256','resource_review_sha256'}
STATE_FIELDS={'device','inode','size','mtime_ns','ctime_ns','mode'}

def obj(value,fields,label):
    if type(value)is not dict or set(value)!=set(fields):raise ValueError('exact '+label+' fields required')
    return value

def raw_object(raw,limit,label):
    if type(raw)is not bytes or not 0<len(raw)<=limit:raise ValueError('bounded immutable '+label+' required')
    value=parse(raw)
    if type(value)is not dict or canonical(value)!=raw:raise ValueError('canonical '+label+' object required')
    return value

def integer(value,minimum,label,maximum=None):
    if type(value)is not int or value<minimum or(maximum is not None and value>maximum):raise ValueError('exact '+label+' integer required')
    return value

def hashes(value,fields,label):
    obj(value,fields,label)
    for v in value.values():hash_value(v)
    return value

def role(value):
    if type(value)is not str or value not in ROLES:raise ValueError('exact named client role required')
    return value

def path(value):
    if type(value)is not str or not 1<len(value)<=4096 or value.startswith('//')or '\\'in value or '\x00'in value:raise ValueError('canonical absolute path required')
    p=PurePosixPath(value)
    if not p.is_absolute()or str(p)!=value or '..'in p.parts or p==p.parent:raise ValueError('canonical non-root path required')
    return p

def source_map(value):
    if type(value)is not dict or not 0<len(value)<=512:raise ValueError('bounded complete source map required')
    for name,digest in value.items():
        if type(name)is not str or not 0<len(name)<=512 or '\\'in name or '\x00'in name:raise ValueError('relative source name required')
        p=PurePosixPath(name)
        if p.is_absolute()or str(p)!=name or '..'in p.parts or name=='.':raise ValueError('canonical relative source name required')
        hash_value(digest)
    return value

def model_pins(model,public,consumer):
    require_admitted(model,public,consumer_source_manifest_sha256=consumer)
    return dict(manifest_sha256=sha(model.manifest_bytes),report_sha256=sha(model.report_bytes),
        outer_sha256=sha(model.outer_bytes),root_review_sha256=sha(model.review_bytes),expected_pins_sha256=sha(model.expected_pins.raw))

def key_pins(ready):
    if type(ready)is not GeneratedKeyReadiness:raise TypeError('independently admitted generated keys required')
    ready.check();value=ready.expected_pins.check()
    return {n:value[n]for n in READY_PIN_FIELDS-{'schema'}}|dict(expected_pins_sha256=sha(ready.expected_pins.raw))

def parent(value,leaf):
    obj(value,('path','device','inode','mode'),'external parent')
    if path(value['path'])!=path(leaf).parent:raise ValueError('external output parent differs')
    integer(value['device'],0,'parent device');integer(value['inode'],1,'parent inode');integer(value['mode'],0,'parent mode',4095)
    return value

def output(value):
    obj(value,('path','parent'),'fresh output')
    path(value['path']);parent(value['parent'],value['path']);return value

def state(value):
    obj(value,STATE_FIELDS,'file state')
    for n in ('device','mtime_ns','ctime_ns'):integer(value[n],0,n)
    integer(value['inode'],1,'inode');integer(value['size'],1,'size');integer(value['mode'],0,'mode',4095)
    return value

def disjoint(leaves,borrowed):
    roots=[path(x)for x in leaves];others=[path(x)for x in borrowed]
    for i,p in enumerate(roots):
        if any(p==q or p in q.parents or q in p.parents for q in roots[:i]+others):raise ValueError('fresh output overlaps owned or borrowed scope')
