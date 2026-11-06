"""HELD / NOT RUN. Root must separately authorize metadata acquisition.

Borrowed files are lstat-only. Declared hashes are copied from pinned public
metadata, never recomputed from parameter bodies. This is not model admission.
"""
from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import argparse, hashlib, json, math, os, stat, sys, time

ROOTS = {
    'qkv': '/tmp/qwen-qkv-attention-join-20261103/public_qkv_parameters',
    'decoder': '/tmp/qwen-decoder-parameters-20261103/actual-v1/public_matrices',
    'gamma': '/tmp/qwen-rmsnorm-owner-20261103',
}
GAMMA_NAME = 'client_gamma_fixtures.npz'
OUTPUT = '/tmp/qwen-model-store-metadata-expectations-20261104'
AGGREGATE = '18189e5f2f2de3a4939c6afca19d79822465d917caf06c4f6e36668bd37c1f55'
CATALOG = '6faa4630169f055b853a62a64088952e2ead48dc2ace5b61b6aa438c0106a0c0'
MODEL = 'ee3daf8b07141134466d02a6e1c141365ac4e0f0117d62f887c416a050b04bfb'

# Literal shared4543 exception-provenance functions; no local-module import.
ATTRIBUTE='_cross_toy4_cleanup_origin'
BODY_ATTRIBUTE='_cross_toy4_body_origin'
def is_cleanup(error):return error is not None and getattr(error,BODY_ATTRIBUTE,False)is not True and getattr(error,ATTRIBUTE,False)is True
def protect_body(error):
    if error is not None and not is_cleanup(error):setattr(error,BODY_ATTRIBUTE,True)
    return error
def mark(error):
    if getattr(error,BODY_ATTRIBUTE,False)is not True:setattr(error,ATTRIBUTE,True)
    return error
def select(primary,errors):
    """Select one exception; errors are labeled cleanup operations, in order."""
    protect_body(primary)
    rows=[]
    for label,error in errors:
        if error is None or error is primary:continue
        rows.append((label,mark(error)))
    if primary is not None and not is_cleanup(primary):chosen=primary
    else:
        ordered=([('earlier cleanup',primary)]if primary is not None else[])+rows
        if not ordered:return None
        chosen=next((e for _,e in ordered if not isinstance(e,Exception)),ordered[0][1])
    if primary is not None and primary is not chosen:chosen.add_note('earlier cleanup: '+repr(primary))
    for label,error in rows:
        if error is not chosen:chosen.add_note(label+': '+repr(error))
    return chosen

def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def require(condition,label):
    if not condition:raise ValueError(label)
def digest(value):
    require(type(value)is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'digest')
    return value
def path(value):
    require(type(value)is str and 1<len(value)<=4096 and '\x00'not in value and '\\'not in value,'path')
    p=PurePosixPath(value)
    require(p.is_absolute() and str(p)==value and not value.startswith('//') and '..'not in p.parts and p!=p.parent,'canonical path')
    return Path(value)
def overlap(a,b):return a==b or a in b.parents or b in a.parents
def pairs(rows):
    out={}
    for k,v in rows:
        require(k not in out,'duplicate JSON key');out[k]=v
    return out
def parse(raw):
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda value:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
def identity(s):return (s.st_dev,s.st_ino,stat.S_IMODE(s.st_mode))
def state(s):
    return dict(device=s.st_dev,inode=s.st_ino,size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,mode=stat.S_IMODE(s.st_mode))
def ancestors(p):
    result={}
    for part in reversed((p,)+tuple(p.parents)):
        s=os.lstat(part);require(stat.S_ISDIR(s.st_mode),'ordinary ancestor directory required')
        result[str(part)]=identity(s)
    return result

@contextmanager
def opened(name,flags,mode=0o600,*,dir_fd=None):
    fd=None;primary=None;errors=[]
    try:
        fd=os.open(name,flags,mode,dir_fd=dir_fd)
        yield fd
    except BaseException as error:
        primary=protect_body(error)
    finally:
        if fd is not None:
            owned=fd;fd=None
            try:os.close(owned)
            except BaseException as error:errors.append(('descriptor close (no retry)',error))
        chosen=select(primary,errors)
        if chosen is not None:raise chosen

def read_public(p,maximum,expected):
    # Only fixed public source/JSON paths enter this helper.
    for root in ROOTS.values():require(not overlap(p,path(root)),'public metadata overlaps borrowed store')
    parent=ancestors(p.parent);before=os.lstat(p)
    require(stat.S_ISREG(before.st_mode) and 0<before.st_size<=maximum,'bounded public file')
    with opened(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK) as fd:
        require(state(os.fstat(fd))==state(before),'public opened identity')
        parts=[];remaining=before.st_size
        while remaining:
            piece=os.read(fd,min(65536,remaining));require(bool(piece),'short public metadata')
            parts.append(piece);remaining-=len(piece)
        require(os.read(fd,1)==b'','public extent changed')
        require(state(os.fstat(fd))==state(os.lstat(p))==state(before),'public identity changed')
    require(ancestors(p.parent)==parent,'public parent changed')
    raw=b''.join(parts);require(sha(raw)==digest(expected),'public source pin differs')
    return raw

def inventory(catalog):
    require(set(catalog)=={'matrices','gammas','gamma_container_sha256','provenance'},'catalog fields')
    require(len(catalog['matrices'])==198 and len(catalog['gammas'])==113,'logical catalog count')
    physical={}
    for row in catalog['matrices'].values():
        family=row['family'];d=row['descriptor'];name=d['file']
        require(family in ('qkv','decoder') and type(name)is str and 0<len(name)<=255 and Path(name).name==name and name not in ('.','..') and '\\'not in name and '\x00'not in name,'flat matrix name')
        shape=d['shape'];require(type(shape)is list and len(shape)==2 and all(type(x)is int and x>0 for x in shape),'matrix shape')
        record=dict(family=family,name=name,file_sha256=digest(d['file_sha256']),raw_sha256=digest(d['sha256']),minimum_bytes=math.prod(shape)*8+1,maximum_bytes=math.prod(shape)*8+16384)
        key=(family,name)
        if key in physical:require(physical[key]==record,'tied descriptor mismatch')
        else:physical[key]=record
    require(len(physical)==197,'physical matrix count')
    require(sum(k[0]=='qkv' for k in physical)==84 and sum(k[0]=='decoder' for k in physical)==113,'family counts')
    physical[('gamma',GAMMA_NAME)]=dict(family='gamma',name=GAMMA_NAME,file_sha256=digest(catalog['gamma_container_sha256']),raw_sha256=None,minimum_bytes=1,maximum_bytes=4<<20)
    rows=[physical[k] for k in sorted(physical)]
    require(len({str(path(ROOTS[r['family']])/r['name']) for r in rows})==198,'duplicate body paths')
    return rows

def observe(rows):
    roots={};chains={};files=[];inodes=set()
    for family,value in ROOTS.items():
        p=path(value);chains[family]=ancestors(p);s=os.lstat(p)
        require(stat.S_ISDIR(s.st_mode),'ordinary borrowed root required')
        roots[family]=dict(path=value,device=s.st_dev,inode=s.st_ino,mode=stat.S_IMODE(s.st_mode))
    require(len({(r['device'],r['inode']) for r in roots.values()})==3,'duplicate root identities')
    for row in rows:
        p=path(ROOTS[row['family']])/row['name'];s=os.lstat(p)
        require(stat.S_ISREG(s.st_mode),'ordinary known file required')
        require(row['minimum_bytes']<=s.st_size<=row['maximum_bytes'],'known file extent')
        require(s.st_dev>=0 and s.st_ino>0 and s.st_mtime_ns>=0 and s.st_ctime_ns>=0,'file state range')
        ident=(s.st_dev,s.st_ino);require(ident not in inodes,'duplicate physical file identity');inodes.add(ident)
        files.append({k:row[k] for k in ('family','name','file_sha256','raw_sha256')}|{'state':state(s)})
    for family,value in ROOTS.items():require(ancestors(path(value))==chains[family],'borrowed root/path changed')
    return dict(roots=roots,files=files),chains

def write_new(directory,name,raw):
    require(type(raw)is bytes and 0<len(raw)<=2<<20,'bounded new public metadata')
    with opened(name,os.O_RDWR|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,dir_fd=directory) as fd:
        first=os.fstat(fd);require(stat.S_ISREG(first.st_mode) and stat.S_IMODE(first.st_mode)==0o600,'new file mode')
        offset=0
        while offset<len(raw):
            count=os.write(fd,raw[offset:offset+65536]);require(type(count)is int and 0<count<=min(65536,len(raw)-offset),'write progress');offset+=count
        os.fsync(fd);os.lseek(fd,0,os.SEEK_SET);parts=[];remaining=len(raw)
        while remaining:
            piece=os.read(fd,min(65536,remaining));require(bool(piece),'short new metadata');parts.append(piece);remaining-=len(piece)
        require(b''.join(parts)==raw and os.read(fd,1)==b'','new metadata readback')
        held=os.fstat(fd);named=os.stat(name,dir_fd=directory,follow_symlinks=False)
        require(identity(held)==identity(named)==identity(first) and held.st_size==named.st_size==len(raw),'new metadata identity')
    os.fsync(directory)

def run(args):
    require(args.execute_metadata_only is True,'separate root authorization required before execution')
    source=Path(__file__).absolute();pins_path=path(args.source_pins);output=path(args.output)
    require(output==path(OUTPUT),'fixed proposed output differs')
    pins_raw=read_public(pins_path,1<<20,args.source_pins_sha256);pins=parse(pins_raw)
    require(set(pins)=={'schema','script','public_inputs','expected_roots','gamma_name','scope'},'source pin fields')
    require(pins['schema']=='qwen-model-metadata-acquisition-pins-v1' and pins['expected_roots']==ROOTS and pins['gamma_name']==GAMMA_NAME,'fixed acquisition scope')
    require(pins['script']['path']==str(source),'script path');script_raw=read_public(source,1<<20,pins['script']['sha256'])
    require(set(pins['public_inputs'])=={'aggregate','catalog','layout_reference'},'fixed public inputs')
    raw={}
    for name,row in pins['public_inputs'].items():
        require(set(row)=={'path','sha256','maximum_bytes'},'public pin row')
        require(type(row['maximum_bytes'])is int and 0<row['maximum_bytes']<=1<<20,'public bound')
        raw[name]=read_public(path(row['path']),row['maximum_bytes'],row['sha256'])
    aggregate=parse(raw['aggregate']);catalog=parse(raw['catalog']);layout=parse(raw['layout_reference'])['public_parameters']
    require(sha(raw['aggregate'])==AGGREGATE and sha(raw['catalog'])==CATALOG and canonical(catalog)==raw['catalog'],'fixed aggregate/catalog')
    require(aggregate['catalog']==catalog and aggregate['catalog_sha256']==CATALOG and aggregate['model_id']==MODEL and sha(canonical(aggregate['content']))==MODEL,'content binding')
    require(layout['qkv_directory']==ROOTS['qkv'] and layout['decoder_directory']==ROOTS['decoder'] and layout['gamma_path']==ROOTS['gamma']+'/'+GAMMA_NAME and layout['gamma_sha256']==catalog['gamma_container_sha256'],'public path provenance')
    roots=[path(p) for p in ROOTS.values()]
    require(all(not overlap(a,b) for i,a in enumerate(roots) for b in roots[i+1:]),'overlapping borrowed roots')
    inputs=[source,pins_path]+[path(r['path']) for r in pins['public_inputs'].values()]
    require(all(not overlap(output,p) for p in roots+inputs),'output/input overlap')
    rows=inventory(catalog);started=time.time_ns()
    first,chains=observe(rows);second,chains2=observe(rows)
    require(first==second and chains==chains2,'metadata changed between repeated observations')
    finished=time.time_ns()
    snapshot=canonical(second);require(len(snapshot)<=2<<20,'expected-store bound')
    require(read_public(pins_path,1<<20,args.source_pins_sha256)==pins_raw,'source pin bytes changed')
    require(read_public(source,1<<20,pins['script']['sha256'])==script_raw,'script bytes changed')
    for name,row in pins['public_inputs'].items():
        require(read_public(path(row['path']),row['maximum_bytes'],row['sha256'])==raw[name],'public provenance changed')
    # Only a fresh evidence leaf is created; existing parent may have protected siblings.
    parent=output.parent;parent_chain=ancestors(parent)
    with opened(parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW) as parent_fd:
        require(identity(os.fstat(parent_fd))==parent_chain[str(parent)],'output held parent')
        os.mkdir(output.name,0o700,dir_fd=parent_fd);os.fsync(parent_fd)
        with opened(output.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent_fd) as directory:
            owned=identity(os.fstat(directory));require(owned[2]==0o700,'output directory mode')
            def output_check():
                require(ancestors(parent)==parent_chain,'output parent changed')
                require(identity(os.fstat(directory))==identity(os.stat(output.name,dir_fd=parent_fd,follow_symlinks=False))==identity(os.lstat(output))==owned,'output directory changed')
            output_check();write_new(directory,'expected_stores.json',snapshot)
            provenance=dict(schema='qwen-model-store-metadata-observation-v1',metadata_observation_completed=True,actual_model_validation=False,body_integrity_admitted=False,execution_authorized=False,body_files_opened=0,body_hashes_recomputed=False,model_id=MODEL,aggregate_manifest_sha256=AGGREGATE,catalog_sha256=CATALOG,source_pins_sha256=sha(pins_raw),script_sha256=sha(script_raw),source_and_provenance_rechecked=True,public_inputs=pins['public_inputs'],roots=ROOTS,file_count=198,matrix_files=197,gamma_files=1,observations=2,started_ns=started,finished_observation_ns=finished,expected_stores_sha256=sha(snapshot),expected_stores_bytes=len(snapshot),pid=os.getpid(),pgid=os.getpgrp(),limitation='Repeated named metadata observations are not an atomic snapshot or body integrity proof. Later held readers must reject any changed expectation. Final close requires independent exit0/reap and exact output hash before use.')
            output_check();write_new(directory,'acquisition.json',canonical(provenance));output_check();os.fsync(directory)
        require(ancestors(parent)==parent_chain and identity(os.lstat(output))==owned,'final output path changed');os.fsync(parent_fd)
    return 0

def deny_body_open(event,args):
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        value=os.fsdecode(args[0]);p=Path(os.path.abspath(value))
        for root in ROOTS.values():
            if p==Path(root) or Path(root) in p.parents:raise PermissionError('borrowed body/directory open forbidden')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pins',required=True);parser.add_argument('--source-pins-sha256',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--execute-metadata-only',action='store_true')
    args=parser.parse_args()
    try:
        sys.addaudithook(deny_body_open)
        result=run(args)
        require(type(result)is int and result==0,'normal zero required')
        return result
    except BaseException as error:
        try:print('Metadata acquisition infrastructure failure: '+type(error).__name__,file=sys.stderr)
        except BaseException:pass
        return 70

if __name__=='__main__':raise SystemExit(main())
