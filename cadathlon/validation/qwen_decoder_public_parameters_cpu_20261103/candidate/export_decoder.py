"""Bounded public O/MLP/embedding/vocabulary extraction; no model/backend."""
from pathlib import Path
from contextlib import contextmanager
import argparse, hashlib, io, json, os, stat, struct, sys, time
import numpy as np
from cleanup import cleanup_signals, termination_signals

HERE=Path(__file__).resolve().parent
ASSETS=Path('/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca')
CHECKPOINT='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
CANDIDATE='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'
CHUNK_ROWS=4096
FORBIDDEN=('torch','transformers','jax','jaxlib','safetensors')

def digest(raw):return hashlib.sha256(raw).hexdigest()
def unique(pairs):
    out={}
    for k,v in pairs:
        if k in out:raise ValueError('duplicate public JSON key')
        out[k]=v
    return out
def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def write_json(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_bytes(canonical(value));temp.replace(path)

@contextmanager
def owned_stream(path,*,write=False):
    """Own exactly one FD; stream never owns it, including failed handoff."""
    flags=os.O_WRONLY|os.O_CREAT|os.O_EXCL if write else os.O_RDONLY
    fd=os.open(path,flags|os.O_NOFOLLOW|os.O_NONBLOCK,0o600)
    stream=None;primary=None;errors=[]
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):raise ValueError('regular public file required')
        stream=os.fdopen(fd,'wb' if write else 'rb',buffering=0,closefd=False)
        yield stream
    except BaseException as error:primary=error;raise
    finally:
        def attempt(label,fn):
            try:fn()
            except BaseException as error:errors.append((label,error))
        attempted=False
        def close_owned():
            nonlocal attempted
            attempted=True
            if stream is not None:attempt('stream close',stream.close)
            attempt('descriptor close',lambda:os.close(fd))
        try:
            with cleanup_signals():close_owned()
        except BaseException as error:errors.append(('cleanup scope',error))
        finally:
            if not attempted:close_owned()
        if primary is not None:
            for label,error in errors:primary.add_note(label+': '+type(error).__name__)
        elif errors:
            chosen=next((e for _,e in errors if not isinstance(e,Exception)),errors[0][1])
            for label,error in errors:
                if error is not chosen:chosen.add_note(label+': '+type(error).__name__)
            raise chosen

def file_sha(path):
    with owned_stream(path) as f:return hashlib.file_digest(f,'sha256').hexdigest()
def public_report(path):
    with owned_stream(path) as f:
        raw=f.read((16<<20)+1)
        if len(raw)>16<<20:raise ValueError('bounded public report required')
        return raw
def write_all(stream,raw):
    view=memoryview(raw)
    while view:
        n=stream.write(view)
        if type(n)is not int or not 0<n<=len(view):raise OSError('incomplete public matrix write')
        view=view[n:]
def file_identity(info):return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
def inventory():
    result={}
    for layer in range(28):
        result[f'model.layers.{layer}.self_attn.o_proj.weight']=(1024,2048)
        for kind in ('gate','up','down'):
            result[f'model.layers.{layer}.mlp.{kind}_proj.weight']=(1024,3072) if kind=='down' else (3072,1024)
    result['lm_head.weight']=(151936,1024)
    result['model.embed_tokens.weight']=(151936,1024)
    return result
def header(stream):
    prefix=stream.read(8)
    if len(prefix)!=8:raise ValueError('truncated public header prefix')
    n=struct.unpack('<Q',prefix)[0]
    if not 2<=n<=16<<20:raise ValueError('bounded public header required')
    raw=stream.read(n)
    if len(raw)!=n:raise ValueError('truncated public header')
    result=json.loads(raw,object_pairs_hook=unique)
    if type(result)is not dict:raise ValueError('public tensor mapping required')
    return result,8+n
def descriptor(metadata,name,shape,base,file_size):
    d=metadata.get(name)
    if (type(d)is not dict or set(d)!={'dtype','shape','data_offsets'} or d['dtype']!='BF16'
        or type(d['shape'])is not list or any(type(n)is not int or n<1 for n in d['shape'])
        or tuple(d['shape'])!=shape):raise ValueError('exact public BF16 matrix descriptor required')
    offsets=d['data_offsets']
    if (type(offsets)is not list or len(offsets)!=2 or any(type(n)is not int for n in offsets)
        or not 0<=offsets[0]<offsets[1] or base+offsets[1]>file_size
        or offsets[1]-offsets[0]!=2*shape[0]*shape[1]):raise ValueError('bounded exact tensor range required')
    return offsets
def blocks(stream,offset,shape):
    stream.seek(offset)
    for start in range(0,shape[0],CHUNK_ROWS):
        rows=min(CHUNK_ROWS,shape[0]-start);raw=stream.read(rows*shape[1]*2)
        if len(raw)!=rows*shape[1]*2:raise ValueError('truncated public tensor body')
        value=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').astype('<f8')
        if not np.isfinite(value).all():raise ValueError('nonfinite public tensor')
        yield raw,value.reshape(rows,shape[1])
def export_matrix(stream,metadata,base,file_size,name,shape,path):
    low,high=descriptor(metadata,name,shape,base,file_size)
    raw_hash=hashlib.sha256();value_hash=hashlib.sha256();count=0
    with owned_stream(path,write=True) as out:
        encoded=io.BytesIO();np.lib.format.write_array_header_2_0(encoded,dict(descr='<f8',fortran_order=False,shape=shape))
        write_all(out,encoded.getvalue())
        for raw,value in blocks(stream,base+low,shape):
            promoted=value.tobytes(order='C');write_all(out,promoted)
            raw_hash.update(raw);value_hash.update(promoted);count+=len(promoted)
    if count!=8*shape[0]*shape[1]:raise ValueError('full public matrix required')
    record=dict(name=name,shape=list(shape),source_dtype='BF16',source_raw_sha256=raw_hash.hexdigest(),
        dtype='<f8',sha256=value_hash.hexdigest(),promotion_exact=True,file=Path(path).name,
        file_sha256=file_sha(path),storage_alias_of=None)
    verify_matrix(path,record)
    return record
def verify_matrix(path,row):
    with owned_stream(path) as f:
        if np.lib.format.read_magic(f)!=(2,0):raise ValueError('exact NPY2 matrix required')
        shape,fortran,dtype=np.lib.format.read_array_header_2_0(f,max_header_size=1024)
        if list(shape)!=row['shape'] or fortran is not False or dtype.str!='<f8':raise ValueError('saved public matrix header differs')
        h=hashlib.sha256();remaining=shape[0]*shape[1]*8
        while remaining:
            raw=f.read(min(remaining,CHUNK_ROWS*shape[1]*8))
            if not raw or len(raw)%8:raise ValueError('truncated public saved matrix')
            if not np.isfinite(np.frombuffer(raw,dtype='<f8')).all():raise ValueError('nonfinite saved matrix')
            h.update(raw);remaining-=len(raw)
        if f.read(1) or h.hexdigest()!=row['sha256']:raise ValueError('saved public matrix payload differs')
    if file_sha(path)!=row['file_sha256']:raise ValueError('saved matrix file changed')
    return True
def tied_alias(stream,metadata,base,file_size,*,source_name,target_name,shape,record):
    """Compare actual BF16 bytes before sharing one physical public matrix."""
    a,_=descriptor(metadata,source_name,shape,base,file_size)
    b,_=descriptor(metadata,target_name,shape,base,file_size)
    raw_hash=hashlib.sha256();value_hash=hashlib.sha256()
    for start in range(0,shape[0],CHUNK_ROWS):
        n=min(CHUNK_ROWS,shape[0]-start)*shape[1]*2
        stream.seek(base+a+start*shape[1]*2);left=stream.read(n)
        stream.seek(base+b+start*shape[1]*2);right=stream.read(n)
        if len(left)!=n or len(right)!=n or left!=right:raise ValueError('configured tied matrices differ in actual BF16 bytes')
        values=(np.frombuffer(right,dtype='<u2').astype('<u4')<<16).view('<f4').astype('<f8')
        if not np.isfinite(values).all():raise ValueError('nonfinite tied public matrix')
        raw_hash.update(right);value_hash.update(values.tobytes())
    if raw_hash.hexdigest()!=record['source_raw_sha256'] or value_hash.hexdigest()!=record['sha256']:
        raise ValueError('tied matrix differs from exported matrix')
    return record|dict(name=target_name,storage_alias_of=source_name,exact_bf16_alias_verified=True)
def backend_gate():
    if any(name.split('.')[0] in FORBIDDEN for name in sys.modules):raise RuntimeError('forbidden backend/model import')

def run(output,body):
    output=Path(output);body=Path(body);output.mkdir(mode=0o700,exist_ok=False);body.mkdir(mode=0o700,exist_ok=False)
    report=dict(passed=False,operation_complete=False,termination_scope_completed=False,pid=os.getpid(),
        model_forward=False,backend_imported=False,he=False,matrices={},all_bodies_retained=True)
    start=time.monotonic()
    try:
        backend_gate();write_json(output/'report.json',report)
        if file_sha(HERE/'inputs/config.json')!=CONFIG or file_sha(HERE/'inputs/candidate_manifest.json')!=CANDIDATE:
            raise ValueError('exact public config/candidate required')
        config=json.loads((HERE/'inputs/config.json').read_bytes())
        if config.get('tie_word_embeddings')is not True:raise ValueError('published tied embedding config required')
        if file_sha(ASSETS/'config.json')!=CONFIG:raise ValueError('current public config differs')
        with owned_stream(ASSETS/'model.safetensors') as checkpoint:
            info=os.fstat(checkpoint.fileno())
            before=hashlib.file_digest(checkpoint,'sha256').hexdigest();checkpoint.seek(0)
            if before!=CHECKPOINT:raise ValueError('pinned public checkpoint required before tensor reads')
            report['checkpoint_sha256_before']=before
            metadata,base=header(checkpoint)
            rows=inventory()
            for name,shape in rows.items():descriptor(metadata,name,shape,base,info.st_size)
            for index,(name,shape) in enumerate(rows.items()):
                if name=='model.embed_tokens.weight':
                    row=tied_alias(checkpoint,metadata,base,info.st_size,source_name='lm_head.weight',target_name=name,shape=shape,record=report['matrices']['lm_head.weight'])
                else:
                    row=export_matrix(checkpoint,metadata,base,info.st_size,name,shape,body/f'matrix_{index:03d}.npy')
                report['matrices'][name]=row
            checkpoint.seek(0);after=hashlib.file_digest(checkpoint,'sha256').hexdigest()
            if after!=CHECKPOINT or file_identity(os.fstat(checkpoint.fileno()))!=file_identity(info):raise ValueError('held public checkpoint changed')
            report['checkpoint_sha256_after']=after
        if file_sha(ASSETS/'model.safetensors')!=CHECKPOINT or file_sha(ASSETS/'config.json')!=CONFIG:
            raise ValueError('public checkpoint/config path changed')
        files={r['file'] for r in report['matrices'].values()}
        if len(report['matrices'])!=114 or len(files)!=113 or set(p.name for p in body.iterdir())!=files:
            raise ValueError('all114 named matrices/113 bodies required')
        for row in report['matrices'].values():verify_matrix(body/row['file'],row)
        backend_gate()
        manifest=dict(schema='qwen-decoder-public-parameters-v1',checkpoint_sha256=CHECKPOINT,config_sha256=CONFIG,
            candidate_sha256=CANDIDATE,matrices=report['matrices'],named_matrix_count=114,physical_matrix_count=113,
            chunk_rows=CHUNK_ROWS,tied_embedding_and_head_verified=True)
        write_json(output/'parameter_manifest.json',manifest)
        report.update(operation_complete=True,passed=True,parameter_manifest_sha256=file_sha(output/'parameter_manifest.json'),
            physical_matrix_count=113,named_matrix_count=114,elapsed_s=time.monotonic()-start,
            scope='Public exact tensor extraction only; no model inference, backend, keys or encrypted evaluation.')
    except BaseException as error:
        report.update(passed=False,error=dict(type=type(error).__name__,message=str(error)),elapsed_s=time.monotonic()-start)
        try:write_json(output/'report.json',report)
        except BaseException as publication:error.add_note('failure publication: '+type(publication).__name__)
        raise
    write_json(output/'report.json',report)
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--body',type=Path,required=True);args=p.parse_args()
    from finish import finish
    result=finish(args.output/'report.json',lambda:run(args.output,args.body),write_json,public_report)
    print(json.dumps(dict(passed=True,named_matrices=114,physical_matrices=113,elapsed_s=result['elapsed_s'])))
