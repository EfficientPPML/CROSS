"""Pinned public Q/K/V tensor extraction without model/backend imports."""
from pathlib import Path
import hashlib,io,json,os,stat,struct,sys,tarfile,time
import numpy as np

ROOT=Path(__file__).resolve().parent
MODEL=Path('/home/jianming_gatech/cross_validation_assets/qwen/c1899de289a04d12100db370d81485cdf75e47ca')
CHECKPOINT='f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b'
CONFIG='660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd'
CANDIDATE='7f6478a6bc710f3d2a370198e075e6ba8b0e55c144d981bca0eed59eaf176333'


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def byte_sha(raw):return hashlib.sha256(raw).hexdigest()
def inventory():
    return {f'model.layers.{layer}.self_attn.{kind}_proj.weight':(2048 if kind=='q' else 1024,1024)
            for layer in range(28) for kind in ('q','k','v')}
def unique_object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('duplicate tensor header key')
        result[key]=value
    return result
def header(stream):
    prefix=stream.read(8)
    if len(prefix)!=8:raise ValueError('truncated tensor header')
    size=struct.unpack('<Q',prefix)[0]
    if not 2<=size<=16*1024**2:raise ValueError('bounded tensor header required')
    raw=stream.read(size)
    if len(raw)!=size:raise ValueError('truncated tensor header')
    value=json.loads(raw,object_pairs_hook=unique_object)
    if type(value)is not dict:raise ValueError('tensor mapping required')
    return value,8+size
def tensor(stream,metadata,data_start,name,expected_shape,file_size):
    row=metadata.get(name)
    if (type(row)is not dict or set(row)!={'dtype','shape','data_offsets'} or
        row['dtype']!='BF16' or type(row['shape'])is not list or
        any(type(n)is not int or n<1 for n in row['shape']) or tuple(row['shape'])!=tuple(expected_shape)):
        raise ValueError('exact public BF16 matrix shape required: '+name)
    offsets=row['data_offsets']
    if (type(offsets)is not list or len(offsets)!=2 or any(type(n)is not int for n in offsets) or
        not 0<=offsets[0]<offsets[1] or data_start+offsets[1]>file_size or
        offsets[1]-offsets[0]!=2*int(np.prod(expected_shape))):
        raise ValueError('invalid public tensor byte range')
    stream.seek(data_start+offsets[0]);raw=stream.read(offsets[1]-offsets[0])
    if len(raw)!=offsets[1]-offsets[0]:raise ValueError('truncated public tensor')
    # BF16 -> binary32 by exact bit placement -> binary64 exact promotion.
    promoted=(np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).view('<f4').astype('<f8').reshape(expected_shape)
    if not np.isfinite(promoted).all():raise ValueError('nonfinite public matrix')
    return promoted,dict(name=name,shape=list(expected_shape),source_dtype='BF16',
        source_raw_sha256=byte_sha(raw),dtype='float64',sha256=byte_sha(promoted.tobytes()),promotion_exact=True)


def collect():
    out=ROOT/'qkv-export-v1';out.mkdir(exist_ok=False)
    body=ROOT/'public_qkv_parameters';body.mkdir(mode=0o700,exist_ok=False)
    names=('export_qkv.py','test_export_qkv.py','inputs/config.json','inputs/candidate_manifest.json')
    report=dict(passed=False,model_forward=False,backend_imported=False,scope='Only84 pinned public BF16 Q/K/V tensors; bodies stay outside public evidence.',arrays={})
    start=time.monotonic();primary=None;stream=None
    def write():
        tmp=out/'report.tmp';tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(out/'report.json')
    try:
        write();before={name:sha(ROOT/name) for name in names};report['source_hashes_before']=before
        with tarfile.open(out/'sources.tar.gz','w:gz') as t:
            for name in names:
                raw=(ROOT/name).read_bytes();m=tarfile.TarInfo(name);m.size=len(raw);t.addfile(m,io.BytesIO(raw))
        if before['inputs/config.json']!=CONFIG or before['inputs/candidate_manifest.json']!=CANDIDATE:
            raise ValueError('fixed public source prerequisites differ')
        assets={'config.json':sha(MODEL/'config.json')}
        fd=os.open(MODEL/'model.safetensors',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        stream=os.fdopen(fd,'rb');info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):raise ValueError('regular pinned public checkpoint required')
        assets['model.safetensors']=hashlib.file_digest(stream,'sha256').hexdigest();stream.seek(0)
        report['asset_sha256_before']=assets
        if assets!={'config.json':CONFIG,'model.safetensors':CHECKPOINT}:
            raise ValueError('actual pinned model/config gate failed before tensor access')
        metadata,start_data=header(stream)
        for name,shape in inventory().items():
            value,descriptor=tensor(stream,metadata,start_data,name,shape,info.st_size)
            layer=name.split('.')[2];kind=name.split('.')[4].removesuffix('_proj')
            filename=f'layer{layer}_{kind}.npy'
            with (body/filename).open('xb') as f:np.save(f,value,allow_pickle=False)
            descriptor['file']=filename;descriptor['file_sha256']=sha(body/filename)
            loaded=np.load(body/filename,mmap_mode='r',allow_pickle=False)
            if (loaded.dtype!=np.dtype('float64') or loaded.shape!=shape or loaded.flags.writeable or
                byte_sha(np.asarray(loaded,dtype='<f8').tobytes())!=descriptor['sha256']):
                raise ValueError('public matrix readback differs')
            report['arrays'][name]=descriptor;del loaded,value
        public=dict(schema='qwen-qkv-public-parameters-v1',checkpoint_sha256=CHECKPOINT,
            config_sha256=CONFIG,candidate_sha256=CANDIDATE,matrices=report['arrays'])
        raw=(json.dumps(public,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
        (out/'parameter_manifest.json').write_bytes(raw)
        report['parameter_manifest_sha256']=byte_sha(raw);report['operation_complete']=True
    except BaseException as error:
        primary=error;report['error']=dict(type=type(error).__name__,message=str(error))
    finally:
        errors=[]
        def attempt(label,fn):
            try:return fn()
            except BaseException as error:errors.append((label,error));return None
        if stream is not None:
            attempt('checkpoint rewind',lambda:stream.seek(0))
            report['held_checkpoint_sha256_after']=attempt('checkpoint readback',lambda:hashlib.file_digest(stream,'sha256').hexdigest())
            attempt('checkpoint close',stream.close)
        report['asset_sha256_after']=attempt('asset final readback',lambda:{n:sha(MODEL/n) for n in ('config.json','model.safetensors')})
        report['source_hashes_after']=attempt('source final readback',lambda:{n:sha(ROOT/n) for n in names})
        def archive_check():
            with tarfile.open(out/'sources.tar.gz') as t:
                ms=t.getmembers()
                return len(ms)==len({m.name for m in ms})==len(names) and all(m.isfile() for m in ms) and {m.name:byte_sha(t.extractfile(m).read()) for m in ms}==report['source_hashes_before']
        report['archive_verified']=attempt('archive readback',archive_check)
        report['source_archive_sha256']=attempt('archive hash',lambda:sha(out/'sources.tar.gz'))
        report['elapsed_s']=time.monotonic()-start
        report['passed']=(primary is None and not errors and report.get('operation_complete')is True
            and len(report['arrays'])==84 and report.get('archive_verified')is True
            and report.get('source_hashes_before')==report.get('source_hashes_after')
            and report.get('asset_sha256_before')==report.get('asset_sha256_after')
            and report.get('held_checkpoint_sha256_after')==CHECKPOINT)
        if any(n in sys.modules for n in ('torch','transformers','jax','jaxlib','safetensors')):
            errors.append(('backend',RuntimeError('forbidden backend/model import')));report['passed']=False
        attempt('report publication',write)
        if primary is not None:
            for label,error in errors:primary.add_note(label+': '+repr(error))
        elif errors:
            primary=next((e for _,e in errors if not isinstance(e,Exception)),errors[0][1])
            for label,error in errors:
                if error is not primary:primary.add_note(label+': '+repr(error))
    if primary is not None:raise primary
    if report['passed']is not True:raise RuntimeError('public export provenance failed')
    print(json.dumps(dict(passed=True,tensors=84,elapsed_s=report['elapsed_s'],parameter_manifest_sha256=report['parameter_manifest_sha256'])))
if __name__=='__main__':collect()
