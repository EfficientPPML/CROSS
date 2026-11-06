"""Concrete bounded key readers. Callers must admit a role before real use.

This preparation exercises only newly created small synthetic files. Public
open_client/open_server remain disabled; these trusted helpers are not grants.
"""
from pathlib import Path
import hashlib,io,math,os,stat,sys
import numpy as np
from cleanup_origin import protect_body,select,mark
from key_descriptor_io import _CloseState,_stream,_sha_fd,_IO_FAILURE_QUARANTINE
from q5_key_contract import ArraySpec,PublicKeyBinding,PrivateKeyBinding,PUBLIC_LIMIT,PRIVATE_LIMIT

_QUARANTINE=[]
HEADER_LIMIT=512


def _file_state(info):return(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)


def _flat(name):
    if type(name)is not str or not name or name in ('.','..')or any(x in name for x in ('/','\\','\x00')):
        raise ValueError('flat key filename required')
    return name


def read_metadata(owner,name,maximum):
    """Original descriptor-only bounded read with held-name/private checks."""
    _flat(name)
    if type(maximum)is not int or not 0<maximum<=PUBLIC_LIMIT:raise ValueError('bounded metadata limit required')
    owner.check();state=_CloseState()
    before=os.stat(name,dir_fd=owner.fd,follow_symlinks=False)
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=owner.fd)
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode)or not 0<info.st_size<=maximum or
                _file_state(info)!=_file_state(before)or
                (owner.private and stat.S_IMODE(info.st_mode)!=0o600)):
            raise ValueError('bounded held key metadata identity or mode differs')
        chunks=[];remaining=maximum+1
        while remaining:
            block=os.read(fd,min(remaining,1<<20))
            if not block:break
            chunks.append(block);remaining-=len(block)
        raw=b''.join(chunks);after=os.fstat(fd)
        visible=os.stat(name,dir_fd=owner.fd,follow_symlinks=False)
        if (len(raw)!=info.st_size or len(raw)>maximum or
                _file_state(after)!=_file_state(info)or _file_state(visible)!=_file_state(info)or
                (owner.private and (stat.S_IMODE(after.st_mode)!=0o600 or stat.S_IMODE(visible.st_mode)!=0o600))):
            raise ValueError('held key metadata changed during read')
        owner.check();return raw
    finally:state.close(fd)


class ReadDirectory:
    def __init__(self,path,*,private=False):
        if type(private)is not bool:raise TypeError('exact private flag required')
        self.path=Path(path);self.private=private;self.fd=None;self.failed=None
        try:
            self.fd=os.open(self.path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            info=os.fstat(self.fd);self.identity=(info.st_dev,info.st_ino);self.check()
        except BaseException as primary:
            protect_body(primary)
            try:self.close()
            except BaseException as secondary:
                chosen=select(primary,[('directory setup close',secondary)])
                if chosen is not primary:raise chosen
            raise
    def check(self):
        if self.failed is not None:raise self.failed
        if self.fd is None:raise ValueError('key directory is closed')
        info=self.path.lstat();held=os.fstat(self.fd)
        if (not stat.S_ISDIR(info.st_mode)or (info.st_dev,info.st_ino)!=self.identity or
                (held.st_dev,held.st_ino)!=self.identity or
                (self.private and (stat.S_IMODE(info.st_mode)!=0o700 or stat.S_IMODE(held.st_mode)!=0o700))):
            raise ValueError('held key directory identity or private permissions differ')
    def close(self):
        if self.failed is not None:raise self.failed
        fd=self.fd;self.fd=None
        if fd is not None:
            try:os.close(fd)
            except BaseException as error:
                self.failed=mark(error);_QUARANTINE.append((self,fd,error));raise


def _header(stream,size,shape):
    """Adapted reviewed public-parameter pre-read cap to uint32 keys/512B."""
    version=np.lib.format.read_magic(stream)
    if version not in ((1,0),(2,0)):raise ValueError('unsupported NPY key version')
    width=2 if version==(1,0)else 4
    prefix=stream.read(width)
    if len(prefix)!=width:raise ValueError('truncated NPY key header length')
    length=int.from_bytes(prefix,'little')
    if not 0<length<=HEADER_LIMIT or stream.tell()+length>min(size,HEADER_LIMIT):
        raise ValueError('NPY key header exceeds pre-read bound')
    header=stream.read(length)
    if len(header)!=length:raise ValueError('truncated NPY key header')
    reader=np.lib.format.read_array_header_1_0 if version==(1,0)else np.lib.format.read_array_header_2_0
    stored,fortran,dtype=reader(io.BytesIO(prefix+header),max_header_size=HEADER_LIMIT)
    if (type(stored)is not tuple or any(type(x)is not int for x in stored)or stored!=shape or
            fortran or dtype!=np.dtype('uint32')):
        raise ValueError('exact uint32 C key header required')
    offset=stream.tell()
    if offset>HEADER_LIMIT or offset+math.prod(shape)*4!=size:raise ValueError('exact NPY key extent differs')
    return offset


def map_array(owner,name,description,shape,moduli):
    """Authenticate and map one SAME bounded fd; no dup or handoff."""
    _flat(name)
    if (type(shape)is not tuple or len(shape)<2 or any(type(x)is not int or x<=0 for x in shape)or
            type(moduli)not in (tuple,list)or len(moduli)!=shape[-2]or
            any(type(x)is not int or not 1<x<2**32 for x in moduli)):
        raise ValueError('typed uint32 residue shape/moduli required')
    ArraySpec(name,shape,tuple(moduli)).validate(description)
    owner.check();state=_CloseState();mapped=None
    fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=owner.fd)
    try:
        info=os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode)or info.st_size!=description['bytes']or
                (owner.private and stat.S_IMODE(info.st_mode)!=0o600)):
            raise ValueError('bounded regular key file or private mode differs')
        with _stream(fd,'rb',state)as stream:offset=_header(stream,info.st_size,shape)
        if _sha_fd(fd)!=description['sha256']:raise ValueError('key file hash differs')
        with _stream(fd,'rb',state)as stream:
            mapped=np.memmap(stream,mode='r',dtype=np.uint32,shape=shape,offset=offset)
        for limb,modulus in enumerate(moduli):
            if np.any(mapped[...,limb,:]>=modulus):raise ValueError('key residue outside modulus')
        owner.check()
        visible=os.stat(name,dir_fd=owner.fd,follow_symlinks=False)
        if _file_state(os.fstat(fd))!=_file_state(info)or _file_state(visible)!=_file_state(info):
            raise ValueError('key file identity/extent changed during mapping')
        result=(mapped,_file_state(info))
    finally:
        active=protect_body(sys.exception())
        try:state.close(fd)
        finally:
            if mapped is not None and (active is not None or sys.exception()is not None):
                state.cleanup('key mapping cleanup',mapped._mmap.close,mapped)
    return result


def read_public(owner,binding):
    if type(binding)is not PublicKeyBinding:raise TypeError('exact public key binding required')
    expected=binding.check();owner.check()
    raw=read_metadata(owner,'public_manifest.json',PUBLIC_LIMIT)
    if raw!=binding.raw:raise ValueError('externally authenticated public manifest changed')
    specs={s.name:(s.shape,s.moduli)for s in binding.profile.arrays}
    if set(os.listdir(owner.fd))!=set(specs)|{'public_manifest.json'}:
        raise ValueError('exact203 public key files required')
    owner.check();return expected,specs,raw


def read_private(owner,binding,raw_public):
    if type(binding)is not PrivateKeyBinding:raise TypeError('exact private key binding required')
    expected=binding.check();owner.check()
    if binding.public.raw!=raw_public:raise ValueError('private reader public identity differs')
    if set(os.listdir(owner.fd))!={'secret.npy','private_manifest.json'}:
        raise ValueError('private key store must contain only secret and manifest')
    for name in ('secret.npy','private_manifest.json'):
        info=os.stat(name,dir_fd=owner.fd,follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode)or stat.S_IMODE(info.st_mode)!=0o600:
            raise ValueError('private key files must be regular0600')
    raw=read_metadata(owner,'private_manifest.json',PRIVATE_LIMIT)
    if raw!=binding.raw:raise ValueError('externally authenticated private manifest changed')
    owner.check();return expected,raw
