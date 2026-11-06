"""Retire only a completed, client-recorded request before the next harness run.

The upstream harness reuses upload/download/results paths. A private completion
receipt records their inode and byte identities after successful decryption.
An interrupted or unrecognized request is retained for review. These are
ownership/concurrency checks, not an OS sandbox against hostile processes.
"""
from pathlib import Path
import hashlib
import os
import shutil
import stat
import sys
from client_keys import _read
from file_stage_io import new_file, read_manifest
from transport_contract import canonical, digest, _sha


def _paths(paths):
    return {name:Path(paths[name]) for name in ('upload','download','server_report','decrypted_results')} | {
        'client_requests':Path(paths['client_private'])/'requests.json'}


def _identity(info):return dict(device=info.st_dev,inode=info.st_ino)


def _tree(fd,kind,nonces):
    if set(os.listdir(fd))!=set(nonces)|{'batch.json'}:
        raise ValueError('completed ciphertext directory inventory changed')
    for nonce in nonces:
        sample=os.open(nonce,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            expected={f'tile{i}' for i in range(8)} if kind=='upload' else {'bundle.json','ct_0.npz','transport.json'}
            if set(os.listdir(sample))!=expected:raise ValueError('completed sample inventory changed')
            if kind=='upload':
                for tile in sorted(expected):
                    tile_fd=os.open(tile,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=sample)
                    try:
                        if set(os.listdir(tile_fd))!={'bundle.json','ct_0.npz','transport.json'}:
                            raise ValueError('completed tile inventory changed')
                        if any(not stat.S_ISREG(os.stat(n,dir_fd=tile_fd,follow_symlinks=False).st_mode) for n in os.listdir(tile_fd)):
                            raise ValueError('completed tile contains nonregular files')
                    finally:os.close(tile_fd)
            elif any(not stat.S_ISREG(os.stat(n,dir_fd=sample,follow_symlinks=False).st_mode) for n in expected):
                raise ValueError('completed response contains nonregular files')
        finally:os.close(sample)


def _observe(path,kind,nonces):
    if kind in ('upload','download'):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            identity=os.fstat(fd);_tree(fd,kind,nonces)
            raw=_read(fd,'batch.json',16<<20)
        finally:os.close(fd)
    else:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            identity=os.fstat(fd)
            if not stat.S_ISREG(identity.st_mode) or identity.st_size>16<<20:
                raise ValueError('bounded regular completed output required')
            with os.fdopen(os.dup(fd),'rb') as stream:raw=stream.read((16<<20)+1)
            if len(raw)>16<<20:raise ValueError('completed output exceeds byte bound')
        finally:os.close(fd)
    current=path.lstat()
    if _identity(current)!=_identity(identity):raise ValueError('completed output changed during capture')
    return _identity(identity)|dict(sha256=hashlib.sha256(raw).hexdigest())


def record_completed(paths,*,context_id,batch):
    _sha(context_id)
    if batch['public']['context_id']!=context_id:raise ValueError('completion key context differs')
    nonces=[r['contract']['request_id'] for r in batch['requests']]
    if not nonces or len(set(nonces))!=len(nonces):raise ValueError('unique completed request inventory required')
    objects={kind:_observe(path,kind,nonces) for kind,path in _paths(paths).items()}
    if objects['upload']['sha256']!=hashlib.sha256(canonical(batch)).hexdigest():
        raise ValueError('completed upload bytes differ from validated batch')
    receipt=dict(schema='cross-bert-client-completion-v1',context_id=context_id,
        public_batch_sha256=digest(batch),nonces=nonces,objects=objects)
    with new_file(Path(paths['client_private'])/'completed.json',canonical(receipt)):pass
    return receipt


def retire_completed(paths,*,context_id):
    _sha(context_id);receipt_path=Path(paths['client_private'])/'completed.json'
    if not receipt_path.exists() and not receipt_path.is_symlink():return False
    receipt=read_manifest(receipt_path);objects=_paths(paths)
    if (type(receipt) is not dict or set(receipt)!={'schema','context_id','public_batch_sha256','nonces','objects'}
            or receipt['schema']!='cross-bert-client-completion-v1' or receipt['context_id']!=context_id
            or type(receipt['objects']) is not dict or set(receipt['objects'])!=set(objects)):
        raise ValueError('unrecognized or foreign completion receipt; artifacts retained')
    _sha(receipt['public_batch_sha256']);nonces=receipt['nonces']
    if (type(nonces) is not list or not 1<=len(nonces)<=408 or any(type(n)is not str or len(n)!=32 or any(c not in '0123456789abcdef' for c in n) for n in nonces)
            or len(set(nonces))!=len(nonces)):
        raise ValueError('completion nonce inventory differs')
    held=[]
    try:
        # Validate every remaining object before deleting any of them. Missing
        #objects are allowed only here, for retry after partial retirement.
        for kind,path in objects.items():
            desc=receipt['objects'][kind]
            if (type(desc)is not dict or set(desc)!={'device','inode','sha256'}
                    or any(type(desc[n])is not int or desc[n]<0 for n in ('device','inode'))):
                raise ValueError('completion ownership descriptor differs')
            _sha(desc['sha256'])
            try:info=path.lstat()
            except FileNotFoundError:continue
            if _identity(info)!={n:desc[n] for n in ('device','inode')}:
                raise ValueError('completed output replaced; replacement retained')
            if _observe(path,kind,nonces)!=desc:raise ValueError('completed output byte identity changed; retained')
            parent=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
            held.append((kind,path,parent,os.fstat(parent),desc))
        receipt_desc=_observe(receipt_path,'receipt',nonces)
        if receipt_desc['sha256']!=hashlib.sha256(canonical(receipt)).hexdigest():
            raise ValueError('completion receipt changed during retirement')
        parent=os.open(receipt_path.parent,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        held.append(('receipt',receipt_path,parent,os.fstat(parent),receipt_desc))
        for kind,path,parent,parent_info,desc in held:
            if _identity(path.parent.lstat())!=_identity(parent_info):
                raise ValueError('completed output parent replaced; replacement retained')
            current=os.stat(path.name,dir_fd=parent,follow_symlinks=False)
            if _identity(current)!={n:desc[n] for n in ('device','inode')}:
                raise ValueError('completed output replaced during retirement; retained')
            if kind in ('upload','download'):shutil.rmtree(path.name,dir_fd=parent)
            else:os.unlink(path.name,dir_fd=parent)
        return True
    finally:
        active=sys.exception();failure=None
        for _,_,fd,_,_ in held:
            try:os.close(fd)
            except BaseException as error:
                if active is not None:active.add_note('completion handle close: '+repr(error))
                elif failure is None:failure=error
                else:failure.add_note('additional completion handle close: '+repr(error))
        if active is None and failure is not None:raise failure
