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
from client_keys import _CloseState
from file_stage_io import new_file, read_manifest
from transport_contract import canonical, digest, _sha


def _paths(paths):
    return {name:Path(paths[name]) for name in ('upload','download','server_report','decrypted_results')} | {
        'client_requests':Path(paths['client_private'])/'requests.json'}


def _identity(info):return dict(device=info.st_dev,inode=info.st_ino)


def _file_state(info):
    if not stat.S_ISREG(info.st_mode):raise ValueError('completed artifact is not a regular file')
    return _identity(info)|dict(size=info.st_size,mtime_ns=info.st_mtime_ns,ctime_ns=info.st_ctime_ns)


def _tree(fd,kind,nonces,*,_close_state=None):
    state=_CloseState()if _close_state is None else _close_state
    if set(os.listdir(fd))!=set(nonces)|{'batch.json'}:
        raise ValueError('completed ciphertext directory inventory changed')
    states={'batch.json':_file_state(os.stat('batch.json',dir_fd=fd,follow_symlinks=False))}
    for nonce in nonces:
        sample=os.open(nonce,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            states[nonce]=_identity(os.fstat(sample))
            expected={f'tile{i}' for i in range(8)} if kind=='upload' else {'bundle.json','ct_0.npz','transport.json'}
            if set(os.listdir(sample))!=expected:raise ValueError('completed sample inventory changed')
            if kind=='upload':
                for tile in sorted(expected):
                    tile_fd=os.open(tile,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=sample)
                    try:
                        states[nonce+'/'+tile]=_identity(os.fstat(tile_fd))
                        if set(os.listdir(tile_fd))!={'bundle.json','ct_0.npz','transport.json'}:
                            raise ValueError('completed tile inventory changed')
                        for name in sorted(os.listdir(tile_fd)):
                            states[nonce+'/'+tile+'/'+name]=_file_state(os.stat(name,dir_fd=tile_fd,follow_symlinks=False))
                    finally:state.close(tile_fd)
            else:
                for name in sorted(expected):
                    states[nonce+'/'+name]=_file_state(os.stat(name,dir_fd=sample,follow_symlinks=False))
        finally:state.close(sample)
    return states


def _observe(path,kind,nonces):
    """Bounded descriptor reads sharing cleanup origin through nested handles."""
    state=_CloseState()
    def read_owned(fd):
        initial=os.fstat(fd);before=_file_state(initial)
        if initial.st_size>16<<20:raise ValueError('bounded regular completed output required')
        chunks=[];remaining=(16<<20)+1
        while remaining:
            block=os.read(fd,min(remaining,1<<20))
            if not block:break
            chunks.append(block);remaining-=len(block)
        raw=b''.join(chunks)
        if len(raw)>16<<20:raise ValueError('completed output exceeds byte bound')
        if len(raw)!=initial.st_size or _file_state(os.fstat(fd))!=before:
            raise ValueError('completed output changed during descriptor read')
        return initial,raw
    if kind in ('upload','download'):
        fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
        try:
            identity=os.fstat(fd);children=_tree(fd,kind,nonces,_close_state=state)
            metadata=os.open('batch.json',os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
            try:
                metadata_identity,raw=read_owned(metadata)
                if _file_state(metadata_identity)!=children['batch.json']:
                    raise ValueError('completed batch metadata replaced before read')
            finally:state.close(metadata)
            if _tree(fd,kind,nonces,_close_state=state)!=children:raise ValueError('completed child identity changed during capture')
        finally:state.close(fd)
    else:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:identity,raw=read_owned(fd)
        finally:state.close(fd)
    current=path.lstat()
    if _identity(current)!=_identity(identity):raise ValueError('completed output changed during capture')
    if kind not in ('upload','download')and _file_state(current)!=_file_state(identity):
        raise ValueError('completed file state changed during capture')
    result=_identity(identity)|dict(sha256=hashlib.sha256(raw).hexdigest())
    if kind in ('upload','download'):result['children']=children
    return result


def completion_path(paths,completion_directory=None):
    """Choose a private receipt destination without changing retirement paths.

    The default retains the original transaction ledger. A component caller may
    instead supply its own private result directory; this never grants authority
    to retire the original borrowed upload, request record or key workspace.
    """
    if completion_directory is None:return Path(paths['client_private'])/'completed.json'
    root=Path(completion_directory);info=root.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_IMODE(info.st_mode)!=0o700:
        raise ValueError('ordinary private0700 completion directory required')
    if root.resolve()==Path(paths['client_private']).resolve():
        raise ValueError('explicit separate completion directory required')
    return root/'completed.json'


def record_completed(paths,*,context_id,batch,completion_directory=None):
    _sha(context_id)
    if batch['public']['context_id']!=context_id:raise ValueError('completion key context differs')
    nonces=[r['contract']['request_id'] for r in batch['requests']]
    if not nonces or len(set(nonces))!=len(nonces):raise ValueError('unique completed request inventory required')
    objects={kind:_observe(path,kind,nonces) for kind,path in _paths(paths).items()}
    if objects['upload']['sha256']!=hashlib.sha256(canonical(batch)).hexdigest():
        raise ValueError('completed upload bytes differ from validated batch')
    receipt=dict(schema='cross-bert-client-completion-v1',context_id=context_id,
        public_batch_sha256=digest(batch),nonces=nonces,objects=objects)
    with new_file(completion_path(paths,completion_directory),canonical(receipt)):pass
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
            fields={'device','inode','sha256'}|({'children'} if kind in ('upload','download') else set())
            if (type(desc)is not dict or set(desc)!=fields
                    or any(type(desc[n])is not int or desc[n]<0 for n in ('device','inode'))):
                raise ValueError('completion ownership descriptor differs')
            _sha(desc['sha256'])
            if kind=='upload' and desc['sha256']!=receipt['public_batch_sha256']:
                raise ValueError('completion upload binding differs')
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
            if _observe(path,kind,nonces)!=desc:
                raise ValueError('completed output children/bytes changed during retirement; retained')
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
