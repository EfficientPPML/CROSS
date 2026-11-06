"""Future dedicated-process audit integration probe; never a capacity grant.

This child reads only its own selected proc metadata after authenticating the
closed runtime sources. All intent/profile/resource objects below are synthetic
scope fixtures. No capacity(), policies(), key/model loader or backend is called.
"""
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

MAP_SHA256='01e5f8681ff5731d8d19d8f488b692bcf38025b83a0dbb3036b536a3d9f1726e'
DEPENDENCIES_SHA256='9e6760172efb58342b4aee5d7a3ae1799930869d0ae479258d691ba9e7365ca4'
FORBIDDEN={'numpy','jax','jaxlib','jaxite_word','libtpu','torch','tensorflow','flax',
    'torch_xla','transformers','safetensors','tokenizers'}
SELECTED=('/proc/self/status','/proc/self/cgroup','/proc/self/mountinfo')
BACKEND_ATTEMPTS=[]


def canonical(value):
    return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()


def authenticate(root,map_path):
    assert root.is_absolute() and root.resolve()==root
    raw=map_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==MAP_SHA256
    mapping=json.loads(raw)
    assert canonical(mapping)==raw and len(mapping)==85
    assert mapping['keygen_dependencies.py']==DEPENDENCIES_SHA256
    names=[]
    for path in sorted(root.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():names.append(path.relative_to(root).as_posix())
    assert names==sorted(mapping)
    for name,digest in mapping.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
    return mapping


def deny_backends(mapping):
    # Flat core aliases are forbidden too; none of the copied core is imported.
    for name in mapping:
        if name.startswith('runtime/jaxite_word/') and name.endswith('.py'):
            first=name[len('runtime/jaxite_word/'):].split('/')[0]
            if first!='__init__.py':FORBIDDEN.add(first.removesuffix('.py'))
    assert not any(name.split('.')[0]in FORBIDDEN for name in sys.modules)
    class Deny(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0]in FORBIDDEN:
                BACKEND_ATTEMPTS.append(fullname)
                raise PermissionError('backend forbidden in proc integration probe')
    sys.meta_path.insert(0,Deny())
    def audit(event,args):
        if event=='import' and args and str(args[0]).split('.')[0]in FORBIDDEN:
            BACKEND_ATTEMPTS.append(str(args[0]))
            raise PermissionError('backend forbidden in proc integration probe')
    sys.addaudithook(audit)


def check_local_origins(root,mapping):
    observed={}
    for name,module in tuple(sys.modules.items()):
        if name=='__main__':continue
        values=(getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None))
        if not any(type(v)is str and (Path(v)==root or root in Path(v).parents)for v in values):continue
        assert all(type(v)is str for v in values)
        paths=tuple(Path(v)for v in values)
        assert paths[0]==paths[1] and paths[0].resolve()==paths[0]
        relative=paths[0].relative_to(root).as_posix()
        assert relative in mapping
        assert hashlib.sha256(paths[0].read_bytes()).hexdigest()==mapping[relative]
        observed[name]=relative
    assert observed['keygen_dependencies']=='keygen_dependencies.py'
    return observed


def close_once(fd):
    # The caller detaches before this call. There is no close retry.
    os.close(fd)


def read_fd(path,flags,**kwargs):
    fd=os.open(path,flags,**kwargs)
    try:
        data=os.read(fd,256)
        assert type(data)is bytes and data
    finally:
        original=fd;fd=None
        close_once(original)


def read_builtin(path,mode='rb'):
    # A builtin may normalize a PathLike before its audit event. Only literal
    # strings are used here; original-type rejection is tested via os.open.
    with open(path,mode)as stream:
        if mode=='rb':assert stream.read(256)


def rejected(operation):
    try:operation()
    except PermissionError as error:
        assert str(error).startswith('keygen audit denied ')
    else:raise AssertionError('unselected proc operation was admitted')


def main(case,root,map_path,owned):
    assert sys.flags.isolated and sys.dont_write_bytecode
    mapping=authenticate(root,map_path)
    deny_backends(mapping)
    sys.path.insert(0,str(root))
    import keygen_dependencies as kd
    origins=check_local_origins(root,mapping)
    assert owned.is_absolute() and owned.resolve()==owned and owned.is_dir()
    assert not tuple(owned.iterdir())
    before=owned.stat()
    assert stat.S_ISDIR(before.st_mode) and before.st_uid==os.geteuid()
    # Setup is restricted to this fresh pytest-owned directory. Directory and
    # alias fixtures precede installation so the audited test is only an open.
    alias=owned/'status-alias'
    if case=='symlink':alias.symlink_to('/proc/self/status')
    directory_fd=None
    if case in('relative_dirfd','absolute_dirfd'):
        directory_fd=os.open(owned,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW|os.O_CLOEXEC)
    intent={'paths':{name:str(owned/name)for name in('public_root','private_root','evidence_root')}}
    profile=SimpleNamespace(arrays=())
    dep={'installed_sources':{}}
    res={'cgroup_memory_current':'/sys/fs/cgroup/SYNTHETIC_UNOPENED/memory.current',
         'cgroup_memory_max':'/sys/fs/cgroup/SYNTHETIC_UNOPENED/memory.max'}
    protected=['/proc/self/status']if case=='protected'else[]
    original_open=os.open
    scope=kd.AccessScope(root,mapping,[],intent,profile,dep,res,protected)
    installed=False;body=None;close_error=None;read_succeeded=False
    negative=case not in('read_status','read_cgroup','read_mountinfo','builtin_status','builtin_cgroup',
        'builtin_mountinfo','flag_plain','flag_cloexec','flag_nofollow','flag_nonblock','flag_all','body_error')
    try:
        scope.install();installed=True
        assert os.open is scope.wrapper and scope.active
        if case.startswith('read_'):
            value=kd.kernel_read('/proc/self/'+case.removeprefix('read_'))
            assert type(value)is str and value
            read_succeeded=True
        elif case.startswith('builtin_') and case!='builtin_write':
            read_builtin('/proc/self/'+case.removeprefix('builtin_'));read_succeeded=True
        elif case in('flag_plain','flag_cloexec','flag_nofollow','flag_nonblock','flag_all'):
            flags={'flag_plain':0,'flag_cloexec':os.O_CLOEXEC,'flag_nofollow':os.O_NOFOLLOW,
                'flag_nonblock':os.O_NONBLOCK,'flag_all':os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK}[case]
            read_fd('/proc/self/status',flags);read_succeeded=True
        elif case in('body_error','swallowed_then_read'):
            if case=='swallowed_then_read':rejected(lambda:read_fd('/proc/self/environ',0))
            read_fd('/proc/self/status',os.O_CLOEXEC);read_succeeded=True
            if case=='body_error':
                body=OSError('synthetic body after permitted metadata read')
                raise body
        else:
            bad_flags={'write_wronly':os.O_WRONLY,'write_rdwr':os.O_RDWR,'write_creat':os.O_CREAT,
                'write_trunc':os.O_TRUNC,'write_append':os.O_APPEND,'flag_directory':os.O_DIRECTORY,
                'flag_path':os.O_PATH,'flag_bool':False}
            if case in bad_flags:rejected(lambda:read_fd('/proc/self/status',bad_flags[case]))
            elif case=='builtin_write':rejected(lambda:read_builtin('/proc/self/status','wb'))
            elif case=='bytes':rejected(lambda:read_fd(b'/proc/self/status',0))
            elif case=='pathlike':rejected(lambda:read_fd(Path('/proc/self/status'),0))
            elif case=='dot':rejected(lambda:read_fd('/proc/self/./status',0))
            elif case=='double_slash':rejected(lambda:read_fd('/proc//self/status',0))
            elif case=='explicit_pid':rejected(lambda:read_fd('/proc/'+str(os.getpid())+'/status',0))
            elif case=='unselected':rejected(lambda:read_fd('/proc/self/environ',0))
            elif case=='relative_dirfd':rejected(lambda:read_fd('status',0,dir_fd=directory_fd))
            elif case=='absolute_dirfd':rejected(lambda:read_fd('/proc/self/status',0,dir_fd=directory_fd))
            elif case=='symlink':rejected(lambda:read_fd(str(alias),0))
            elif case=='protected':rejected(lambda:read_fd('/proc/self/status',0))
            else:raise AssertionError('unknown fixed probe case')
        if negative:
            assert scope.denials
            try:scope.check()
            except PermissionError as error:assert str(error)=='sticky keygen audit denial'
            else:raise AssertionError('swallowed denial did not remain sticky')
        else:scope.check();assert scope.denials==[]
    except OSError as error:
        if body is None or error is not body:raise
    finally:
        # Close the original scope even if a body assertion fails. Its own
        # method must restore original os.open despite sticky denial.
        try:
            if installed:
                try:scope.close()
                except BaseException as error:close_error=error
        finally:
            if directory_fd is not None:
                original_fd=directory_fd;directory_fd=None
                close_once(original_fd)
    assert installed and os.open is original_open and scope.active is False
    assert getattr(scope.local,'open',None)is None
    if negative:
        assert type(close_error)is PermissionError and str(close_error)=='sticky keygen audit denial'
    else:assert close_error is None and scope.denials==[]
    assert case!='body_error' or type(body)is OSError
    assert kd._QUARANTINE==[]
    assert not BACKEND_ATTEMPTS
    assert not any(name.split('.')[0]in FORBIDDEN for name in sys.modules)
    assert check_local_origins(root,mapping)==origins
    assert authenticate(root,map_path)==mapping
    after=owned.stat()
    assert (before.st_dev,before.st_ino,before.st_uid)==(after.st_dev,after.st_ino,after.st_uid)
    assert sorted(p.name for p in owned.iterdir())==(['status-alias']if case=='symlink'else[])
    result={'schema':'qwen-proc-self-integration-observation-v1','case':case,'passed':True,
        'pid':os.getpid(),'pgid':os.getpgrp(),'source_manifest_sha256':MAP_SHA256,
        'runtime_sources':85,'local_module_origins':origins,'scope_restored':True,
        'sticky_denial_expected':negative,'denial_count':len(scope.denials),
        'permitted_metadata_read_completed':read_succeeded,'backend_attempts':[],
        'backend_modules':[],'resource_capacity_evaluated':False,'actual_generation_authorized':False}
    sys.stdout.buffer.write(canonical(result));sys.stdout.buffer.flush()


if __name__=='__main__':
    try:
        assert len(sys.argv)==5
        main(sys.argv[1],Path(sys.argv[2]),Path(sys.argv[3]),Path(sys.argv[4]))
    except BaseException as error:
        try:sys.stderr.write(type(error).__name__+': '+str(error)+'\n')
        except BaseException:pass
        raise SystemExit(70)
