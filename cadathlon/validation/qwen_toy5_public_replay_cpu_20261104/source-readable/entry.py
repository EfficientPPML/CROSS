"""Unexecuted CPU-public-toy harness. A separately pinned external grant is required.

No grant is shipped. Exit0 means replay consistency, never HE success; a finite
original numeric/domain failure remains explicit diagnostic data. Exit70 reserves
infrastructure/escaped BaseException, including late output-close failures.
"""
import argparse
import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys

FORBIDDEN={'jax','jaxlib','jaxite_word','torch','transformers','tensorflow','safetensors','libtpu'}


def canonical(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()


def read(path,limit):
    path=Path(path);before=path.lstat()
    if not stat.S_ISREG(before.st_mode)or before.st_size>limit:raise ValueError('bounded regular public source required')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW);primary=None
    try:
        opened=os.fstat(fd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
        if identity(opened)!=identity(before):raise ValueError('public source identity changed')
        pieces=[];remaining=limit+1
        while remaining:
            chunk=os.read(fd,min(remaining,1048576))
            if not chunk:break
            pieces.append(chunk);remaining-=len(chunk)
        raw=b''.join(pieces)
        if len(raw)!=opened.st_size or identity(os.fstat(fd))!=identity(opened)or identity(path.lstat())!=identity(opened):raise ValueError('public source changed during read')
        return raw
    except BaseException as error:primary=error;raise
    finally:
        try:os.close(fd)
        except BaseException:
            if primary is None:raise


def source_check(root,expected):
    if type(expected)is not dict or not expected:raise ValueError('exact source map')
    observed={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        for name in dirs:
            if Path(directory,name).is_symlink():raise ValueError('symlink source directory')
        for name in files:
            path=Path(directory,name);relative=path.relative_to(root).as_posix()
            if relative not in expected:raise ValueError('unexpected source file')
            observed[relative]=sha(read(path,8*1024**2))
    if observed!=expected:raise ValueError('source closure differs')
    for name,module in list(sys.modules.items()):
        for origin in (getattr(module,'__file__',None),getattr(getattr(module,'__spec__',None),'origin',None)):
            if origin and origin not in ('built-in','frozen'):
                p=Path(origin).resolve()
                if p.is_relative_to(root) and not (name=='__main__' and p==root/'entry.py'):
                    raise ValueError('unexpected preloaded local source')
    return observed


def unchanged(root,expected):
    # End checks admit only the three exact lazy local helper origins.
    for name in ('replay','literal_graph','dense_reference'):
        module=sys.modules.get(name)
        if module is not None:
            target=str(root/(name+'.py'))
            if str(Path(module.__file__).resolve())!=target or str(Path(module.__spec__.origin).resolve())!=target:raise ValueError('local loaded origin mismatch')
    actual={}
    for directory,dirs,files in os.walk(root,followlinks=False):
        if any(Path(directory,n).is_symlink()for n in dirs):raise ValueError('changed source directory')
        for name in files:
            p=Path(directory,name);actual[p.relative_to(root).as_posix()]=sha(read(p,8*1024**2))
    if actual!=expected:raise ValueError('end source closure differs')
    return actual


def write_owned(fd,name,raw):
    child=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd);primary=None
    try:
        view=memoryview(raw)
        while view:
            n=os.write(child,view)
            if n<=0:raise OSError('short public output write')
            view=view[n:]
        os.fsync(child)
    except BaseException as error:primary=error;raise
    finally:
        try:os.close(child)
        except BaseException:
            if primary is None:raise


def dispatch(args):
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise PermissionError('use Python -I -B')
    if os.sched_getaffinity(0)!={108,109,110,111}:raise PermissionError('reviewed CPU lane only')
    root=Path(__file__).resolve().parent
    raw=read(args.manifest,2*1024**2)
    if sha(raw)!=args.manifest_sha256:raise ValueError('external source digest')
    expected=json.loads(raw)
    if canonical(expected)!=raw:raise ValueError('canonical source map')
    before=source_check(root,expected)
    rawgrant=read(args.grant,65536)
    if sha(rawgrant)!=args.grant_sha256:raise PermissionError('external grant digest')
    grant=json.loads(rawgrant)
    if canonical(grant)!=rawgrant:raise PermissionError('canonical external grant')
    target=Path(args.output)
    if not target.is_absolute()or target!=target.resolve()or target==root or target.is_relative_to(root)or root.is_relative_to(target):raise ValueError('disjoint canonical output')
    required={'schema':'qwen-toy5-public-cpu-replay-grant-v1','execute':True,
              'source_manifest_sha256':args.manifest_sha256,'entry_sha256':expected['entry.py'],
              'output':str(target),'affinity':[108,109,110,111],
              'scope':'fixed-public-toy5-original-polynomial-and-independent-dense',
              'pretrained_model':False,'key_generation':False,'backend_context':False,'HE':False}
    if canonical(grant)!=canonical(required):raise PermissionError('exact CPU-public grant required')
    runtime=json.loads(read(root/'runtime_pins.json',65536))
    runtime_before={p:sha(read(p,8*1024**2))for p in runtime['sources']}
    if runtime_before!=runtime['sources']:raise ValueError('NumPy public source mismatch')
    if any(n.split('.')[0] in FORBIDDEN for n in sys.modules):raise PermissionError('backend already loaded')
    denied=[]
    def audit(event,arguments):
        backend=event=='import'and arguments and str(arguments[0]).split('.')[0]in FORBIDDEN
        process=event in {'subprocess.Popen','os.system','os.fork','os.forkpty','os.posix_spawn','os.exec','pty.spawn'}
        network=event.startswith('socket.')
        if backend or process or network:
            denied.append({'event':event,'module':str(arguments[0])if backend else None})
            raise PermissionError('public CPU replay forbids backend/process/network')
    class DenyBackend(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0]in FORBIDDEN:
                denied.append({'event':'find_spec','module':fullname})
                raise PermissionError('public CPU replay forbids backend discovery')
    sys.addaudithook(audit);sys.meta_path.insert(0,DenyBackend())
    # This is a source-bound integrity check, not OS confinement. The admitted
    # code has only fixed public source paths; no arbitrary data path is accepted.
    parent=target.parent
    pst=parent.lstat()
    if not stat.S_ISDIR(pst.st_mode)or parent!=parent.resolve()or not(pst.st_uid==os.getuid()or(pst.st_uid==0 and pst.st_mode&stat.S_ISVTX)):raise ValueError('trusted output parent')
    target.mkdir(mode=0o700)
    fd=None;primary=None;report={'schema':'qwen-toy5-replay-outer-v1','passed':False,'HE':False,'pretrained_model':False,'key_generation':False,'source_manifest_sha256':args.manifest_sha256,'grant_sha256':args.grant_sha256,'pid':os.getpid(),'pgid':os.getpgrp(),'affinity':sorted(os.sched_getaffinity(0)),'acceptance_requires_independent_exit_and_exact_report_hash':True}
    try:
        fd=os.open(target,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);owned=os.fstat(fd)
        if owned.st_uid!=os.getuid()or stat.S_IMODE(owned.st_mode)!=0o700:raise ValueError('fresh owned output')
        sys.path.insert(0,str(root))
        import replay
        details=replay.run(root)
        report['diagnostic']=details
        report['passed']=bool(details['diagnostic_complete']and details['literal_matches_independent_dense_at_1e_10'])
    except BaseException as error:
        primary=error;report['error_type']=type(error).__name__;report['passed']=False
    finally:
        try:
            report['source_before']=before;report['source_after']=unchanged(root,expected)
            report['runtime_sources_before']=runtime_before
            report['runtime_sources_after']={p:sha(read(p,8*1024**2))for p in runtime['sources']}
            if report['runtime_sources_after']!=runtime_before:raise ValueError('NumPy source changed')
            origins={}
            for name,path in runtime['required_origins'].items():
                module=sys.modules.get(name)
                if module is None:raise ValueError('required NumPy module absent')
                pair={'file':str(Path(module.__file__).resolve()),'spec':str(Path(module.__spec__.origin).resolve())}
                if pair!={'file':path,'spec':path}:raise ValueError('NumPy module origin')
                origins[name]=pair
            report['runtime_origins']=origins
            report['backend_modules']=[n for n in sys.modules if n.split('.')[0]in FORBIDDEN]
            report['audit_denials']=list(denied)
            if denied or report['backend_modules']:raise PermissionError('sticky audit veto')
            if fd is not None:
                visible=target.lstat()
                if (visible.st_dev,visible.st_ino)!=(owned.st_dev,owned.st_ino):raise ValueError('output replacement')
                if (parent.lstat().st_dev,parent.lstat().st_ino)!=(pst.st_dev,pst.st_ino):raise ValueError('output parent replacement')
                write_owned(fd,'report.json',canonical(report));os.fsync(fd)
        except BaseException as error:
            if primary is None:primary=error
        finally:
            if fd is not None:
                try:os.close(fd)
                except BaseException as error:
                    if primary is None:primary=error
    if primary is not None:raise primary
    if denied:raise PermissionError('post-publication audit veto')
    return 0 if report['passed'] else 1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','manifest-sha256','grant','grant-sha256','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    try:
        result=dispatch(args)
        if type(result)is not int or result not in(0,1):raise TypeError('exact replay status')
        return result
    except BaseException as error:
        try:print('Public replay infrastructure failure: '+type(error).__name__,file=sys.stderr)
        except BaseException:pass
        return 70

if __name__=='__main__':raise SystemExit(main())
