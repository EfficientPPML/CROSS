"""Dedicated targeted child; source is held unexecuted pending root grant.

Imports the source-authenticated pure client dependency module only. The phase,
profile and unopened data paths are explicit recording seams, not admission.
Only six positive cases may read this child's three public /proc/self metadata
files. No backend, private/key/model/CT body, cgroup or capacity check is run.
"""
import argparse
import hashlib
import importlib
import importlib.abc
import json
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace

FORBIDDEN=frozenset(('numpy','jax','jaxlib','jaxite_word','torch','transformers',
    'tensorflow','flax','torch_xla','safetensors','tokenizers','libtpu',
    'key_gen','ckks_ctx','polynomial','support'))
LEAVES=('status','cgroup','mountinfo')
SPELLINGS=('bytes','pathlike','dot','double_slash','explicit_pid','dir_fd','unselected_self','owned_alias')
FLAGS=('wronly','rdwr','trunc','append','create_excl','directory','path')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def canonical(value):return (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:raise ImportError('forbidden genuine backend in kernel metadata probe')
        return None

def authenticate(root,map_path,digest):
    raw=map_path.read_bytes()
    if sha(raw)!=digest:raise ValueError('external test runtime manifest pin differs')
    expected=json.loads(raw)
    if type(expected)is not dict or len(expected)!=464 or canonical(expected)!=raw:
        raise ValueError('exact compact464 source map required')
    if root.resolve()!=root or any(p.is_symlink()for p in root.rglob('*')):raise ValueError('exact source tree required')
    actual={str(p.relative_to(root)):sha(p.read_bytes())for p in root.rglob('*')if p.is_file()}
    if actual!=expected:raise ValueError('complete test runtime bytes differ')
    return expected

def main(args):
    if not sys.flags.isolated or not sys.dont_write_bytecode:raise ValueError('fresh -I -B child required')
    root=Path(args.root);expected=authenticate(root,Path(args.manifest),args.manifest_sha256)
    if any(n.split('.')[0]in FORBIDDEN for n in sys.modules):raise ValueError('preloaded forbidden backend')
    sys.meta_path.insert(0,DenyBackend());sys.path.insert(0,str(root))
    deps=importlib.import_module('qwen_client_dependencies')
    resource=importlib.import_module('qwen_client_resource')
    if Path(deps.__file__)!=root/'qwen_client_dependencies.py':raise ValueError('selected dependency origin differs')
    scratch=Path(args.scratch)
    if scratch.resolve()!=scratch or not scratch.is_dir():raise ValueError('existing newly owned scratch required')
    alias=scratch/'status-alias';alias.symlink_to('/proc/self/status')
    original_open=os.open;parent_fd=original_open(scratch,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    scope=None;observed=None;cleanup_error=None;body=None
    receipt=dict(schema='qwen-client-own-proc-probe-v1',case=args.case,baseline=args.baseline,
        pid=os.getpid(),pgid=os.getpgrp(),source_manifest_sha256=args.manifest_sha256,
        recording_admission=True,real_role_admission=False,own_proc_metadata_only=True,
        actual_capacity_check=False,backend_imported=False,body_content_recorded=False)
    try:
        # Minimal recording construction of the actual scope. No model/key
        # proof is passed or fabricated; those unopened paths are never read.
        grant=dict(role='request_producer',outputs={n:dict(path=str(scratch/n))for n in('result','evidence')},
            key_stores={n:dict(path=str(scratch/('unopened-'+n)))for n in('public','private')},
            input=dict(path=str(scratch/'unopened-embedding.npy')))
        phase=SimpleNamespace(check=lambda:None,grant_bytes=canonical(grant),request_bytes=canonical({'input':{'input_tile_count':1}}))
        dep={'installed_sources':{}};res={'cgroup_memory_current':'/sys/fs/cgroup/memory.current','cgroup_memory_max':'/sys/fs/cgroup/memory.max'}
        scope=deps.AccessScope(str(root),expected,(),phase,SimpleNamespace(arrays=()),dep,res,())
        scope.install();scope.check()
        try:
            family,case=args.case.split(':',1)
            if family=='helper':
                if case not in LEAVES:raise ValueError('unselected positive leaf')
                value=resource._kernel('/proc/self/'+case)
                if type(value)is not str or not 0<len(value)<=1<<20:raise AssertionError('bounded own-process metadata expected')
                # Avoid retaining/publishing any metadata content or digest.
                observed={'read_completed':True,'nonempty':True};del value
            elif family=='builtin':
                if case not in LEAVES:raise ValueError('unselected audit-event leaf')
                with open('/proc/self/'+case,'rb')as stream:
                    value=stream.read((1<<20)+1)
                if not 0<len(value)<=1<<20:raise AssertionError('bounded own-process metadata expected')
                observed={'read_completed':True,'nonempty':True};del value
            elif family in('flag','spelling'):
                value='/proc/self/status';flags=os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW;kw={}
                if family=='flag':
                    choices=dict(wronly=os.O_WRONLY,rdwr=os.O_RDWR,trunc=os.O_TRUNC,
                        append=os.O_APPEND,create_excl=os.O_CREAT|os.O_EXCL,
                        directory=os.O_DIRECTORY,path=os.O_PATH)
                    if case not in FLAGS:raise ValueError('unselected denied flag')
                    flags|=choices[case]
                else:
                    if case not in SPELLINGS:raise ValueError('unselected denied spelling')
                    if case=='bytes':value=value.encode()
                    elif case=='pathlike':value=Path(value)
                    elif case=='dot':value='/proc/self/./status'
                    elif case=='double_slash':value='/proc/self//status'
                    elif case=='explicit_pid':value='/proc/'+str(os.getpid())+'/status'
                    elif case=='dir_fd':kw={'dir_fd':parent_fd}
                    elif case=='unselected_self':value='/proc/self/io'
                    elif case=='owned_alias':value=str(alias)
                fd=os.open(value,flags,**kw)
                # A regression must not read or write through an unexpectedly
                # admitted descriptor. Retire it once, then fail the test.
                try:raise AssertionError('unselected proc spelling or flags admitted')
                finally:os.close(fd)
            else:raise ValueError('unselected probe family')
        except PermissionError as error:
            observed={'read_completed':False,'error_type':type(error).__name__,'message':str(error)}
        denied=args.baseline or args.case.split(':',1)[0]in('flag','spelling')
        if denied:
            if observed is None or observed.get('error_type')!='PermissionError' or not scope.denials:
                raise AssertionError('expected audit denial before syscall/body')
            if args.baseline and scope.denials!=['open alias']:raise AssertionError('old literal-alias control missed original gate')
            try:scope.check()
            except PermissionError:pass
            else:raise AssertionError('audit denial was not sticky')
        else:
            if observed is None or observed.get('read_completed')is not True or scope.denials:
                raise AssertionError('exact read-only kernel path did not pass')
            scope.check()
        receipt.update(expected_denial=denied,observation=observed,audit_denials=list(scope.denials))
    except BaseException as error:body=error
    finally:
        if scope is not None:
            try:scope.close()
            except BaseException as error:cleanup_error=error
        # Both original owners are retired even when sticky denial is raised.
        restored=os.open is original_open
        try:os.close(parent_fd)
        except BaseException as error:
            if body is None:body=error
        if not restored and body is None:body=AssertionError('original os.open was not restored')
        if scope is not None and scope.active and body is None:body=AssertionError('audit hook was not deactivated')
        expect_denial=receipt.get('expected_denial')is True
        if expect_denial and not isinstance(cleanup_error,PermissionError) and body is None:
            body=AssertionError('sticky cleanup denial expected')
        if not expect_denial and cleanup_error is not None and body is None:body=cleanup_error
        receipt.update(original_open_restored=restored,audit_inactive=scope is not None and not scope.active,
            cleanup_error_type=None if cleanup_error is None else type(cleanup_error).__name__,
            passed=body is None,error_type=None if body is None else type(body).__name__)
        receipt['forbidden_loaded_modules']=sorted(n for n in sys.modules if n.split('.')[0]in FORBIDDEN)
        if receipt['forbidden_loaded_modules']:receipt['passed']=False
        with open(args.receipt,'xb')as stream:stream.write(canonical(receipt))
    if body is not None:raise body
    if not receipt['passed']:raise AssertionError('forbidden backend import')
    return 0

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    for name in('root','manifest','manifest-sha256','case','scratch','receipt'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--baseline',action='store_true')
    raise SystemExit(main(parser.parse_args()))
