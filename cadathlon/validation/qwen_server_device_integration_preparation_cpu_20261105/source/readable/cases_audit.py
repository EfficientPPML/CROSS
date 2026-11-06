"""Real AccessScope/wrapper against one tiny owned file, no native device."""
import json
import os
from pathlib import Path
import stat
from types import SimpleNamespace

def _setup(f):
    dep=f.module('qwen_server_dependencies')
    target=f.owned/'native-fixture.bin';target.write_bytes(b'owned synthetic fixture\n')
    state=dict(target=target,fds=set(),opened=[],closed=[],mismatch=False,resolve=[],original_open=os.open,original_close=os.close)
    f.audit_state=state
    def documents(role,value):
        if role=='access':value['device_nodes'][0]['path']=str(target)
    f.rebuild_request(json.loads(f.phase.request_bytes),mutate_document=documents)
    profile=f.key_contract.PublicProfile.from_bytes(f.blobs['recording_sources/reader43/public_inputs/profile_audit.json'],f.blobs['recording_sources/reader43/public_inputs/actual_key_plan.json'])
    def public_profile(value):assert value is f.public;return profile.parameters
    f.patch(f.file_module.PublicContract,'profile',public_profile,'A: public profile metadata recording; real PublicProfile validation')
    roots={n:dict(path=str(f.owned/('unopened-'+n)))for n in('qkv','decoder','gamma')}
    stores=dict(roots=roots,files=[dict(family='qkv',name=f'matrix{i:03}.npy')for i in range(197)]+[dict(family='gamma',name='gamma.npz')])
    def model_stores(value):assert value is f.model;return SimpleNamespace(check=lambda:stores)
    f.patch(f.model_module.ModelAdmission,'stores',property(model_stores),'A: model-store path/count metadata only; no current store stat/read')
    ready_report=f.canonical(dict(stores=dict(public=f.bundle['grant']['public_key_store'],private=dict(path=str(f.owned/'unopened-private')))))
    object.__setattr__(f.ready,'report_bytes',ready_report)
    original_lstat=Path.lstat;original_fstat=os.fstat
    def as_character(info,minor=0):
        return SimpleNamespace(**{n:getattr(info,n)for n in('st_dev','st_ino','st_size','st_mtime_ns','st_ctime_ns','st_uid')},st_mode=stat.S_IFCHR|0o600,st_rdev=os.makedev(0,minor))
    def lstat(path,*args,**kwargs):
        info=original_lstat(path,*args,**kwargs)
        return as_character(info)if path==target else info
    def fstat(fd):
        info=original_fstat(fd)
        return as_character(info,1 if state['mismatch']else 0)if fd in state['fds']else info
    def original_open(path,flags,mode=0o777,*,dir_fd=None):
        fd=state['original_open'](path,flags,mode,dir_fd=dir_fd)
        if type(path)is str and path==str(target)and dir_fd is None:
            state['fds'].add(fd);state['opened'].append(fd);f.counts['open']+=1
        return fd
    def close(fd):
        if fd in state['fds']:state['closed'].append(fd);state['fds'].remove(fd)
        return state['original_close'](fd)
    f.patch(Path,'lstat',lstat,'A: synthetic char metadata only for exact tiny owned target')
    f.patch(os,'fstat',fstat,'A: synthetic char/rdev only for the original owned returned FD')
    f.patch(os,'open',original_open,'A: recording original-open forwarding, original native FD retained')
    f.patch(os,'close',close,'A: original owned FD close counter; forwards all other descriptors')
    public_paths=tuple(str(f.test_root/name)for name in f.loader.full_map)
    audit=dep.AccessScope(str(f.root),f.expected,public_paths,f.phase,profile,f.dep,f.res,(),always_protected=())
    f.audit=audit;f.audit_methods={n:getattr(audit,n)for n in('check','install','close','enable_initialization','enable_public_keys','enable_server_bodies')}
    audit.install();f.seams.append(dict(owner='AccessScope',export='install',label='real Python audit/wrapper installed; fresh child exits after original wrapper retirement'))

def run(f,case):
    checks=[];closed=False
    f.prepare_owner(before_capture=_setup);audit=f.audit;original_close=audit.close;state=f.audit_state
    try:
        f.start();assert audit.initialization_enabled and audit.native_binding[0]is f.device
        flags=os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK
        target=str(state['target'])
        if case=='audit_owned_character_descriptor_control':
            fd=os.open(target,flags)
            try:assert fd in state['fds']and state['opened']==[fd]
            finally:os.close(fd)
            assert state['closed']==[fd]and not state['fds'];audit.check()
            checks.append('real open/close of tiny owned regular file under exact named/FD character-metadata seam; no read')
        elif case=='audit_opened_descriptor_mismatch':
            state['mismatch']=True
            f.expect(PermissionError,lambda:os.open(target,flags),'opened native device differs')
            assert len(state['opened'])==1 and state['closed']==state['opened']and not state['fds']
            f.expect(PermissionError,audit.check,'sticky')
            checks.append('post-open mismatch closes the original returned FD once, denial remains sticky')
        elif case=='audit_literal_flags_spelling_rejection':
            variants=[(target,flags|os.O_RDWR,None),(target,flags|os.O_CREAT,None),(target,flags|os.O_APPEND,None),
                (state['target'],flags,None),(os.fsencode(target),flags,None),(target,flags,123456)]
            for path,bits,dirfd in variants:
                def attempt():
                    fd=os.open(path,bits,dir_fd=dirfd)
                    try:raise AssertionError('unselected native open unexpectedly returned')
                    finally:os.close(fd)
                f.expect((PermissionError,ValueError),attempt)
            assert state['opened']==[];f.expect(PermissionError,audit.check)
            checks.append('unselected flag tuples and wrapper argument spellings reject before any FD opens')
        elif case=='audit_metadata_body_overlap_rejection':
            # A second scope is never activated. Call the genuine admission
            # method on a fresh uninstalled scope with metadata-only variants.
            dep=f.module('qwen_server_dependencies')
            for path in('/dev/unopened-owned-marker',str(f.owned/'unopened-private'/'secret.npy'),str(f.owned/'unopened-qkv'/'matrix000.npy')):
                other=object.__new__(dep.AccessScope);other.__dict__.update(audit.__dict__)
                other.initialization_enabled=False;other.native_binding=None;other._original_native_binding=None
                raw=json.loads(f.bundle['documents']['access']);raw['metadata_files']=[path]
                f.expect(ValueError,lambda:other.enable_initialization(f.device,f.canonical(raw)))
            assert state['opened']==[];checks.append('metadata exceptions reject protected/private/model/device paths without opening')
        elif case=='audit_native_binding_accessor_replacement':
            def changed(*args,**kwargs):f.counts['replacement']+=1;raise AssertionError('changed native checker invoked')
            f.mutate(f.device.__class__,'require_initialization_scope',changed)
            f.expect(ValueError,audit.check,'original native scope checker changed')
            f.restore_mutations();assert f.counts['replacement']==0
            # The real inventory tuple identity check is a sticky denial route.
            saved=audit.native_binding;replacement=tuple(list(saved));assert replacement==saved and replacement is not saved
            audit.native_binding=replacement
            f.expect(PermissionError,lambda:audit.access(Path(target),flags,through_wrapper=True),'inventory replaced')
            audit.native_binding=saved;f.expect(PermissionError,audit.check,'sticky')
            checks.append('original accessor and inventory identities checked before foreign invocation; sticky deny survives restore')
        elif case=='audit_proc_policy_only':
            dep=f.module('qwen_server_dependencies');original_resolve=Path.resolve
            literals=dep._PROC_SELF_KERNEL
            def resolve(path,*args,**kwargs):
                if str(path)in literals:
                    state['resolve'].append(str(path));return Path('/proc')/str(os.getpid())/path.name
                if str(path)=='/proc'or str(path).startswith('/proc/'):
                    raise AssertionError('unselected proc resolution is not a selected test seam')
                return original_resolve(path,*args,**kwargs)
            f.patch(Path,'resolve',resolve,'A: only exact3 literal /proc/self paths map to this process PID; no proc reads; all other paths forward or deny')
            for literal in sorted(literals):
                path=Path(literal);assert dep._literal_proc_self(literal)
                audit.access(path,flags,literal_kernel=True)
                for wrong_flags,literal_bit in((flags|os.O_WRONLY,True),(flags,False)):
                    f.expect(PermissionError,lambda:audit.access(path,wrong_flags,literal_kernel=literal_bit),'proc-self')
                explicit=str(Path('/proc')/str(os.getpid())/path.name)
                assert not dep._literal_proc_self(explicit)and Path(explicit)not in audit.kernel
            assert sorted(state['resolve'])==sorted(literals)and state['opened']==[]
            checks.append('real access positive three literals with exact resolution recording; real early-denial flags/nonliteral bits')
            checks.append('explicitPID checked only by original pure literal helper and fixed kernel-set exclusion; no full resolve/access claim')
        else:raise AssertionError('unselected audit case '+case)
    finally:
        # Original scope close always restores its original open wrapper even
        # when a deliberate sticky denial or body failure is already present.
        try:original_close()
        except (PermissionError,ValueError):
            if not audit.denials:raise
        finally:
            closed=not audit.active and os.open is state['original_open']or(not audit.active and os.open is audit.original_open)
            for fd in tuple(state['fds']):os.close(fd)
    assert closed and not state['fds']
    assert state['opened']==state['closed']
    return dict(passed=True,case=case,checks=checks,recording_only=True,actual_authority=False,
        selected_fd_open_count=len(state['opened']),selected_fd_close_count=len(state['closed']),native_device_opened=False)
