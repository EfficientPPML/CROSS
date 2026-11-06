"""Explicit integration seam for a verified, dedicated no-write public loader.

Only instantiated after the existing loader has authenticated every key byte.
No private files, key generation, codecs or cryptographic operations occur here.
Readonly mmap/stat checks are supplementary: the caller must own an exclusive
no-write lifetime for the authenticated producer's public directory.
"""
import os
import stat
import sys


class PreparedLoaderSession:
    def __init__(self,context,public,raw_public,public_owner,states,arrays,*,dedicated_no_write):
        if dedicated_no_write is not True:
            raise ValueError('explicit dedicated authenticated-key no-write lifetime required')
        from prepared_rotation import AuthenticatedRotationLease
        import client_keys as keys
        if keys.canonical(public)!=raw_public:
            raise ValueError('authenticated public manifest was replaced')
        self._context=context;self._public=public;self._raw_public=raw_public
        self._owner=public_owner;self._arrays=arrays
        self._states={name:(owner,state) for owner,name,state in states}
        self._maps=dict(context._param_cache.coef_maps)
        self._active=True;self._owners=[];self._failure=None;self.cleanup_completed=False
        self.lease=AuthenticatedRotationLease(context,context_id=public['context_id'],backing_guard=self._guard)
        self._lease_bound=self.lease

    def _guard(self,index,a,b,coef):
        import client_keys as keys
        if not self._active or self._failure is not None:
            raise RuntimeError('prepared loader lease is retired or failed')
        if self.lease is not self._lease_bound or self._public['context_id']!=self.lease.context_id:
            raise ValueError('prepared loader lease or public identity changed')
        if keys.canonical(self._public)!=self._raw_public:
            raise ValueError('authenticated public manifest changed')
        if coef is not self._maps[index]:raise ValueError('authenticated coefficient map replaced')
        self._owner.check()
        for part,array in (('a',a),('b',b)):
            name=f'rot_{index}_{part}.npy';owner,bound=self._states[name]
            if array is not self._arrays[name] or owner is not self._owner:
                raise ValueError('authenticated key mapping replaced')
            owner.check();current=os.stat(name,dir_fd=owner.fd,follow_symlinks=False)
            state=(current.st_dev,current.st_ino,current.st_size,current.st_mtime_ns,current.st_ctime_ns)
            if not stat.S_ISREG(current.st_mode) or state!=bound:
                raise ValueError('authenticated key file identity or state changed')
            if (array.mode!='r' or array.flags.writeable or array._mmap.closed):
                raise ValueError('authenticated readonly mapping lifetime ended')

    def create(self,*,max_entries=2,max_bytes=2<<30,low_memory=False,preserve_dispatch_cache=False):
        if not self._active or self._failure is not None:raise RuntimeError('prepared loader is not active')
        if self.lease is not self._lease_bound:raise ValueError('prepared loader lease replaced')
        from prepared_rotation import create_prepared_arithmetic
        try:
            owner=create_prepared_arithmetic(self._context,source_lease=self.lease,
                max_entries=max_entries,max_bytes=max_bytes,low_memory=low_memory,
                preserve_dispatch_cache=preserve_dispatch_cache)
            self._owners.append(owner)
            return owner
        except BaseException as error:
            # Even if a caller catches construction failure inside the stage,
            # exit must retain backing when pending setup is not observable.
            self._failure=error;raise

    def close(self):
        if self.cleanup_completed:return True
        self._active=False
        failure=self._failure
        def attempt(fn):
            nonlocal failure
            try:fn()
            except BaseException as error:
                if failure is None:failure=error
                elif not isinstance(error,Exception) and isinstance(failure,Exception):
                    error.add_note('earlier prepared cleanup: '+repr(failure));failure=error
                elif error is not failure:failure.add_note('additional prepared cleanup: '+repr(error))
        # Attempt every registered owner even if one has uncertain device work.
        for owner in self._owners:
            failed=getattr(owner.rotation,'_failed',None)
            if failure is None and failed is not None:failure=failed
            attempt(owner._clear_working_caches)
            attempt(owner.rotation.close)
        if failure is not None:
            self._failure=failure
            # No lease retirement/ref dropping; loader must quarantine mappings.
            raise failure
        self.lease.retire()
        self._owners.clear();self._arrays={};self._maps={};self._states={}
        self._context=None;self._owner=None
        self.cleanup_completed=True
        return True
