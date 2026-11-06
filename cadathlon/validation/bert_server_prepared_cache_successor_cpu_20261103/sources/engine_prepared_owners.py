"""Two explicit engine-owned caches under an authenticated loader lifetime.

No global cache, key reader or arithmetic. Loader registration owns backing
retirement; this controller retires original compiled owners before NTT cleanup.
"""
from typing import NamedTuple

MAX_ENTRIES=2
MAX_BYTES=2<<30
ROLES=('graph','linear')

def _types():
    from loader_prepared_rotation import PreparedLoaderSession
    from encrypted_packed_qwen import CompiledArithmetic
    from prepared_rotation import PreparedCompiledRotation
    return PreparedLoaderSession,CompiledArithmetic,PreparedCompiledRotation

def identity(method):return getattr(method,'__self__',None),getattr(method,'__func__',method)

class Owner(NamedTuple):
    role: str
    arithmetic: object
    rotation: object
    clear: object
    rotation_clear: object
    rotation_close: object

class PreparedEngineOwners:
    def __init__(self,handle,ctx):
        Loader,_,_= _types();factory=getattr(handle,'prepared_arithmetic',None);session=getattr(factory,'__self__',None)
        if type(session)is not Loader or getattr(factory,'__func__',None)is not Loader.create:
            raise ValueError('actual authenticated loader factory required')
        if handle.ctx is not ctx or session._context is not ctx or session._active is not True or session._failure is not None:
            raise ValueError('same live authenticated keyless context required')
        session.lease._check()
        self.handle=handle;self.ctx=ctx;self.session=session;self.factory=factory
        self._bound=(handle,ctx,session,identity(factory),session.lease)
        self._maps=tuple((k,v) for k,v in ctx._param_cache.coef_maps.items())
        self.owners=[];self._owner_list=self.owners;self._registered=();self.closed=False;self.failed=None
        self.check()

    def check(self):
        if self.closed or self.failed is not None:raise RuntimeError('prepared engine owner is closed or failed')
        h,c,s,f,lease=self._bound
        if (self.handle is not h or self.ctx is not c or self.session is not s or identity(self.factory)!=f
                or h.ctx is not c or identity(h.prepared_arithmetic)!=f or s.lease is not lease or s._context is not c
                or s._active is not True or s._failure is not None):raise ValueError('prepared engine loader binding changed')
        lease._check()
        if len(c._param_cache.coef_maps)!=len(self._maps) or any(c._param_cache.coef_maps.get(k)is not v for k,v in self._maps):
            raise ValueError('authenticated coefficient map inventory changed during engine construction/use')
        self._check_inventory()
        for item in self._registered:self._check_owner(item)
        return True

    def _check_inventory(self):
        if (self.owners is not self._owner_list or type(self.owners)is not list or len(self.owners)!=len(self._registered)
                or any(a is not b for a,b in zip(self.owners,self._registered))):
            raise ValueError('original prepared owner inventory changed')
        return True

    def _check_owner(self,item):
        a=item.arithmetic;r=item.rotation
        if (a.ctx is not self.ctx or a.rotation is not r or identity(a._clear_working_caches)!=identity(item.clear)
                or identity(r.clear)!=identity(item.rotation_clear) or identity(r.close)!=identity(item.rotation_close)
                or r._lease is not self._bound[4] or r.ctx is not self.ctx or r._max_entries!=MAX_ENTRIES or type(r._max_entries)is not int
                or r._max_bytes!=MAX_BYTES or type(r._max_bytes)is not int):raise ValueError('original prepared owner/budget/cleanup changed')
        return True

    def create(self,role,ctx,*,low_memory,preserve_dispatch_cache):
        self.check()
        if type(role)is not str or len(self._registered)>=2 or role!=ROLES[len(self._registered)]:
            raise ValueError('exact graph then linear owner construction required')
        if ctx is not self.ctx or low_memory is not True or preserve_dispatch_cache is not True:
            raise ValueError('unchanged low-memory/dispatch policy required')
        _,Arithmetic,Rotation=_types()
        try:
            value=self.factory(max_entries=MAX_ENTRIES,max_bytes=MAX_BYTES,low_memory=True,preserve_dispatch_cache=True)
            if type(value)is not Arithmetic or type(value.rotation)is not Rotation:raise ValueError('verified concrete prepared arithmetic required')
            item=Owner(role,value,value.rotation,value._clear_working_caches,value.rotation.clear,value.rotation.close)
            self._registered=(*self._registered,item);self.owners.append(item);self._check_owner(item)
            return value
        except BaseException as error:self.failed=error;raise

    def linear_factory(self,ctx,**options):return self.create('linear',ctx,**options)

    def close(self):
        if self.closed:return True
        failure=self.failed
        def attempt(call):
            nonlocal failure
            try:return call()
            except BaseException as error:
                if failure is None:failure=error
                elif not isinstance(error,Exception) and isinstance(failure,Exception):
                    error.add_note('earlier prepared-engine cleanup: '+type(failure).__name__);failure=error
                elif error is not failure:failure.add_note('additional prepared-engine cleanup: '+type(error).__name__)
        # Keep original readiness failure identity ahead of later inventory
        #errors, even when a caller has also removed the public owner entries.
        for item in reversed(self._registered):
            def old_failure(item=item):
                error=getattr(item.rotation,'_failed',None)
                if error is not None:raise error
            attempt(old_failure)
        attempt(self._check_inventory)
        for item in reversed(self._registered):
            # A replaced public attribute must not hide original uncertain work.
            okay=attempt(lambda item=item:self._check_owner(item)) is True
            attempt(lambda item=item:item.rotation_clear(executables=True))
            if okay:
                attempt(item.clear);attempt(item.rotation_close)
        if failure is not None:
            self.failed=failure
            # Keep all references; loader sees failure and quarantines backing.
            raise failure
        self.closed=True
        return True
