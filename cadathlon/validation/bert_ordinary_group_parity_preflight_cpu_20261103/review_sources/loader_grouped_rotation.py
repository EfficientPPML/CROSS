"""Explicit ordinary-group owner registration in the reviewed public loader.

The original prepared-key factory/manager stay unchanged. This subclass reuses
its authenticated lease, original-method registration and close/quarantine
ordering, replacing only the fresh arithmetic factory for this opt-in handle.
"""
from loader_prepared_rotation import PreparedLoaderSession,_RegisteredOwner

class GroupedLoaderSession(PreparedLoaderSession):
    def create(self,*,max_results=16,max_result_bytes=2<<30,max_digit_bytes=512<<20,
               low_memory=False,preserve_dispatch_cache=False):
        if not self._active or self._failure is not None:raise RuntimeError('grouped loader is not active')
        if self.lease is not self._lease_bound:raise ValueError('grouped loader lease replaced')
        from grouped_rotation import create_grouped_arithmetic
        try:
            owner=create_grouped_arithmetic(self._context,source_lease=self.lease,
                max_results=max_results,max_result_bytes=max_result_bytes,max_digit_bytes=max_digit_bytes,
                low_memory=low_memory,preserve_dispatch_cache=preserve_dispatch_cache)
            self._registered.append(_RegisteredOwner(owner,owner.rotation,owner._clear_working_caches,
                owner.rotation.clear,owner.rotation.close,self._context))
            return owner
        except BaseException as error:
            self._failure=error
            raise
