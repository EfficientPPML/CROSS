"""Reviewed server boundary plus explicit whole-model execution denial."""
from contextlib import contextmanager
import boundaries
from cleanup import cleanup_signals

@contextmanager
def rotation_only(calls,*,private_paths=()):
    import file_stages,full_model_engine,server_public_model
    patches=[];primary=None;failure=None
    def denied(label):
        def stop(*args,**kwargs):
            calls.append(label);raise RuntimeError('forbidden rotation parity action: '+label)
        return stop
    with boundaries.server_only(calls,private_paths=private_paths):
        try:
            for owner,name in ((file_stages,'server_encrypted_compute'),(full_model_engine,'server_engine'),(server_public_model,'load')):
                original=getattr(owner,name);patches.append((owner,name,original))
                setattr(owner,name,denied(owner.__name__+'.'+name))
            yield
        except BaseException as error:primary=error;raise
        finally:
            with cleanup_signals():
                for owner,name,original in reversed(patches):
                    try:setattr(owner,name,original)
                    except BaseException as error:
                        if primary is not None:primary.add_note('parity guard restore: '+repr(error))
                        elif failure is None:failure=error
                        elif not isinstance(error,Exception) and isinstance(failure,Exception):
                            error.add_note('earlier guard restore: '+repr(failure));failure=error
                        else:failure.add_note('additional guard restore: '+repr(error))
                if primary is None and failure is not None:raise failure
