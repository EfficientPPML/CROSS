"""Lifecycle for the real joined server graph; contains no client codec."""
from contextlib import contextmanager
from dataclasses import dataclass
import sys
import signal
import threading


def _dependencies():
    from encrypted_ntt import enable_scoped_ntt_subsets
    from encrypted_packed_qwen import CompiledArithmetic
    from encrypted_chained_streaming_linear import ChainedStreamingLinear
    from encrypted_streaming_linear import RowTileLayout
    from encrypted_refresh import BoundedSecureRefresh
    from encrypted_bert_full_model import BertFullModel
    from support import require_secure_execution
    return (enable_scoped_ntt_subsets, CompiledArithmetic, ChainedStreamingLinear,
            RowTileLayout, BoundedSecureRefresh, BertFullModel, require_secure_execution)


@dataclass
class ServerEngine:
    ctx: object
    arithmetic: object
    linear: object
    refresher: object
    owner_type: object
    secure: object
    manifest_bytes: bytes
    prepared_owners: object = None

    def __post_init__(self):
        self._prepared_bound = None if self.prepared_owners is None else (
            self.ctx,self.arithmetic,self.linear,self.linear.arithmetic,self.refresher,
            self.prepared_owners,self.owner_type,self.secure,self.manifest_bytes,
            tuple(self.refresher.compiled_primitives),self.refresher.evaluator)

    def _check_prepared(self):
        bound=self._prepared_bound
        if bound is None:
            if self.prepared_owners is not None:raise ValueError('prepared mode cannot be enabled after construction')
            return
        c,a,l,la,r,p,o,s,m,helpers,evaluator=bound
        if (self.ctx is not c or self.arithmetic is not a or self.linear is not l or l.arithmetic is not la
                or self.refresher is not r or self.prepared_owners is not p or self.owner_type is not o
                or self.secure is not s or self.manifest_bytes is not m or r.evaluator is not evaluator
                or len(r.compiled_primitives)!=len(helpers)
                or any(x is not y for x,y in zip(r.compiled_primitives,helpers))):
            raise ValueError('constructed prepared engine owner or ordinary refresh helper changed')
        p.check()

    def evaluate(self, tiles, weights, *, original_positions, context_id, request_id):
        self._check_prepared()
        # Both inputs and the synchronized final output expose the actual
        #context/payload rings to the normal security gate.
        for tile in tiles:
            self.secure(self.ctx, tile)
        owner = self.owner_type(self.arithmetic, self.linear, self.refresher,
            weights, self.manifest_bytes, original_positions=tuple(original_positions),
            context_id=context_id, request_id=request_id)
        result = owner.evaluate(tiles, capture_checkpoints=False)
        result.output.polynomial.block_until_ready()
        self.secure(self.ctx, result.output)
        self._check_prepared()
        return result


@contextmanager
def server_engine(ctx, *, manifest_bytes, prepared_handle=None):
    provider = arithmetic = linear = refresher = prepared = None
    try:
        enable, Arithmetic, Linear, Layout, Refresh, Owner, secure = _dependencies()
        # The concrete arithmetic/refresh constructors check the initialized
        #context. Actual input/output payloads are checked again in evaluate.
        if (ctx.secret_key is not None or ctx._param_cache.secret_key is not None
                or ctx.parameters.get('secret_key') is not None):
            raise ValueError('server graph requires a keyless context')
        if prepared_handle is not None:
            from engine_prepared_owners import PreparedEngineOwners
            prepared = PreparedEngineOwners(prepared_handle, ctx)
        provider = enable(ctx)
        arithmetic = (Arithmetic(ctx,low_memory=True,preserve_dispatch_cache=True) if prepared is None else
            prepared.create('graph',ctx,low_memory=True,preserve_dispatch_cache=True))
        options = {} if prepared is None else {'arithmetic_factory':prepared.linear_factory}
        linear = Linear(ctx,Layout(65536),low_memory=True,preserve_dispatch_cache=True,**options)
        refresher = Refresh(ctx,period=65536,level_budget=(4,4),public_bound=2.,
            meta_precision=10,compiled=True,low_memory=True,preserve_dispatch_cache=True)
        if prepared is not None:prepared.check()
        yield ServerEngine(ctx,arithmetic,linear,refresher,Owner,secure,manifest_bytes,prepared)
    finally:
        active = sys.exception(); failure = None
        def attempt(label, operation):
            nonlocal failure
            try: return operation()
            except BaseException as error:
                if active is not None: active.add_note(label+': '+repr(error))
                elif failure is None: failure=error
                elif not isinstance(error, Exception) and isinstance(failure, Exception):
                    error.add_note('earlier engine cleanup: '+repr(failure)); failure=error
                else: failure.add_note(label+': '+repr(error))
        # The outer key loader must remain alive until every compiled owner is
        #retired. Defer repeated cancellation through these inner cleanups too.
        shield = []; deferred = []
        cancellation_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
        if threading.current_thread() is threading.main_thread():
            def remember_signal(signum, frame): deferred.append(signum)
            # Query before the mutating call: a block operation may change
            #the mask and then raise without returning the original mask.
            previous_mask = attempt('engine cleanup signal mask query',
                lambda:signal.pthread_sigmask(signal.SIG_BLOCK, ()))
            if previous_mask is not None:
                try:
                    attempt('engine cleanup signal block',
                        lambda:signal.pthread_sigmask(signal.SIG_BLOCK, cancellation_signals))
                    for signum in cancellation_signals:
                        def install(sig=signum):
                            original=signal.getsignal(sig)
                            # A setter can mutate process state and then raise.
                            shield.append((sig, original))
                            signal.signal(sig, remember_signal)
                        attempt('engine cleanup signal shield', install)
                finally:
                    attempt('engine cleanup setup mask restoration',
                        lambda:signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
        # Release executable owners while their context and NTT provider still
        #exist. The outer key loader owns final context eviction and JAX caches.
        def clear_arithmetic(value, label):
            attempt(label+' working cleanup',lambda:value._clear_working_caches())
            for name in ('rotation','_plaintext_multiply','_rescale'):
                helper=attempt(label+' helper lookup',lambda name=name:getattr(value,name,None))
                if helper is not None:attempt(label+' helper cleanup',lambda helper=helper:helper.clear(executables=True))
            compiler=attempt(label+' multiplication lookup',lambda:getattr(value,'_multiplication_compiler',None))
            if compiler is not None:attempt(label+' multiplication cleanup',lambda:compiler.clear_executables())
        if refresher is not None:
            def refresh_helpers():
                helpers=refresher.compiled_primitives
                return None if helpers is None else tuple(helpers)
            helpers=attempt('refresh helper inventory',refresh_helpers)
            if helpers is not None:
                for helper in helpers:
                    attempt('refresh helper cleanup',lambda helper=helper:helper.clear(executables=True))
            value=attempt('refresh evaluator lookup',lambda:refresher.evaluator)
            if value is not None:clear_arithmetic(value,'refresh evaluator')
        if linear is not None:
            value=attempt('linear arithmetic lookup',lambda:linear.arithmetic)
            if value is not None:clear_arithmetic(value,'linear arithmetic')
        if arithmetic is not None:
            clear_arithmetic(arithmetic,'arithmetic')
        if prepared is not None:attempt('original prepared owners cleanup',prepared.close)
        if provider is not None:attempt('NTT provider cleanup',lambda:provider.close())
        if shield:
            previous_mask = attempt('engine cleanup restoration mask query',
                lambda:signal.pthread_sigmask(signal.SIG_BLOCK, ()))
            try:
                if previous_mask is not None:
                    attempt('engine cleanup restoration signal block',
                        lambda:signal.pthread_sigmask(signal.SIG_BLOCK, cancellation_signals))
                for signum, original in reversed(shield):
                    attempt('engine cleanup signal restoration', lambda s=signum,o=original:signal.signal(s,o))
            finally:
                if previous_mask is not None:
                    # Pending cancellation may run the restored caller handler
                    #here; preserve its precedence without replacing a primary.
                    attempt('engine cleanup final mask restoration',
                        lambda:signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
        if deferred:
            def cancellation():
                raise KeyboardInterrupt('signals deferred until engine cleanup completed: '+repr(deferred))
            attempt('deferred engine cleanup cancellation', cancellation)
        if active is None and failure is not None:raise failure
