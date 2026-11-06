"""Lifecycle for the real joined server graph; contains no client codec."""
from contextlib import contextmanager
from dataclasses import dataclass
import sys


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

    def evaluate(self, tiles, weights, *, original_positions, context_id, request_id):
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
        return result


@contextmanager
def server_engine(ctx, *, manifest_bytes):
    provider = arithmetic = linear = refresher = None
    try:
        enable, Arithmetic, Linear, Layout, Refresh, Owner, secure = _dependencies()
        # The concrete arithmetic/refresh constructors check the initialized
        #context. Actual input/output payloads are checked again in evaluate.
        if (ctx.secret_key is not None or ctx._param_cache.secret_key is not None
                or ctx.parameters.get('secret_key') is not None):
            raise ValueError('server graph requires a keyless context')
        provider = enable(ctx)
        arithmetic = Arithmetic(ctx,low_memory=True,preserve_dispatch_cache=True)
        linear = Linear(ctx,Layout(65536),low_memory=True,preserve_dispatch_cache=True)
        refresher = Refresh(ctx,period=65536,level_budget=(4,4),public_bound=2.,
            meta_precision=10,compiled=True,low_memory=True,preserve_dispatch_cache=True)
        yield ServerEngine(ctx,arithmetic,linear,refresher,Owner,secure,manifest_bytes)
    finally:
        active = sys.exception(); failure = None
        def attempt(label, operation):
            nonlocal failure
            try: operation()
            except BaseException as error:
                if active is not None: active.add_note(label+': '+repr(error))
                elif failure is None: failure=error
                else: failure.add_note(label+': '+repr(error))
        # Release executable owners while their context and NTT provider still
        #exist. The outer key loader owns final context eviction and JAX caches.
        def clear_arithmetic(value, label):
            attempt(label+' working cleanup',value._clear_working_caches)
            for name in ('rotation','_plaintext_multiply','_rescale'):
                helper=getattr(value,name,None)
                if helper is not None:attempt(label+' helper cleanup',lambda helper=helper:helper.clear(executables=True))
            compiler=getattr(value,'_multiplication_compiler',None)
            if compiler is not None:attempt(label+' multiplication cleanup',compiler.clear_executables)
        if refresher is not None:
            if refresher.compiled_primitives is not None:
                for helper in refresher.compiled_primitives:
                    attempt('refresh helper cleanup',lambda helper=helper:helper.clear(executables=True))
            clear_arithmetic(refresher.evaluator,'refresh evaluator')
        if linear is not None:clear_arithmetic(linear.arithmetic,'linear arithmetic')
        if arithmetic is not None:
            clear_arithmetic(arithmetic,'arithmetic')
        if provider is not None:attempt('NTT provider cleanup',provider.close)
        if active is None and failure is not None:raise failure
