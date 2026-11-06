"""Isolated opt-in completed rotation operands; no arithmetic/kernel changes.

The caller must supply a trusted authenticated-source lease. Its backing guard
belongs to the key loader, which has verified manifest digests and holds mapped
files alive. This module does not turn a readonly mmap into immutable storage,
does not authenticate key files, and is not yet wired into the real loader.
Mutable/custom host arrays use ordinary preparation on every call.

Entries contain only completed immutable device arrays and retained provenance
references. A serial owner clears them before loader/context retirement. Any
uncertain preparation/execution/readiness failure quarantines the owner and
traceback until process exit; clear cannot report successful retirement then.
"""
from collections import Counter, OrderedDict
from dataclasses import dataclass
import math
import re
import sys

from encrypted_rotation import CompiledRotation, rotation_state_without_controls

# Failure retention only, never a reusable cross-context optimization cache.
_FAILURE_QUARANTINE=[]


def create_prepared_arithmetic(context,*,source_lease,max_entries=2,max_bytes=2<<30,
                               low_memory=False,preserve_dispatch_cache=False):
    """Construct a fresh compiled owner with the explicit optional rotation seam.

    This factory never replaces a working owner. The unchanged superclass calls
    ``rotation.clear`` before dropping other controls/parameter caches, so the
    prepared arrays obey the same level-switch/checkpoint retirement boundary.
    Caller must complete that owner cleanup before retiring its source lease.
    """
    from encrypted_packed_qwen import CompiledArithmetic
    source_lease._check()
    rotation=PreparedCompiledRotation(context,source_lease=source_lease,
                                     max_entries=max_entries,max_bytes=max_bytes)
    owner=CompiledArithmetic(context,low_memory=low_memory,
                             preserve_dispatch_cache=preserve_dispatch_cache)
    owner.rotation.clear(executables=True)
    owner.rotation=rotation
    return owner


def _typed(value):
    if isinstance(value,(tuple,list)):
        return type(value),tuple(_typed(x) for x in value)
    return type(value),value


def _host_chain(array):
    import numpy as np
    chain=[];seen=set();current=array
    while isinstance(current,np.ndarray):
        if id(current) in seen:raise ValueError('cyclic public array backing')
        seen.add(id(current));chain.append(current)
        current=current.base
    return tuple(chain),current


def _array_state(array):
    """Return retained identity/metadata and eligibility, without content reads."""
    import jax
    import numpy as np
    if isinstance(array,jax.Array):
        return ((array,),('device',id(array),tuple(array.shape),str(array.dtype),
                          frozenset(array.devices())),True)
    if isinstance(array,np.ndarray):
        chain,root=_host_chain(array)
        records=tuple((type(x),id(x),tuple(x.shape),x.dtype.str,tuple(x.strides),
                       bool(x.flags.writeable),int(x.__array_interface__['data'][0]),
                       getattr(x,'mode',None)) for x in chain)
        plain=all(type(x) in (np.ndarray,np.memmap) for x in chain)
        immutable_bytes=type(root) is bytes
        # mmap content stability is the loader's authenticated lease obligation.
        readonly_map=any(type(x) is np.memmap and x.mode=='r' for x in chain)
        eligible=plain and not any(x.flags.writeable for x in chain) and (immutable_bytes or readonly_map)
        return ((*chain,root),('host',records,type(root),id(root)),eligible)
    return ((array,),('unsupported',type(array),id(array)),False)


class AuthenticatedRotationLease:
    """Bind already-authenticated keys to a dedicated live loader/context.

    ``backing_guard(index,a,b,coef_map)`` must verify the original authenticated
    directory/file identities and state and reject retirement. It must also
    enforce the dedicated loader's no-write ownership contract: inode/timestamps
    do NOT prove unchanged contents (rapid same-inode writes can share them).
    It must retain the loader's mapped owners. The exact
    callback and context_id are construction-bound. No hashes of mutable keys
    are inferred from Python identity. This seam needs real-loader wiring and
    review before a genuine proof; current tests provide explicit mock leases.
    """
    def __init__(self,context,*,context_id,backing_guard):
        if type(context_id) is not str or re.fullmatch('[0-9a-f]{64}',context_id) is None:
            raise ValueError('authenticated public context SHA256 required')
        if not callable(backing_guard):raise TypeError('authenticated backing guard required')
        self.context=context;self.context_id=context_id;self.backing_guard=backing_guard
        self._context=context;self._cache=context._param_cache
        self._context_id=context_id;self._backing_guard=backing_guard
        self._keys=self._cache.raw_rotation_keys;self._maps=self._cache.coef_maps
        self._profile=self._profile_state();self._closed=False
        self._check()

    def _profile_state(self):
        ctx=self.context;cache=ctx._param_cache;field=cache.ff_context_cls
        return (tuple(_typed(getattr(ctx,k,None)) for k in
                      ('degree','num_slots','degree_layout','dnum','composite_degree')),
                tuple(_typed(getattr(cache,k,None)) for k in
                      ('degree','batch','degree_layout','dnum','q_towers','p_towers',
                       'per_level_rotation_keys','cache_rotation_keys','composite_degree')),
                id(field),id(getattr(field,'to_computation_format',None)),
                _typed(getattr(field,'computation_format_is_standard',None)))

    def _check(self):
        ctx=self.context;cache=getattr(ctx,'_param_cache',None)
        if (self._closed or ctx is not self._context or cache is not self._cache
                or self.context_id!=self._context_id or self.backing_guard is not self._backing_guard
                or cache.raw_rotation_keys is not self._keys or cache.coef_maps is not self._maps
                or self._profile_state()!=self._profile):
            raise ValueError('authenticated rotation owner/context/provenance changed')
        if (ctx.secret_key is not None or cache.secret_key is not None
                or ctx.parameters.get('secret_key') is not None
                or cache.per_level_rotation_keys is not False):
            raise ValueError('prepared rotation requires a keyless supplied-key context')

    def source(self,index):
        self._check()
        raw=self._keys.get(index);coef=self._maps.get(index)
        if raw is None or coef is None or raw.get('a') is None or raw.get('b') is None:
            raise ValueError('missing authenticated public rotation key/map')
        arrays=(raw['a'],raw['b'],coef)
        self.backing_guard(index,*arrays)
        self._check()
        states=tuple(_array_state(x) for x in arrays)
        refs=(raw,*arrays,*(ref for state in states for ref in state[0]))
        signature=(self._context_id,id(raw),tuple(x[1] for x in states))
        return refs,signature,all(x[2] for x in states)

    def retire(self):
        # Caller must clear/close its compiled owner BEFORE this loader marker.
        self._closed=True


@dataclass
class _Entry:
    state: tuple
    source_refs: tuple
    source_signature: tuple
    device_signature: tuple
    size: int


class PreparedCompiledRotation(CompiledRotation):
    """Explicit opt-in, at most two completed key pairs by default.

    No executable, key-switch arithmetic, rescale, or coefficient-map ordering
    changes. Memory accounting is exact logical device-array bytes; allocator
    capacity, raw host maps, compiler controls and preparation temporaries are
    separate and may exceed this cap.
    """
    def __init__(self,context,*,source_lease,max_entries=2,max_bytes=2<<30):
        if type(max_entries) is not int or max_entries<1:raise ValueError('positive integer entry limit required')
        if type(max_bytes) is not int or max_bytes<1:raise ValueError('positive integer byte budget required')
        if type(source_lease) is not AuthenticatedRotationLease or source_lease.context is not context:
            raise ValueError('exact authenticated lease for this context required')
        source_lease._check()
        super().__init__(context)
        self._lease=source_lease;self._lease_bound=source_lease
        self._entries=OrderedDict();self._max_entries=max_entries;self._max_bytes=max_bytes
        self._bytes=0;self._failed=None;self._closed=False;self.stats=Counter()

    @property
    def retained_bytes(self):return self._bytes

    @property
    def retained_entries(self):return len(self._entries)

    def _usable(self):
        if self._failed is not None:raise RuntimeError('uncertain rotation buffers quarantined; process must exit') from self._failed
        if self._closed:raise RuntimeError('prepared rotation owner is closed')
        if self._lease is not self._lease_bound or self.ctx is not self._lease.context:
            raise ValueError('prepared rotation lease/context replaced')

    def _quarantine(self,error,*references):
        if self._failed is None:
            self._failed=error
            _FAILURE_QUARANTINE.append((self,error,references))
        self.stats['quarantined']+=1

    @staticmethod
    def _state_signature(state,device):
        import jax
        if type(state) is not tuple or len(state)!=3:raise ValueError('three explicit key operands required')
        for value in state:
            if not isinstance(value,jax.Array) or value.devices()!={device}:
                raise ValueError('prepared operands require immutable arrays on the ciphertext device')
        if tuple(str(x.dtype) for x in state)!=('uint32','uint32','int32'):
            raise ValueError('prepared rotation operand representation changed')
        return tuple((id(x),tuple(x.shape),str(x.dtype),frozenset(x.devices())) for x in state)

    def _drop(self,key):
        import jax
        entry=self._entries[key]
        try:jax.block_until_ready(entry.state)
        except BaseException as error:
            self._quarantine(error,entry);raise
        del self._entries[key];self._bytes-=entry.size

    def _reserve(self,size):
        # Evict completed entries before the next conversion starts. The budget
        # still excludes conversion temporaries and compiler/control arrays.
        while self._entries and (len(self._entries)>=self._max_entries or self._bytes+size>self._max_bytes):
            self._drop(next(iter(self._entries)));self.stats['evictions']+=1

    def clear(self,*,executables=False):
        if self._failed is not None:raise RuntimeError('uncertain rotation buffers quarantined; process must exit') from self._failed
        if self._closed:raise RuntimeError('prepared rotation owner is closed')
        # Barrier all entries before releasing ANY, preserving all on failure.
        import jax
        try:
            for entry in self._entries.values():jax.block_until_ready(entry.state)
        except BaseException as error:
            self._quarantine(error);raise
        self._entries.clear();self._bytes=0
        super().clear(executables=executables)

    def close(self):
        if self._closed:return
        self.clear(executables=True);self._closed=True

    def _publish(self,key,state,refs,source_signature,device):
        import jax
        # Only completed buffers may enter the bounded cache, even when result
        # completion happened to imply their readiness on a particular backend.
        jax.block_until_ready(state)
        signature=self._state_signature(state,device)
        size=sum(math.prod(x.shape)*x.dtype.itemsize for x in state)
        if size>self._max_bytes:
            self.stats['oversized']+=1;return
        self._reserve(size)
        self._entries[key]=_Entry(state,refs,source_signature,signature,size);self._bytes+=size
        self.stats['published']+=1

    def rotate(self,value,index):
        import jax
        from support import require_secure_execution
        self._usable();require_secure_execution(self,value)
        if type(index) is not int:raise ValueError('rotation index must be an integer')
        cache=self.ctx._param_cache;level=cache.level_for_num_q(value.num_moduli)
        expected=(cache.batch,2,*cache.degree_layout,value.num_moduli)
        if tuple(value.shape)!=expected:raise ValueError('ciphertext layout differs from initialized rotation context')
        if len(value.polynomial.devices())!=1:raise ValueError('compiled rotation requires one physical device')
        device=next(iter(value.polynomial.devices()))
        signature=(level,tuple(value.shape),str(value.polynomial.dtype),device)
        # A changed level/layout/device releases all prepared operands before
        # replacing the control binding; ordinary compiler executable reuse stays.
        if self._binding is not None and self._binding[0]!=signature:self.clear()
        try:refs,source_signature,eligible=self._lease.source(index)
        except BaseException as primary:
            try:self.clear()
            except BaseException as cleanup:primary.add_note('prepared-key cleanup: '+repr(cleanup))
            raise
        key=(signature,index)
        q=cache.q_moduli_at_level(level)
        alpha=(len(cache.q_towers)+cache.dnum-1)//cache.dnum
        digits=(len(q)+alpha-1)//alpha
        state_shape=(digits,*cache.degree_layout,len(q)+len(cache.p_towers))
        state_bytes=2*math.prod(state_shape)*4+cache.degree*4
        entry=self._entries.get(key)
        if entry is not None and (entry.source_signature!=source_signature
                or entry.device_signature!=self._state_signature(entry.state,device)):
            self._drop(key);entry=None;self.stats['invalidations']+=1
        state=result=None
        try:
            if entry is None and eligible:self._reserve(state_bytes)
            if self._binding is None:
                operation=self.ctx.he_rot[level,index]
                state=operation._rotation_state()
                rotate=self.ctx.he_rot.take_explicit_rotation_fn((level,index))
                binding=self._compiler.bind(rotate,(value.polynomial,*state),signature=signature,device=device)
                self._binding=(signature,binding)
                self.stats['binding_misses']+=1
            elif entry is not None and eligible:
                state=entry.state;self._entries.move_to_end(key);self.stats['hits']+=1
            else:
                state=rotation_state_without_controls(cache,level,index);self.stats['preparations']+=1
            if (tuple(x.shape for x in state)!=(state_shape,state_shape,(cache.degree,))
                    or self._state_signature(state,device) is None):
                raise ValueError('prepared operands differ from the actual public partition/layout')
            result=value._clone_with_payload(self._binding[1](value.polynomial,*state))
            jax.block_until_ready(result.polynomial)
            # A source swap during transfer/execution must not publish stale
            # state under new provenance. Keep the completed result private.
            after_refs,after_signature,after_eligible=self._lease.source(index)
            if after_signature!=source_signature:raise ValueError('public key source changed during rotation')
            if entry is None and eligible and after_eligible:
                self._publish(key,state,after_refs,after_signature,device)
            cache._formatted_rotation_keys.pop(index,None)
            self.ctx.he_rot._instances.pop((level,index),None)
            return result
        except BaseException as error:
            self._quarantine(error,value,state,result,refs);raise
