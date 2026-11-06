"""Opt-in serial ordinary-rotation groups; ciphertext digits never enter a cache.

Requires an authenticated public-key lease and immutable device payloads. Host
mapping immutability remains the loader's dedicated no-write obligation. Group
bytes count digits/results only, excluding controls, key conversion, compiler,
allocator and original backing. No HE performance or precision claim.
"""
from dataclasses import dataclass,field
import math
from encrypted_rotation import CompiledRotation,rotation_state_without_controls
from prepared_rotation import AuthenticatedRotationLease,_typed,_array_state

_FAILURE_QUARANTINE=[]  # Failure retention only; never a reusable shared cache.

def _method(value):
    return (id(getattr(value,'__self__',None)),id(getattr(value,'__func__',value)))


def _array_signature(value):
    import jax
    if not isinstance(value,jax.Array):raise TypeError('immutable JAX array required')
    return (id(value),tuple(value.shape),str(value.dtype),frozenset(value.devices()))


def _value_signature(value):
    return (_array_signature(value.polynomial),id(value.ntt_ctx),tuple(_typed(getattr(value,n,None)) for n in
        ('moduli','degree','degree_layout','batch','num_elements','precision','_ckks_scale','_ckks_nsd')))


@dataclass
class _Group:
    source: object
    source_state: tuple
    indices: tuple
    provenance: dict
    outputs: list=field(default_factory=list)
    output_states: list=field(default_factory=list)
    states: list=field(default_factory=list)
    bindings: list=field(default_factory=list)
    binding_states: list=field(default_factory=list)
    kernel: object=None
    kernel_state: tuple=None
    digits: object=None
    digit_state: tuple=None
    mode: str='ordinary'
    started: bool=False


class GroupedCompiledRotation(CompiledRotation):
    """Bounded one-shot groups; ordinary rotate remains the default method.

    Every duplicate nonzero request executes separately; zero aliases the input.
    Results are returned only after readiness and final provenance checks. Any
    post-start exception conservatively quarantines original backing until
    process exit; caller must retire this owner before its key lease/context.
    """
    def __init__(self,context,*,source_lease,max_results=16,max_result_bytes=2<<30,max_digit_bytes=512<<20):
        if type(source_lease)is not AuthenticatedRotationLease or source_lease.context is not context:
            raise ValueError('exact authenticated lease for this context required')
        for number in (max_results,max_result_bytes,max_digit_bytes):
            if type(number)is not int or number<1:raise ValueError('positive exact integer limits required')
        source_lease._check();super().__init__(context)
        from encrypted_compilation import ExplicitArrayCompiler
        self._group_compiler=ExplicitArrayCompiler(max_executables=4)
        self._lease=source_lease;self._owner=(context,source_lease,self._group_compiler)
        self.max_results=max_results;self.max_result_bytes=max_result_bytes;self.max_digit_bytes=max_digit_bytes
        self._limits=(max_results,max_result_bytes,max_digit_bytes)
        self._accessor=context.he_rot
        self._methods=self._method_state()
        self._group=None;self._active=False;self._failed=None;self._closed=False;self.last_group=None

    def _method_state(self):
        cache=self.ctx._param_cache;accessor=self.ctx.he_rot
        return (id(accessor),tuple(_method(getattr(cache,n)) for n in ('get_rot_key','q_moduli_at_level','level_for_num_q')),
                _method(accessor.take_explicit_rotation_fn),_method(type(accessor).__getitem__))

    def _usable(self):
        if self._failed is not None:raise RuntimeError('rotation group backing quarantined; process must exit') from self._failed
        if self._closed:raise RuntimeError('rotation group owner is closed')
        if (self.ctx is not self._owner[0] or self._lease is not self._owner[1] or self._group_compiler is not self._owner[2]
                or _typed((self.max_results,self.max_result_bytes,self.max_digit_bytes))!=_typed(self._limits)):
            raise ValueError('rotation group owner or limits changed')
        self._lease._check()
        if self.ctx.he_rot is not self._accessor or self._method_state()!=self._methods:
            raise ValueError('bound rotation control/key accessors changed')

    def _quarantine(self,error,group):
        if self._failed is None:
            self._failed=error;_FAILURE_QUARANTINE.append((self,error,group,self._binding))

    def clear(self,*,executables=False):
        if self._active:raise RuntimeError('cannot clear an active same-input group')
        if self._failed is not None:raise RuntimeError('rotation group backing quarantined; process must exit') from self._failed
        if self._closed:return
        super().clear(executables=executables)
        if executables:self._group_compiler.clear_executables()

    def close(self):
        if self._closed:return
        self.clear(executables=True);self._closed=True

    def rotate(self,value,index):
        self._usable()
        if self._active:raise RuntimeError('serial rotation owner cannot be reentered')
        if type(index)is not int:raise ValueError('rotation index must be exact integer')
        refs,signature,eligible=self._lease.source(index)
        try:
            result=super().rotate(value,index)
            _,after,ok=self._lease.source(index)
            if after!=signature or ok!=eligible:raise ValueError('authenticated key backing changed during ordinary rotation')
            return result
        except BaseException as error:
            self._quarantine(error,_Group(value,None,(index,),{index:(refs,signature,eligible)}));raise

    def _preflight(self,value,indices,progress):
        import jax
        from finite_field import BarrettContext
        from support import require_secure_execution
        self._usable()
        if self._active:raise RuntimeError('serial rotation owner cannot be reentered')
        if type(indices)not in (tuple,list) or len(indices)>self.max_results or any(type(i)is not int for i in indices):
            raise ValueError('bounded exact ordered integer rotation request required')
        if progress is not None and not callable(progress):raise TypeError('progress must be callable or None')
        require_secure_execution(self,value)
        cache=self.ctx._param_cache
        if type(self.ctx.num_slots)is not int or self.ctx.num_slots<1:raise ValueError('invalid slot count')
        indices=tuple(i%self.ctx.num_slots for i in indices)
        level=cache.level_for_num_q(value.num_moduli);q=tuple(cache.q_moduli_at_level(level));p=tuple(cache.p_towers)
        expected=(cache.batch,2,*cache.degree_layout,len(q))
        if (type(level)is not int or tuple(value.shape)!=expected or tuple(value.moduli)!=q
                or _typed(value.degree_layout)!=_typed(cache.degree_layout) or value.degree!=cache.degree
                or type(value._ckks_nsd)is not int or value._ckks_nsd!=1
                or type(value._ckks_scale)is not float or not math.isfinite(value._ckks_scale) or value._ckks_scale<=0):
            raise ValueError('input metadata differs from exact context Q prefix/layout/scale')
        result_bytes=sum(bool(i) for i in indices)*math.prod(value.shape)*value.polynomial.dtype.itemsize
        if result_bytes>self.max_result_bytes:raise ValueError('rotation result byte budget exceeded before work')
        provenance={}
        eligible=True
        alpha=(len(cache.q_towers)+cache.dnum-1)//cache.dnum
        max_parts=(len(cache.q_towers)+alpha-1)//alpha
        for index in dict.fromkeys(i for i in indices if i):
            refs,signature,ok=self._lease.source(index);provenance[index]=(refs,signature,ok);eligible &= ok
            pair=cache.raw_rotation_keys[index];mapping=cache.coef_maps[index]
            if (any(tuple(pair[n].shape)!=(max_parts,len(cache.q_towers)+len(p),cache.degree) or str(pair[n].dtype)!='uint32' for n in ('a','b'))
                    or tuple(mapping.shape)!=(cache.degree,) or str(mapping.dtype)!='int32'):
                raise ValueError('authenticated public key/map structure differs')
        devices=value.polynomial.devices() if isinstance(value.polynomial,jax.Array) else set()
        supported=(isinstance(value.polynomial,jax.Array) and len(devices)==1 and str(value.polynomial.dtype)=='uint32'
            and type(value.precision)is int and value.precision==32 and cache.ff_context_cls is BarrettContext
            and cache.ff_context_cls.computation_format_is_standard is True
            and type(self.ctx.composite_degree)is int and self.ctx.composite_degree==2
            and type(cache.composite_degree)is int and cache.composite_degree==2
            and all(type(m)is int and 1<m<(1<<31) for m in (*cache.q_towers,*p))
            and q==tuple(cache.q_towers[:len(q)]) and eligible)
        parts=(len(q)+alpha-1)//alpha
        digit_bytes=parts*cache.batch*cache.degree*(len(q)+len(p))*4
        supported &= digit_bytes<=self.max_digit_bytes
        if len(devices)==1:
            device=next(iter(devices))
            for index in provenance:
                for a in (*cache.raw_rotation_keys[index].values(),cache.coef_maps[index]):
                    if isinstance(a,jax.Array) and a.devices()!={device}:supported=False
        else:device=None
        # Unsupported cases use the original completion only after all shared
        # no-secret, authenticated-source and bounded-result checks succeeded.
        return _Group(value,_value_signature(value),indices,provenance,mode='digits' if supported else 'ordinary'),level,device,digit_bytes,result_bytes

    @staticmethod
    def _kernel_state(kernel):
        names=('r','c','batch','sizeQl','sizeP','sizeQlP','numPartQl','overall_moduli','extend_moduli','select_tower_index','non_select_tower_index','restore_indices','_controls_ready')
        return (tuple(_typed(getattr(kernel,n,None)) for n in names),id(kernel.ff_context_cls),
                tuple(id(getattr(kernel,n,None)) for n in ('ct_full','ct_in','ct_approx','bconv')))

    def _validate_kernel(self,kernel,level):
        from finite_field import BarrettContext
        cache=self.ctx._param_cache;q=tuple(cache.q_moduli_at_level(level));p=tuple(cache.p_towers)
        alpha=(len(cache.q_towers)+cache.dnum-1)//cache.dnum
        partitions=[list(range(start,min(start+alpha,len(q)))) for start in range(0,len(q),alpha)]
        if (kernel.ff_context_cls is not BarrettContext or kernel._controls_ready is not True
                or _typed(kernel.select_tower_index)!=_typed(partitions)
                or tuple(kernel.overall_moduli)!=q+p or tuple(kernel.extend_moduli)!=p
                or _typed((kernel.r,kernel.c,kernel.batch,kernel.sizeQl,kernel.sizeQlP,kernel.numPartQl))
                   !=_typed((*cache.degree_layout,cache.batch,len(q),len(q)+len(p),len(partitions)))):
            raise ValueError('controls differ from ordered Q/P/max-Q partition contract')

    def _validate_state(self,state,level,device):
        cache=self.ctx._param_cache;q=cache.q_moduli_at_level(level)
        alpha=(len(cache.q_towers)+cache.dnum-1)//cache.dnum;parts=(len(q)+alpha-1)//alpha
        expected=(parts,*cache.degree_layout,len(q)+len(cache.p_towers))
        if (type(state)is not tuple or len(state)!=3 or tuple(tuple(a.shape) for a in state)!=(expected,expected,(cache.degree,))
                or any(a.devices()!={device} for a in state) or tuple(str(a.dtype) for a in state)!=('uint32','uint32','int32')):
            raise ValueError('explicit operands differ from partition/layout/device/representation')

    def _guard(self,group):
        self._usable()
        if self._group is not group or not self._active:raise ValueError('active group identity changed')
        if _value_signature(group.source)!=group.source_state:raise ValueError('same-input ciphertext changed')
        if len(group.outputs)!=len(group.output_states) or any(_value_signature(v)!=s for v,s in zip(group.outputs,group.output_states)):
            raise ValueError('retained rotation result changed')
        for index,(_,signature,eligible) in group.provenance.items():
            _,now,ok=self._lease.source(index)
            if now!=signature or ok!=eligible:raise ValueError('authenticated key backing changed during group')
        if group.kernel is not None and self._kernel_state(group.kernel)!=group.kernel_state:raise ValueError('rotation controls changed')
        if group.digits is not None and _array_signature(group.digits)!=group.digit_state:raise ValueError('same-input digits changed')
        for binding,state in zip(group.bindings,group.binding_states):
            current=(id(binding),id(binding.executable),tuple(_array_signature(a) for a in binding.controls))
            if current!=state:raise ValueError('explicit compiled controls changed')

    def _bind(self,group,function,operands,signature,device):
        binding=self._group_compiler.bind(function,operands,signature=signature,device=device)
        group.bindings.append(binding)
        group.binding_states.append((id(binding),id(binding.executable),tuple(_array_signature(a) for a in binding.controls)))
        return binding

    @staticmethod
    def _barriers(group):
        import jax
        # Attempt every readiness barrier, retaining ALL backing on any failure.
        values=[group.source.polynomial,*group.states]
        if group.digits is not None:values.append(group.digits)
        values += [a for b in group.bindings for a in b.controls]
        values += [v.polynomial for v in group.outputs]
        errors=[]
        for value in values:
            try:jax.block_until_ready(value)
            except BaseException as error:errors.append(error)
        if errors:
            chosen=next((e for e in errors if not isinstance(e,Exception)),errors[0])
            for error in errors:
                if error is not chosen:chosen.add_note('additional readiness failure: '+repr(error))
            raise chosen

    def rotate_many(self,value,indices,*,progress=None):
        import jax
        group,level,device,digit_bytes,result_bytes=self._preflight(value,indices,progress)
        self._group=group;self._active=True;self.last_group=None;primary=None
        try:
            self._guard(group)
            nonzero=next((i for i in group.indices if i),None)
            completion=None;first_state=None;first_index=None
            if group.mode=='digits' and nonzero is not None:
                group.started=True
                with jax.default_device(device):
                    operation=self.ctx.he_rot[level,nonzero];group.kernel=operation._herot
                    self._validate_kernel(group.kernel,level)
                    group.kernel_state=self._kernel_state(group.kernel)
                    first_state=operation._rotation_state();group.states.extend(first_state);first_index=nonzero
                    self._validate_state(first_state,level,device)
                    # Detach the facade while retaining the ORIGINAL kernel and
                    # its public backing for the complete bounded group lifetime.
                    self.ctx.he_rot.take_explicit_rotation_fn((level,nonzero))
                    group.kernel_state=self._kernel_state(group.kernel)
                    decomposition=self._bind(group,group.kernel._decompose_array,(value.polynomial,),('group-decompose',level,tuple(value.shape)),device)
                    group.digits=decomposition(value.polynomial);group.digit_state=_array_signature(group.digits)
                    jax.block_until_ready(group.digits)
                    completion=self._bind(group,group.kernel._rotate_from_digits_array,(value.polynomial,group.digits,*first_state),('group-complete',level,tuple(value.shape)),device)
            for ordinal,index in enumerate(group.indices):
                self._guard(group)
                if not index:result=value
                elif completion is None:
                    group.started=True;result=super().rotate(value,index)
                else:
                    with jax.default_device(device):
                        state=first_state if first_state is not None and index==first_index else rotation_state_without_controls(self.ctx._param_cache,level,index)
                        group.states.extend(a for a in state if all(a is not old for old in group.states))
                        self._validate_state(state,level,device)
                        result=value._clone_with_payload(completion(value.polynomial,group.digits,*state))
                        jax.block_until_ready(result.polynomial)
                    # Complete each key before releasing its transient facade.
                    self.ctx._param_cache._formatted_rotation_keys.pop(index,None)
                    self.ctx.he_rot._instances.pop((level,index),None)
                result_signature=_value_signature(result)
                if result_signature[0][1:]!=group.source_state[0][1:] or result_signature[1:]!=group.source_state[1:]:
                    raise ValueError('rotation payload layout/device/dtype or metadata changed')
                group.outputs.append(result);group.output_states.append(_value_signature(result));self._guard(group)
                if progress is not None:progress(ordinal,index,result)
                self._guard(group)
                # Result list is bounded. Completed per-key states are not
                # stacked or retained across iterations after their barrier.
                if completion is not None and index:
                    for a in group.states:jax.block_until_ready(a)
                    group.states.clear();first_state=None;state=None
            self._barriers(group);self._guard(group)
            self.last_group=dict(mode=group.mode,requests=len(group.indices),decompositions=int(completion is not None),digit_bytes=digit_bytes if completion else 0,result_bytes=result_bytes)
            return tuple(group.outputs)
        except BaseException as error:
            primary=error
            if group.started:self._quarantine(error,group)
            raise
        finally:
            try:self._barriers(group)
            except BaseException as cleanup:
                self._quarantine(cleanup,group)
                if primary is not None:primary.add_note('rotation group cleanup: '+repr(cleanup))
                else:raise
            finally:
                self._active=False
                if self._failed is None:self._group=None


def create_grouped_arithmetic(context,*,source_lease,low_memory=False,preserve_dispatch_cache=False,**limits):
    """Fresh explicit opt-in; no existing owner or module is monkeypatched."""
    from encrypted_packed_qwen import CompiledArithmetic
    class GroupedArithmetic(CompiledArithmetic):
        def rotate_many(self,value,indices,*,progress=None):
            self.rotation._preflight(value,indices,progress)  # reject before cache work
            self._prepare(self._level(value))
            # Group completion already barriers every array. Its clones all share
            # the input NTT context; skipping _finish avoids touching zero aliases.
            return self.rotation.rotate_many(value,indices,progress=progress)
    replacement=GroupedCompiledRotation(context,source_lease=source_lease,**limits)
    owner=None
    try:
        owner=GroupedArithmetic(context,low_memory=low_memory,preserve_dispatch_cache=preserve_dispatch_cache)
        owner.rotation.clear(executables=True);owner.rotation=replacement
        return owner
    except BaseException as primary:
        actions=[replacement.close]
        if owner is not None:actions.insert(0,owner._clear_working_caches)
        for action in actions:
            try:action()
            except BaseException as cleanup:primary.add_note('group owner construction cleanup: '+repr(cleanup))
        raise
