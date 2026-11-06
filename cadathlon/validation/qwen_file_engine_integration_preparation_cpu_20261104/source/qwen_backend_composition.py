"""Qwen's ordinary borrowed-context composition, with no executable role entry.

The public factory denies before imports or argument access. The internal engine
is a trusted component seam: an eventual independently admitted loader must own
its key context for the engine's entire lifetime. Recording tests supply doubles;
they do not instantiate a CKKS context, import a backend, or read model/key data.
"""
from dataclasses import dataclass
import json, math, threading
from qwen_file_contract import (PublicContract, canonical, same, scale,
    validate_request, ciphertext_binding, checked_artifact, response)
from qwen_key_adapter import Admission
from qwen_public_parameters import ParameterCatalog
from qwen_finalizer import Finalizer

_QUARANTINE = []
_ACTIVE = threading.Lock()


def create_reference_session(*args, **kwargs):
    raise PermissionError('Qwen backend/session role awaits independent execution grant and active loader proof')


def _full_contract(public, request, contract_type):
    b = public.check()
    return contract_type(b['inputs/candidate_manifest.json'], b['inputs/config.json'],
        b['inputs/site_metadata.json'], b['inputs/certificates.json'],
        b['inputs/join_plan.json'], b['qkv-export-v1/parameter_manifest.json'],
        b['inputs/decoder_parameters.json'], b['inputs/actual_key_plan.json'],
        tuple(request['input']['original_positions']))


def _materialize(record):
    """Same Polynomial construction as default125 transport, on admitted bytes."""
    import jax.numpy as jnp
    import numpy as np
    from jaxite_word import Polynomial
    binding = json.loads(record.binding)
    payload = record.array()
    if (type(payload) is not np.ndarray or payload.dtype != np.dtype('uint32') or
            list(payload.shape) != binding['shape'] or payload.flags.writeable):
        raise ValueError('immutable admitted physical RNS payload required')
    result = Polynomial.from_array(jnp.asarray(payload, dtype=jnp.uint32),
        dict(batch=1, num_elements=2, degree=131072,
             num_moduli=len(binding['moduli']), precision=32, degree_layout=(256, 512)),
        dict(moduli=binding['moduli']))
    result._ckks_scale = binding['scale']
    result._ckks_nsd = 1
    return result


@dataclass(frozen=True)
class Dependencies:
    enable: object
    arithmetic: object
    linear: object
    layout: object
    refresh: object
    contract: object
    parameters: object
    owner: object
    secure: object
    materialize: object


def _runtime_dependencies():
    """Unselected until a future active factory is separately admitted."""
    from encrypted_ntt import enable_scoped_ntt_subsets
    from encrypted_packed_qwen import CompiledArithmetic
    from encrypted_chained_streaming_linear import ChainedStreamingLinear
    from encrypted_streaming_linear import RowTileLayout
    from encrypted_refresh import BoundedSecureRefresh
    from encrypted_qwen_full_model import QwenParameterSource, QwenFullModel
    from full_contract import FullQwenContract
    from support import require_secure_execution
    return Dependencies(enable_scoped_ntt_subsets, CompiledArithmetic,
        ChainedStreamingLinear, RowTileLayout, BoundedSecureRefresh,
        FullQwenContract, QwenParameterSource, QwenFullModel,
        require_secure_execution, _materialize)


def _context(public, ctx):
    p = public.profile()
    if (ctx.secret_key is not None or ctx._param_cache.secret_key is not None or
            ctx.parameters.get('secret_key') is not None):
        raise ValueError('borrowed server context must be keyless in all three locations')
    # Full owner's fixed_profile also validates this, after construction. This
    # first guard rejects an incompatible borrowed context before any primitive.
    for name in ('degree', 'num_slots', 'max_level', 'dnum', 'scaling_factor'):
        same(getattr(ctx, name), p[name], 'actual context profile differs: ' + name)
    for name in ('q_towers', 'p_towers', 'degree_layout'):
        same(list(getattr(ctx, name)), p[name], 'actual context ordered ring differs: ' + name)
        same(list(getattr(ctx._param_cache, name)), p[name], 'parameter cache differs: ' + name)
    same(ctx._param_cache.dnum, p['dnum'], 'parameter cache dnum differs')
    same(ctx.parameters.get('sigma'), p['sigma'], 'actual sigma differs')
    same(ctx.parameters.get('secret_distribution'), p['secret_distribution'], 'actual secret distribution differs')


def _metadata(value, binding):
    value.validate()
    same(list(value.shape), binding['shape'], 'actual ciphertext physical shape differs')
    same(list(value.get_moduli()), binding['moduli'], 'actual ciphertext Q prefix differs')
    same(list(value.degree_layout), binding['degree_layout'], 'actual ciphertext layout differs')
    same(value.precision, 32, 'actual ciphertext precision differs')
    same(value._ckks_nsd, 1, 'actual ciphertext NSD differs')
    same(scale(value._ckks_scale), binding['scale'], 'actual ciphertext scale differs')


def _snapshot(value):
    return (value, value.polynomial, value.moduli_array, canonical(dict(
        shape=list(value.shape), moduli=list(value.get_moduli()), layout=list(value.degree_layout),
        precision=value.precision, scale=value._ckks_scale, nsd=value._ckks_nsd)))


def _unchanged(saved):
    value, payload, moduli, metadata = saved
    if value.polynomial is not payload or value.moduli_array is not moduli:
        raise ValueError('borrowed or returned ciphertext arrays changed')
    now = _snapshot(value)
    # Exact scalar/container types are checked by the normal metadata gate.
    if now[3] != metadata:
        raise ValueError('retained ciphertext metadata changed')


class BorrowedServerEngine:
    """One request, ordinary arithmetic, and explicit borrowed-owner retirement.

    This internal composition accepts trusted dependencies/file-owner objects for
    recording tests. It is not an admission capability, loader, or actual entry.
    The actual factory remains disabled; it must enforce concrete reader types
    and externally pinned context/model/source IDs before opening any bodies.
    """
    def __init__(self, public, request, admission, *, ctx, key_owner,
                 request_files, parameter_files, dependencies, cleanup_shield):
        # Pure expected-ID and profile admission precedes file-owner callbacks.
        if type(public) is not PublicContract or type(admission) is not Admission:
            raise TypeError('exact source-bound public contract and key admission required')
        c = validate_request(public, request)
        identity = admission.check()
        if admission.public.role != 'server' or admission.private is not None:
            raise ValueError('non-overlapping keyless server admission required')
        same(identity, c['public'], 'external key and file identities differ')
        same(admission.public.profile.parameters, public.profile(), 'key/file profile differs')
        if key_owner is None or type(dependencies) is not Dependencies or not callable(cleanup_shield):
            raise TypeError('borrowed key owner, exact dependency set and cleanup shield required')
        for fn in dependencies.__dict__.values():
            if not callable(fn): raise TypeError('explicit trusted dependency callable required')
        self.public = public; self.request = canonical(c); self.admission = admission
        self.ctx = ctx; self.key_owner = key_owner; self.upload = request_files
        self.parameters = parameter_files; self.deps = dependencies; self.shield = cleanup_shield
        self.provider = self.arithmetic = self.linear = self.refresher = self.owner = None
        self.parameter_source = self.full_contract = self.result = None
        self.refs = []; self._original_refs = self.refs; self._retained = (); self.started = False; self.retired = False; self.failed = None
        self.cleanup = {}; self._binding = (public, self.request, admission, ctx, key_owner,
            request_files, parameter_files, dependencies, cleanup_shield)
        self._check()

    def _check(self):
        if _QUARANTINE or self.failed is not None:
            raise RuntimeError('failed Qwen composition requires dedicated-process exit')
        now = (self.public, self.request, self.admission, self.ctx, self.key_owner,
               self.upload, self.parameters, self.deps, self.shield)
        if any(a is not b for a, b in zip(now, self._binding)):
            raise ValueError('construction-bound borrowed engine changed')
        c = validate_request(self.public, json.loads(self.request))
        same(self.admission.check(), c['public'], 'external key/file binding changed')
        _context(self.public, self.ctx)
        expected_catalog = ParameterCatalog.from_public(self.public)
        if self.parameters.catalog != expected_catalog:
            raise ValueError('fixed source-bound parameter catalog required')
        if (self.upload.public is not self.public or self.upload.contract != self.request or
                len(self.upload.records) != c['input']['input_tile_count']):
            raise ValueError('held upload belongs to a different request')
        self.upload.check(); self.parameters.check()
        if self.refs is not self._original_refs or len(self.refs) != len(self._retained) or any(a is not b for a,b in zip(self.refs,self._retained)):
            raise ValueError('retained array owner inventory changed')
        for saved in self._retained: _unchanged(saved)
        return c

    def _retain(self, value):
        # First retain wrapper and whatever arrays exist before inspecting it.
        self.refs.append((value, getattr(value, 'polynomial', None),
                          getattr(value, 'moduli_array', None)))
        self.refs[-1] = _snapshot(value)
        self._retained = tuple(self.refs)
        return value

    def run(self, publish_block):
        if not callable(publish_block): raise TypeError('opaque output publication callback required')
        if self.started or self.retired: raise ValueError('one request per borrowed engine')
        c = self._check()
        # These run-local references are independent of callback-writable engine
        # attributes, including _binding/_retained. They also own partial returns.
        binding = tuple(self._binding)
        public, request_bytes, admission, ctx, key_owner, upload, parameters, d, shield = binding
        cleanup_report = self.cleanup
        check_method = self._check
        refs = []; owned = []; actions = []; helper_guards = []
        provider = arithmetic = linear = refresher = owner = parameter_source = full_contract = result_value = None
        published_result = None; entered = False; primary = None
        lock = _ACTIVE
        file_methods = tuple((value,name,getattr(value,name)) for value,names in (
            (upload,('check','consumer_entered','consumers_retired')),
            (parameters,('check','matrix','gamma','consumer_entered','consumers_retired','readback'))) for name in names)
        upload_check,upload_enter,upload_retire = (row[2] for row in file_methods[:3])
        parameter_check,matrix_loader,gamma_loader,parameter_enter,parameter_retire,parameter_readback = (row[2] for row in file_methods[3:])

        def own(name, value):
            owned.append((name, value))
            setattr(self, name, value)
            return value

        def retain(value):
            raw = (value, getattr(value, 'polynomial', None), getattr(value, 'moduli_array', None))
            refs.append(raw)  # retained even when the following snapshot raises
            saved = _snapshot(value)
            refs[-1] = saved
            self.refs.append(saved); self._retained = tuple(self.refs)
            return value

        def check():
            now = (self.public,self.request,self.admission,self.ctx,self.key_owner,
                   self.upload,self.parameters,self.deps,self.shield)
            if any(a is not b for a,b in zip(now,binding)) or self._binding is not initial_binding:
                raise ValueError('original borrowed ownership binding changed')
            if self.cleanup is not cleanup_report:
                raise ValueError('original cleanup report changed')
            if any(getattr(self,name,None) is not value for name,value in owned):
                raise ValueError('original owned engine helper changed')
            if self.refs is not original_ref_list or len(self.refs)!=len(refs) or any(a is not b for a,b in zip(self.refs,refs)):
                raise ValueError('original retained array inventory changed')
            for value,name,original in helper_guards:
                current=getattr(value,name,None)
                if current is not original:
                    raise ValueError('original owned helper inventory changed')
            for value,name,original in file_methods:
                current=getattr(value,name,None)
                if getattr(current,'__self__',None)is not getattr(original,'__self__',None)or getattr(current,'__func__',current)is not getattr(original,'__func__',original):
                    raise ValueError('original borrowed file accessor changed')
            for saved in refs: _unchanged(saved)
            check_method()

        def capture_arithmetic(value, label):
            if value is None:return
            actions.append((label+' working cache',value._clear_working_caches))
            for name in ('rotation','_plaintext_multiply','_rescale'):
                helper=getattr(value,name,None);helper_guards.append((value,name,helper))
                if helper is not None:
                    clear=helper.clear
                    actions.append((label+' helper clear',lambda fn=clear:fn(executables=True)))
            compiler=getattr(value,'_multiplication_compiler',None)
            helper_guards.append((value,'_multiplication_compiler',compiler))
            if compiler is not None:actions.append((label+' executable clear',compiler.clear_executables))

        initial_binding=self._binding;original_ref_list=self.refs
        if not lock.acquire(blocking=False): raise RuntimeError('serial Qwen engine already active')
        try:
            self.started=True
            upload_enter();parameter_enter()
            entered=True
            provider=own('provider',d.enable(ctx))
            arithmetic=own('arithmetic',d.arithmetic(ctx,low_memory=True,preserve_dispatch_cache=True))
            linear=own('linear',d.linear(ctx,d.layout(65536),low_memory=True,preserve_dispatch_cache=True))
            refresher=own('refresher',d.refresh(ctx,period=65536,level_budget=(4,4),public_bound=2.,
                meta_precision=10,compiled=True,low_memory=True,preserve_dispatch_cache=True))
            prepared=[]
            for index,record in enumerate(upload.records):
                input_binding=ciphertext_binding(public,c,'request',index)
                same(json.loads(record.binding),input_binding,'admitted input binding differs')
                fresh=retain(d.materialize(record))
                fresh.polynomial.block_until_ready();fresh.moduli_array.block_until_ready()
                _metadata(fresh,input_binding);d.secure(ctx,fresh);check()
                reduced=retain(ctx.he_level_reduce[49,25].level_reduce(fresh))
                reduced.polynomial.block_until_ready();reduced.moduli_array.block_until_ready()
                target=dict(input_binding,level=25,moduli=public.profile()['q_towers'][:52],shape=[1,2,256,512,52])
                _metadata(reduced,target);d.secure(ctx,reduced);check();prepared.append(reduced)
            full_contract=own('full_contract',_full_contract(public,c,d.contract))
            parameter_source=own('parameter_source',d.parameters(full_contract,matrix_loader=matrix_loader,gamma_loader=gamma_loader))
            owner=own('owner',d.owner(arithmetic,linear,full_contract,parameter_source,refresher=refresher))
            result_value=own('result',owner.evaluate(tuple(prepared)))
            values=getattr(result_value,'logit_blocks',None)
            if type(values)is not tuple:raise ValueError('exact38 logit tuple required')
            for value in values:retain(value)
            if (len(values)!=38 or result_value.blocks is not full_contract.vocabulary_blocks or
                    type(result_value.original_positions)is not tuple or result_value.original_positions!=tuple(c['input']['original_positions']) or
                    type(result_value.last_position)is not int or result_value.last_position!=c['response']['last_original_position'] or
                    type(result_value.metadata)is not tuple or len(result_value.metadata)!=38):
                raise ValueError('actual complete ordered Qwen full-model result differs')
            same(list(result_value.original_positions),c['input']['original_positions'],'typed original result positions differ')
            bindings=[]
            for index,value in enumerate(values):
                value.polynomial.block_until_ready();value.moduli_array.block_until_ready()
                output_binding=ciphertext_binding(public,c,'response',index,actual_scale=scale(value._ckks_scale),input_digest=upload.input_digest)
                _metadata(value,output_binding);d.secure(ctx,value)
                descriptor=c['response']['blocks'][index]
                expected=dict(block=index,start=descriptor['start'],stop=descriptor['stop'],level=2,scale=value._ckks_scale,last_position=result_value.last_position)
                same(result_value.metadata[index],expected,'owner result scale/order metadata differs')
                bindings.append(output_binding)
            # Capture helper objects and cleanup callables BEFORE publication.
            helpers=tuple(refresher.compiled_primitives or ())
            for helper in helpers:
                clear=helper.clear;actions.append(('refresh helper clear',lambda fn=clear:fn(executables=True)))
            evaluator=refresher.evaluator;helper_guards.append((refresher,'evaluator',evaluator))
            capture_arithmetic(evaluator,'refresh')
            linear_arithmetic=linear.arithmetic;helper_guards.append((linear,'arithmetic',linear_arithmetic))
            capture_arithmetic(linear_arithmetic,'linear');capture_arithmetic(arithmetic,'graph')
            actions.append(('NTT provider close',provider.close))
            artifacts=[]
            for index,(value,output_binding)in enumerate(zip(values,bindings)):
                check()
                row=publish_block(index,value,json.loads(canonical(output_binding)))
                check();_metadata(value,output_binding)
                checked_artifact(public,c,row,'response',index,input_digest=upload.input_digest)
                same(row['binding'],output_binding,'publisher actual ciphertext binding differs');artifacts.append(row)
            published_result=response(public,c,upload.input_digest,artifacts)
        except BaseException as error:
            primary=error
        finally:
            final=Finalizer(cleanup_report,primary)
            def retire():
                seen=set()
                for saved in refs:
                    for array in saved[1:3]:
                        if array is not None and id(array)not in seen:
                            seen.add(id(array))
                            final.attempt('original retained array barrier',lambda a=array:a.block_until_ready())
                final.attempt('held upload readback',lambda:upload_check(readback=True))
                final.attempt('held parameter readback',parameter_readback)
                if entered and primary is not None:return
                if final.failure is not None:return
                for label,operation in actions:final.attempt(label,operation)
                if final.failure is None and entered:
                    final.attempt('parameter consumer retirement',parameter_retire)
                    final.attempt('upload consumer retirement',upload_retire)
            try:
                final.shielded(shield,retire)
                error=final.selected()
                if error is not None and(self.started or entered):
                    self.failed=error
                    _QUARANTINE.append((self,binding,tuple(owned),tuple(refs),tuple(helper_guards),tuple(actions),file_methods,result_value,error))
                elif error is None:
                    self.retired=True
                    original_ref_list.clear();self._retained=()
                    self.provider=self.arithmetic=self.linear=self.refresher=self.owner=None
                    self.parameter_source=self.result=None
            finally:
                lock.release()
            final.raise_if_failed()
        return published_result
