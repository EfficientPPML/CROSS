"""One-process retained-state guard after full pure server-packet admission.

This is not a grant, file reader, resource policy or cross-run outcome cache.
The caller independently binds this helper and its pure dependency sources and
must retain the original guard/methods. Full pure checks remain unchanged.
"""
import os
import sys
import threading
from types import FunctionType,MethodType,ModuleType

import qwen_server_authority_packet as packet_module
import qwen_server_admission as phase_module
import qwen_scientific_admission as scientific_module
import qwen_file_contract as file_module
import model_admission as model_module
import keygen_authority as key_module
import client_role_common as common_module
import freeze_admission as freeze_module
import freeze_issuer as issuer_module
import qwen_key_contract as key_contract_module
import qwen_public_parse as parse_module
import public_source_pins as public_pins_module
import keygen_boundary_pins as key_pins_module
import scientific_source_pins as scientific_pins_module


# Fixed fields from the exact reviewed packet27 dependencies. No dynamic
# dataclass introspection, property evaluation or unknown object traversal.
_FIELDS=(
 (packet_module.ServerPacket,('public','raw','expected_sha256','phase','scientific_proofs','dependency_policy','role_policy','resource_policy','resource_review')),
 (phase_module.ServerPhaseAdmission,('public','model','readiness','request_bytes','request_manifest_bytes','grant_bytes','expected_pins','scientific')),
 (phase_module.ExpectedServerPhasePins,('raw',)),
 (scientific_module.ScientificAdmission,('public','model','readiness','request_bytes','request_manifest_bytes','statement_bytes','proof_bytes','expected_pins')),
 (scientific_module.ExpectedScientificPins,('raw',)),
 (file_module.PublicContract,('blobs',)),
 (model_module.ModelAdmission,('public','manifest_bytes','report_bytes','outer_bytes','review_bytes','expected_pins')),
 (model_module.ExpectedModelPins,('raw',)),
 (key_module.ActualKeygenIntent,('public','raw')),
 (key_module.ExpectedLaunchPins,('raw',)),
 (key_module.LaunchAdmission,('intent','grant_bytes','model','expected_pins')),
 (key_module.ExpectedReadyPins,('raw',)),
 (key_module.GeneratedKeyReadiness,('launch','public_manifest_bytes','report_bytes','outer_bytes','root_review_bytes','expected_pins')),
)
_MODULES=(packet_module,phase_module,scientific_module,file_module,model_module,key_module,
    common_module,freeze_module,issuer_module,key_contract_module,parse_module,
    public_pins_module,key_pins_module,scientific_pins_module)
_PACKET_TYPE=packet_module.ServerPacket
_FULL_PACKET_CHECK=packet_module.ServerPacket.check
_REGISTRY={}
_ISSUED=False


def _defaults(value):
    if value is None or type(value)in(str,bytes,bool,int,float):return
    if type(value)is tuple:
        for child in value:_defaults(child)
        return
    raise TypeError('mutable or unknown callable default is outside snapshot scope')


def _function(fn):
    if type(fn)is not FunctionType:raise TypeError('source-fixed Python function required')
    _defaults(fn.__defaults__)
    kw=fn.__kwdefaults__
    if kw is not None:
        if type(kw)is not dict or any(type(k)is not str for k in kw):raise TypeError('exact callable keyword defaults required')
        for value in kw.values():_defaults(value)
    return(fn,fn.__code__,fn.__defaults__,kw,tuple(sorted(kw.items()))if kw is not None else())


def _function_unchanged(saved):
    fn,code,defaults,kw,items=saved
    if(fn.__code__ is not code or fn.__defaults__ is not defaults or fn.__kwdefaults__ is not kw):
        raise ValueError('original helper function code/defaults changed')
    if kw is not None and(set(kw)!=set(k for k,v in items)or any(kw[k]is not v for k,v in items)):
        raise ValueError('original helper keyword defaults changed')


def _callback(fn):
    if type(fn)is FunctionType:return(fn,None,_function(fn))
    if type(fn)is MethodType:return(fn,fn.__self__,_function(fn.__func__))
    raise TypeError('independent source-binding check must be an original Python function or method')


def _callback_unchanged(saved):
    fn,owner,function=saved
    if owner is not None and(fn.__self__ is not owner or fn.__func__ is not function[0]):
        raise ValueError('original independent binding checker changed')
    _function_unchanged(function)


def _bindings():
    modules=[];classes=[];functions=[];seen_classes=set();seen_functions=set()
    names={m.__name__ for m in _MODULES}
    def function(fn):
        if id(fn)not in seen_functions:seen_functions.add(id(fn));functions.append(_function(fn))
    for module in _MODULES:
        if type(module)is not ModuleType or sys.modules.get(module.__name__)is not module:
            raise TypeError('original pure helper module required')
        spec=module.__spec__;file=module.__file__;origin=spec.origin
        if type(file)is not str or type(origin)is not str:raise TypeError('independently bound source module origins required')
        exports=[]
        for name,value in tuple(module.__dict__.items()):
            if type(value)is FunctionType and value.__module__ in names:
                exports.append((name,value));function(value)
            elif type(value)is type and value.__module__ in names:
                exports.append((name,value))
                if id(value)not in seen_classes:
                    seen_classes.add(id(value));items=tuple(value.__dict__.items());classes.append((value,items))
                    for attr,member in items:
                        if type(member)is FunctionType:function(member)
                        elif type(member)is property:
                            for fn in(member.fget,member.fset,member.fdel):
                                if fn is not None:function(fn)
                        elif type(member)in(staticmethod,classmethod):function(member.__func__)
        modules.append((module,module.__name__,spec,file,origin,tuple(exports)))
    return(tuple(modules),tuple(classes),tuple(functions))


def _bindings_unchanged(saved):
    modules,classes,functions=saved
    for module,name,spec,file,origin,exports in modules:
        if(sys.modules.get(name)is not module or module.__name__!=name or module.__spec__ is not spec
                or module.__file__ is not file or spec.origin is not origin):raise ValueError('original pure module/origin changed')
        for key,value in exports:
            if module.__dict__.get(key)is not value:raise ValueError('original pure helper export changed')
    # Check class descriptors before touching instance fields, so a substituted
    # property or __getattribute__ cannot run in place of a retained field read.
    for cls,items in classes:
        current=cls.__dict__
        if set(current)!=set(k for k,v in items)or any(current[k]is not v for k,v in items):
            raise ValueError('original pure class descriptor changed')
    for function in functions:_function_unchanged(function)


def _graph(packet):
    fields=dict(_FIELDS);rows=[];seen=set()
    def visit(value):
        if type(value)in(bytes,str):return
        if id(value)in seen:return
        if len(seen)>=8192:raise ValueError('bounded retained admission object inventory exceeded')
        seen.add(id(value))
        if type(value)is tuple:
            # Tuples cannot change in place; recurse into any admitted objects.
            rows.append((value,tuple,None,tuple(value)))
            for child in value:visit(child)
            return
        cls=type(value)
        if cls not in fields:raise TypeError('unknown or mutable value in admitted object graph')
        data=object.__getattribute__(value,'__dict__')
        wanted=fields[cls]
        if type(data)is not dict or set(data)!=set(wanted):raise ValueError('closed original admission instance fields required')
        items=tuple((name,data[name])for name in wanted)
        rows.append((value,cls,data,items))
        for name,child in items:visit(child)
    visit(packet)
    return tuple(rows)


def _graph_unchanged(rows):
    for value,cls,data,items in rows:
        if type(value)is not cls:raise TypeError('original admission type changed')
        if cls is tuple:
            if len(value)!=len(items)or any(a is not b for a,b in zip(value,items)):
                raise ValueError('original immutable admission tuple changed')
        else:
            current=object.__getattribute__(value,'__dict__')
            if current is not data or set(current)!=set(k for k,v in items)or any(current[k]is not v for k,v in items):
                raise ValueError('original admitted field/object/byte identity changed')


class AdmissionSnapshot:
    """Opaque single-use process/thread guard; contains no writable fields."""
    __slots__=()
    def check(self,packet):
        if type(self)is not AdmissionSnapshot:raise TypeError('exact original snapshot required')
        state=_REGISTRY.get(id(self))
        if state is None:raise PermissionError('snapshot is retired or was not created here')
        guard,original,pid,thread,bindings,graph,checker,status=state
        if guard is not self:raise PermissionError('original snapshot identity differs')
        if status!='active':raise PermissionError('failed or reentrant snapshot requires process exit')
        _REGISTRY[id(self)]=(self,original,pid,thread,bindings,graph,checker,'checking')
        try:
            if(packet is not original or os.getpid()!=pid or threading.get_ident()!=thread[0]
                    or threading.current_thread()is not thread[1]):
                raise PermissionError('snapshot belongs to one original packet/process/thread')
            _callback_unchanged(checker)
            checker[0]()  # Independent caller-owned source/origin/scope gate.
            _callback_unchanged(checker)
            _bindings_unchanged(bindings);_graph_unchanged(graph)
        except BaseException:
            _REGISTRY[id(self)]=(self,original,pid,thread,bindings,graph,checker,'failed')
            raise
        _REGISTRY[id(self)]=(self,original,pid,thread,bindings,graph,checker,'active')
        return packet
    def retire(self,packet):
        # Clean retirement releases proof references; any failure retains them
        # in the private process registry. Caller cleanup chooses body priority.
        self.check(packet)
        del _REGISTRY[id(self)]
    @property
    def execution_authorized(self):return False


def capture_admitted(packet,*,check_bindings):
    """Full check once at this boundary, then cheap retained-state checks.

    check_bindings is an independently selected caller-owned function, not an
    arbitrary trust token. It must bind this source, helper sources/origins and
    process policy before capture and every callback check. A no-op supplied by
    an untrusted caller establishes no authority. The actual role stays denied.
    """
    global _ISSUED
    if _ISSUED:raise PermissionError('only one admission snapshot may be issued per process')
    _ISSUED=True
    checker=_callback(check_bindings);_callback_unchanged(checker);checker[0]()
    if type(packet)is not _PACKET_TYPE:raise TypeError('exact fully admitted server packet required')
    bindings=_bindings()
    if packet_module.ServerPacket.check is not _FULL_PACKET_CHECK:raise ValueError('original complete packet validator changed')
    # Snapshot before validation, then compare afterwards: a final caller
    # binding callback cannot replace validated bytes just before capture.
    graph=_graph(packet)
    _FULL_PACKET_CHECK(packet)
    _graph_unchanged(graph)
    _callback_unchanged(checker);checker[0]();_bindings_unchanged(bindings)
    _graph_unchanged(graph)
    guard=AdmissionSnapshot()
    _REGISTRY[id(guard)]=(guard,packet,os.getpid(),(threading.get_ident(),threading.current_thread()),bindings,graph,checker,'active')
    guard.check(packet)
    return guard


def create_actual_snapshot(*args,**kwargs):
    raise PermissionError('retained packet guard is not an actual role/source/resource grant')
