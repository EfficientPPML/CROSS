"""One-process device observations after separately authorized initialization.

Unselected source component. The current server lacks the required process
hooks and still selects its unconditional denial. No backend imports at module
import, no device enumeration or default capacity. Trusted hooks are part of
the independently reviewed server source closure, not plugin callbacks.
"""
import os
import sys
import threading
from pathlib import Path
from types import FunctionType,ModuleType
from qwen_device_policy import (DevicePolicyAdmission,HOOKS,STAGES,canonical,sha,parse,obj,same,integer)

_ISSUED=False
_REGISTRY={}

class InitializationPermit:
    __slots__=()
    @property
    def execution_authorized(self):return False

class CapacityAdmission:
    __slots__=()
    @property
    def execution_authorized(self):return False

def create_actual_server(*args,**kwargs):
    raise PermissionError('device observations are not an actual server role factory')

def _origin(module,wanted):
    if type(module)is not ModuleType or type(module.__file__)is not str or module.__file__!=wanted:
        raise ValueError('original selected module file required')
    spec=module.__spec__
    if spec is None or type(spec.origin)is not str or spec.origin!=wanted:raise ValueError('independent selected module spec required')
    p=Path(wanted)
    if p.resolve()!=p or p.is_symlink():raise ValueError('canonical selected source required')
    return spec

def _functions(module):
    output=[]
    for name in HOOKS:
        fn=getattr(module,name,None)
        if (type(fn)is not FunctionType or fn.__module__!=module.__name__ or fn.__name__!=name
                or fn.__qualname__!=name or fn.__globals__ is not module.__dict__
                or fn.__code__.co_filename!=module.__file__ or fn.__closure__ is not None
                or fn.__defaults__ is not None or fn.__kwdefaults__ is not None):
            raise PermissionError('missing original source-fixed process hook: '+name)
        output.append((name,fn,fn.__code__))
    return tuple(output)

def _caller(policy):
    phase=policy.phase;pp=phase.expected_pins.check();grant=parse(phase.grant_bytes,'phase')
    wanted=str(Path(grant['source_root'])/'qwen_server_process.py')
    module=sys.modules.get('qwen_server_process');spec=_origin(module,wanted)
    if sha(Path(wanted).read_bytes())!=pp['source_map']['qwen_server_process.py']:raise ValueError('selected process source changed')
    for name in('qwen_device_policy','qwen_device_capacity'):
        helper=sys.modules.get(name);file=str(Path(grant['source_root'])/(name+'.py'))
        _origin(helper,file)
        if sha(Path(file).read_bytes())!=pp['source_map'][name+'.py']:raise ValueError('selected device helper source changed')
    return module,spec,_functions(module)

def _lookup(permit):
    record=_REGISTRY.get(id(permit))
    if type(permit)is not InitializationPermit or record is None or record['permit']is not permit:
        raise PermissionError('original process-local initialization permit required')
    return record

def _context(record):
    if (os.getpid()!=record['pid']or threading.get_ident()!=record['thread_id']or threading.current_thread()is not record['thread']):
        raise PermissionError('original dedicated process/thread required')
    policy=record['policy'];policy.check()
    caller,spec,hooks=_caller(policy)
    if caller is not record['caller']or spec is not record['spec']or len(hooks)!=len(record['hooks']):raise ValueError('original process module binding changed')
    for now,old in zip(hooks,record['hooks'],strict=True):
        if now[0]!=old[0]or now[1]is not old[1]or now[2]is not old[2]:raise ValueError('original process hook changed')
    return policy

def _hook(record,index,*args):
    # The original function is used even if a callback tries to replace exports.
    fn=record['hooks'][index][1]
    result=fn(record['policy'].phase,record['policy'].blobs,*args)
    # Capture a returned native/device owner before any post-callback check.
    # The selected process also retains its initializer's partially built
    # owner if that hook raises before returning anything.
    if result is not None:record['returned'].append(result)
    if record['state']!='busy':raise RuntimeError('nested or failed device transition is sticky')
    _context(record)
    return result

def _before(record):
    if record['state']=='busy':
        record['state']='failed'
        raise RuntimeError('nested device admission is permanently failed')
    if record['state']in('failed','retired'):raise PermissionError('failed or retired device attempt')
    record['state']='busy'
    _context(record)
    if _hook(record,0)is not None:raise ValueError('binding hook must complete normally without authority return')
    if _hook(record,2)is not None:raise ValueError('original exclusive lease check must complete normally')

def _host(record,stage):
    raw=_hook(record,1,stage)
    value=parse(raw,'genuine host observation')
    obj(value,('schema','pid','stage','resource_policy_sha256','observation'),'host admission observation')
    same(value['schema'],'qwen-device-host-observation-v1','host hook schema differs')
    same(value['pid'],record['pid'],'host hook process differs');same(value['stage'],stage,'host hook stage differs')
    same(value['resource_policy_sha256'],sha(dict(record['policy'].blobs)['resource_policy']),'host resource bytes differ')
    observation=value['observation']
    if type(observation)is not dict:raise ValueError('actual selected host checker result required')
    if (observation.get('after_import')is not(stage!='before_init')or observation.get('borrowed_body_reads')is not False
            or observation.get('capacity_reserved')is not False or observation.get('device_capacity_checked')is not False
            or observation.get('resource_admission_complete')is not False):
        raise ValueError('host-only observation must remain separate from device admission')
    record['host'].append(raw)
    return raw

def issue_initialization(policy):
    """No initialization here. Requires real factual policies and selected hooks."""
    global _ISSUED
    if _ISSUED:
        for previous in _REGISTRY.values():
            if previous['state']in('capturing','busy'):previous['state']='failed'
        raise PermissionError('one device initialization attempt per process')
    _ISSUED=True
    permit=InitializationPermit()
    record=dict(permit=permit,policy=policy,pid=os.getpid(),thread_id=threading.get_ident(),thread=threading.current_thread(),
        caller=None,spec=None,hooks=(),state='capturing',device=None,jax=None,host=[],observations=[],returned=[],stage_index=-1,current=None)
    _REGISTRY[id(permit)]=record
    try:
        if type(policy)is not DevicePolicyAdmission:raise TypeError('exact pure policy admission required')
        policy.check()
        record['caller'],record['spec'],record['hooks']=_caller(policy)
        _before(record);_host(record,'before_init')
        record['state']='permitted'
        return permit
    except BaseException:
        record['state']='failed'
        raise

def _jax_binding(record,device):
    policy=dict(record['policy'].blobs);allocator=parse(policy['allocator'],'allocator');selection=parse(policy['selection'],'selection')
    jax=sys.modules.get('jax');info=allocator['jax_module'];spec=_origin(jax,info['path'])
    if sha(Path(info['path']).read_bytes())!=info['sha256']:raise ValueError('original JAX source differs')
    cls=getattr(jax,'Device',None)
    # This implementation supports the exact selected JAX Device type only.
    # A different installed subtype needs a declared reviewed successor.
    if cls is None or type(device)is not cls:raise TypeError('exact genuine selected JAX Device required')
    descriptor=vars(cls).get('memory_stats')
    if descriptor is None:raise TypeError('genuine memory_stats descriptor required')
    methods=tuple((name,getattr(jax,name),getattr(getattr(jax,name),'__code__',None))for name in('devices','local_devices','process_count'))
    if not all(callable(fn)for _,fn,_ in methods):raise TypeError('original JAX selection methods required')
    observed={n:getattr(device,n)for n in('platform','id','process_index','local_hardware_id','device_kind')}
    same(observed,selection['device'],'actual selected device descriptor differs')
    for n in('id','process_index','local_hardware_id'):integer(observed[n],0,n)
    if type(jax.config.jax_enable_x64)is not bool or jax.config.jax_enable_x64 is not True:raise ValueError('genuine x64 configuration required')
    # The initializer owns a retained default-device scope until all consumers
    # retire. This checks that scope; it does not create a context or transfer.
    if jax.config.jax_default_device is not device:raise ValueError('original selected default-device scope required')
    actual=methods[0][1](backend='tpu');local=methods[1][1](backend='tpu');count=methods[2][1](backend='tpu')
    if type(actual)is not list or type(local)is not list or len(actual)!=1 or len(local)!=1 or actual[0]is not device or local[0]is not device or type(count)is not int or count!=1:
        raise ValueError('this device checker supports one addressable selected device and one process')
    return jax,spec,cls,descriptor,descriptor.__get__(device,cls),methods,canonical(observed)

def _device_guard(record):
    device=record['device'];jax,spec,cls,descriptor,method,methods,observed=record['jax']
    if sys.modules.get('jax')is not jax or jax.__spec__ is not spec or getattr(jax,'Device',None)is not cls or type(device)is not cls:
        raise ValueError('original JAX/device identity changed')
    if vars(cls).get('memory_stats')is not descriptor:raise ValueError('original memory_stats descriptor changed')
    info=parse(dict(record['policy'].blobs)['allocator'],'allocator')['jax_module']
    if _origin(jax,info['path'])is not spec:raise ValueError('original JAX origins changed')
    for name,fn,code in methods:
        if getattr(jax,name)is not fn or getattr(fn,'__code__',None)is not code:raise ValueError('original JAX selection callable changed')
    if canonical({n:getattr(device,n)for n in('platform','id','process_index','local_hardware_id','device_kind')})!=observed:
        raise ValueError('original device metadata changed')
    if jax.config.jax_default_device is not device or jax.config.jax_enable_x64 is not True:raise ValueError('original device/x64 scope changed')
    # Binding hook checks installed dependency/native origins/configuration;
    # class and callable retention complements, rather than replaces, it.
    return method

def initialize_once(permit):
    """Consumes the permit before the selected process initializes its backend."""
    record=_lookup(permit)
    try:
        if record['state']!='permitted':raise PermissionError('unconsumed original initialization permit required')
        _before(record)
        # The pinned process function must perform only the admitted init,
        # retaining the native/backend/default-device owner through retirement.
        device=_hook(record,3)
        record['device']=device
        record['jax']=_jax_binding(record,device)
        if _hook(record,0)is not None:raise ValueError('post-init bindings did not complete normally')
        record['state']='initialized'
        return permit
    except BaseException:
        record['state']='failed'
        raise

def observe_capacity(permit,stage):
    """A stage-specific conditional admission, never a memory reservation."""
    record=_lookup(permit)
    try:
        next_index=record['stage_index']+1
        if type(stage)is not str or next_index>=len(STAGES)or stage!=STAGES[next_index]or record['state']not in('initialized','observed'):
            raise PermissionError('exact ordered original device stage required')
        _before(record)
        if _hook(record,4,stage)is not None:raise ValueError('original synchronization must complete normally')
        host=_host(record,stage);method=_device_guard(record)
        stats=method()
        record['returned'].append(stats)
        if type(stats)is not dict or not 2<=len(stats)<=32 or not {'bytes_in_use','bytes_limit'}<=set(stats):
            raise ValueError('genuine complete allocator statistics required; no None/missing fallback')
        if any(type(k)is not str or not k or len(k)>128 or type(v)is not int or v<0 for k,v in stats.items()):
            raise ValueError('exact nonnegative allocator integer fields required')
        limit=integer(stats['bytes_limit'],1,'allocator limit');used=stats['bytes_in_use']
        if used>limit or('peak_bytes_in_use'in stats and stats['peak_bytes_in_use']<used):raise ValueError('inconsistent allocator observation')
        workspace=parse(dict(record['policy'].blobs)['workspace'],'workspace');bound=workspace['stages'][stage]
        required=sum(bound.values())
        if limit-used<required:raise MemoryError('reviewed stage additional peak/workspace/reserve does not fit')
        _device_guard(record)
        if _hook(record,0)is not None or _hook(record,2)is not None:raise ValueError('post-observation binding/lease check did not complete normally')
        raw=canonical(dict(schema='qwen-tpu-capacity-observation-v1',pid=record['pid'],stage=stage,
            device=parse(dict(record['policy'].blobs)['selection'],'selection')['device'],statistics=dict(stats),
            stage_bounds=bound,required_additional_bytes=required,host_observation_sha256=sha(host),
            expected_pins_sha256=sha(record['policy'].expected_pins),documents={n:sha(b)for n,b in record['policy'].blobs},
            comparison_passed=True,capacity_reserved=False,global_free_memory_measured=False,
            sufficient_only_under_independently_reviewed_evidence=True,actual_server_role_authorized=False))
        value=CapacityAdmission();record['observations'].append(raw);record['stage_index']=next_index
        record['current']=value;record['state']='observed'
        return value
    except BaseException:
        record['state']='failed'
        raise

def require_current(permit,value,stage):
    """Check original current observation before caller's immediate transition.

    This rechecks bindings/lease, not memory. A caller must not represent it as
    another live sample. Only the process can enable its body/engine scopes.
    """
    record=_lookup(permit)
    try:
        if record['state']!='observed'or type(value)is not CapacityAdmission or record['current']is not value or type(stage)is not str or STAGES[record['stage_index']]!=stage:
            raise PermissionError('original current stage observation required')
        _before(record);_device_guard(record)
        raw=record['observations'][-1];record['state']='observed';return raw
    except BaseException:
        record['state']='failed';raise

def retire(permit):
    """Retire only this guard after terminal observation; never release hardware."""
    record=_lookup(permit)
    try:
        if record['state']!='observed'or record['stage_index']!=len(STAGES)-1:raise PermissionError('terminal device observation required')
        _before(record);_device_guard(record)
        result=tuple(record['observations']);record['state']='retired';del _REGISTRY[id(permit)]
        return result
    except BaseException:
        record['state']='failed';raise
