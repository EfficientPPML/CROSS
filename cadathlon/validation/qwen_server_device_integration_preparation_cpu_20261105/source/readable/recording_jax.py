"""Owned Python descriptor/selection recording, never an installed backend."""
from types import SimpleNamespace
EVENTS=[]
MODE='clean'
REPLACEMENT_CALLS=0
STATS={'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':90}
SAMPLES=0
MARKER=RuntimeError('owned recording default scope failure')
config=SimpleNamespace(jax_enable_x64=True,jax_default_device=None)

class Device:
    platform='tpu';id=0;process_index=0;local_hardware_id=0
    device_kind='synthetic-recording-device'
    def memory_stats(self):
        global SAMPLES
        SAMPLES+=1;EVENTS.append('sample');return STATS

DEVICE=Device()

class Array:
    def __init__(self,label,device=None,shape=(1,)):
        self.label=label;self.device=DEVICE if device is None else device
        self.shape=shape;self.waits=0
    def devices(self):return{self.device}
    def block_until_ready(self):
        self.waits+=1;EVENTS.append('wait:'+self.label);return self

ORIGINAL_ARRAY=Array
class ForeignMeta(type):
    def __instancecheck__(cls,value):
        global REPLACEMENT_CALLS
        REPLACEMENT_CALLS+=1;raise AssertionError('changed Array instancecheck was invoked')
class ForeignArray(metaclass=ForeignMeta):pass

class DefaultScope:
    def __init__(self,device):self.device=device;self.enters=0;self.exits=0
    def __enter__(self):
        global Array
        self.enters+=1;EVENTS.append('default.enter');config.jax_default_device=self.device
        if MODE=='enter_array':Array=ForeignArray
        if MODE=='enter_failure':raise MARKER
        return self.device
    def __exit__(self,*args):
        self.exits+=1;EVENTS.append('default.exit')
        if MODE=='exit_failure':raise MARKER
        config.jax_default_device=None

SCOPES=[]
def default_device(device):
    global Array
    EVENTS.append('default.factory')
    scope=DefaultScope(device);SCOPES.append(scope)
    if MODE=='factory_array':Array=ForeignArray
    return scope
def devices(*,backend):
    assert backend=='tpu';EVENTS.append('devices');return[DEVICE]
def local_devices(*,backend):
    assert backend=='tpu';return[DEVICE]
def process_count(*,backend):
    assert backend=='tpu';return 1
