"""Explicit temporary synthetic double; never the installed JAX backend."""
from types import SimpleNamespace

CALLS=0
STATS={'bytes_in_use':90,'bytes_limit':100,'peak_bytes_in_use':90}
class Device:
    platform='tpu'
    id=0
    process_index=0
    local_hardware_id=0
    device_kind='synthetic-recording-device'
    def memory_stats(self):
        global CALLS
        CALLS+=1
        return None if STATS is None else dict(STATS)
DEVICE=Device()
config=SimpleNamespace(jax_enable_x64=True,jax_default_device=DEVICE)
def devices(*,backend):
    assert backend=='tpu'
    return [DEVICE]
def local_devices(*,backend):
    assert backend=='tpu'
    return [DEVICE]
def process_count(*,backend):
    assert backend=='tpu'
    return 1
