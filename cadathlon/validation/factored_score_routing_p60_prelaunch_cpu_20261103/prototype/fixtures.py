"""Independent fixed public gather/scatter oracle; never called by server routing."""
import hashlib
import numpy as np
SLOTS=65536
LABELS=('head compaction','local key placement','key placement','key tile extraction','key extraction','head expansion')
LEVELS=(4,3,3,3,2,1)

def values():
    result=np.random.default_rng(2026110317).uniform(-40.,40.,SLOTS)
    anchors=[row*4096+head*64 for row in range(16) for head in range(12)]
    result[anchors]=np.linspace(-7.,7.,192)
    return result

def oracle(source,gain):
    compact=np.zeros(SLOTS);local=np.zeros(SLOTS);global_keys=np.zeros(SLOTS);expanded=np.zeros(SLOTS)
    for row in range(16):
        local_key=(row+15)%16
        for head in range(12):
            value=source[row*4096+head*64]*gain
            compact[row*4096+head]=value
            local[row*4096+local_key*16+head]=value
            global_keys[row*4096+(112+local_key)*16+head]=value
            expanded[row*4096+head*64]=value
    return dict(zip(LABELS,(compact,local,global_keys,local.copy(),compact.copy(),expanded)))
