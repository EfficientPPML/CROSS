"""Owned source-fixed recording hooks only; no real server or host/device work."""
STATE={}

def device_capacity_check_bindings(phase,blobs):
    STATE['arguments'](phase,blobs);STATE['log'].append(('bindings',None))
    STATE['sources']()
    if STATE['mode']=='binding_failure':raise STATE['marker']
    if STATE['mode']in('reentry_caught','reentry_uncaught'):
        if STATE['mode']=='reentry_uncaught':STATE['capacity'].issue_initialization(STATE['policy'])
        else:
            try:STATE['capacity'].issue_initialization(STATE['policy'])
            except PermissionError as error:STATE['caught_reentry']=error
            else:raise AssertionError('nested issue unexpectedly admitted')

def device_capacity_check_host(phase,blobs,stage):
    STATE['arguments'](phase,blobs);STATE['log'].append(('host',stage))
    return STATE['host'](stage)

def device_capacity_check_lease(phase,blobs):
    STATE['arguments'](phase,blobs);STATE['log'].append(('lease',None))
    if STATE['mode']=='lease_failure':raise STATE['marker']

def device_capacity_initialize(phase,blobs):
    STATE['arguments'](phase,blobs);STATE['log'].append(('initialize',None));STATE['initializations']+=1
    partial=object();STATE['partial_owners'].append(partial)
    if STATE['mode']=='initialization_failure':raise STATE['marker']
    module=STATE['install_recording_jax']();STATE['returned_device']=module.DEVICE
    if STATE['mode']=='policy_blobs':
        replacement=tuple(list(blobs));assert replacement==blobs and replacement is not blobs
        object.__setattr__(STATE['policy'],'blobs',replacement)
    return module.DEVICE

def device_capacity_synchronize(phase,blobs,stage):
    STATE['arguments'](phase,blobs);STATE['log'].append(('synchronize',stage))
