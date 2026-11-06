"""Preserve cleanup provenance across this dedicated process's helper layers.

The marker is private bookkeeping for caught exceptions, not a security token.
A real body failure keeps its identity. With cleanup-only failures, the first
cancellation outranks ordinary failures even after an intermediate helper exits.
"""
ATTRIBUTE='_cross_toy4_cleanup_origin'
BODY_ATTRIBUTE='_cross_toy4_body_origin'

def is_cleanup(error):return error is not None and getattr(error,BODY_ATTRIBUTE,False)is not True and getattr(error,ATTRIBUTE,False)is True

def protect_body(error):
    # Capture provenance before cleanup actions may re-raise the identical
    #exception object (for example a sticky readiness failure).
    if error is not None and not is_cleanup(error):setattr(error,BODY_ATTRIBUTE,True)
    return error

def mark(error):
    if getattr(error,BODY_ATTRIBUTE,False)is not True:setattr(error,ATTRIBUTE,True)
    return error

def select(primary,errors):
    """Select one exception; errors are labeled cleanup operations, in order."""
    protect_body(primary)
    rows=[]
    for label,error in errors:
        if error is None or error is primary:continue
        rows.append((label,mark(error)))
    if primary is not None and not is_cleanup(primary):chosen=primary
    else:
        ordered=([('earlier cleanup',primary)]if primary is not None else[])+rows
        if not ordered:return None
        chosen=next((e for _,e in ordered if not isinstance(e,Exception)),ordered[0][1])
    if primary is not None and primary is not chosen:chosen.add_note('earlier cleanup: '+repr(primary))
    for label,error in rows:
        if error is not chosen:chosen.add_note(label+': '+repr(error))
    return chosen
