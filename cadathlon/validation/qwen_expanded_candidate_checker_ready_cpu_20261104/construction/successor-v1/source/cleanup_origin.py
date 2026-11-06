"""Dedicated-worker exception provenance; not a security token.

Protect a genuine body exception before attempting cleanup. Sticky cleanup
actions can rethrow that same object without changing its original provenance.
"""
BODY = '_qwen_observer_body_origin'
CLEANUP = '_qwen_observer_cleanup_origin'


def cleanup_origin(error):
    return error is not None and getattr(error, BODY, False) is not True and getattr(error, CLEANUP, False) is True


def protect_body(error):
    if error is not None and not cleanup_origin(error):
        setattr(error, BODY, True)
    return error


def mark_cleanup(error):
    if error is not None and getattr(error, BODY, False) is not True:
        setattr(error, CLEANUP, True)
    return error


def choose(primary, error, label, *, body_primary=None):
    if error is None or error is primary:
        return primary
    if body_primary is not None:
        protect_body(body_primary)
    mark_cleanup(error)
    if primary is None:
        return error
    if body_primary is not None:
        primary.add_note(label+': '+type(error).__name__)
        return primary
    if cleanup_origin(primary) and isinstance(primary, Exception) and not isinstance(error, Exception):
        error.add_note('Earlier cleanup failure: '+type(primary).__name__)
        return error
    primary.add_note(label+': '+type(error).__name__)
    return primary
