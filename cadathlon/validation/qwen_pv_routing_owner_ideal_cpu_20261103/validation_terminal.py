"""Fail closed if the cleanup scope raises after terminal publication."""
from contextlib import contextmanager


@contextmanager
def terminal_scope(report,path,write,shield,prior):
    report['terminal_scope_completed']=False
    try:
        with shield():
            yield
        report['terminal_scope_completed']=True
        write(path,report)
    except BaseException as error:
        primary,cleanup_error=prior()
        chosen=primary
        if chosen is None:
            if cleanup_error is None:
                chosen=error
            elif not isinstance(error,Exception) and isinstance(cleanup_error,Exception):
                chosen=error
            else:
                chosen=cleanup_error
        for label,other in (('cleanup scope',error),('earlier cleanup',cleanup_error)):
            if other is not None and other is not chosen:
                chosen.add_note(label+': '+repr(other))
        report['passed']=False
        report['terminal_scope_completed']=False
        if 'cleanup_completed' in report:report['cleanup_completed']=False
        report.setdefault('error',dict(type=type(chosen).__name__,message=str(chosen)))
        report['terminal_failure_republished']=True
        try:
            with shield():
                write(path,report)
        except BaseException as publication:
            if publication is not chosen:
                chosen.add_note('failed terminal republication: '+repr(publication))
        raise chosen
