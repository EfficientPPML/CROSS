"""Publish outer-scope completion or a coherent public failure receipt."""
import json
from cleanup import termination_signals,cleanup_signals

def finish(path,call,write,read,*,termination=termination_signals,shield=cleanup_signals):
    report=None
    try:
        with termination():report=call()
        if type(report)is not dict:raise ValueError('terminal report object required')
        report['termination_scope_completed']=True
        write(path,report)
        return report
    except BaseException as primary:
        if type(report)is not dict:
            try:
                report=json.loads(read(path))
                if type(report)is not dict:report={}
            except BaseException:report={}
        report.update(passed=False,termination_scope_completed=False,terminal_scope_completed=False,
            error={'type':type(primary).__name__,'message':'public parameter terminal failure'})
        try:
            with shield():write(path,report)
        except BaseException as secondary:
            if secondary is not primary:primary.add_note('terminal failed republication: '+type(secondary).__name__)
        raise
