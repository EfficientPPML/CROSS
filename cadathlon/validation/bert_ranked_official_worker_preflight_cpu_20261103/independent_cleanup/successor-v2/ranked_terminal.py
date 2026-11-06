"""Publish completion only after the outer termination scope exits cleanly."""
from pathlib import Path
import json
import key_gate
from cleanup import termination_signals,cleanup_signals


def finish(path,call,write,*,termination=termination_signals,shield=cleanup_signals):
    report=None
    try:
        with termination():report=call()
        if type(report)is not dict:raise ValueError('terminal report object required')
        report['termination_scope_completed']=True
        write(path,report)
        return report
    except BaseException as primary:
        if report is None:
            try:
                raw=key_gate.read(path,16<<20)
                report=json.loads(raw)
                if type(report)is not dict:report={}
            except BaseException:report={}
        report.update(passed=False,termination_scope_completed=False,terminal_scope_completed=False,cleanup_completed=False,
            error={'type':type(primary).__name__,'message':'outer ranked terminal failure; private exception text withheld'})
        try:
            with shield():write(path,report)
        except BaseException as secondary:
            if secondary is not primary:primary.add_note('outer terminal failed republication: '+type(secondary).__name__)
        raise
