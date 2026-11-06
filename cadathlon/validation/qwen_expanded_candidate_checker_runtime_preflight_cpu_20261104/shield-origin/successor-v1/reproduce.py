"""Exact AST cleanup test with fake signals/output; no backend/body access."""
import ast
from contextlib import contextmanager
import hashlib,json,time
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
text=(HERE/'sources/checker_process.py').read_text()
tree=ast.parse(text)
names={'signals','accumulate','supervise'}
selected=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names],type_ignores=[])

def probe(cancel_type,body=None):
    events=[];setup=OSError('synthetic shield setup');cancellation=cancel_type('synthetic cleanup cancellation')
    class Signal:
        SIGTERM=15;SIGINT=2;SIGHUP=1;SIG_BLOCK=0;SIG_SETMASK=2
        calls=0
        @staticmethod
        def pthread_sigmask(*args):return set()
        @staticmethod
        def getsignal(*args):return None
        @classmethod
        def signal(cls,*args):
            cls.calls+=1
            # Three successful outer installations; first inner installation fails.
            if cls.calls==4:raise setup
    class Output:
        path='synthetic';owner=(0,1)
        def close(self):events.append('output_close')
        def json(self,name,report):events.append('publication');return None
        @classmethod
        def reopen(cls,*args):return cls()
    env=dict(contextmanager=contextmanager,signal=Signal,SIGNALS=(15,2,1),time=time,
        Output=Output,source_end=lambda *args:events.append('source_check')or{})
    exec(compile((HERE/'sources/cleanup_origin.py').read_text(),'reviewed-cleanup-origin','exec'),env)
    exec(compile(selected,'reviewed-checker-process-AST','exec'),env)
    def action():
        events.append('body')
        if body is not None:raise body
    def cleanup():events.append('cleanup');raise cancellation
    report=dict(passed=True)
    try:env['supervise'](Output(),report,action,[cleanup],Path('/synthetic'),{})
    except BaseException as error:observed=error
    else:raise AssertionError('Failure unexpectedly accepted')
    wanted=body if body is not None else cancellation
    assert all(n in events for n in ['cleanup','source_check','output_close','publication'])
    return dict(passed=observed is wanted,observed=type(observed).__name__,expected=type(wanted).__name__,
        setup_marked_as_body=getattr(setup,'_qwen_observer_body_origin',False),events=events,
        scientific_pass=report['passed'],cleanup_completed=report['cleanup_completed'])

rows=[probe(KeyboardInterrupt),probe(SystemExit),probe(KeyboardInterrupt,ValueError('genuine body'))]
pins=json.loads((HERE/'before.json').read_text())
after={n:hashlib.sha256((HERE/'sources'/n).read_bytes()).hexdigest()for n in pins}
assert pins==after
report=dict(scope='Exact source AST and real cleanup marker, fake signals/output only; no actual signals/backend/input/model/HE',
    checks=rows,expected_negative_failures=0,genuine_body_control=True,sources_before=pins,sources_after=after,
    archive_sha256=hashlib.sha256((HERE/'sources.tar.gz').read_bytes()).hexdigest())
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if all(row['passed']for row in rows)else 1)
