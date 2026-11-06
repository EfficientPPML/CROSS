"""Stdlib parent: original child ownership, direct waits, no health deadline."""
from pathlib import Path
import json,os,signal,subprocess,sys
ROOT=Path(__file__).resolve().parent

def _dump(path,value):path.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n')
def _gone(group):
    try:os.killpg(group,0)
    except ProcessLookupError:return True
    return False

def _start_ticks(pid):
    raw=Path('/proc')/str(pid)/'stat'
    fields=raw.read_text().rsplit(')',1)[1].split()
    return int(fields[19])

def run_case(name,tmp_path):
    variants=('help','usage')if name=='test_help_and_usage_keep_zero_and_two_without_dispatch'else('',)
    for variant in variants:
        out=tmp_path/('child-'+(variant or 'case'));out.mkdir(mode=0o700)
        log=out/'stdout.log';err=out/'stderr.log'
        primary=None;secondary=[];observed=None;start_ticks=None;code=None;direct_wait_completed=False
        expected=0 if not variant else(0 if variant=='help'else 2)
        if name.startswith('test_actual_')and '_cli_'in name:expected=70
        argv=[sys.executable,'-I','-B',str(ROOT/'case_child.py'),'--case',name,'--variant',variant,'--output',str(out)]
        env={k:os.environ[k]for k in ('PATH','LANG','LC_ALL','TMPDIR')if k in os.environ}
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
        with log.open('wb')as stdout,err.open('wb')as stderr:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
            try:
                try:observed=os.getpgid(child.pid)
                except ProcessLookupError:pass
                try:start_ticks=_start_ticks(child.pid)
                except FileNotFoundError:pass
                _dump(out/'parent-live.json',dict(pid=child.pid,pgid=observed,start_ticks=start_ticks,argv=argv))
                code=child.wait();direct_wait_completed=True
            except BaseException as error:primary=error
            finally:
                # Kill errors cannot bypass the original child's direct wait.
                if not direct_wait_completed:
                    try:
                        if child.poll()is None:
                            try:os.killpg(child.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                    except BaseException as error:secondary.append(('owned group cleanup',error))
                    finally:
                        try:code=child.wait();direct_wait_completed=True
                        except BaseException as error:secondary.append(('owned direct wait',error))
                absent=None
                try:absent=_gone(child.pid)
                except BaseException as error:secondary.append(('terminal group observation',error))
                terminal=dict(case=name,variant=variant,pid=child.pid,observed_pgid=observed,start_ticks=start_ticks,exit_code=code,direct_wait_completed=direct_wait_completed,group_absent=absent,primary_type=None if primary is None else type(primary).__name__,cleanup_types=[dict(label=label,error_type=type(error).__name__)for label,error in secondary])
                try:_dump(out/'parent-terminal.json',terminal)
                except BaseException as error:secondary.append(('parent terminal publication',error))
                chosen=primary if primary is not None else(secondary[0][1]if secondary else None)
                if chosen is not None:
                    for label,error in secondary:
                        if error is not chosen:
                            try:chosen.add_note(label+': '+type(error).__name__)
                            except BaseException:pass
                    raise chosen
        early=json.loads((out/'child-early.json').read_bytes());terminal=json.loads((out/'child-terminal.json').read_bytes())
        assert direct_wait_completed
        assert type(early['pid'])is int and early['pid']==child.pid
        assert type(early['pgid'])is int and early['pgid']==child.pid
        assert type(early['start_ticks'])is int and early['start_ticks']>0
        assert observed is None or observed==early['pgid']
        assert start_ticks is None or start_ticks==early['start_ticks']
        assert terminal['pid']==child.pid and terminal['pgid']==early['pgid']and terminal['start_ticks']==early['start_ticks']and terminal['case']==name
        assert _gone(early['pgid'])
        assert code==expected,(name,code,expected,err.read_text()[-10000:])
        result=json.loads((out/'result.json').read_bytes());assert result['passed']is True,result
        assert terminal['passed']is True and terminal['forbidden_imports']==[],terminal
        assert result['runtime_map_sha256']=='d82d68101c1f3ddc7f18b6ae4492aa7618d779012babaf9ed85355dfab927fea'
