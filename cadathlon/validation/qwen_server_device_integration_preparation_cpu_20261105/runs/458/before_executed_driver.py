"""Unrun enclosing Q5 427/55 proof collector; external launch grant required."""
from pathlib import Path
import hashlib, importlib.util, json, os, signal, stat, subprocess, sys, tarfile, time
import xml.etree.ElementTree as ET

CONFIG_SHA256 = '8b423a4ddbfda4ad4c50aa039f38fa1876f2ba74034e0e60300ac676417d917c'
WORKER_SHA256 = '57fd322b69eaf64b6a1ca58dcf5e9b23dac707b933b2285c15a5b191748ec04d'
INSPECTOR_SHA256 = '5beb2916d31d09bfb267859ae659412b1bd81c2928e1496e24b0f37001996fa2'
HERE = Path(__file__).resolve().parent
sha = lambda raw: hashlib.sha256(raw).hexdigest()
canonical = lambda value: (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
map_canonical = lambda value: (json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()

def dump(name,value): (HERE/name).write_bytes(canonical(value))
def ticks(pid): return int((Path('/proc')/str(pid)/'stat').read_text().rsplit(')',1)[1].split()[19])
def inventory(root):
    assert not any(p.is_symlink()for p in root.rglob('*'))
    return {p.relative_to(root).as_posix():sha(p.read_bytes())
            for p in sorted(root.rglob('*'))if p.is_file()}
def absent(pid,group=False):
    try: (os.killpg if group else os.kill)(pid,0)
    except ProcessLookupError: return True
    return False

def main():
    raw=(HERE/'run_config.json').read_bytes(); assert sha(raw)==CONFIG_SHA256
    config=json.loads(raw)
    assert config['schema']=='qwen-toy5-role-focused-proof-v1'
    assert config['synthetic_only']is True and config['actual_execution_authorized']is False
    assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
    assert sys.executable==config['python'] and os.sched_getaffinity(0)=={108,109,110,111}
    source=Path(config['source_root']); out=Path(config['proof_root']); execution=out/'execution'
    assert HERE==out and source.is_absolute() and out.is_absolute()
    nodes=config['selected_nodes']; xml_cases=config['selected_xml_cases']
    assert len(nodes)==len(set(nodes))==len(xml_cases)==55
    assert sha(canonical(nodes))==config['selected_nodes_sha256']
    assert sha(canonical(xml_cases))==config['selected_xml_cases_sha256']
    manifest=(out/'sources-before.json').read_bytes(); assert sha(manifest)==config['source_manifest_sha256']
    expected=json.loads(manifest); assert manifest==map_canonical(expected) and len(expected)==427
    assert inventory(source)==inventory(execution)==expected
    assert sha((out/'sources.tar.gz').read_bytes())==config['source_archive_sha256']
    def archived():
        with tarfile.open(out/'sources.tar.gz')as archive:
            members=archive.getmembers()
            assert len(members)==len({m.name for m in members})==427
            assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in members)
            return {m.name:sha(archive.extractfile(m).read())for m in members}
    assert archived()==expected
    assert sha((out/'worker_driver.py').read_bytes())==WORKER_SHA256
    inspector_raw=(out/'inspect_children.py').read_bytes(); assert sha(inspector_raw)==INSPECTOR_SHA256
    assert sha((execution/'ARTIFACT_SELECTION.json').read_bytes())==config['artifact_plan_sha256']
    assert json.loads((execution/'ARTIFACT_SELECTION.json').read_bytes())==config['artifact_plan']
    assert sha((execution/'selected_tests.json').read_bytes())==config['selection_sha256']
    assert sha((execution/'numpy-pins.json').read_bytes())==config['numpy_pins_sha256']
    numpy_before={p:sha(Path(p).read_bytes())for p in config['numpy_pins']['sources']}
    assert numpy_before==config['numpy_pins']['sources']
    temporary=Path(config['owned_tmpdir']); basetemp=Path(config['pytest_basetemp'])
    assert temporary==out/'tmp' and basetemp==out/'pytest-owned'
    assert temporary.parent.resolve()==basetemp.parent.resolve()==out.resolve()
    assert not os.path.lexists(temporary)and not os.path.lexists(basetemp)
    temporary.mkdir(mode=0o700)
    env=dict(PATH='/usr/bin:/bin',HOME='/home/jianming_gatech',LANG='C.UTF-8',TZ='UTC',TMPDIR=str(temporary),
             PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    command=['taskset','-c','108-111',config['python'],'-I','-B',str(out/'worker_driver.py'),str(out/'run_config.json'),CONFIG_SHA256]
    started=time.monotonic(); own_ticks=ticks(os.getpid())
    child_pgid=None; child_ticks=None; code=None; waited=False; primary=None; secondary=[]
    with (out/'tests.log').open('xb')as log:
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            # Immediate ownership, before any fallible observation/publication.
            try: child_pgid=os.getpgid(child.pid)
            except ProcessLookupError: pass
            assert child_pgid is None or child_pgid==child.pid
            try: child_ticks=ticks(child.pid)
            except FileNotFoundError: pass
            dump('live-observation.json',dict(outer_pid=os.getpid(),outer_pgid=os.getpgrp(),outer_start_ticks=own_ticks,
                worker_pid=child.pid,worker_pgid=child_pgid,worker_start_ticks=child_ticks,
                owned_worker_session_pgid=child.pid,command=command,environment=env,config_sha256=CONFIG_SHA256))
            code=child.wait(); waited=True  # No healthy-execution deadline.
        except BaseException as error: primary=error
        finally:
            if not waited:
                try:
                    try: os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError: pass
                except BaseException as error: secondary.append(('owned group cleanup',error))
                finally:
                    try: code=child.wait(); waited=True
                    except BaseException as error: secondary.append(('owned direct wait',error))
            group_absent=None; pid_absent=None
            try: group_absent=absent(child.pid,True); pid_absent=absent(child.pid)
            except BaseException as error: secondary.append(('worker terminal observation',error))
            try:
                dump('worker-owner-terminal.json',dict(worker_pid=child.pid,owned_session_pgid=child.pid,
                    observed_pgid=child_pgid,start_ticks=child_ticks,exit_code=code,
                    direct_wait_completed=waited,child_reaped=waited,pid_absent=pid_absent,group_absent=group_absent,
                    parent_pid=os.getpid(),parent_pgid=os.getpgrp(),parent_start_ticks=own_ticks,
                    primary_type=None if primary is None else type(primary).__name__,
                    cleanup_types=[dict(label=label,error_type=type(error).__name__)for label,error in secondary]))
            except BaseException as error: secondary.append(('worker owner publication',error))
            chosen=primary if primary is not None else(secondary[0][1]if secondary else None)
            if chosen is not None:
                for label,error in secondary:
                    if error is not chosen:
                        try: chosen.add_note(label+': '+type(error).__name__)
                        except BaseException: pass
                raise chosen
    current,after=inventory(source),inventory(execution)
    (out/'sources-after.json').write_bytes(map_canonical(current))
    (out/'execution-after.json').write_bytes(map_canonical(after))
    archive_after=archived(); numpy_after={p:sha(Path(p).read_bytes())for p in config['numpy_pins']['sources']}
    spec=importlib.util.spec_from_file_location('q5_receipt_inspector',out/'inspect_children.py')
    module=importlib.util.module_from_spec(spec)
    # This is the separately pinned stdlib evidence inspector, never a runtime helper.
    exec(compile(inspector_raw,str(out/'inspect_children.py'),'exec'),module.__dict__)
    artifacts=module.inspect(out,config)
    dump('child-artifact-review.json',artifacts)
    xml=ET.parse(out/'tests.xml').getroot()
    counts={name:len(list(xml.iter(tag)))for name,tag in [('tests','testcase'),('failures','failure'),('errors','error'),('skipped','skipped')]}
    observed_xml=[dict(classname=c.attrib['classname'],name=c.attrib['name'])for c in xml.iter('testcase')]
    audit=json.loads((out/'worker-audit.json').read_bytes())
    early=json.loads((out/'worker-early.json').read_bytes()); terminal=json.loads((out/'worker-terminal.json').read_bytes())
    worker_identity=(all(type(r['pid'])is int and r['pid']==child.pid and type(r['pgid'])is int and r['pgid']==child.pid for r in (early,terminal,audit))
        and type(early['start_ticks'])is int and early['start_ticks']>0
        and early['start_ticks']==terminal['start_ticks']==audit['start_ticks']
        and(child_ticks is None or child_ticks==early['start_ticks'])
        and early['config_sha256']==CONFIG_SHA256 and early['source_manifest_sha256']==config['source_manifest_sha256'])
    passed=(code==0 and waited and group_absent and pid_absent and worker_identity
        and current==after==archive_after==expected and numpy_after==numpy_before
        and counts==dict(tests=55,failures=0,errors=0,skipped=0)and observed_xml==xml_cases
        and audit['collected_nodes']==nodes and audit['denied']==audit['loaded_backend']==[]
        and audit['forbidden_roots']==config['worker_forbidden_roots']and audit['numpy_imported']is False
        and type(audit['pytest_exit_code'])is int and audit['pytest_exit_code']==0
        and audit['planned_exit_code']==0 and audit['primary_type']is None and audit['cleanup_types']==[]
        and audit['affinity']==[108,109,110,111]and audit['temporary_root']==str(temporary)
        and audit['runtime_sources_before']==audit['runtime_sources_after']==config['runtime_source_expectations']
        and audit['numpy_sources_before']==audit['numpy_sources_after']==numpy_before
        and audit['runtime_module_origins']=={}and terminal['normal_main_return']is True
        and artifacts['passed']is True)
    report=dict(passed=bool(passed),tests=counts,selected_nodes=nodes,selected_xml_cases=observed_xml,
        source_count=427,runtime_count=389,source_manifest_sha256=config['source_manifest_sha256'],
        source_archive_sha256=config['source_archive_sha256'],sources_unchanged=current==after==archive_after==expected,
        child_returncode=code,child_reaped=waited,child_pid_absent=pid_absent,child_group_absent=group_absent,
        worker_identity_verified=worker_identity,elapsed_s=time.monotonic()-started,
        driver_sha256=sha(Path(__file__).read_bytes()),worker_driver_sha256=WORKER_SHA256,
        inspector_sha256=INSPECTOR_SHA256,config_sha256=CONFIG_SHA256,
        validated_children=artifacts['validated_children'],retained_regular_files=artifacts['regular_files'],
        artifact_review_sha256=sha((out/'child-artifact-review.json').read_bytes()),
        numpy_four_sources_unchanged=numpy_after==numpy_before,numpy_imported_in_worker=False,
        numpy_child_origin_scope=config['numpy_pins']['scope'],scope=config['scope'],
        synthetic_only=True,actual_execution_authorized=False,backend_context_construction=False,
        actual_key_generation=False,actual_model_execution=False,encrypted_execution=False,
        genuine_capacity_measured=False,fixture_bodies_copied=False,aliases_followed=False,
        independent_collector_exit_reap_required=True)
    dump('report.json',report)
    print(json.dumps(report,sort_keys=True),flush=True)
    return 0 if passed else 1

if __name__=='__main__':
    status=None; primary=None
    try:
        status=main()
        if type(status)is not int or status not in (0,1): raise ValueError('invalid proof collector status')
    except BaseException as error:
        primary=error; status=70
    try:
        dump('collector-terminal.json',dict(pid=os.getpid(),pgid=os.getpgrp(),start_ticks=ticks(os.getpid()),
            normal_main_return=primary is None,planned_exit_code=status,
            error_type=None if primary is None else type(primary).__name__,
            external_owner_wait_and_reap_required=True,
            source_claim='pre-exit collector record only; root/tool owns actual collector terminal'))
    except BaseException as error:
        if primary is not None:
            try: primary.add_note('collector terminal publication: '+type(error).__name__)
            except BaseException: pass
        status=70
    if primary is not None:
        try: print('Q5 proof collector infrastructure failure: '+type(primary).__name__,file=sys.stderr,flush=True)
        except BaseException: pass
    raise SystemExit(status)
