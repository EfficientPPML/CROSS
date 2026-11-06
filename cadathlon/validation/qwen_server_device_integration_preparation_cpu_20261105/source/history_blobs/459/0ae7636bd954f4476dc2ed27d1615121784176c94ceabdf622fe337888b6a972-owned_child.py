"""One independently reaped recording case; no actual CLI/bootstrap claim."""
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

DENIED=[];TEMPORARY={};PROTECTED=();OWNED=None;ROOT=None

def canonical(value):return(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def dump(path,value):
    raw=canonical(value);assert len(raw)<=2<<20
    path.write_bytes(raw)
def ticks(pid):
    raw=Path('/proc')/str(pid)/'stat'
    text=raw.read_text();return int(text[text.rfind(')')+2:].split()[19])

def audit(event,args):
    if event=='import'and args and str(args[0]).split('.')[0]in ('numpy','jax','jaxlib','jaxite_word','libtpu','jax_plugins','torch','tensorflow','transformers','tokenizers','safetensors','flax','torch_xla'):
        DENIED.append(['import',str(args[0])]);raise ImportError('genuine native imports unavailable')
    if event in('ctypes.dlopen','subprocess.Popen','os.system','os.fork','os.posix_spawn')or event.startswith('socket.'):
        DENIED.append(['effect',event]);raise PermissionError('native/network/child effect not selected')
    if event=='open'and args and type(args[0])in(str,bytes):
        path=os.fsdecode(args[0])
        if any(path==p or path.startswith(p+'/')for p in('/proc','/sys','/dev',*PROTECTED)):
            DENIED.append(['probe_or_body',path]);raise PermissionError('no probe or protected body selected')
        flags=args[2]if len(args)>2 and type(args[2])is int else 0
        if flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            p=Path(path)
            if OWNED is not None and p!=OWNED and OWNED not in p.parents:
                DENIED.append(['outside_owned_write',path]);raise PermissionError('child writes limited to its own fixture/result root')

def main():
    global TEMPORARY,PROTECTED,OWNED,ROOT
    assert sys.flags.isolated and sys.dont_write_bytecode and __debug__
    case,root_arg,out_arg,map_arg,map_sha=sys.argv[1:]
    ROOT=Path(root_arg).resolve();OWNED=Path(out_arg).resolve();assert OWNED.is_dir()
    start=ticks(os.getpid())
    terminal=dict(schema='full28-device-owned-child-v1',case=case,pid=os.getpid(),pgid=os.getpgrp(),
        starttime_ticks=start,status='entered',normal_return=False)
    dump(OWNED/'child-terminal.json',terminal)
    raw=Path(map_arg).read_bytes();assert sha(raw)==map_sha
    full_map=json.loads(raw);assert canonical(full_map)==raw
    # Bootstrap only this source-pinned test helper; no candidate imports.
    helper=ROOT/'source_loader.py';assert sha(helper.read_bytes())==full_map['source_loader.py']
    spec=importlib.util.spec_from_file_location('source_loader',helper);module=importlib.util.module_from_spec(spec)
    sys.modules['source_loader']=module;spec.loader.exec_module(module)
    module.verify_tree(ROOT,full_map)
    expected=json.loads((ROOT/'runtime_sources.json').read_bytes());assert len(expected)==403
    assert {n[8:]:h for n,h in full_map.items()if n.startswith('runtime/')}==expected
    nodes=json.loads((ROOT/'selected_tests.json').read_bytes())['cases'];assert case in nodes
    selection=json.loads((ROOT/'helper_selection.json').read_bytes())
    finder=module.SourceFinder(ROOT,full_map,selection);finder.temporary=TEMPORARY
    assert not any(n.split('.')[0]in module.FORBIDDEN for n in sys.modules)
    sys.meta_path.insert(0,finder);sys.addaudithook(audit)
    provenance=json.loads((ROOT/'entry_fixture_provenance.json').read_bytes())
    entry,definitions=module.entry_definitions(ROOT/'runtime',provenance)
    original_main=sys.modules['__main__'];fixture=None;result=None;primary=None
    try:
        sys.modules['__main__']=entry;exec(definitions,entry.__dict__)
        PROTECTED=entry.EARLY_PROTECTED
        blobs=entry.snapshot(expected);entry.origins(expected)
        if case=='entry_status_denial':
            from cases_startup import entry_denial
            detail=entry_denial(entry,expected,blobs,finder);counts={};events=[];seams=[];details={}
        else:
            from fixture_base import Fixture
            fixture=Fixture(ROOT,ROOT/'runtime',entry,expected,blobs,OWNED,finder,TEMPORARY,case)
            group=json.loads((ROOT/'case_groups.json').read_bytes())[case]
            name={'P':'cases_packet','F':'cases_packet','O':'cases_device','T':'cases_device',
                'A':'cases_audit','R':'cases_process','L':'cases_loader_engine','E':'cases_loader_engine','B':'cases_startup'}[group]
            detail=importlib.import_module(name).run(fixture,case)
            counts=dict(fixture.counts);events=list(fixture.events);seams=list(fixture.seams);details=dict(fixture.details)
            entry.origins(expected,core_enabled=fixture.gate.core_enabled)
        assert detail['passed']is True and detail['case']==case
        finder.check();assert not DENIED
        result=dict(schema='full28-device-recording-result-v1',case=case,passed=True,pid=os.getpid(),pgid=os.getpgrp(),
            starttime_ticks=start,detail=detail,details=details,counters=counts,recording_seams=seams,
            event_counts=dict(collections.Counter(events)),event_tail=events[-32:],
            loaded_helpers=sorted(finder.loaded),definition_only_entry=True,genuine_cli_bootstrap=False,
            model_key_ciphertext_body_reads=False,native_backend_imported=False,capacity_measured=False,
            actual_execution_authorized=False,backend_denials=[])
    except BaseException as error:
        primary=error;raise
    finally:
        try:
            if fixture is not None:fixture.close()
        except BaseException as cleanup:
            if primary is None:primary=cleanup;raise
            if cleanup is not primary:primary.add_note('fixture cleanup: '+repr(cleanup))
        finally:
            sys.modules['__main__']=original_main
            try:
                finder.check();assert not TEMPORARY and sys.getprofile()is None and not DENIED
                module.verify_tree(ROOT,full_map)
            except BaseException as cleanup:
                if primary is None:primary=cleanup;raise
                if cleanup is not primary:primary.add_note('final source/origin restoration: '+repr(cleanup))
    assert result is not None
    result.update(original_main_restored=True,temporary_names_removed=True,profile_restored=True,
        declared_exports_restored=True,source_unchanged=True)
    dump(OWNED/'result.json',result)
    terminal.update(status='normal_return',normal_return=True,result_sha256=sha((OWNED/'result.json').read_bytes()),
        backend_denials=[]);dump(OWNED/'child-terminal.json',terminal)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        # Early receipt remains exact; partial failures never manufacture a
        # normal-return or cleanup certificate.
        traceback.print_exception(error)
        raise SystemExit(70)
