"""Source-only test-parent ownership repair; no imports, collection or probe."""
import ast
import difflib
import hashlib
import io
import json
from pathlib import Path
import tarfile

OLD=Path('/tmp/qwen-client-kernel-allowlist-tests-20261104')
NEW=Path('/tmp/qwen-client-kernel-allowlist-tests-v2-20261104')
PRIOR=Path('/tmp/qwen-client-kernel-allowlist-successor-handoff-20261104')
OUT=Path(__file__).parent
P1=Path('/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v1')
P2=Path('/tmp/qwen-client-kernel-focused-proof-20261104/preparation-v2')
def h(raw):return hashlib.sha256(raw).hexdigest()
def compact(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def pretty(v):return(json.dumps(v,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def snap(root):return{str(p.relative_to(root)):h(p.read_bytes())for p in sorted(root.rglob('*'))if p.is_file()}
oldmap_raw=(PRIOR/'focused-sources.json').read_bytes()
assert h(oldmap_raw)=='adc8de53818dfadf05dd088587203f75a3583a7dbd7bfa857db38b9f3470b3f1'
oldmap=json.loads(oldmap_raw);assert len(oldmap)==472 and snap(OLD)==oldmap
assert not NEW.exists()
for name,digest in oldmap.items():
    data=(OLD/name).read_bytes();assert h(data)==digest
    target=NEW/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
before=(OLD/'test_kernel_allowlist.py').read_text()
old='''        finally:
            if child.poll()is None:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            child.wait()
'''
new='''        finally:
            primary=sys.exception()
            try:
                try:
                    if child.poll()is None:
                        try:os.killpg(child.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                except BaseException as cleanup:
                    if primary is None:raise
                    if cleanup is not primary:primary.add_note('owned child group cleanup: '+repr(cleanup))
            finally:
                # Attempt direct wait even if poll/killpg failed. Retain an
                # active body or group-cleanup error if this wait also fails.
                wait_primary=primary if primary is not None else sys.exception()
                try:child.wait()
                except BaseException as cleanup:
                    if wait_primary is None:raise
                    if cleanup is not wait_primary:wait_primary.add_note('owned child direct wait: '+repr(cleanup))
'''
assert before.count(old)==1;after=before.replace(old,new,1)
old="    assert type(report['pgid'])is int and report['pgid']>0\n"
new="    assert type(report['pgid'])is int and report['pgid']==child.pid\n"
assert after.count(old)==1;after=after.replace(old,new,1)
ast.parse(after);(NEW/'test_kernel_allowlist.py').write_text(after)
current=snap(NEW);assert len(current)==472
assert {n for n in oldmap if oldmap[n]!=current[n]}=={'test_kernel_allowlist.py'}
assert snap(NEW/'runtime')==json.loads((NEW/'runtime-sources.json').read_bytes())
for name,data in [('parent472-sources.json',oldmap_raw),('before_test_kernel_allowlist.py',before.encode())]:
    (OUT/name).write_bytes(data)
(OUT/'test_parent_ownership.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='held472-unrun/test_kernel_allowlist.py',tofile='ownership-successor472/test_kernel_allowlist.py')))
map_raw=compact(current);(OUT/'sources.json').write_bytes(map_raw)
with tarfile.open(OUT/'sources.tar.gz','w:gz')as archive:
    for name,digest in current.items():
        data=(NEW/name).read_bytes();assert h(data)==digest
        info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o644;info.mtime=0
        archive.addfile(info,io.BytesIO(data))
archive_sha=h((OUT/'sources.tar.gz').read_bytes())
correction={'schema':'qwen-client-kernel-test-ownership-correction-v1',
    'predecessor_source':str(OLD),'predecessor_source_manifest_sha256':h(oldmap_raw),
    'predecessor_harness':str(P1),'predecessor_harness_unrun':True,
    'successor_source':str(NEW),'source_count':472,'source_manifest_sha256':h(map_raw),
    'source_archive_sha256':archive_sha,'unchanged_source_files':471,
    'only_changed_file':'test_kernel_allowlist.py','before_sha256':h(before.encode()),'after_sha256':h(after.encode()),
    'runtime464_unchanged':True,'static24_selection_unchanged':True,
    'changes':['Nest direct child.wait in finally regardless poll/killpg error; preserve active primary through notes where possible.',
        'Require recorded child pgid==original child.pid even when parent live getpgid raced with child exit.'],
    'runtime_failure_observed':False,'imports_collection_probes_tests':False,'actual_authority':False}
(OUT/'correction.json').write_bytes(pretty(correction))

# Same admitted parent driver; only final source/config/pins and paths change.
config=json.loads((P1/'run_config.json').read_bytes())
config.update(source_root=str(NEW),proof_root=str(P2),source_manifest_sha256=h(map_raw),source_archive_sha256=archive_sha)
config_raw=pretty(config);(P2/'run_config.json').write_bytes(config_raw)
worker=(P1/'worker_driver.py').read_bytes();(P2/'worker_driver.py').write_bytes(worker)
old_driver=(P1/'executed_driver.py').read_text()
old_config_sha=h((P1/'run_config.json').read_bytes())
assert old_driver.count(old_config_sha)==1
driver=old_driver.replace(old_config_sha,h(config_raw),1)
ast.parse(driver);(P2/'executed_driver.py').write_text(driver)
(P2/'executed_driver.py.patch').write_text(''.join(difflib.unified_diff(old_driver.splitlines(True),driver.splitlines(True),fromfile='unrun-preparation-v1/executed_driver.py',tofile='preparation-v2/executed_driver.py')))
for name,data in [('sources-before.json',map_raw),('sources.tar.gz',(OUT/'sources.tar.gz').read_bytes()),('source-correction.json',pretty(correction))]:
    (P2/name).write_bytes(data)
execution=P2/'execution';assert not execution.exists()
for name,digest in current.items():
    data=(NEW/name).read_bytes();assert h(data)==digest
    target=execution/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
proposal=json.loads((P1/'launch-proposal.json').read_bytes())
proposal.update(driver_sha256=h(driver.encode()),worker_sha256=h(worker),config_sha256=h(config_raw),
    source_manifest_sha256=h(map_raw),source_archive_sha256=archive_sha)
proposal['argv']=[str(P2/'executed_driver.py')if v==str(P1/'executed_driver.py')else v for v in proposal['argv']]
(P2/'launch-proposal.json').write_bytes(pretty(proposal))
provenance=json.loads((P1/'driver_provenance.json').read_bytes())
provenance.update(unrun_preparation_v1_preserved=True,unrun_predecessor_harness=str(P1),
    unrun_predecessor_driver_sha256=h(old_driver.encode()),
    selected_source_sha256=h(map_raw),test_parent_ownership_correction_sha256=h(pretty(correction)),
    worker_exact_unrun_preparation_v1=True,driver_change_from_unrun_v1='CONFIG_SHA256 only')
(P2/'driver_provenance.json').write_bytes(pretty(provenance))
(P2/'README.md').write_text((P1/'README.md').read_text()+'''

This preparation-v2 selects the fresh test-parent ownership successor only.
The unrun preparation-v1 and its472 sources are preserved. The sole test change
always attempts direct wait after group-cleanup errors and requires the child's
reported actual PGID to equal its original PID. All464 nested runtime files,
24 static nodes, probe, parent guard and worker bytes are unchanged. No runtime
failure, collection, probe or test execution occurred before this correction.
''')
handoff={'schema':'qwen-client-kernel-focused-harness-handoff-v2','proof_root':str(P2),
    'source_root':str(NEW),'source_count':472,'static_selected_count':24,
    'source_manifest_sha256':h(map_raw),'source_archive_sha256':archive_sha,
    'driver_sha256':h(driver.encode()),'worker_sha256':h(worker),'config_sha256':h(config_raw),
    'proposal_sha256':h(pretty(proposal)),'driver_provenance_sha256':h(pretty(provenance)),
    'source_correction_sha256':h(pretty(correction)),'test_sha256':h(after.encode()),
    'runtime464_unchanged':True,'source_execution_archive_equal':True,'cpu_affinity':[108,109,110,111],
    'runtime_module_origins':{},'runtime_source_expectations':{},'collected':False,'executed':False,
    'probe_authorized':False,'scope':config['scope']}
for path in (OUT/'handoff.json',P2/'handoff.json'):path.write_bytes(pretty(handoff))
assert snap(OLD)==oldmap and snap(NEW)==snap(execution)==current
print(json.dumps(handoff,sort_keys=True))
