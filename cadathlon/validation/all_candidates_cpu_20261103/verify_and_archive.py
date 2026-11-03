"""Verify and archive the isolated combined candidate regression result."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET

LIVE = Path('/home/jianming_gatech/CROSS')
LAB = Path('/tmp/cross-all-candidates-20261103')
ROOT = LAB / 'checkout'
EVIDENCE = LAB / 'evidence'
OUT = LIVE / 'cadathlon/validation/all_candidates_cpu_20261103'
report = json.loads((EVIDENCE/'report.json').read_text())
before = json.loads((EVIDENCE/'source-before.json').read_text())
after = json.loads((EVIDENCE/'source-after.json').read_text())
assert report['passed'] and report['returncode'] == 0 and before == after
with tarfile.open(EVIDENCE/'sources.tar.gz','r:gz') as archive:
    members = archive.getmembers()
    assert len(members) == len(before) and {m.name for m in members} == set(before)
    for member in members:
        assert member.isfile()
        data = archive.extractfile(member).read()
        assert hashlib.sha256(data).hexdigest() == before[member.name]
        assert (ROOT/member.name).read_bytes() == data
changes = json.loads((LAB/'changed_sources.json').read_text())
for rel, entry in changes.items():
    assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest() == entry['after_sha256']
    old = LIVE / rel
    assert (hashlib.sha256(old.read_bytes()).hexdigest() if old.exists() else None) == entry['before_sha256']
with tempfile.TemporaryDirectory(prefix='cross-patch-reconstruction-') as temp:
    base = Path(temp)
    shutil.copytree(LAB/'changed_base_sources',base,dirs_exist_ok=True)
    subprocess.run(['git','apply','--check',str(LAB/'combined.patch')],cwd=base,check=True)
    subprocess.run(['git','apply',str(LAB/'combined.patch')],cwd=base,check=True)
    for rel, entry in changes.items():
        assert hashlib.sha256((base/rel).read_bytes()).hexdigest() == entry['after_sha256']
log = (EVIDENCE/'pytest.log').read_text()
result = re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed in ([\d.]+)s',log)
assert result, log[-2000:]
passed, skipped, subtests = map(int,result.groups()[:3])
xml = ET.parse(EVIDENCE/'junit.xml')
skips = [{'classname':case.attrib.get('classname'),'name':case.attrib['name'],
          'reason':case.find('skipped').attrib.get('message')}
         for case in xml.findall('.//testcase') if case.find('skipped') is not None]
assert len(skips) == skipped
OUT.mkdir(exist_ok=False)
shutil.copytree(EVIDENCE,OUT,dirs_exist_ok=True)
for name in ('combined.patch','changed_sources.json','dependencies.json','normalization.json','wrapper.log'):
    shutil.copy2(LAB/name,OUT/name)
for name in ('prerequisites','changed_base_sources'):
    shutil.copytree(LAB/name,OUT/name)
shutil.copy2('/tmp/prepare_cross_all_candidates_20261103.py',OUT/'prepare_candidate.py')
shutil.copy2(__file__,OUT/'verify_and_archive.py')
shutil.copy2(ROOT/'cadathlon/validation/check_all_candidates_20261103.py',OUT/'run_checkpoint.py')
summary = {
    'scope':'Isolated integration of five reviewed runtime/collector patches; live runtime remains unchanged.',
    'passed':True,'pytest_passed':passed,'pytest_skipped':skipped,'pytest_subtests_passed':subtests,
    'pytest_seconds':float(result.group(4)),'external_seconds':report['parent_elapsed_s'],
    'source_count':len(before),'exact_archive_verified':True,'recorded_sources_unchanged':True,
    'combined_patch_reconstructs_all_changed_sources':True,'changed_files':len(changes),
    'combined_patch_sha256':hashlib.sha256((LAB/'combined.patch').read_bytes()).hexdigest(),
    'skips':skips,'full_model_evaluated':False,'accelerator_evaluated':False,
    'existing_ring_profiles_changed':False,'core_bootstrap_stage_clears_changed':False,
}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'README.md').write_text(f'''The five reviewed candidates passed together in a fresh isolated checkout:
**{passed} tests and {subtests} subtests passed**, with **{skipped} skips**, in
{float(result.group(4)):.2f} seconds ({report['parent_elapsed_s']:.2f} seconds externally).
All {len(before)} recorded source and fixture files were unchanged; every exact
source-archive member was independently checked against both manifests.

The 28-file combined patch contains bounded giant-step queuing, native key
allocation, optional HERot/HEMul NTT reuse, guarded dispatch-cache retention,
experimental model opt-in wiring, and owned-key-directory collector cleanup.
It reconstructs every changed source from the archived base bytes. The five
prerequisite patches are preserved exactly; the only normalization removes
one trailing empty line in a test file.

The suite covers all CADathlon tests plus CKKS contexts, codec preparation,
uniform encoding, Mapping compilation, BSGS, packing, prime generation,
bootstrap, key generation, rotation and multiplication regressions. Exact
commands, package versions, raw logs, JUnit output, resources and individual
skip reasons are retained. Explicitly opt-in large/secure checks and the
optional second-device guard are outside this run; their earlier separate
evidence remains tied to its original snapshots.

This candidate has not been applied to the live checkout. Existing ring
profiles and core Bootstrap stage clears are unchanged. This CPU regression
does not validate a complete encrypted transformer, TPU memory/performance,
full pretrained file stages, or a scoring baseline.
''')
(OUT/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},indent=2)+'\n')
print(json.dumps(summary,indent=2))
