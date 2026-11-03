"""Verify the integrated runtime candidate without applying it to live sources."""
from pathlib import Path
import hashlib
import json
import math
import re
import shutil
import subprocess
import tarfile
import tempfile
import xml.etree.ElementTree as ET

LAB=Path('/tmp/runtime-followup-integration-20261103');ROOT=LAB/'checkout'
LIVE=Path('/home/jianming_gatech/CROSS')
OUT=LIVE/'cadathlon/validation/runtime_followup_integration_cpu_20261103'
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_archive(path,before,after):
    assert before==after
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        assert len(members)==len(before) and {m.name for m in members}==set(before)
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==before[member.name]

run=LAB/'evidence';report=json.loads((run/'report.json').read_text())
before=json.loads((run/'source-before.json').read_text());after=json.loads((run/'source-after.json').read_text())
assert report['passed'] and report['returncode']==0 and report['source_unchanged']
verify_archive(run/'sources.tar.gz',before,after)
assert all(digest(ROOT/name)==expected for name,expected in before.items())
result=re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed in ([\d.]+)s',(run/'pytest.log').read_text())
assert result
device=json.loads((LAB/'second_device/report.json').read_text())
assert device['passed'] and device['returncode']==0 and device['logical_cpu_devices']==2
verify_archive(LAB/'second_device/sources.tar.gz',device['source_hashes_before'],device['source_hashes_after'])
changes=json.loads((LAB/'changed_sources.json').read_text())
for name,entry in changes.items():
    p=LIVE/name
    assert (digest(p) if p.exists() else None)==entry['before_sha256']
    assert digest(ROOT/name)==entry['after_sha256']
dependencies=json.loads((LAB/'prerequisites.json').read_text())
assert all(digest(LAB/row['copy'])==row['sha256']==digest(LIVE/row['source']) for row in dependencies)
with tempfile.TemporaryDirectory(prefix='runtime-combined-reconstruction-') as directory:
    target=Path(directory);shutil.copytree(LAB/'before/sources',target,dirs_exist_ok=True)
    subprocess.run(['git','apply',str(LAB/'combined.patch')],cwd=target,check=True)
    assert all(digest(target/name)==entry['after_sha256'] for name,entry in changes.items())
subprocess.run(['git','apply','--check',str(LAB/'combined.patch')],cwd=LIVE,check=True)
records=json.loads((run/'secure-records.json').read_text())
assert all(records.get(name) for name in ('test_lazy_bconv','test_scoped_mul_ntt','test_scoped_rescale_ntt','test_scoped_nested_rescale'))
errors=[entry['max_abs_error'] for rows in records.values() for entry in rows if 'max_abs_error' in entry]
assert errors and all(math.isfinite(error) and error<2e-6 for error in errors)
skips=[{'name':case.attrib['name'],'reason':case.find('skipped').attrib.get('message')}
       for case in ET.parse(run/'junit.xml').findall('.//testcase') if case.find('skipped') is not None]
summary=dict(scope=__doc__,passed=True,pytest_passed=int(result.group(1)),skipped=int(result.group(2)),
    subtests_passed=int(result.group(3)),pytest_seconds=float(result.group(4)),
    external_seconds=report['parent_elapsed_s'],source_count=len(before),sources_unchanged=True,
    exact_source_archives_verified=True,second_device_checks_passed=2,skip_reasons=skips,
    secure_component_records={name:len(rows) for name,rows in records.items()},
    secure_component_max_abs_error=max(errors),changed_files=len(changes),
    combined_patch_sha256=digest(LAB/'combined.patch'),patch_reconstruction_verified=True,
    applies_to_current_live_runtime=True,applied_to_live=False,
    full_model_evaluated=False,accelerator_evaluated=False,ring_profiles_changed=False,
    core_bootstrap_arithmetic_and_stage_clears_changed=False)
OUT.mkdir(exist_ok=False)
shutil.copytree(run,OUT/'combined')
shutil.copytree(LAB/'second_device',OUT/'second_device')
for folder in ('before','prerequisites'):
    shutil.copytree(LAB/folder,OUT/folder)
for name in ('combined.patch','changed_sources.json','prerequisites.json','build_combined_patch.py','second_device_driver.py','wrapper.log'):
    shutil.copy2(LAB/name,OUT/name)
shutil.copy2(LAB/'second_device.log',OUT/'second_device/pytest.log')
shutil.copy2('/tmp/prepare_runtime_followups_20261103.py',OUT/'prepare_candidate.py')
shutil.copy2(__file__,OUT/'verify_and_archive.py')
for name in ('check_followup_integration.py','followup_pytest_child.py'):
    shutil.copy2(ROOT/'cadathlon/validation'/name,OUT/name)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'README.md').write_text(f'''All three reviewed follow-ups compose with the earlier five-candidate base:
scoped rescale-facade controls, nested/public HEMul rescale controls, and
serialization-safe lazy public BConv construction. The resulting32-file patch
passed **{summary['pytest_passed']} tests and {summary['subtests_passed']} subtests**, with
**{summary['skipped']} skips**, in {summary['pytest_seconds']:.2f} seconds
({summary['external_seconds']:.2f} seconds externally). All {len(before)} recorded
source and fixture files stayed unchanged; every exact archive member and
every reconstructed changed source was independently verified.

The run covers real secure tensor/multiplication/rescale and lazy-versus-eager
parity, control ownership and retirement, serialization, model opt-in wiring,
all twelve public ideal graph contracts, staged-helper/release packaging and
collector cancellation/cleanup. Secure component errors were at most
{max(errors):.8g}; each test retained its original gate. The ordinary one-device
run skipped two optional second-device guards and two opt-in large secure
refresh tests. Both device guards subsequently passed in a fresh two-device
process. The large refreshes were not run by this checkpoint.

The exact prerequisite patches, combined patch, changed base bytes, source
archives, commands, raw output, JUnit, resource records and numerical records
are retained. This combined candidate has not been applied to the live runtime.
It preserves existing ring profiles and core Bootstrap arithmetic/stage clears.
This is not a complete encrypted-model, pretrained file-stage, TPU-memory or
scoring-calibration result. The live Qwen run keeps its original frozen sources.
'''.replace('resulting32-file','resulting 32-file'))
(OUT/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):digest(p)
    for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},indent=2)+'\n')
print(json.dumps(summary,indent=2))
