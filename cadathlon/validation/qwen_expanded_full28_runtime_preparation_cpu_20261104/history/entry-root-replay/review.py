"""Replay identical synthetic CLI admission fixtures against preserved old/new entries."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import tarfile

sha = lambda b: hashlib.sha256(b).hexdigest()
base = Path('/tmp/qwen-expanded-preparation-entry-root-replay-20261104')
base.mkdir(exist_ok=False)
old = Path('/tmp/qwen-expanded-full-preparation-driver-draft-20261104/prepare_expanded.py')
new = Path('/tmp/qwen-expanded-full-preparation-driver-v3-20261104/prepare_expanded.py')
previous = Path('/tmp/qwen-expanded-full-preparation-driver-v2-20261104/prepare_expanded.py')
probe = Path('/tmp/qwen-expanded-preparation-entry-successor-probe-20261104/probe.py')
old_fixture = base/'old-driver-fixture'
old_fixture.mkdir()
(old_fixture/'prepare_expanded.py').write_bytes(old.read_bytes())
for name in ('freeze_io.py', 'cleanup_origin.py'):
    raw = (new.parent/name).read_bytes()
    assert raw == (previous.parent/name).read_bytes()
    (old_fixture/name).write_bytes(raw)
(base/'probe.py').write_bytes(probe.read_bytes())
(base/'review.py').write_bytes(Path(__file__).read_bytes())
selected = [old, new, previous, probe] + [new.parent/n for n in ('freeze_io.py','cleanup_origin.py')]
before = {str(p):sha(p.read_bytes()) for p in selected}
(base/'selected-before.json').write_text(json.dumps(before, indent=2, sort_keys=True)+'\n')
rows = []
for label, driver, expected_exit in [('old', old_fixture/'prepare_expanded.py', 1), ('corrected', new, 0)]:
    command = ['/tmp/cross-pretrained-validation-env/bin/python', '-I', '-B', str(base/'probe.py'), '--driver', str(driver), '--output', str(base/label)]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    stdout, stderr = process.communicate()
    (base/(label+'-stdout.log')).write_bytes(stdout)
    (base/(label+'-stderr.log')).write_bytes(stderr)
    try:
        os.killpg(process.pid, 0)
    except ProcessLookupError:
        group_absent = True
    else:
        group_absent = False
    assert process.returncode == expected_exit and group_absent
    report = json.loads((base/label/'report.json').read_bytes())
    expected_passes = [False,False,False,True] if label == 'old' else [True]*4
    assert [r['passed'] for r in report['cases']] == expected_passes
    assert report['source_count'] == 36 and report['declared_source_bytes_unchanged'] is True
    assert all(r['reaped'] and r['process_group_absent'] and r['returncode']==70 for r in report['cases'])
    if label == 'corrected':
        child_report = json.loads((base/label/'declared_helper_control/owned-output/report.json').read_bytes())
        assert child_report['synthetic_tests_started'] is False
        assert child_report['synthetic_component_tests_completed'] is False
        assert child_report['passed'] is False
        assert 'synthetic_parent_loop_executed' not in child_report
    rows.append(dict(label=label, driver_sha256=report['driver_sha256'], probe_sha256=report['probe_sha256'],
                     direct_exit=process.returncode, pid_pgid=process.pid, reaped=True, group_absent=group_absent,
                     expected_case_passes=expected_passes, report_sha256=sha((base/label/'report.json').read_bytes())))
assert rows[0]['probe_sha256'] == rows[1]['probe_sha256']
after = {str(p):sha(p.read_bytes()) for p in selected}
assert before == after
(base/'selected-after.json').write_text(json.dumps(after,indent=2,sort_keys=True)+'\n')
report = dict(passed=True, scope='Synthetic CLI admission replay only; no certificate derivation, 176-test suite, model, inputs, backend, HE, or private outcomes.',
              fixture_schema_extension='Two authenticated I/O helper files added identically to both old and new synthetic fixtures; original four assertions unchanged.',
              selected_sources_unchanged=True, cases=rows, corrected_early_failure_reports_tests_not_started=True)
(base/'review.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
with tarfile.open(base/'review-artifacts.tar.gz','w:gz') as archive:
    for p in sorted(base.rglob('*')):
        if p.is_file() and not p.is_symlink() and p.name != 'review-artifacts.tar.gz':
            archive.add(p,arcname=str(p.relative_to(base)),recursive=False)
print(json.dumps(dict(passed=True, review=str(base/'review.json'), sha256=sha((base/'review.json').read_bytes()),
                     old='3 expected failures, 1 passing control', corrected='4 passes', scope=report['scope'])))
