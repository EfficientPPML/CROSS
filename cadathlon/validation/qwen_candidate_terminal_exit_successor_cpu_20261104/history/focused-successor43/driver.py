"""Public source-bound focused terminal probe; no model or genuine run."""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET

SOURCE = Path('/tmp/qwen-expanded-candidate-checker-terminal-exit-successor-20261104')
OUTPUT = Path('/tmp/qwen-candidate-terminal-contract-review-20261104/focused-successor-v1')
PARENT = Path('/tmp/qwen-expanded-candidate-checker-ready-sources-v2-20261104.json')
NUMERIC = ('checker_runtime.py','strict_metrics.py','candidate_domains.py',
           'test_checker_runtime.py','test_strict_metrics.py','test_candidate_domains.py')

def sha(raw): return hashlib.sha256(raw).hexdigest()
def encoded(row): return (json.dumps(row, sort_keys=True, indent=2)+'\n').encode()
def inventory(root):
    out = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink(): raise ValueError('source symlink')
        if path.is_file(): out[str(path.relative_to(root))] = sha(path.read_bytes())
    return out

def main():
    OUTPUT.mkdir()
    before = inventory(SOURCE)
    parent = json.loads(PARENT.read_bytes())
    assert len(before) == 141 and len(parent) == 139
    assert set(before)-set(parent) == {'TERMINAL_SUCCESSOR.md','test_terminal_exit.py'}
    assert {name for name in parent if before[name] != parent[name]} == {'entry.py'}
    assert all(before[name] == parent[name] for name in NUMERIC)
    (OUTPUT/'source_manifest.json').write_bytes(encoded(before))
    execution = OUTPUT/'execution_source'
    execution.mkdir()
    with tarfile.open(OUTPUT/'sources.tar.gz','w:gz') as archive:
        for name in before:
            path = execution/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes((SOURCE/name).read_bytes())
            archive.add(path, arcname=name, recursive=False)
    assert inventory(execution) == before
    code = ('import sys,pytest; sys.path.insert(0,'+repr(str(execution))+'); '
            'raise SystemExit(pytest.main('+repr(['-q','-p','no:cacheprovider',str(execution/'test_terminal_exit.py'),
                '--junitxml='+str(OUTPUT/'tests.xml'),'--basetemp='+str(OUTPUT/'tmp')])+'))')
    environment = dict(os.environ, PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
    with (OUTPUT/'tests.log').open('wb') as log:
        child = subprocess.Popen([sys.executable,'-I','-B','-c',code],stdout=log,stderr=subprocess.STDOUT,
                                 env=environment,start_new_session=True)
        returncode = child.wait()
    try: os.killpg(child.pid,0)
    except ProcessLookupError: absent = True
    else: absent = False
    xml = ET.parse(OUTPUT/'tests.xml').getroot()
    cases = list(xml.iter('testcase'))
    bad = sum(len(list(xml.iter(name))) for name in ('failure','error','skipped'))
    receipts = [json.loads(path.read_bytes()) for path in sorted((OUTPUT/'tmp').rglob('child_receipt.json'))]
    # pytest's "current" convenience symlinks may alias existing directories.
    receipts = list({(row['pid'],row['pgid']):row for row in receipts}.values())
    assert len(receipts) == 42
    for row in receipts:
        assert row['reaped'] is True and row['group_absent'] is True
        try: os.killpg(row['pgid'],0)
        except ProcessLookupError: pass
        else: raise AssertionError('CLI probe group remains')
    after = inventory(SOURCE)
    executed_after = inventory(execution)
    passed = returncode == 0 and absent and len(cases) == 43 and bad == 0 and before == after == executed_after
    report = dict(schema='qwen-candidate-terminal-focused-successor-v1',passed=passed,
        scope='Focused real-CLI and exact terminal-helper synthetic regression only; not a full candidate preflight',
        source_count=141,source_hashes_start=before,source_hashes_end=after,execution_source_hashes_end=executed_after,
        source_manifest_sha256=sha((OUTPUT/'source_manifest.json').read_bytes()),
        source_archive_sha256=sha((OUTPUT/'sources.tar.gz').read_bytes()),
        changed_parent_files=['entry.py'],unchanged_parent_files=138,numerical_six_unchanged=True,
        tests=len(cases),xml_bad=bad,child_returncode=returncode,child_pid=child.pid,child_pgid=child.pid,
        child_reaped=True,child_group_absent=absent,cli_receipts=receipts,
        actual_candidate_outcome_read=False,actual_private_body_read=False,actual_model_execution=False,
        new_execution_grant=False,requires_independent_outer_terminal=True)
    (OUTPUT/'report.json').write_bytes(encoded(report))
    print(json.dumps({key:report[key] for key in ('passed','source_count','tests','xml_bad','child_returncode','child_pid','child_group_absent','source_manifest_sha256','source_archive_sha256')}),flush=True)
    return 0 if passed else 1

if __name__ == '__main__':
    raise SystemExit(main())
