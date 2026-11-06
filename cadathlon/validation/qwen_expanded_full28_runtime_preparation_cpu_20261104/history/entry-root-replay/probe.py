"""Synthetic CLI admission canaries only; no model/HE or scientific tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile

sha = lambda b: hashlib.sha256(b).hexdigest()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--driver', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.absolute()
    root.mkdir(exist_ok=False)
    driver = args.driver.read_bytes()
    (root/'probe.py').write_bytes(Path(__file__).read_bytes())
    (root/'driver.py.txt').write_bytes(driver)
    cases = []
    for name, declared, isolated, alias in [
        ('undeclared_helper', False, True, False),
        ('nonisolated_startup', True, False, False),
        ('output_alias_into_source', False, True, True),
        ('declared_helper_control', True, True, False),
    ]:
        case = root/name
        case.mkdir()
        original, execution = case/'original', case/'execution'
        original.mkdir(); execution.mkdir()
        out = case/'owned-output'
        if alias:
            (case/'source-alias').symlink_to(original, target_is_directory=True)
            out = case/'source-alias'/'unexpected-output'
        marker = out/'tmp'/'declared-or-unbound-helper-executed.txt'
        helper = ('from pathlib import Path\n'
                  + 'Path('+repr(str(marker))+').write_text("synthetic canary only\\n")\n'
                  + 'raise RuntimeError("canary stops before derivation or pytest")\n').encode()
        expected = {'prepare_expanded.py': sha(driver)}
        helper_bytes = {n: (args.driver.parent/n).read_bytes() for n in ('freeze_io.py','cleanup_origin.py')}
        expected.update({n: sha(b) for n,b in helper_bytes.items()})
        for source in (original, execution):
            (source/'prepare_expanded.py').write_bytes(driver)
            for n,b in helper_bytes.items(): (source/n).write_bytes(b)
            if not alias:
                (source/'derive_expanded.py').write_bytes(helper)
        if declared:
            expected['derive_expanded.py'] = sha(helper)
        manifest = case/'manifest.json'
        manifest.write_text(json.dumps(expected, sort_keys=True, indent=2)+'\n')
        cases.append(dict(name=name, root=case, original=original, execution=execution,
                          output=out, marker=marker, manifest=manifest,
                          isolated=isolated, declared=declared, alias=alias))
    source_files = [p for p in root.rglob('*') if p.is_file() and not p.is_symlink()]
    before = {str(p.relative_to(root)): sha(p.read_bytes()) for p in source_files}
    (root/'sources-before.json').write_text(json.dumps(before, sort_keys=True, indent=2)+'\n')
    with tarfile.open(root/'sources.tar.gz', 'w:gz') as archive:
        for p in source_files:
            archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
    rows = []
    for case in cases:
        command = ['taskset', '-c', '112-115', '/tmp/cross-pretrained-validation-env/bin/python']
        if case['isolated']:
            command.append('-I')
        command += ['-B', str(case['execution']/'prepare_expanded.py'),
                    '--source-manifest', str(case['manifest']), '--manifest-sha256', sha(case['manifest'].read_bytes()),
                    '--original-root', str(case['original']), '--output', str(case['output'])]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        stdout, stderr = process.communicate()
        (case['root']/'stdout.log').write_bytes(stdout)
        (case['root']/'stderr.log').write_bytes(stderr)
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            group_absent = True
        else:
            group_absent = False
        marker_exists = case['marker'].exists()
        alias_output_exists = (case['original']/'unexpected-output').exists()
        expected_marker = case['name'] == 'declared_helper_control'
        passed = process.returncode == 70 and marker_exists == expected_marker and not alias_output_exists and group_absent
        rows.append(dict(name=case['name'], passed=passed, returncode=process.returncode,
                         pid_pgid=process.pid, reaped=True, process_group_absent=group_absent,
                         helper_marker_present=marker_exists, output_created_inside_source=alias_output_exists,
                         command=command, scope='synthetic CLI admission only; helper deliberately raises before derivation/tests'))
    after = {n: sha((root/n).read_bytes()) for n in before}
    assert before == after
    (root/'sources-after.json').write_text(json.dumps(after, sort_keys=True, indent=2)+'\n')
    report = dict(scope='Synthetic entry admission canaries. No real source tree, private input, model, backend, HE or scientific test accessed.',
                  driver_sha256=sha(driver), probe_sha256=sha(Path(__file__).read_bytes()), source_count=len(before),
                  declared_source_bytes_unchanged=True, cases=rows, passed=all(r['passed'] for r in rows))
    (root/'report.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
    print(json.dumps(report, sort_keys=True))
    return 0 if report['passed'] else 1
if __name__ == '__main__':
    raise SystemExit(main())
