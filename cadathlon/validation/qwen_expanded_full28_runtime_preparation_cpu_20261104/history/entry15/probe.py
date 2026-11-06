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
        ('private_read_denied', True, True, False),
        ('subprocess_denied', True, True, False),
        ('numpy_pytest_control', True, True, False),
        ('preloaded_canonical', True, True, False),
        ('preloaded_spec_alias', True, True, False),
        ('preloaded_foreign_file_local_spec', True, True, False),
        ('undeclared_namespace', True, True, False),
        ('undeclared_pyc', True, True, False),
        ('undeclared_native', True, True, False),
        ('symlink_directory', True, True, False),
        ('existing_output', True, True, False),
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
        if name == 'private_read_denied':
            forbidden = case/'synthetic-private-like.json'
            forbidden.write_text('{"synthetic": true}\n')
            helper = ('from pathlib import Path\nPath('+repr(str(forbidden))+').read_bytes()\n').encode()+helper
        elif name == 'subprocess_denied':
            helper = b'import subprocess,sys\nsubprocess.run([sys.executable,"-c","pass"],check=True)\n'+helper
        elif name == 'numpy_pytest_control':
            helper = b'import numpy,pytest\n'+helper
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
        for source in (original,execution):
            if name == 'undeclared_namespace': (source/'unknown_namespace').mkdir()
            elif name == 'undeclared_pyc': (source/'shadow.pyc').write_bytes(b'public synthetic non-bytecode')
            elif name == 'undeclared_native': (source/'shadow.so').write_bytes(b'public synthetic non-library')
            elif name == 'symlink_directory': (source/'alias').symlink_to(case,target_is_directory=True)
        if name == 'existing_output':
            out.mkdir();(out/'report.json').write_text('preexisting public sentinel\n')
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
        if case['name'].startswith('preloaded_'):
            entry=str(case['execution']/'prepare_expanded.py')
            alias='derive_expanded' if case['name']=='preloaded_canonical' else 'unknown_local_alias'
            shim='import sys,types,runpy; m=types.ModuleType('+repr(alias)+'); ' 
            if case['name']!='preloaded_canonical':
                shim+='m.__spec__=types.SimpleNamespace(origin='+repr(str(case['execution']/'derive_expanded.py'))+'); '
            if case['name']=='preloaded_foreign_file_local_spec':shim+='m.__file__="/usr/lib/foreign.py"; '
            shim+='sys.modules['+repr(alias)+']=m; sys.argv=['+repr(entry)+']+sys.argv[1:]; runpy.run_path('+repr(entry)+',run_name="__main__")'
            index=command.index(entry);command[index:index+1]=['-c',shim]
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
        expected_marker = case['name'] in ('declared_helper_control','numpy_pytest_control')
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
