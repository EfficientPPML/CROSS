"""Own one isolated public/synthetic test child; preserve exact source closure."""
from pathlib import Path
import hashlib, json, os, signal, subprocess, sys, tarfile, time
import xml.etree.ElementTree as ET

source = Path('/tmp/qwen-client-stage-glue-v2-20261104')
out = Path('/tmp/qwen-client-stage-boundary-proof-20261104/development-v1')
execution = out / 'execution'
manifest_sha = '7837d7c4d2a3d23ca77c7d3ff58052b45b4cda309f5cb6c234afb08fa5772f1b'
sha = lambda b: hashlib.sha256(b).hexdigest()
def dump(name, value):
    (out / name).write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')
def inventory(root):
    assert not any(p.is_symlink() for p in root.rglob('*'))
    return {str(p.relative_to(root)): sha(p.read_bytes())
            for p in sorted(root.rglob('*')) if p.is_file()}
assert sha((out / 'sources-before.json').read_bytes()) == manifest_sha
expected = json.loads((out / 'sources-before.json').read_bytes())
assert len(expected) == 373 and inventory(source) == inventory(execution) == expected
with tarfile.open(out / 'sources.tar.gz') as archive:
    members = archive.getmembers()
    assert len(members) == 373 and all(m.isfile() for m in members)
    archived = {m.name: sha(archive.extractfile(m).read()) for m in members}
assert archived == expected
assert Path(__file__).read_bytes() == (out / 'executed_driver.py').read_bytes()
scope = ('Exactly nineteen new client boundary cases. Real owned synthetic files at '
         'Qwen physical dimensions; recording key maps, context, security and signal '
         'scopes. No inherited suites or actual backend/model/key/private/ciphertext '
         'bodies, context construction or role execution.')
env = dict(PATH='/usr/bin:/bin', HOME='/home/jianming_gatech', LANG='C.UTF-8', TZ='UTC',
           PYTHONDONTWRITEBYTECODE='1', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
           OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
command = ['taskset', '-c', '108-111', '/tmp/cross-pretrained-validation-env/bin/python',
           '-I', '-B', str(out / 'worker_driver.py'), str(execution), str(out)]
start = time.monotonic()
with (out / 'tests.log').open('xb') as log:
    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True)
    dump('live-observation.json', dict(outer_pid=os.getpid(), outer_pgid=os.getpgrp(),
         worker_pid=child.pid, worker_pgid=os.getpgid(child.pid), command=command,
         environment=env, source_manifest_sha256=manifest_sha))
    try:
        code = child.wait()
    except BaseException:
        try: os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError: pass
        finally: child.wait()
        raise
try: os.killpg(child.pid, 0)
except ProcessLookupError: group_absent = True
else: group_absent = False
current, after = inventory(source), inventory(execution)
dump('sources-after.json', current)
dump('execution-after.json', after)
with tarfile.open(out / 'sources.tar.gz') as archive:
    archived = {m.name: sha(archive.extractfile(m).read()) for m in archive.getmembers()}
xml = ET.parse(out / 'tests.xml').getroot()
counts = {name: len(list(xml.iter(tag))) for name, tag in
          [('tests', 'testcase'), ('failures', 'failure'), ('errors', 'error'), ('skipped', 'skipped')]}
nodes = ['test_client_stage.py::' + case.attrib['name'] for case in xml.iter('testcase')]
expected_nodes = json.loads((execution / 'client_test_origin.json').read_bytes())['selected_nodes']
audit = json.loads((out / 'worker-audit.json').read_bytes())
passed = (code == 0 and group_absent and current == after == archived == expected
          and counts == dict(tests=19, failures=0, errors=0, skipped=0)
          and nodes == expected_nodes and audit['denied'] == audit['loaded_backend'] == [])
report = dict(passed=passed, tests=counts, selected_nodes=nodes,
    source_count=373, source_manifest_sha256=manifest_sha,
    source_archive_sha256=sha((out / 'sources.tar.gz').read_bytes()),
    sources_unchanged=current == after == archived == expected,
    child_returncode=code, child_reaped=True, child_group_absent=group_absent,
    elapsed_s=time.monotonic()-start, synthetic_only=True, model_execution=False,
    encrypted_execution=False, actual_role_execution=False, scope=scope,
    driver_sha256=sha(Path(__file__).read_bytes()),
    worker_driver_sha256=sha((out / 'worker_driver.py').read_bytes()))
dump('report.json', report)
print(json.dumps(report, sort_keys=True), flush=True)
raise SystemExit(0 if passed else 1)
