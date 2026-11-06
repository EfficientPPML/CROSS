from pathlib import Path
import ast
import hashlib
import json
import os
import re
import subprocess
import sys

source = Path('/tmp/qwen-softmax-cohort-full-owner-integration-v2-20261104')
evidence = Path('/tmp/qwen-cohort-full-owner-combined-preflight-20261104/collection-v1')
evidence.mkdir(parents=True)
def digest(raw):
    return hashlib.sha256(raw).hexdigest()
def inventory():
    return {p.relative_to(source).as_posix(): digest(p.read_bytes()) for p in source.rglob('*') if p.is_file()}
before = inventory()
assert len(before) == 208
names = json.loads((source / 'selected_preparation.json').read_bytes())['test_files']
tree = ast.parse((source / 'prepare_expanded.py').read_bytes())
assignment = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'TESTS' for t in n.targets))
assert tuple(names) == ast.literal_eval(assignment.value)
command = [sys.executable, '-I', '-B', '-m', 'pytest', '--collect-only', '-q', '--noconftest', '-p', 'no:cacheprovider', '--rootdir=' + str(source), '--confcutdir=' + str(source), *names]
with (evidence / 'stdout.log').open('wb') as stdout, (evidence / 'stderr.log').open('wb') as stderr:
    child = subprocess.Popen(command, cwd=source, stdout=stdout, stderr=stderr, start_new_session=True)
    pid, pgid = child.pid, os.getpgid(child.pid)
    code = child.wait()
text = (evidence / 'stdout.log').read_text()
match = re.search(r'(\d+) tests collected', text)
count = int(match.group(1)) if match else None
after = inventory()
assert before == after
try:
    os.killpg(pgid, 0)
except ProcessLookupError:
    absent = True
else:
    absent = False
report = dict(schema='qwen-cohort-combined-collection-v1', passed=code == 0 and count == 330 and absent,
              tests_collected=count, tests_executed=0, child_returncode=code, child_reaped=True,
              child_pid=pid, child_pgid=pgid, child_group_absent=absent, source_count=208,
              source_start=before, source_end=after, command=command, driver_sha256=digest(Path(__file__).read_bytes()))
(evidence / 'report.json').write_text(json.dumps(report, sort_keys=True, indent=2) + '\n')
print({key: report[key] for key in ('passed', 'tests_collected', 'tests_executed', 'child_returncode', 'child_pid', 'child_pgid', 'child_group_absent')})
raise SystemExit(0 if report['passed'] else 1)
