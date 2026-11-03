import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import time
from urllib.parse import unquote, urlsplit

root = Path('/home/jianming_gatech/CROSS')
archive = root/'cadathlon/contestant-release.tar.gz'
report = dict(scope='Local release packaging and plaintext checks; no encrypted model validation or publication.',
              archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), checks=[])


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.hrefs = []
    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.hrefs.extend(value for key, value in attrs if key == 'href')


started = time.perf_counter()
with tempfile.TemporaryDirectory(prefix='cross-release-check-') as directory:
    extracted = Path(directory)
    with tarfile.open(archive) as source:
        names = source.getnames()
        assert len(names) == len(set(names))
        assert all(not Path(name).is_absolute() and '..' not in Path(name).parts for name in names)
        assert all(not name.endswith(('.golden', '.pyc')) and '__pycache__' not in name for name in names)
        assert all(not any(part.startswith('.') for part in Path(name).parts) for name in names)
        source.extractall(extracted, filter='data')
    report['files'] = len(names)
    for name in ('common/encrypted_bounded_arithmetic.py', 'common/encrypted_bounded_transformer.py',
                 'agent_pset/p2_bert/src/p2_runtime.py'):
        assert name in names, name
    suite = extracted / 'non_agent_pset/p1_benchmark/src/suite'
    baseline = json.loads((suite / 'baseline_manifest_student.json').read_text())
    assert baseline['gate_version'] == 'student-v2'
    proof_path = suite / baseline['evaluation_report']
    assert proof_path.resolve().is_relative_to(suite.resolve()) and proof_path.is_file()
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and proof['evaluation_id'] == baseline['evaluation_id']
    report['p1_student_v2_evaluation_id'] = baseline['evaluation_id']
    report['p1_published_proof_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    report['p1_immutable_proof_files'] = sum(p.is_file() for p in proof_path.parent.rglob('*'))
    compact = json.loads((extracted / 'validation/p3_compact_baseline.json').read_text())
    assert compact['passed'] and compact['within_budget'] and compact['elapsed_s'] <= 300
    report['p3_compact_internal_elapsed_s'] = compact['elapsed_s']
    report['source_snapshot_matches_checkout'] = all(
        (root/'cadathlon'/name).read_bytes() == (extracted/name).read_bytes() for name in names)
    local_links = []
    for path in extracted.rglob('*.html'):
        parser = Links()
        parser.feed(path.read_text())
        for href in parser.hrefs:
            parsed = urlsplit(href)
            if not parsed.scheme and not parsed.netloc and parsed.path:
                destination = (path.parent/unquote(parsed.path)).resolve()
                assert destination.is_relative_to(extracted.resolve()) and destination.is_file(), (path, href)
                local_links.append((str(path.relative_to(extracted)), href))
    report['resolved_local_html_links'] = len(local_links)
    env = dict(os.environ, JAX_PLATFORMS='cpu', CROSS_ROOT=str(root), OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
               PYTHON='/home/jianming_gatech/miniconda3/envs/jaxite/bin/python')
    for folder in ('non_agent_pset/p1_benchmark', 'agent_pset/p2_bert',
                   'non_agent_pset/p3_cnn', 'agent_pset/p4_qwen'):
        command = ['make', '-C', str(extracted/folder/'src'), 'all', 'reference-test']
        result = subprocess.run(command, env=env, text=True, capture_output=True)
        report['checks'].append(dict(problem=folder, returncode=result.returncode,
                                    stdout=result.stdout, stderr=result.stderr))
        print(folder, result.returncode, flush=True)
        assert result.returncode == 0, result.stderr
    report['passed'] = report['source_snapshot_matches_checkout'] and all(check['returncode'] == 0 for check in report['checks'])
report['elapsed_s'] = time.perf_counter()-started
(root/'cadathlon/validation/release_integration_20261103.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: value for key, value in report.items() if key != 'checks'}, indent=2))
