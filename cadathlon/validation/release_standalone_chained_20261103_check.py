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
                 'agent_pset/p2_bert/src/p2_runtime.py', 'common/encrypted_ntt.py',
                 'agent_pset/p2_bert/src/bert_optimized_reference.py',
                 'agent_pset/p4_qwen/src/qwen_optimized_reference.py', 'common/encrypted_streaming_linear.py', 'common/encrypted_chained_streaming_linear.py'):
        assert name in names, name
    streaming = extracted / 'common/encrypted_streaming_linear.py'
    report['streaming_linear_sha256'] = hashlib.sha256(streaming.read_bytes()).hexdigest()
    assert report['streaming_linear_sha256'] == '83e12b9cf00f1393a47aadddffaaa1ff6b4d0745cb68d9393d2dd38a2a165ecb'
    chained_promotion = json.loads((root / 'cadathlon/validation/standalone_chained_linear_promotion_20261103/promotion.json').read_text())
    chained_hash = hashlib.sha256((extracted / 'common/encrypted_chained_streaming_linear.py').read_bytes()).hexdigest()
    assert chained_hash == chained_promotion['source_sha256']['cadathlon/common/encrypted_chained_streaming_linear.py']
    report['chained_linear_sha256'] = chained_hash
    proposed = json.loads((root / 'cadathlon/validation/integer_newton_promotion_cpu_20261103/candidate_files.json').read_text())
    report['integer_helper_sha256'] = {}
    for module in ('encrypted_integer_affine', 'encrypted_scaled_newton'):
        helper = extracted / 'common' / (module + '.py')
        helper_hash = hashlib.sha256(helper.read_bytes()).hexdigest()
        assert helper_hash == proposed['source_sha256']['cadathlon/common/' + module + '.py']
        report['integer_helper_sha256'][module] = helper_hash
    smoke_code = ("import pathlib,sys;sys.path.insert(0,sys.argv[1]);"
                  "import encrypted_integer_affine as a,encrypted_scaled_newton as b,encrypted_chained_streaming_linear as c;"
                  "assert all(pathlib.Path(m.__file__).resolve().parent==pathlib.Path(sys.argv[1]).resolve() for m in (a,b,c));"
                  "assert c.ChainedStreamingLinear.__name__=='ChainedStreamingLinear';assert 'jax' not in sys.modules;print('release-local integer and chained helpers imported without JAX')")
    smoke_env = dict(os.environ, JAX_PLATFORMS='release_import_must_not_initialize_jax')
    smoke = subprocess.run(['/home/jianming_gatech/miniconda3/envs/jaxite/bin/python','-c',smoke_code,str(extracted/'common')],
                           cwd=extracted,env=smoke_env,text=True,capture_output=True,check=False)
    report['public_helper_import'] = dict(returncode=smoke.returncode,stdout=smoke.stdout,stderr=smoke.stderr)
    assert smoke.returncode == 0, smoke.stderr
    calibration = json.loads((root / 'cadathlon/validation/streaming_linear_qkv_cpu_20261103/closure_followup/calibration_source_check.json').read_text())
    for data in calibration['calibrations'].values():
        for name, record in data['sources'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == record['expected']
    report['p1_p3_measured_sources_unchanged'] = True
    suite = extracted / 'non_agent_pset/p1_benchmark/src/suite'
    baseline = json.loads((suite / 'baseline_manifest_student.json').read_text())
    assert baseline['gate_version'] == 'student-v2'
    proof_path = suite / baseline['evaluation_report']
    assert proof_path.resolve().is_relative_to(suite.resolve()) and proof_path.is_file()
    proof = json.loads(proof_path.read_text())
    assert proof['passed'] and proof['evaluation_id'] == baseline['evaluation_id']
    public_p1 = json.loads((extracted / 'validation/p1_student_suite.json').read_text())
    assert all(public_p1.get(key) == value for key, value in proof.items())
    assert public_p1['source_files_unchanged'] and public_p1['cleanup_passed']
    fresh_p1 = json.loads((root / 'cadathlon/validation/p1_student_post_promotion_20261103/collection.json').read_text())
    assert fresh_p1['passed'] and fresh_p1['source_files_unchanged']
    assert fresh_p1['returncode'] == 0 and fresh_p1['external_evaluation_elapsed_s'] <= 300
    report['p1_external_elapsed_s'] = fresh_p1['external_evaluation_elapsed_s']
    report['p1_student_v2_evaluation_id'] = baseline['evaluation_id']
    report['p1_published_proof_sha256'] = hashlib.sha256(proof_path.read_bytes()).hexdigest()
    report['p1_immutable_proof_files'] = sum(p.is_file() for p in proof_path.parent.rglob('*'))
    compact = json.loads((extracted / 'validation/p3_compact_baseline.json').read_text())
    assert compact['passed'] and compact['within_budget'] and compact['elapsed_s'] <= 300
    fresh_p3 = json.loads((root / 'cadathlon/validation/p3_compact_post_promotion_20261103/process.json').read_text())
    assert fresh_p3['passed'] and fresh_p3['source_files_unchanged']
    assert compact == fresh_p3['result']
    assert fresh_p3['returncode'] == 0 and fresh_p3['evaluator_process_elapsed_s'] <= 300
    report['p3_external_elapsed_s'] = fresh_p3['evaluator_process_elapsed_s']
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
report['checker'] = str(Path(__file__).relative_to(root))
report['checker_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(root/'cadathlon/validation/release_standalone_chained_20261103.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({key: value for key, value in report.items() if key != 'checks'}, indent=2))
