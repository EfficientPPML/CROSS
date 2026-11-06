"""Archive lazy BConv integration, its serialization regression and fixed rerun."""
from pathlib import Path
import difflib
import hashlib
import json
import re
import shutil
import subprocess
import tarfile
import tempfile

LIVE=Path('/home/jianming_gatech/CROSS')
OLD=Path('/tmp/lazy-bconv-20261103')
LAB=Path('/tmp/lazy-bconv-v2-20261103')
OUT=LIVE/'cadathlon/validation/lazy_bconv_cpu_20261103'

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_run(folder):
    report=json.loads((folder/'report.json').read_text())
    before=json.loads((folder/'source-before.json').read_text())
    after=json.loads((folder/'source-after.json').read_text())
    assert report['passed'] and report['returncode']==0 and before==after
    with tarfile.open(folder/'sources.tar.gz','r:gz') as archive:
        members=archive.getmembers()
        assert len(members)==len(before) and {m.name for m in members}==set(before)
        for member in members:
            assert member.isfile()
            assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==before[member.name]
    result=re.search(r'(\d+) passed, (\d+) skipped, (\d+) subtests passed in ([\d.]+)s',
                     (folder/'pytest.log').read_text())
    assert result
    return dict(passed=int(result.group(1)),skipped=int(result.group(2)),
                subtests=int(result.group(3)),pytest_seconds=float(result.group(4)),
                external_seconds=report['parent_elapsed_s'],recorded_sources=len(before),
                exact_source_archive_verified=True,recorded_sources_unchanged=True)

initial=verify_run(OLD/'evidence')
final=verify_run(LAB/'evidence')
negative=json.loads((OLD/'pickle_repro.json').read_text())
fixed=json.loads((LAB/'pickle_repro.json').read_text())
assert not negative['passed'] and negative['sources_unchanged']
assert negative['cases']['eager']['passed'] and not negative['cases']['lazy']['passed']
assert fixed['passed'] and fixed['sources_unchanged']
records=json.loads((LAB/'evidence/secure-records.json').read_text())['test_lazy_bconv']
assert len(records)==1 and records[0]['payload_and_metadata_exact']
assert records[0]['lazy_bconv_builder_forbidden'] and records[0]['keyless_server']
assert records[0]['max_abs_error']<records[0]['fixed_max_abs_error_gate']
device=json.loads((LAB/'second_device/report.json').read_text())
assert device['passed'] and device['returncode']==0 and device['sources_unchanged']
assert device['source_hashes_before']==device['source_hashes_after']
with tarfile.open(LAB/'second_device/sources.tar.gz','r:gz') as archive:
    members=archive.getmembers();expected=device['source_hashes_before']
    assert len(members)==len(expected) and {m.name for m in members}==set(expected)
    for member in members:
        assert member.isfile()
        assert hashlib.sha256(archive.extractfile(member).read()).hexdigest()==expected[member.name]

OUT.mkdir(exist_ok=False)
for source,name in [(OLD/'evidence','v1_incomplete'),(LAB/'evidence','v2'),
                    (LAB/'second_device','second_device')]:
    shutil.copytree(source,OUT/name)
for source,name in [(OLD/'pickle_observation.json','v1_incomplete/pickle_observation.json'),
                    (OLD/'pickle_repro.json','v1_incomplete/pickle_repro.json'),
                    (OLD/'pickle_repro.log','v1_incomplete/pickle_repro.log'),
                    (OLD/'initial_tests.log','v1_incomplete/development_public_tests.log'),
                    (OLD/'secure_tests.log','v1_incomplete/development_secure_tests.log'),
                    (LAB/'pickle_repro.json','v2/pickle_repro.json'),
                    (LAB/'pickle_repro.log','v2/pickle_repro.log'),
                    (LAB/'second_device.log','second_device/pytest.log'),
                    (LAB/'builder_equivalence.json','builder_equivalence.json'),
                    (LAB/'check_builder_equivalence.py','check_builder_equivalence.py'),
                    (Path('/tmp/check_deferred_bconv_pickle_20261103.py'),'check_pickle.py'),
                    (Path(__file__),'verify_and_archive.py'),
                    (LAB/'checkout/cadathlon/validation/check_lazy_bconv_20261103.py','run_checkpoint.py'),
                    (LAB/'checkout/cadathlon/validation/lazy_bconv_pytest_child.py','run_pytest_child.py'),
                    (LAB/'second_device_driver.py','second_device_driver.py')]:
    shutil.copy2(source,OUT/name)
base_patch=LIVE/'cadathlon/validation/all_candidates_cpu_20261103/combined.patch'
shutil.copy2(base_patch,OUT/'prerequisite_five_candidates.patch')
changes={};patch=[]
for rel in ('jaxite_word/he_params.py','cadathlon/tests/test_lazy_bconv.py'):
    before=LAB/'before'/rel;after=LAB/'checkout'/rel
    a=before.read_text() if before.exists() else ''
    b=after.read_text()
    patch.extend(difflib.unified_diff(a.splitlines(True),b.splitlines(True),
                 fromfile='a/'+rel if before.exists() else '/dev/null',tofile='b/'+rel))
    changes[rel]={'before_sha256':digest(before) if before.exists() else None,'after_sha256':digest(after)}
    if before.exists():
        target=OUT/'before/sources'/rel;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(before,target)
(OUT/'candidate.patch').write_text(''.join(patch))
with tempfile.TemporaryDirectory(prefix='lazy-bconv-patch-check-') as temporary:
    root=Path(temporary)
    shutil.copytree(OUT/'before/sources',root,dirs_exist_ok=True)
    subprocess.run(['git','apply','--check',str(OUT/'candidate.patch')],cwd=root,check=True)
    subprocess.run(['git','apply',str(OUT/'candidate.patch')],cwd=root,check=True)
    assert all(digest(root/name)==entry['after_sha256'] for name,entry in changes.items())
(OUT/'changed_sources.json').write_text(json.dumps(changes,indent=2)+'\n')
summary=dict(scope='Isolated lazy public BConv controls; not applied to live runtime.',
    passed=True,v1_broad_tests=initial,v1_known_incomplete='Unmaterialized LevelParams could not be pickled; this green broad run lacked that test.',
    v2_broad_tests=final,serialization_repro_fixed=True,second_device_check=device,
    secure_check=records[0],builder_arithmetic_ast_unchanged=True,
    candidate_patch_sha256=digest(OUT/'candidate.patch'),prerequisite_sha256=digest(base_patch),
    patch_reconstruction_verified=True,full_model_evaluated=False,accelerator_evaluated=False)
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(OUT/'README.md').write_text(f'''The fixed isolated candidate passed **{final['passed']} tests and {final['subtests']} subtests**,
with **{final['skipped']} skips**, in {final['pytest_seconds']:.2f} seconds.
All {final['recorded_sources']} recorded source and fixture files stayed unchanged;
every member of the exact source archive was independently verified. The
second-device delayed-read check also passed in a fresh process.

Ordinary finite-field, NTT and psi lookups no longer build unused cached BConv
controls in deterministic lazy mode. Direct `LevelParams.bconv_params` reads,
the public getter, dataclass helpers and serialization materialize the same
controls on demand. The stored specification contains only immutable public
metadata and the construction device, so retained parameters can outlive their
key-owning cache. Eager and synthetic performance-test generation order stays
unchanged. An AST comparison confirms unchanged control arithmetic/partitions.

A genuine secure N8192 batch-two affine-then-square circuit exactly matched
the eager ciphertext payloads and metadata while cached BConv construction,
secret-dependent key generation and server encryption/decryption were forbidden.
All 8,192 client output slots were checked; maximum error was
{records[0]['max_abs_error']:.8g} against the unchanged 2e-6 gate. This covers
that component only; it adds no complete model, refresh or TPU timing claim.

The initial candidate passed {initial['passed']} broad tests but failed a separate
pickle reproducer: its deferred JAX Device was not serializable. That version
is retained under `v1_incomplete` and is not promotion-ready. The fixed version
materializes public controls before serialization; protocol 4/5, shallow/deep
copy, direct reads, manual replacement, retry, cache eviction, both reduction
backends and retained-owner lifetime are covered. The original failure and
its fixed rerun remain separate, with exact source hashes.

`candidate.patch` is a two-file follow-up after the archived five-candidate
integration prerequisite. It reconstructs the tested files from archived base
bytes and remains unapplied to the live checkout. No running proof or published
calibration was changed, and no model speedup is inferred from avoiding this work.
''')
(OUT/'artifact_manifest.json').write_text(json.dumps({str(p.relative_to(OUT)):digest(p)
    for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json'},indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='second_device_check'},indent=2))
