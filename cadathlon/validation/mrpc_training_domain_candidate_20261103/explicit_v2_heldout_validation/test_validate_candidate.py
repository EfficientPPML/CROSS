from pathlib import Path
import tempfile,json,unittest,hashlib,tarfile,io
from validate_candidate import require_training_coverage

class Prerequisite(unittest.TestCase):
    def test_requires_complete_same_candidate_training_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';manifest.write_text('{}')
            report=root/'report.json';valid=dict(completed_examples=3668,complete_model_outputs=3668,source_archive_verified=True,validation_evaluated=False,all_training_examples_evaluated=True,
                approximate_trajectory_coverage=True,source_files_unchanged=True,no_jax_imported=True,
                labels_used=False,training_labels_loaded=False,domain_failures=[],
                candidate_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),passed=False)
            raw=manifest.read_bytes();digest=hashlib.sha256(raw).hexdigest()
            with tarfile.open(root/'sources.tar.gz','w:gz') as archive:
                info=tarfile.TarInfo(str(manifest).lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
            valid.update(source_hashes_before={str(manifest):digest},source_hashes_after={str(manifest):digest},
                source_archive_sha256=hashlib.sha256((root/'sources.tar.gz').read_bytes()).hexdigest())
            report.write_text(json.dumps(valid))
            self.assertEqual(require_training_coverage(report,manifest),valid)
            for key,value in [('completed_examples',3),('complete_model_outputs',3),('source_archive_verified',False),('validation_evaluated',True),('all_training_examples_evaluated',False),('approximate_trajectory_coverage',False),
                    ('source_files_unchanged',False),('no_jax_imported',False),('labels_used',True),('training_labels_loaded',True),
                    ('domain_failures',[{'failure':True}]),('candidate_manifest_sha256','0'*64),('source_hashes_after',{}),('source_archive_sha256','0'*64)]:
                report.write_text(json.dumps(dict(valid,**{key:value})))
                with self.subTest(key=key),self.assertRaises(ValueError):require_training_coverage(report,manifest)

    def test_rejects_changed_dependency_and_archive_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);manifest=root/'manifest.json';manifest.write_text('{}')
            dependency=root/'arithmetic.py';dependency.write_text('old')
            before={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (manifest,dependency)}
            def archive_source():
                with tarfile.open(root/'sources.tar.gz','w:gz') as archive:
                    for name in before:
                        raw=Path(name).read_bytes();info=tarfile.TarInfo(name.lstrip('/'));info.size=len(raw);archive.addfile(info,io.BytesIO(raw))
            archive_source()
            valid=dict(completed_examples=3668,complete_model_outputs=3668,source_archive_verified=True,validation_evaluated=False,
                all_training_examples_evaluated=True,approximate_trajectory_coverage=True,source_files_unchanged=True,
                no_jax_imported=True,labels_used=False,training_labels_loaded=False,domain_failures=[],
                candidate_manifest_sha256=before[str(manifest)],source_hashes_before=before,source_hashes_after=before,
                source_archive_sha256=hashlib.sha256((root/'sources.tar.gz').read_bytes()).hexdigest())
            report=root/'report.json';report.write_text(json.dumps(valid));require_training_coverage(report,manifest)
            dependency.write_text('changed')
            with self.assertRaisesRegex(ValueError,'current arithmetic'):require_training_coverage(report,manifest)
            archive_source();valid['source_archive_sha256']=hashlib.sha256((root/'sources.tar.gz').read_bytes()).hexdigest();report.write_text(json.dumps(valid))
            with self.assertRaisesRegex(ValueError,'archive bytes'):require_training_coverage(report,manifest)

if __name__=='__main__':unittest.main()
