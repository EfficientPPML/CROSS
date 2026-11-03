"""Public identity checks shared by P1 evaluation, archiving and scoring."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def model_identity(problem):
    folder = Path(problem) / 'weights'
    manifest = folder / 'MANIFEST.json'
    weights = folder / 'mlp_784_50_10.pth'
    declared = json.loads(manifest.read_text())['files'][weights.name]
    actual = sha256(weights)
    if actual != declared['sha256'] or weights.stat().st_size != declared['bytes']:
        raise ValueError('P1 ranked weights differ from weights/MANIFEST.json; restore the frozen asset')
    return {'weights_sha256': actual, 'manifest_sha256': sha256(manifest)}


def case_identity(problem, suite_root, instance_name):
    folder = Path(suite_root) / 'datasets' / instance_name / 'intermediate'
    return {'model': model_identity(problem),
            'pixels_sha256': sha256(folder / 'test_pixels.txt'),
            'labels_sha256': sha256(folder / 'test_labels.txt')}


def archive_hashes(folder, runs):
    return {name: sha256(Path(folder) / name)
            for name in ['case.json', *(f'results-{i}.json' for i in range(1, runs+1))]}
