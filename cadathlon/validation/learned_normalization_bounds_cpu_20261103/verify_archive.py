"""Offline source and exact rational endpoint checks; no model imports."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import tarfile

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT/'result-v2/report.json').read_text())
verification = json.loads((ROOT/'verification.json').read_text())
assert report['passed'] and report['no_jax_imported']
assert report['counts'] == {'bert': 25, 'qwen': 113}
assert report['source_hashes_before'] == report['source_hashes_after']
for name, expected in verification['artifact_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected
archive_path = ROOT/'result-v2/sources.tar.gz'
assert hashlib.sha256(archive_path.read_bytes()).hexdigest() == report['source_archive_sha256']
with tarfile.open(archive_path) as archive:
    members = archive.getmembers()
    assert len(members) == 3
    assert {'/'+m.name for m in members} == set(report['source_hashes_before'])
    for member in members:
        assert member.isfile() and not Path(member.name).is_absolute()
        assert '..' not in Path(member.name).parts
        actual = hashlib.sha256(archive.extractfile(member).read()).hexdigest()
        assert actual == report['source_hashes_before']['/'+member.name]
for row in report['normalizations'].values():
    exact = Fraction(*map(int, row['upper_fraction']))
    assert Fraction.from_float(row['exact_real_abs_output_upper']) >= exact
    power = row['power_of_two_cover']
    assert power & (power-1) == 0
    assert Fraction(power//2) < exact <= power
    assert Fraction.from_float(row['sqrt_upper'])**2 >= row['width']-int(row['centered'])
assert verification['terminal_exit_code'] == 0 and verification['authoritatively_reaped']
print(json.dumps({'passed': True, 'exact_sources': 3, 'normalizations': 138,
                  'encrypted_execution': False}))
