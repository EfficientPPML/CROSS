"""Offline verification of ideal-only FFN composition evidence."""
from pathlib import Path
import hashlib,json,tarfile
HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    artifacts=json.loads((HERE/'artifact_sha256.json').read_text())
    assert {n:sha(HERE/n) for n in artifacts}==artifacts
    for name in ('source-bound-final-v1','development-full-v1'):
        p=HERE/name;r=json.loads((p/'report.json').read_text())
        assert r['passed'] and r['child_reaped'] and r['executed_snapshot_unchanged']
        assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
        with tarfile.open(p/'sources.tar.gz') as archive:
            members=archive.getmembers();assert len(members)==len({m.name for m in members})
            assert all(m.isfile() for m in members)
            assert {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}==r['source_hashes_before']
    r=json.loads((HERE/'source-bound-final-v1/report.json').read_text())
    assert r['final_source_bound'] and r['original_sources_unchanged'] and r['assets_unchanged']
    assert len(r['source_hashes_before'])==125 and set(r['cases'])=={'15','128'}
    assert r['test_returncode']==0 and not r['encrypted_execution'] and not r['actual_refresh'] and not r['model_arrays_archived']
    for key,case in r['cases'].items():
        count=1 if key=='15' else 8
        assert case['passed'] and case['tiles']==count and case['physical_slots_checked']==count*65536
        assert max(case['max_abs_errors'].values())<case['actual_loop_vs_independent_polynomial_gate']==1e-8
        assert case['actual_primitive_counts']=={name:count*n for name,n in {'rotate':286,'weighted_sum':190,'multiply':80,'add':164}.items()}
        assert case['public_plaintext_products']==count*7857 and case['identity_refresh_calls']==2*count
    print('Verified125 final source/public-plan files, all9 complete ideal tiles and artifact hashes; no HE claim.')
if __name__=='__main__':main()
