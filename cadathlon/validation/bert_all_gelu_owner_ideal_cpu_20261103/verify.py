"""Offline source and case accounting for ideal all-layer GELU evidence."""
from pathlib import Path
import hashlib,json,math,tarfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    for n,h in json.loads((HERE/'artifact_sha256.json').read_text()).items():assert sha(HERE/n)==h,n
    p=HERE/'measured';r=json.loads((p/'report.json').read_text())
    assert r['passed'] and r['returncode']==0 and r['child_reaped'] and r['source_files_unchanged'] and r['snapshot_unchanged']
    assert not r['encrypted_execution'] and len(r['source_hashes_before'])==119
    assert r['source_hashes_before']==r['source_hashes_after']
    assert sha(p/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(p/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==119 and all(m.isfile() for m in ms)
        contents={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in contents.items()}==r['source_hashes_before']
    manifest=json.loads(contents['candidate_manifest.json']);cases=r['cases'];assert len(cases)==72
    assert r['physical_slots_checked']==sum(c['physical_slots_checked'] for c in cases)==4718592
    assert r['max_abs_vs_independent_chebval']==max(c['max_abs_vs_independent_chebval'] for c in cases)
    count=0
    for layer in range(12):
        rule=manifest['gelu'][str(layer)];degree=rule['degree'];depth=2*math.ceil(math.log2(degree))+2
        rows=[c for c in cases if c['layer']==layer];assert len(rows)==6
        assert {(str(c['level_choice']),c['source_scale']) for c in rows}.__len__()==6
        assert {c['level_choice'] for c in rows}=={'minimum',24,49}
        count+=50-(depth+2)
        for c in rows:
            assert c['degree']==degree and c['domain']==rule['polynomial']['domain']
            assert c['coefficient_sha256']==rule['polynomial']['coefficient_f64le_sha256']
            assert c['physical_slots_checked']==65536 and c['max_abs_vs_independent_chebval']<c['gate']==1e-10
            plan=c['actual_plan'];assert plan['consumed_levels']==depth and plan['output_level']==c['source_level']-depth>=2
            assert plan['ciphertext_products']==degree-1 and plan['weighted_operations']==degree+1 and plan['actual_composite_rescales']==2*degree
    assert count==388
    terminal=json.loads((HERE/'terminal.json').read_text());assert terminal['session_id']==24006 and terminal['observed_exit_code']==0 and terminal['reaped']
    print('Verified119 source files,72 ideal full-slot cases and388 level schedules; no HE claim.')
if __name__=='__main__':main()
