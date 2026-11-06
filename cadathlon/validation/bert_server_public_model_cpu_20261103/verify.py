"""Read-only source/provenance and full public-tensor inventory verification."""
from pathlib import Path
import ast,hashlib,json,tarfile
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def archive(path,expected):
    with tarfile.open(path) as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==len(expected)==7 and all(m.isfile() for m in ms)
        data={m.name:t.extractfile(m).read() for m in ms}
    assert {n:hashlib.sha256(b).hexdigest() for n,b in data.items()}==expected
    return data

def main():
    for n,h in json.loads((HERE/'artifact_sha256.json').read_text()).items():assert sha(HERE/n)==h,n
    for version in (1,2,3):
        d=HERE/f'source-bound-v{version}';r=json.loads((d/'report.json').read_text())
        assert r['passed'] and r['returncode']==0 and r['child_reaped'] and r['temporary_public_arrays_removed']
        assert r['source_files_unchanged'] and r['assets_unchanged'] and not r['HE_executed'] and not r['model_forward']
        assert r['source_hashes_before']==r['source_hashes_after']==r['snapshot_hashes_after']
        assert r['asset_hashes_before']==r['asset_hashes_after'] and len(r['asset_hashes_before'])==2
        assert sha(d/'sources.tar.gz')==r['source_archive_sha256'];data=archive(d/'sources.tar.gz',r['source_hashes_before'])
        case=json.loads((d/'case.json').read_text());assert case==r['case'] and case['passed']
        assert case['tensors']==196 and case['matrices']==74 and case['vector_tensors']==122
        assert case['raw_tensor_bytes']==4*(12*(4*(768*768+768)+(3072*768+3072)+(768*3072+768)+4*768)+(768*768+768)+(2*768+2))==342586376
        assert case['immutable_exact_fp64_lifts'] and case['no_backend_or_model_forward']
        source=ast.parse(data['server_public_model.py'])
        pins=next(ast.literal_eval(n.value) for n in source.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TENSOR_SHA256' for t in n.targets))
        public=json.loads(data['server_tensor_pins.json']);assert pins==public['tensors'] and len(pins)==196 and all('embeddings' not in n for n in pins)
        count=28 if version==3 else 23
        assert f'{count} passed' in (d/'tests.log').read_text()
    current=HERE/'source'
    assert all(sha(current/n)==h for n,h in r['source_hashes_before'].items())
    terminal=json.loads((HERE/'terminal.json').read_text());assert terminal['session_id']==92223 and terminal['observed_exit_code']==0 and terminal['reaped']
    assert not list(HERE.rglob('*.npy')) and not list(HERE.rglob('*.npz'))
    print('Verified7 final sources,28 checks,196 pinned server tensors/74 matrices; no keys, inference, or five-handler claim.')
if __name__=='__main__':main()
