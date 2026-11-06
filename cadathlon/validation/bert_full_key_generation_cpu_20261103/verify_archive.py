"""Verify public first-handler evidence only; never access live key storage."""
from pathlib import Path
import hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
canonical=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def verify():
    d=ROOT/'measured';s=json.loads((ROOT/'summary.json').read_text());r=json.loads((d/'report.json').read_text());w=json.loads((d/'worker/worker_report.json').read_text());m=json.loads((d/'public_manifest.json').read_text())
    assert sha((d/'report.json').read_bytes())==s['report_sha256']=='022cbdf687273e765668184f32cb7cd501d371c4220916e1c8f02574cf3a7f08'
    assert sha((d/'sources.tar.gz').read_bytes())==s['source_archive_sha256']=='083851cf0667d2bb500514b5599028312c7d387236347ce3de3b585c3dcd2a29'
    assert all(r[n]is True for n in ('passed','execute','actual_key_generation','workspace_retained','all_processes_reaped','source_files_unchanged','source_archive_verified','cleanup_completed'))
    p=r['worker_process'];assert type(p['returncode'])is int and p['returncode']==0 and not p.get('error')
    assert all(p[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
    assert p['result']['worker_report_sha256']==sha((d/'worker/worker_report.json').read_bytes())
    assert all(w[n]is True for n in ('passed','actual_key_generation_started','actual_key_generation_completed','private_binding_verified','private_permissions_verified','source_files_unchanged'))
    assert w['private_file_count']==2 # Historical producer assertion only.
    assert m==w['public_manifest'] and m['context_id']==r['context_id']==p['result']['context_id']==s['context_id']
    assert sha(canonical({k:v for k,v in m.items() if k!='context_id'}))==s['context_id']
    assert len(m['arrays'])==263 and len(m['rotations'])==130 and sum(v['bytes'] for v in m['arrays'].values())==125380363136
    assert not any(n.lower().startswith(('secret','private','sk')) for n in m['arrays'])
    expected=r['source_hashes_start'];assert len(expected)==58 and expected==r['source_hashes_end']==r['isolated_execution_source_hashes']==w['source_hashes_start']==w['source_hashes_end']==json.loads((d/'source_manifest.json').read_text())
    with tarfile.open(d/'sources.tar.gz') as tar:
        rows=tar.getmembers();assert len(rows)==len({m.name for m in rows})==58 and all(m.isfile() for m in rows)
        assert {m.name:sha(tar.extractfile(m).read()) for m in rows}==expected
    inventory=json.loads((ROOT/'archive_inventory.json').read_text())
    assert inventory=={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='archive_inventory.json' and '__pycache__' not in p.parts}
    print(json.dumps(dict(passed=True,sources=58,public_arrays=263,context_id=s['context_id'],live_key_workspace_read=False)))
if __name__=='__main__':verify()
