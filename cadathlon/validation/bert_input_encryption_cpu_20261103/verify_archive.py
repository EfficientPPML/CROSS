"""Public actual-stage3 evidence only; no retained payloads or runtime imports."""
from pathlib import Path
import hashlib,json,tarfile
HERE=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def read(p):return json.loads(p.read_bytes())
def verify():
    r=read(HERE/'report.json');w=read(HERE/'worker/worker_report.json');p=read(HERE/'worker/process.json');s=read(HERE/'summary.json')
    assert sha((HERE/'report.json').read_bytes())==s['report_sha256']=='add5e6a3143be3931ad29c018cdee40bdf059497b8bfa987114f39baeb88631e'
    assert all(r[n]is True for n in ('passed','execute','actual_encryption','actual_execution_attempted','io_workspace_retained','all_processes_reaped','source_files_unchanged','source_archive_verified','cleanup_completed','terminal_scope_completed'))
    assert all(w[n]is True for n in ('passed','actual_client_encode_started','actual_client_encode_completed','readback_complete','all_eight_ordered_payloads_validated','keygen_decrypt_native_model_forbidden','source_files_unchanged','terminal_scope_completed'))
    assert not w['forbidden_attempts'] and not any(w[n] for n in ('decrypt','model_forward','server_compute'))
    assert r['worker_process']==p and p['returncode']==0 and p['passed'] is False
    assert all(p[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
    assert p['result']==read(HERE/'worker/stdout.json') and p['result']['passed'] is True
    assert p['result']['worker_report_sha256']==sha((HERE/'worker/worker_report.json').read_bytes())
    assert not r.get('error') and not r.get('cleanup_errors') and not w.get('error') and not p.get('error')
    b=r['source_hashes_start'];assert len(b)==80 and b==read(HERE/'source_manifest.json')==r['source_hashes_end']==r['execution_source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']
    assert sha((HERE/'sources.tar.gz').read_bytes())==r['source_archive_sha256']==s['source_archive_sha256']
    with tarfile.open(HERE/'sources.tar.gz') as t:
        ms=t.getmembers();assert len(ms)==len({m.name for m in ms})==80
        assert all(m.isfile() and m.size<=8<<20 and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts for m in ms)
        assert sum(m.size for m in ms)<=16<<20
        assert {m.name:sha(t.extractfile(m).read()) for m in ms}==b
    assert r['prerequisite']==w['prerequisite'] and r['readback']==w['readback']
    readback=r['readback'];assert readback['tile_count']==8
    rows=readback['tiles'];assert len(rows)==8 and [x['tile'] for x in rows]==list(range(8))
    assert sum(x['payload_bytes'] for x in rows)==838860800
    assert all(x['shape']==[1,2,256,512,100] for x in rows)
    assert readback['private_binding_verified'] and readback['private_request_permissions_verified']
    assert readback['private_input_sha256']=='c30ffc79d8150cf6cf3d544bca23a07150d8a0d0f7b7924788e91df3aa55991f'
    assert r['asset_hashes_start']==r['asset_hashes_end']==w['asset_hashes_start']==w['asset_hashes_end']
    terminal=read(HERE/'terminal.json');assert terminal['session_id']==54713 and terminal['exit_code']==0 and terminal['reaped'] and terminal['pids_absent']
    assert terminal['current_source_hashes']==b and terminal['dummy_owned_key_directory_absent']
    assert s['archived_payloads'] is False and s['server_compute'] is False and s['numerical_accuracy_checked'] is False
    print(json.dumps(dict(passed=True,actual_client_encryption=True,sources=80,tiles=8,raw_payload_bytes=838860800,server_or_accuracy_claim=False)))
if __name__=='__main__':verify()
