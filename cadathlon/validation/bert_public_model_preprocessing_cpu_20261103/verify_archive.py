"""Verify captured public metadata without opening any retained model/key store."""
from pathlib import Path
import ast,hashlib,json,tarfile
ROOT=Path(__file__).resolve().parent
sha=lambda raw:hashlib.sha256(raw).hexdigest()
canonical=lambda v:(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def verify():
    d=ROOT/'measured';s=json.loads((ROOT/'summary.json').read_text());r=json.loads((d/'report.json').read_text());w=json.loads((d/'worker/worker_report.json').read_text());m=json.loads((d/'model_manifest.json').read_text());p=r['worker_process']
    assert sha((d/'report.json').read_bytes())==s['report_sha256']=='5b9589b51828d066706c4c46a6404d8c52fcd67a2c9aa9f3f57c73df8caeefe0'
    assert sha((d/'sources.tar.gz').read_bytes())==s['source_archive_sha256']=='e7c5ac0b864cbcabecdaec84deb6a85c568914e95e655191c092465653b7bcec'
    assert all(r[n]is True for n in ('passed','execute','actual_preprocessing','model_workspace_retained','all_processes_reaped','source_files_unchanged','source_archive_verified','cleanup_completed'))
    assert p['passed']is False and type(p['returncode'])is int and p['returncode']==0 and p['result']['passed']is True and not p.get('error')
    assert all(p[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
    assert p['result']['worker_report_sha256']==sha((d/'worker/worker_report.json').read_bytes())
    assert all(w[n]is True for n in ('passed','actual_server_preprocess_called','readback_complete','all_readback_fp64_immutable','source_files_unchanged','keygen_and_codecs_forbidden','context_construction_forbidden'))
    assert w['tensor_count']==196 and w['matrix_count']==74 and not w['forbidden_attempts'] and not any(w[n] for n in ('model_forward','encryption','context_constructed'))
    assert w['prerequisite']==r['prerequisite'] and w['asset_hashes_start']==w['asset_hashes_end']==r['asset_hashes_start']==r['asset_hashes_end']
    assert w['public_model_manifest']==m and m['context_id']==s['context_id']==r['context_id']==p['result']['context_id']
    assert m['model_id']==s['model_id']==r['model_id']==p['result']['model_id']==sha(canonical({k:v for k,v in m.items() if k!='model_id'}))
    assert sha((d/'model_manifest.json').read_bytes())==r['model_manifest_sha256']
    before=r['source_hashes_start'];assert len(before)==65 and before==r['source_hashes_end']==w['source_hashes_start']==w['source_hashes_end']==json.loads((d/'source_manifest.json').read_text())
    with tarfile.open(d/'sources.tar.gz') as tar:
        rows=tar.getmembers();assert len(rows)==len({m.name for m in rows})==65 and all(m.isfile() for m in rows)
        raw={m.name:tar.extractfile(m).read() for m in rows};assert {n:sha(b) for n,b in raw.items()}==before
    tree=ast.parse(raw['checkout/stage/server_public_model.py']);pins=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TENSOR_SHA256' for t in n.targets))
    assert len(pins)==len(m['tensors'])==196 and sum(len(v['shape'])==2 for v in m['tensors'].values())==74
    assert all(v['tensor_sha256']==pins[n] and v['dtype']=='<f4' for n,v in m['tensors'].items())
    fn=next(n for n in ast.parse(raw['checkout/check_transformer_suite.py']).body if isinstance(n,ast.FunctionDef) and n.name=='run_case')
    assert not any(isinstance(n,ast.Subscript) and isinstance(n.ctx,ast.Store) and isinstance(n.value,ast.Name) and n.value.id=='record' and isinstance(n.slice,ast.Constant) and n.slice.value=='passed' for n in ast.walk(fn))
    inventory=json.loads((ROOT/'archive_inventory.json').read_text())
    assert inventory=={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='archive_inventory.json' and '__pycache__' not in p.parts}
    print(json.dumps(dict(passed=True,sources=65,tensors=196,matrices=74,raw_process_passed_field=False,raw_process_placeholder_preserved=True,retained_payloads_read=False)))
if __name__=='__main__':verify()
