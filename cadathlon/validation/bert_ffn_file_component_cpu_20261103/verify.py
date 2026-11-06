"""Offline verification of preserved FFN component evidence; does not run HE."""
from pathlib import Path
import hashlib,json,tarfile
HERE=Path(__file__).resolve().parent
sha=lambda data:hashlib.sha256(data).hexdigest()
def main():
 run=HERE/'run-v1';r=json.loads((run/'report.json').read_text());start=r['source_hashes_start']
 assert r['passed'] is True and start==r['source_hashes_end'] and len(start)==135
 assert all(r[k] is True for k in ('all_component_gates_passed','all_processes_reaped','owned_component_directory_removed','owned_directory_handle_closed','source_files_unchanged'))
 assert sha((run/'sources.tar.gz').read_bytes())==r['source_archive_sha256']
 prefix='/tmp/bert-ffn-file-component-20261103/checkout/'
 with tarfile.open(run/'sources.tar.gz') as t:
  members=t.getmembers();assert len(members)==135 and len({m.name for m in members})==135
  for name,digest in start.items():
   member='repository/'+name.removeprefix(prefix) if name.startswith(prefix) else 'prototype/'+Path(name).name
   assert sha(t.extractfile(member).read())==digest,name
 for role in ('client-preprocess','client-encrypt','keyless-server','client-decrypt'):
  p=json.loads((run/role/'process.json').read_text());assert p['returncode']==0 and p['process_reaped'] and p['public_key_store_removed']
 result=json.loads((run/'client-decrypt/stdout.json').read_text())
 assert result['passed'] and result['all_slots_checked']==3*32768
 for row in result['checks'].values():
  assert row['passed'] and row['slots_checked']==32768 and row['gate']==1e-6 and row['max_abs_error']<=1e-6
 assert not result['full_bert_evaluated'] and not result['residual_or_layernorm_evaluated']
 print(json.dumps({'passed':True,'exact_archived_sources':135,'all_slots_checked':98304,'worst_abs_error':result['worst_abs_error_vs_fixed_polynomial']}))
if __name__=='__main__':main()
