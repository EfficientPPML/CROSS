from pathlib import Path
import tempfile,sys,json
sys.path[:0]=['/tmp/bert-full-model-file-stages-20261103','/tmp/bert-full-model-transport-20261103']
from request_lifecycle import record_completed,retire_completed
from transport_contract import canonical
root=Path(tempfile.mkdtemp(prefix='review-completion-retirement-'));private=root/'private';private.mkdir()
paths={name:str(root/name) for name in ('upload','download','server_report','decrypted_results')};paths['client_private']=str(private)
nonce='a'*32;context='b'*64;batch={'public':{'context_id':context},'requests':[{'contract':{'request_id':nonce}}]}
for kind in ('upload','download'):
 parent=Path(paths[kind]);parent.mkdir();(parent/'batch.json').write_bytes(canonical(batch));sample=parent/nonce;sample.mkdir()
 for directory in ([sample/f'tile{i}' for i in range(8)] if kind=='upload' else [sample]):
  if directory!=sample:directory.mkdir()
  for name in ('bundle.json','ct_0.npz','transport.json'):(directory/name).write_bytes(b'original-owned-artifact')
for name in ('server_report','decrypted_results'):Path(paths[name]).write_bytes(b'completed')
(private/'requests.json').write_bytes(b'completed')
receipt=record_completed(paths,context_id=context,batch=batch)
path=Path(paths['upload'])/nonce/'tile0/ct_0.npz';original_inode=path.stat().st_ino
replacement=path.with_name('foreign-new-file');replacement.write_bytes(b'replacement-not-in-completion-receipt');new_inode=replacement.stat().st_ino;replacement.replace(path)
result=retire_completed(paths,context_id=context)
report={'root':str(root),'original_inode':original_inode,'replacement_inode':new_inode,'replacement_distinct':new_inode!=original_inode,'retirement_returned':result,'replacement_preserved':path.exists()}
Path('/tmp/review_file_stages_retirement.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
