"""Private size0 metrics and isolated completion receipt; aggregates only."""
from pathlib import Path
import hashlib,stat
import client_io as io
from client_metrics import compare

def validate(paths,*,request_batch,reference,completion_directory):
    from transport_contract import digest
    import request_lifecycle as lifecycle
    result_path=Path(paths['decrypted_results']);state=result_path.lstat()
    if not stat.S_ISREG(state.st_mode)or stat.S_IMODE(state.st_mode)!=0o600:raise ValueError('private0600 decoded result required')
    raw=io.read(result_path,65536);lines=raw.decode('utf-8').splitlines()
    if len(lines)!=1:raise ValueError('exactly one private size0 result required')
    row=io.parse(lines[0])
    if type(row)is not dict or set(row)!={'target_idx','he_logits'}or type(row['target_idx'])is not int or row['target_idx']!=reference['target_idx']:
        raise ValueError('private target/order binding differs')
    metrics=compare(row['he_logits'],reference)
    receipt_path=lifecycle.completion_path(paths,completion_directory);state=receipt_path.lstat()
    if not stat.S_ISREG(state.st_mode)or stat.S_IMODE(state.st_mode)!=0o600:raise ValueError('private0600 isolated receipt required')
    receipt_raw=io.read(receipt_path,16<<20);receipt=io.parse(receipt_raw)
    nonces=[r['contract']['request_id']for r in request_batch['requests']]
    objects={kind:lifecycle._observe(path,kind,nonces)for kind,path in lifecycle._paths(paths).items()}
    expected=dict(schema='cross-bert-client-completion-v1',context_id=request_batch['public']['context_id'],
        public_batch_sha256=digest(request_batch),nonces=nonces,objects=objects)
    canonical=lambda v:(__import__('json').dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
    if canonical(receipt)!=canonical(expected):raise ValueError('isolated receipt differs from retained original inputs/new outputs')
    return dict(sample_count=1,metrics=metrics,decoded_result_sha256=hashlib.sha256(raw).hexdigest(),completed_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        private_receipt_validated=True,result_permissions_0600=True,public_batch_sha256=digest(request_batch),
        isolated_completion_receipt=True,retirement_performed=False)
