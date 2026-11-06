"""Ordered full-batch readback and ranked acceptance; no crypto/model constructor.

Callers must authenticate case/source/key/model producer evidence before calling
the disk readers. These functions do not turn a self-consistent report into an
authenticated producer. Tests use only newly owned synthetic artifacts.
"""
from pathlib import Path
import hashlib,math,stat
import file_stage_contract as batches
from file_stage_io import read_manifest
import transport_contract as tc


def ranked_count(value):
    if type(value)is not int or value not in (100,200,408):
        raise ValueError('one exact ranked count required')
    return value


def _inventory(path,names):
    path=Path(path)
    if not stat.S_ISDIR(path.lstat().st_mode) or {p.name for p in path.iterdir()}!=set(names):
        raise ValueError('exact ordinary ranked directory inventory required')


def requests(paths,*,public,count,expected_batch_sha256,expected_summary=None):
    from transport import validate_ciphertext,request_digest,manifest_digest
    count=ranked_count(count)
    batch=batches.validate_public_batch(read_manifest(Path(paths['upload'])/'batch.json'),public=public,count=count)
    if tc.digest(batch)!=tc._sha(expected_batch_sha256):
        raise ValueError('authenticated ranked upload digest differs')
    _inventory(paths['upload'],[r['contract']['request_id'] for r in batch['requests']]+['batch.json'])
    rows=[];p=public['parameters'];shape=[1,2,*p['degree_layout'],len(p['q_towers'])]
    per=math.prod(shape)*4
    for ordinal,request in enumerate(batch['requests']):
        nonce=request['contract']['request_id'];sample=Path(paths['upload'])/nonce
        _inventory(sample,[f'tile{i}' for i in range(8)])
        manifests=[];tiles=[]
        for i in range(8):
            payload,scale,binding,manifest=validate_ciphertext(sample/f'tile{i}',contract=request['contract'],role='request',tile_index=i)
            if (list(payload.shape)!=shape or payload.nbytes!=per or str(payload.dtype)!='uint32'
                    or type(scale)is not float or scale!=p['scaling_factor'] or binding['level']!=49):
                raise ValueError('canonical complete ranked input metadata differs')
            tiles.append(dict(tile=i,payload_bytes=payload.nbytes,shape=list(payload.shape),scale=scale,manifest_sha256=manifest_digest(manifest)))
            manifests.append(manifest);del payload
        actual=request_digest(manifests,request['contract'])
        if actual!=request['input_digest']:raise ValueError('ranked ordered input digest differs')
        rows.append(dict(ordinal=ordinal,request_id=nonce,input_digest=actual,tiles=tiles))
    summary=dict(sample_count=count,tile_count=8*count,public_batch_sha256=tc.digest(batch),
        ciphertext_payload_bytes=count*8*per,requests=rows)
    if expected_summary is not None and tc.canonical(summary)!=tc.canonical(expected_summary):
        raise ValueError('completed ranked producer summary differs')
    return summary,batch


def responses(paths,*,batch,expected_response_sha256=None):
    from transport import validate_ciphertext,manifest_digest
    import file_stages
    count=ranked_count(batch['sample_count'])
    batches.validate_public_batch(batch,public=batch['public'],count=count)
    inputs,candidate=file_stages._inputs()
    response=batches.validate_response_batch(read_manifest(Path(paths['download'])/'batch.json'),batch,
        check_scale=file_stages._scale_checker(inputs,candidate))
    actual=tc.digest(response)
    if expected_response_sha256 is not None and actual!=tc._sha(expected_response_sha256):
        raise ValueError('authenticated ranked response digest differs')
    _inventory(paths['download'],[r['request_id'] for r in response['responses']]+['batch.json'])
    rows=[];p=batch['public']['parameters'];shape=[1,2,*p['degree_layout'],6];per=math.prod(shape)*4
    for ordinal,(request,row) in enumerate(zip(batch['requests'],response['responses'])):
        payload,scale,binding,manifest=validate_ciphertext(Path(paths['download'])/row['request_id'],
            contract=request['contract'],role='response',tile_index=0,expected_scale=row['scale'],input_digest=request['input_digest'])
        if (manifest_digest(manifest)!=row['response_manifest_sha256'] or list(payload.shape)!=shape
                or payload.nbytes!=per or str(payload.dtype)!='uint32' or binding['level']!=2 or scale!=row['scale']):
            raise ValueError('ranked response payload/metadata differs')
        rows.append(dict(ordinal=ordinal,request_id=row['request_id'],input_digest=row['input_digest'],
            response_manifest_sha256=manifest_digest(manifest),payload_bytes=per,shape=shape,scale=scale,level=2))
        del payload
    timing=read_manifest(paths['server_report']);seconds=timing.get('full_model_compute_and_transport_seconds')
    if (set(timing)!={'full_model_compute_and_transport_seconds'} or type(seconds)not in (float,int)
            or not math.isfinite(seconds) or seconds<0):raise ValueError('finite original server diagnostic time required')
    return dict(sample_count=count,response_count=count,public_batch_sha256=tc.digest(batch),
        response_batch_sha256=actual,ciphertext_payload_bytes=count*per,responses=rows,
        full_model_compute_and_transport_seconds=seconds)


def label(logits):
    if type(logits)is not list or len(logits)!=2 or any(type(x)not in (int,float) for x in logits):
        raise ValueError('two real JSON logits required')
    try:values=[float(x) for x in logits]
    except (OverflowError,ValueError):raise ValueError('finite binary64 logits required') from None
    if not all(math.isfinite(x) for x in values):raise ValueError('finite binary64 logits required')
    return int(values[1]>=values[0])


def evaluate_records(records,references,*,count,predictions=None):
    """Private in-memory validation; returns aggregate-only ranked gates.

references are already byte-authenticated client rows from the completed
publication. Per-logit/approximate agreement is deliberately not an HE gate.
"""
    count=ranked_count(count)
    if type(records)is not list or type(references)is not list or len(records)!=count or len(references)!=count:
        raise ValueError('complete ranked private result/reference inventory required')
    ids=[];correct=official=agreement=0;decisions=[]
    for result,reference in zip(records,references):
        if (type(result)is not dict or set(result)!={'target_idx','he_logits'} or type(result['target_idx'])is not int
                or result['target_idx']<0 or type(reference)is not dict or type(reference.get('target_idx'))is not int
                or result['target_idx']!=reference['target_idx'] or type(reference.get('label'))is not int
                or reference['label'] not in (0,1)):
            raise ValueError('ranked private target/order/label schema differs')
        pred=label(result['he_logits']);native=label(reference.get('native_fp32_logits'))
        ids.append(result['target_idx']);decisions.append(pred)
        correct+=pred==reference['label'];official+=native==reference['label'];agreement+=pred==native
    if len(set(ids))!=count:raise ValueError('unique ranked private targets required')
    postprocessed=False
    if predictions is not None:
        if (type(predictions)is not list or len(predictions)!=count or any(type(x)is not int or x not in (0,1) for x in predictions)
                or predictions!=decisions):raise ValueError('postprocessed labels differ from original-order decrypted logits')
        postprocessed=True
    floor=4*correct>=3*count;drop=100*(official-correct)<=2*count
    return dict(sample_count=count,he_correct=correct,official_correct=official,accuracy=correct/count,
        official_accuracy=official/count,ranked_accuracy_floor_passed=floor,ranked_accuracy_drop_passed=drop,
        prediction_agreement=agreement/count,prediction_agreement_is_diagnostic=True,
        all_ordered_finite_logits_validated=True,postprocessed_predictions_validated=postprocessed,
        ranked_labels_passed=floor and drop,final_ranked_acceptance=floor and drop and postprocessed,
        per_logit_tolerance_required_for_ranked_acceptance=False)


def _private_read(path,maximum):
    # Reuse the reviewed exact held-fd/before/opened/after bounded read, with
    # explicit uncertain-close quarantine; never use public response readers.
    from ranked_private_io import PrivateReader
    import os
    if type(maximum)is not int or not 0<maximum<=16<<20:raise ValueError('bounded private read required')
    path=Path(path);owner=PrivateReader()
    try:
        parent=owner._open_dir(path.parent);raw=owner._read(path,parent)
        if len(raw)>maximum:raise ValueError('private result exceeds declared bound')
        return raw
    finally:owner.close(__import__('sys').exception())


def client_results(paths,*,batch,references,count,predictions=None):
    import key_gate,request_lifecycle as lifecycle
    count=ranked_count(count)
    batches.validate_public_batch(batch,public=batch['public'],count=count)
    raw=_private_read(paths['decrypted_results'],16<<20)
    try:records=[key_gate.parse(line) for line in raw.decode('utf8').splitlines()]
    except Exception:raise ValueError('ranked private output JSON could not be validated') from None
    metrics=evaluate_records(records,references,count=count,predictions=predictions)
    receipt_path=Path(paths['client_private'])/'completed.json'
    receipt_raw=_private_read(receipt_path,16<<20);receipt=key_gate.parse(receipt_raw)
    nonces=[r['contract']['request_id'] for r in batch['requests']]
    objects={kind:lifecycle._observe(path,kind,nonces) for kind,path in lifecycle._paths(paths).items()}
    expected=dict(schema='cross-bert-client-completion-v1',context_id=batch['public']['context_id'],
        public_batch_sha256=tc.digest(batch),nonces=nonces,objects=objects)
    if tc.canonical(receipt)!=tc.canonical(expected):raise ValueError('ranked completion receipt differs from every retained object')
    return dict(sample_count=count,metrics=metrics,decoded_result_sha256=hashlib.sha256(raw).hexdigest(),
        completed_receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),private_receipt_validated=True,
        result_permissions_0600=True,public_batch_sha256=tc.digest(batch),receipt_does_not_authorize_retirement_by_itself=True)
