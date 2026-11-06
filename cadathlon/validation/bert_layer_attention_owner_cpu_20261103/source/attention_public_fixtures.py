"""Pinned public tensors and fixed native component inputs; never a server path."""
from pathlib import Path
import hashlib,json,os
import numpy as np
HERE=Path(__file__).resolve().parent
MODEL=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c/model.safetensors')
MODEL_SHA='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
FIXTURE_SHA='54fc08da5e682ac6c73519b5310631a43519be95ee8c5501e7574355f5fb956e'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def load():
    from layer_attention_contract import validate_projection_arrays,PREFIX
    from encrypted_bert_all_row_norm import AFFINE_SHA256
    fixture=Path(os.environ.get('BERT_LAYER_ATTENTION_FIXTURE',HERE/'client_all_attention_fixtures.npz'))
    if sha(MODEL)!=MODEL_SHA or sha(fixture)!=FIXTURE_SHA:raise ValueError('pinned checkpoint/native fixture changed')
    arrays={};records={};models={}
    with MODEL.open('rb') as stream:
        length=int.from_bytes(stream.read(8),'little');header=json.loads(stream.read(length));base=8+length
        for layer in range(12):
            prefix='bert.encoder.layer.'+str(layer)+'.';model={}
            for role in ('query','key','value','output'):
                model[role]={}
                for kind in ('weight','bias'):
                    source=prefix+PREFIX[role]+'.'+kind;desc=header[source];shape=(768,768) if kind=='weight' else (768,)
                    if desc['dtype']!='F32' or desc['shape']!=list(shape):raise ValueError('pinned projection dtype/shape differs')
                    start,end=desc['data_offsets'];stream.seek(base+start);raw=stream.read(end-start)
                    if len(raw)!=np.prod(shape)*4:raise ValueError('truncated tensor')
                    model[role][kind]=np.frombuffer(raw,dtype='<f4').reshape(shape).astype(np.float64)
                    records[source]=dict(shape=list(shape),source_f32_sha256=hashlib.sha256(raw).hexdigest(),exact_f64_lift=True)
            models[layer]=validate_projection_arrays(model,layer)
            for index,(name,kind) in enumerate((('gamma','weight'),('beta','bias'))):
                source=prefix+'attention.output.LayerNorm.'+kind;desc=header[source]
                if desc['dtype']!='F32' or desc['shape']!=[768]:raise ValueError('pinned norm dtype/shape differs')
                start,end=desc['data_offsets'];stream.seek(base+start);raw=stream.read(end-start)
                if len(raw)!=768*4 or hashlib.sha256(raw).hexdigest()!=AFFINE_SHA256[f'{layer}.attention'][index]:raise ValueError('pinned norm bytes differ')
                arrays[f'layer{layer}_{name}']=np.frombuffer(raw,dtype='<f4').astype(np.float64)
                records[source]=dict(shape=[768],source_f32_sha256=hashlib.sha256(raw).hexdigest(),exact_f64_lift=True)
    with np.load(fixture,allow_pickle=False) as data:
        for name in data.files:
            a=data[name]
            if a.dtype.kind not in 'fi' or not np.isfinite(a).all():raise ValueError('invalid fixed native fixture')
            arrays[name]=a.copy();records[name]=dict(shape=list(a.shape),dtype=a.dtype.str,sha256=hashlib.sha256(a.tobytes()).hexdigest())
    return arrays,models,dict(passed=True,scope=__doc__,checkpoint_sha256=MODEL_SHA,native_fixture_sha256=FIXTURE_SHA,arrays=records)
