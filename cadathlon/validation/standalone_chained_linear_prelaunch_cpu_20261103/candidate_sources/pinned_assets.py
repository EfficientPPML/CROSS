"""Pinned public tensors and fixed client-only linear fixtures."""
from pathlib import Path
import hashlib
import json
import os
import numpy as np

REVISION='f150c1d609d1e50dd5e2e5408661cfac8339277c'
CHECKPOINT_SHA256='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
EXPORT_SHA256='85ca3df4220f5442c02b94a2b850c9447a2e8320304b0ad88b59aba9321a5e34'
TENSORS={
 'query_weight':('bert.encoder.layer.0.attention.self.query.weight',(768,768),'1a8914c482024a643b1503123f2c1bac81ce565d0ab7c98f06d8951d8b277f47'),
 'query_bias':('bert.encoder.layer.0.attention.self.query.bias',(768,),'85b53d121a7751dbd243a12ce73c6be48132471667f817fde6020e1baaaed9c4'),
 'intermediate_weight':('bert.encoder.layer.0.intermediate.dense.weight',(3072,768),'9a0c8aa79b2bac48fc35f37b6695d253080f7eedd556b70b65411aab9c319fd3'),
 'intermediate_bias':('bert.encoder.layer.0.intermediate.dense.bias',(3072,),'b279856af79c545cdb2f914e02e9a953204152319f160e484ebfcc8d6b825774'),
 'output_weight':('bert.encoder.layer.0.output.dense.weight',(768,3072),'754a718ad071ca9ec27790aee1b82dc9db41cc3ef90c6c73a90ada8668f6fa0d'),
 'output_bias':('bert.encoder.layer.0.output.dense.bias',(768,),'744629ad72bdc326f22cd92c6429610c06ed41dd3a2fedd307d3f01f861638b9')}

def sha(data):return hashlib.sha256(data).hexdigest()

def load_matrices(export,checkpoint):
    """Verify exported bytes against pinned source offsets, before encryption."""
    export=Path(export);checkpoint=Path(checkpoint)
    if not export.is_file() or export.is_symlink() or not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError('regular public assets required')
    if sha(export.read_bytes())!=EXPORT_SHA256:raise ValueError('pinned exported NPZ differs')
    hasher=hashlib.sha256()
    with checkpoint.open('rb') as stream:
        while block:=stream.read(1024*1024):hasher.update(block)
        stream.seek(0);length=int.from_bytes(stream.read(8),'little')
        if length<=0 or length>16*1024**2:raise ValueError('invalid checkpoint header')
        header=json.loads(stream.read(length));base=8+length
        if hasher.hexdigest()!=CHECKPOINT_SHA256:raise ValueError('pinned checkpoint differs')
        result={};descriptors={}
        with np.load(export,allow_pickle=False) as arrays:
            if set(arrays.files)!=set(TENSORS):raise ValueError('unexpected exported inventory')
            for name,(tensor,shape,digest) in TENSORS.items():
                array=arrays[name];desc=header[tensor]
                if array.dtype!=np.dtype('<f4') or array.shape!=shape or not np.isfinite(array).all():raise ValueError('invalid exported tensor')
                if desc['dtype']!='F32' or tuple(desc['shape'])!=shape:raise ValueError('checkpoint tensor shape/dtype mismatch')
                begin,end=desc['data_offsets'];stream.seek(base+begin);raw=stream.read(end-begin)
                if len(raw)!=array.nbytes or sha(raw)!=digest or raw!=array.tobytes():raise ValueError('export differs from actual pinned tensor')
                result[name]=array.astype(np.float64)
                descriptors[name]={'tensor_name':tensor,'shape':list(shape),'dtype':'<f4','sha256':digest}
    matrices={name:(result[name+'_weight'],result[name+'_bias']) for name in ('query','intermediate','output')}
    return matrices,dict(revision=REVISION,checkpoint_sha256=CHECKPOINT_SHA256,export_sha256=EXPORT_SHA256,tensors=descriptors,
        export_verified_against_actual_checkpoint_offsets=True)

def make_fixtures(matrices,layout):
    result={}
    for index,(name,(weight,bias)) in enumerate(matrices.items()):
        rng=np.random.default_rng(20261103+index)
        valid=1 if name=='intermediate' else layout.rows_per_tile
        raw=rng.uniform(8.,16.,layout.slots).reshape(layout.rows_per_tile,layout.features)
        raw[:valid,:weight.shape[1]]=rng.uniform(-.25,.25,(valid,weight.shape[1]))+np.arange(valid)[:,None]/32
        mask=np.where(np.arange(layout.slots)%3,.5,-1.).astype(float) if name=='output' else None
        if mask is not None:mask[::7]=0.
        expected=np.zeros_like(raw)
        active=raw[:valid,:weight.shape[1]]
        if mask is not None:active=active*mask.reshape(raw.shape)[:valid,:weight.shape[1]]
        expected[:valid,:weight.shape[0]]=active@weight.T+bias
        if valid>1 and (np.array_equal(raw[0],raw[1]) or np.array_equal(expected[0],expected[1])):raise AssertionError('distinct rows required')
        result[name]=dict(input=raw.ravel(),expected=expected.ravel(),valid_rows=valid,input_mask=mask,
            contract=dict(valid_rows=valid,distinct_physical_rows=valid>1,partial_tile=valid<layout.rows_per_tile,
                signed_public_source_mask=mask is not None,hostile_feature_and_inactive_row_padding=True,
                input_sha256=sha(raw.astype('<f8').tobytes()),expected_sha256=sha(expected.astype('<f8').tobytes()),
                input_minimum=float(raw.min()),input_maximum=float(raw.max()),
                expected_minimum=float(expected.min()),expected_maximum=float(expected.max())))
    return result
