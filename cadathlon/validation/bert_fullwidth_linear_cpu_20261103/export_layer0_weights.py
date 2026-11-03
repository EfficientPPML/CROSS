"""Export pinned public checkpoint arrays only; never repository weights."""
import hashlib, json
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file
snapshot=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
checkpoint=snapshot/'model.safetensors'
expected='e1f8085ef673c9eb8a1a93a84270742e8aa0ccbceadef01eef15e26c44b46034'
with checkpoint.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
assert actual==expected
values=load_file(checkpoint)
names=['bert.encoder.layer.0.attention.self.query.'+field for field in ('weight','bias')]
names += ['bert.encoder.layer.0.'+part+'.dense.'+field for part in ('intermediate','output') for field in ('weight','bias')]
short_names=['query_weight','query_bias','intermediate_weight','intermediate_bias','output_weight','output_bias']
arrays={short:np.ascontiguousarray(values[name]) for short,name in zip(short_names,names)}
expected_shapes=[(768,768),(768,),(3072,768),(3072,),(768,3072),(768,)]
assert [arrays[name].shape for name in short_names]==expected_shapes
out=Path('/tmp/bert-layer0-linear-weights.npz')
with out.open('wb') as f: np.savez(f,**arrays)
with out.open('rb') as f: filehash=hashlib.file_digest(f,'sha256').hexdigest()
report={'scope':'Public pinned pretrained BERT layer0 linear weights; no inference.',
'model_revision':snapshot.name,'checkpoint_file':str(checkpoint),'checkpoint_sha256':actual,
'original_tensor_names':dict(zip(short_names,names)),
'export_file':str(out),'export_sha256':filehash,
'array_hash_convention':'SHA256 over C-order ndarray raw bytes; shape/dtype recorded separately; original checkpoint dtype preserved.',
'arrays':{name:{'shape':list(value.shape),'dtype':value.dtype.str,'sha256':hashlib.sha256(value.tobytes(order='C')).hexdigest()} for name,value in arrays.items()},
'export_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out.with_suffix('.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
