import hashlib,importlib.util,json,sys,time
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
path=Path('/tmp/streaming-bert-layout/row_attention.py');before=hashlib.sha256(path.read_bytes()).hexdigest()
spec=importlib.util.spec_from_file_location('cnn_row',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.path.insert(0,str(HERE));import layout_checks as independent
start=time.monotonic();rng=np.random.default_rng(9327);engine=m.RowPackedAttention();checks=[]
tile=np.arange(65536.).reshape(16,4096)
expected=np.zeros_like(tile)
for row in range(16):
 for head in range(12):expected[row,head*64:(head+1)*64]=tile[row,head*64:(head+1)*64].sum()
np.testing.assert_array_equal(engine.head_broadcast(engine.head_reduce(tile.ravel())).reshape(16,4096),expected)
checks.append({'name':'head_reduce_broadcast_hostile_padding','exact':True})
for offset in range(16):np.testing.assert_array_equal(engine.rotate_rows(tile.ravel(),offset).reshape(16,4096),np.roll(tile,-offset,axis=0))
checks.append({'name':'all16_cyclic_tile_offsets','exact':True})
for sequence in (17,128):
 q,k,v=(rng.normal(0,.2,(sequence,768)) for _ in range(3))
 cases=[('allvalid',np.ones(sequence,bool)),('singlelast',np.arange(sequence)==sequence-1),('interleaved',np.arange(sequence)%3==1)]
 for name,mask in cases:
  e=m.RowPackedAttention();actual=e.attention(q,k,v,mask)
  qq=q.reshape(sequence,12,64).transpose(1,0,2);kk=k.reshape(sequence,12,64).transpose(1,0,2);vv=v.reshape(sequence,12,64).transpose(1,0,2)
  scores=qq@kk.swapaxes(-1,-2)/8;scores=np.where(mask[None,None,:],scores,-np.inf)
  z=np.exp(scores-scores.max(axis=-1,keepdims=True));expected=((z/z.sum(axis=-1,keepdims=True))@vv).transpose(1,0,2).reshape(sequence,768)
  np.testing.assert_allclose(actual,expected,atol=2e-14,rtol=0)
  plan=e.plan(sequence)
  for key,val in plan['expected_counts'].items():assert e.counts[key]==val,(key,e.counts[key],val)
  checks.append({'name':f'{sequence}_{name}','max_abs':float(np.max(np.abs(actual-expected))),'counts_match':True,'padded_query_max_abs':float(np.max(np.abs(actual[~mask]-expected[~mask]))) if (~mask).any() else None})
report={'passed':True,'scope':'Independent CNN simulator boundary and dense exact-softmax comparison, no encrypted or polynomial claim.','checks':checks,'seconds':time.monotonic()-start,'cnn_source_sha256':before,'cnn_source_unchanged':before==hashlib.sha256(path.read_bytes()).hexdigest(),'test_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
assert report['cnn_source_unchanged'];(HERE/'check_cnn_layout.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
