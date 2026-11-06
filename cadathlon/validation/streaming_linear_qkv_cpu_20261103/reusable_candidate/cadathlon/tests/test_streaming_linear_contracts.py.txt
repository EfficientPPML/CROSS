"""Backend-free client/preflight and keyless boundary checks; no encryption."""
from pathlib import Path
from types import SimpleNamespace
import importlib,json,os,subprocess,sys
import numpy as np
import pytest
ROOT=Path(os.environ.get('CROSS_STREAMING_ROOT',Path(__file__).resolve().parents[2]))
sys.path[:0]=[str(ROOT),str(ROOT/'cadathlon/common')]
from encrypted_streaming_linear import RowTileLayout,StreamingLinear

def test_client_codec_import_requires_no_crypto_backend():
 code='''import importlib.abc,sys
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in ('jax','jaxlib','jaxite_word','encrypted_packed_qwen','encrypted_packed_transformer'):raise AssertionError('crypto backend import: '+fullname)
sys.meta_path.insert(0,Deny())
import encrypted_streaming_linear as m
import numpy as np
layout=m.RowTileLayout(8192)
rows=np.arange(3*768,dtype=float).reshape(3,768)
np.testing.assert_array_equal(layout.unpack_client_rows(layout.pack_client_rows(rows),rows=3,width=768,padding_atol=0),rows)
'''
 env=os.environ.copy();env['PYTHONPATH']=str(ROOT/'cadathlon/common')
 completed=subprocess.run([sys.executable,'-c',code],env=env,text=True,capture_output=True)
 assert completed.returncode==0,completed.stderr

def test_declared_package_closure_includes_module():
 import package
 assert 'encrypted_streaming_linear.py' in package.COMMON_RUNTIME_FILES
 package.required_files(ROOT/'cadathlon/common',package.COMMON_RUNTIME_FILES)

@pytest.mark.parametrize('where',['context','cache','parameters'])
def test_secret_in_any_location_is_rejected_on_every_call(where):
 evaluator=object.__new__(StreamingLinear)
 evaluator.ctx=SimpleNamespace(secret_key=None,_param_cache=SimpleNamespace(secret_key=None),parameters={})
 if where=='parameters':evaluator.ctx.parameters['secret_key']=object()
 elif where=='cache':evaluator.ctx._param_cache.secret_key=object()
 else:evaluator.ctx.secret_key=object()
 with pytest.raises(ValueError,match='remain keyless'):evaluator._validate_ciphertext(np.ones(8192))

def test_plaintext_is_not_an_evaluator_input():
 evaluator=object.__new__(StreamingLinear)
 evaluator.ctx=SimpleNamespace(secret_key=None,_param_cache=SimpleNamespace(secret_key=None),parameters={})
 with pytest.raises(TypeError,match='canonical Polynomial ciphertext'):evaluator._validate_ciphertext(np.ones(8192))

@pytest.mark.parametrize('options',[{'low_memory':1},{'preserve_dispatch_cache':'yes'}])
def test_options_rejected_before_backend_construction(options):
 with pytest.raises(TypeError,match='explicit booleans'):StreamingLinear(SimpleNamespace(num_slots=8192),RowTileLayout(8192),**options)

def test_inconsistent_context_slots_rejected():
 with pytest.raises(ValueError,match='context slots'):StreamingLinear(SimpleNamespace(num_slots=65536),RowTileLayout(8192))

@pytest.mark.parametrize('kind',['weight_complex','bias_complex','mask_complex','weight_nan','bias_nan','mask_nan','weight_shape','bias_shape','mask_shape'])
def test_malformed_public_data_rejected_before_ciphertext_access(kind):
 evaluator=object.__new__(StreamingLinear);evaluator.layout=RowTileLayout(8192)
 evaluator._validate_ciphertext=lambda value:(_ for _ in ()).throw(AssertionError('ciphertext accessed'))
 w=np.eye(3);b=np.arange(3.);m=np.ones(8192)
 if kind=='weight_complex':w=w.astype(complex)+1j
 if kind=='bias_complex':b=b.astype(complex)+1j
 if kind=='mask_complex':m=m.astype(complex)+1j
 if kind=='weight_nan':w[0,0]=np.nan
 if kind=='bias_nan':b[0]=np.nan
 if kind=='mask_nan':m[0]=np.nan
 if kind=='weight_shape':w=np.ones(3)
 if kind=='bias_shape':b=np.ones(4)
 if kind=='mask_shape':m=np.ones(4)
 with pytest.raises(ValueError):evaluator.linear_tile(object(),w,b,valid_rows=1,input_mask=m)

@pytest.mark.parametrize('slots,rows',[(8192,2),(8192,3),(65536,16),(65536,19)])
def test_exact_public_tile_contract(slots,rows):
 layout=RowTileLayout(slots);per=slots//4096;count=(rows+per-1)//per
 contract=layout.linear_contract(rows=rows,input_width=768,output_width=3072)
 assert contract=={'schema':'distinct-row-linear-v1','slots':slots,'features':4096,'rows':rows,'input_width':768,'output_width':3072,'rows_per_tile':per,'tile_count':count,'last_tile_rows':rows-(count-1)*per,'row_order':'contiguous-input-order','feature_padding':'zero','inactive_rows':'zero-including-bias'}

@pytest.mark.parametrize('kwargs',[{'rows':True,'input_width':768,'output_width':768},{'rows':2,'input_width':0,'output_width':768},{'rows':2,'input_width':768,'output_width':4097}])
def test_invalid_contract_shape(kwargs):
 with pytest.raises(ValueError):RowTileLayout(8192).linear_contract(**kwargs)

def test_partial_physical_ring_context_rejected_before_arithmetic():
 with pytest.raises(ValueError,match='every physical CKKS slot'):
  StreamingLinear(SimpleNamespace(num_slots=8192,degree=32768),RowTileLayout(8192))

def test_batch_context_rejected_before_arithmetic():
 with pytest.raises(ValueError,match='batch-one'):
  StreamingLinear(SimpleNamespace(num_slots=8192,degree=16384,_param_cache=SimpleNamespace(batch=2)),RowTileLayout(8192))
