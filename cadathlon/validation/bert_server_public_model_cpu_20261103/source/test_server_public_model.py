"""Full196-tensor extraction/readback and meaningful public-cache regressions."""
from pathlib import Path
import copy,hashlib,json,os,shutil,sys
import numpy as np
import pytest
import server_public_model as public
SNAPSHOT=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
CANDIDATE=(Path(__file__).parent/'server_inputs/candidate_manifest.json').read_bytes()
PROFILE=(Path(__file__).parent/'server_inputs/p60_profile_and_refresh.json').read_bytes()
CONTEXT='0123456789abcdef'*4
@pytest.fixture(scope='module')
def cache(tmp_path_factory):
    path=tmp_path_factory.mktemp('bert_public_cache')/'cache'
    manifest=public.preprocess(SNAPSHOT,path,context_id=CONTEXT,candidate_bytes=CANDIDATE,profile_bytes=PROFILE)
    return path,manifest

def test_complete_public_server_tensors_and_exact_readback(cache):
    path,manifest=cache;arrays,checked=public.load(path,context_id=CONTEXT)
    assert checked==manifest and len(arrays)==196 and len(public.TENSOR_SHA256)==196
    assert len([n for n in arrays if 'embeddings' in n])==0
    assert sum(v.ndim==2 for v in arrays.values())==74
    for name,value in arrays.items():
        assert not value.flags.writeable and value.dtype==np.float64 and value.shape==public.tensor_shapes()[name]
        assert hashlib.sha256(value.astype('<f4').tobytes()).hexdigest()==public.TENSOR_SHA256[name]
    assert not any(n in sys.modules for n in ('jax','torch','transformers','jaxite_word'))
    dest=os.environ.get('SERVER_MODEL_CASE_REPORT')
    if dest:Path(dest).write_text(json.dumps(dict(passed=True,tensors=196,matrices=74,vector_tensors=122,model_id=manifest['model_id'],context_id=CONTEXT,source_checkpoint_sha256=public.CHECKPOINT_SHA256,raw_tensor_bytes=sum(v.size*4 for v in arrays.values()),immutable_exact_fp64_lifts=True,no_backend_or_model_forward=True),indent=2)+'\n')

def mutate_manifest(path,m):
    m['model_id']=public.digest({k:v for k,v in m.items() if k!='model_id'})
    (path/'model_manifest.json').write_bytes(public.canonical(m))

def clone_manifest(cache,tmp_path):
    original,manifest=cache;path=tmp_path/'cache';path.mkdir()
    for source in original.iterdir():
        if source.name=='model_manifest.json':shutil.copyfile(source,path/source.name)
        else:os.link(source,path/source.name)
    return path,copy.deepcopy(manifest)
@pytest.mark.parametrize('kind',('context','circuit','packing_type','missing_tensor','extra_tensor','tensor_shape','untrusted_tensor_hash','traversal','extra_file'))
def test_rehashed_manifest_cannot_change_fixed_contract(cache,tmp_path,kind):
    path,m=clone_manifest(cache,tmp_path);first=next(iter(m['tensors']))
    if kind=='context':m['context_id']='f'*64
    elif kind=='circuit':m['candidate_sha256']='f'*64
    elif kind=='packing_type':m['packing']['encoder_layers']=12.
    elif kind=='missing_tensor':m['tensors'].pop(first)
    elif kind=='extra_tensor':m['tensors']['secret_key']={}
    elif kind=='tensor_shape':m['tensors'][first]['shape']=[2,768]
    elif kind=='untrusted_tensor_hash':m['tensors'][first]['tensor_sha256']='f'*64
    elif kind=='traversal':m['tensors'][first]['file']='../secret.npy'
    else:(path/'extra.bin').write_bytes(b'extra')
    mutate_manifest(path,m)
    with pytest.raises((ValueError,OSError)):public.load(path,context_id=CONTEXT)
@pytest.mark.parametrize('kind',('payload','symlink','duplicate_json'))
def test_actual_file_rejections(cache,tmp_path,kind):
    path,m=clone_manifest(cache,tmp_path)
    if kind=='duplicate_json':
        raw=(path/'model_manifest.json').read_bytes();(path/'model_manifest.json').write_bytes(b'{"schema":"ignored",'+raw[1:])
    else:
        name='classifier.bias';desc=m['tensors'][name];target=path/desc['file'];target.unlink()
        if kind=='symlink':target.symlink_to(cache[0]/desc['file'])
        else:
            np.save(target,np.zeros(2,dtype='<f4'),allow_pickle=False)
            # Even rehashing every self-authenticated field cannot impersonate
            #the pinned public learned tensor.
            desc['file_sha256']=public.file_sha(target);desc['tensor_sha256']=hashlib.sha256(np.zeros(2,dtype='<f4').tobytes()).hexdigest();mutate_manifest(path,m)
    with pytest.raises((ValueError,OSError)):public.load(path,context_id=CONTEXT)
@pytest.mark.parametrize('field',('candidate','profile','context'))
def test_invalid_preprocessing_identity_writes_nothing(tmp_path,field):
    output=tmp_path/'cache';kwargs=dict(context_id=CONTEXT,candidate_bytes=CANDIDATE,profile_bytes=PROFILE)
    if field=='context':kwargs['context_id']=True
    else:kwargs[field+'_bytes']+=b' '
    with pytest.raises(ValueError):public.preprocess(SNAPSHOT,output,**kwargs)
    assert not output.exists()

def test_existing_output_is_never_replaced(cache):
    path,m=cache;before=public.file_sha(path/'model_manifest.json')
    with pytest.raises(FileExistsError):public.preprocess(SNAPSHOT,path,context_id=CONTEXT,candidate_bytes=CANDIDATE,profile_bytes=PROFILE)
    assert public.file_sha(path/'model_manifest.json')==before

@pytest.mark.parametrize('kind',('primary_only','cleanup_error','cleanup_and_close_error','replacement','moved'))
def test_owned_cleanup_preserves_primary_and_replacement(tmp_path,monkeypatch,kind):
    output=tmp_path/'cache';moved=tmp_path/'moved_original';primary=KeyboardInterrupt('primary interruption');real_close=public.os.close
    def fail_save(*args,**kwargs):
        if kind in ('replacement','moved'):
            output.rename(moved)
            if kind=='replacement':output.mkdir();(output/'unowned-sentinel').write_text('keep')
        raise primary
    def fail_cleanup(*args,**kwargs):raise OSError('secondary cleanup')
    def fail_close(fd):real_close(fd);raise OSError('secondary close')
    monkeypatch.setattr(public.np,'save',fail_save)
    if kind in ('cleanup_error','cleanup_and_close_error'):monkeypatch.setattr(public.shutil,'rmtree',fail_cleanup)
    if kind=='cleanup_and_close_error':monkeypatch.setattr(public.os,'close',fail_close)
    with pytest.raises(KeyboardInterrupt) as caught:public.preprocess(SNAPSHOT,output,context_id=CONTEXT,candidate_bytes=CANDIDATE,profile_bytes=PROFILE)
    assert caught.value is primary
    if kind=='primary_only':assert not output.exists()
    if kind=='replacement':assert (output/'unowned-sentinel').read_text()=='keep' and moved.is_dir()
    if kind=='moved':assert not output.exists() and moved.is_dir()
    if kind!='primary_only':assert primary.__notes__
    if kind=='cleanup_and_close_error':assert any('handle close' in n for n in primary.__notes__)

def test_identity_checked_before_manifest_publication(tmp_path,monkeypatch):
    output=tmp_path/'cache';moved=tmp_path/'moved';real_save=public.np.save;calls=[]
    def save_then_move(*args,**kwargs):
        real_save(*args,**kwargs);calls.append(1)
        if len(calls)==196:output.rename(moved);output.mkdir();(output/'unowned').write_text('preserve')
    monkeypatch.setattr(public.np,'save',save_then_move)
    with pytest.raises(RuntimeError,match='moved or replaced'):public.preprocess(SNAPSHOT,output,context_id=CONTEXT,candidate_bytes=CANDIDATE,profile_bytes=PROFILE)
    assert (output/'unowned').read_text()=='preserve' and not (output/'model_manifest.json').exists()
    assert moved.is_dir() and not (moved/'model_manifest.json').exists()

@pytest.mark.parametrize('kind',('huge_shape','object_dtype','fortran','trailing','version2'))
def test_npy_header_is_bounded_before_any_array_allocation(cache,tmp_path,monkeypatch,kind):
    import io
    path,m=clone_manifest(cache,tmp_path);record=m['tensors']['classifier.bias'];target=path/record['file'];target.unlink()
    shape=(2**60,) if kind=='huge_shape' else (2,)
    header=dict(descr='|O' if kind=='object_dtype' else '<f4',fortran_order=kind=='fortran',shape=shape)
    stream=io.BytesIO()
    (np.lib.format.write_array_header_2_0 if kind=='version2' else np.lib.format.write_array_header_1_0)(stream,header)
    raw=stream.getvalue()+bytes(8)+(b'extra' if kind=='trailing' else b'');target.write_bytes(raw)
    record['file_sha256']=hashlib.sha256(raw).hexdigest();mutate_manifest(path,m)
    def forbidden(*args,**kwargs):raise AssertionError('unvalidated header reached np.load')
    monkeypatch.setattr(public.np,'load',forbidden)
    with pytest.raises(ValueError,match='NPY version|header shape'):public.load(path,context_id=CONTEXT)
