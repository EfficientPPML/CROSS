"""Synthetic layout checks with scalar-index oracles; no backend imports."""
from pathlib import Path
import ast
import gc
import hashlib
import json
import weakref
import numpy as np
import pytest
import qwen_layout as layout

HERE = Path(__file__).resolve().parent


def descriptors():
    return tuple(dict(index=i, start=4096*i, stop=min(4096*i+4096, 151936),
                      physical_row=0, level=2) for i in range(38))


@pytest.fixture(scope='module')
def decoded():
    # Every physical row differs: selecting or averaging aliases is observable.
    return tuple(np.frombuffer((np.arange(65536, dtype=np.float64) + i*1000000).tobytes(),
                               dtype=np.float64) for i in range(38))


@pytest.mark.parametrize('count', range(1, 33))
def test_every_length_independent_physical_index_oracle(count):
    source = np.arange(count*1024, dtype=np.float64).reshape(1, count, 1024)
    positions = tuple(7 + i*11 for i in range(count))
    result = layout.place_embeddings(source, positions)
    assert type(result) is tuple and len(result) == (count+15)//16
    for tile_index, tile in enumerate(result):
        slot = np.arange(65536)
        logical_row = 16*tile_index + slot//4096
        feature = slot%4096
        active = (logical_row<count) & (feature<1024)
        expected = np.zeros(65536, dtype=np.float64)
        expected[active] = logical_row[active]*1024 + feature[active]
        np.testing.assert_array_equal(tile.view(np.uint64), expected.view(np.uint64))
        assert tile.shape == (65536,) and tile.dtype == np.dtype('float64')
        assert not tile.flags.writeable and not np.shares_memory(tile, source)
    assert positions == tuple(7 + i*11 for i in range(count))


def test_noncontiguous_embeddings_signed_zero_subnormal_and_independent_storage():
    source = np.arange(1*19*2048, dtype=np.float64).reshape(1,19,2048)[:, :, ::-2]
    source[0, 0, :4] = [-0.0, 0.0, np.nextafter(0., 1.), -np.nextafter(0., 1.)]
    saved = source.copy()
    result = layout.place_embeddings(source, tuple(range(1,39,2)))
    assert not source.flags.c_contiguous and not np.shares_memory(result[0],result[1])
    source.fill(999.)
    for row in range(19):
        start = (row%16)*4096
        np.testing.assert_array_equal(result[row//16][start:start+1024].view(np.uint64),
                                      saved[0,row].view(np.uint64))
    for tile in result:
        with pytest.raises(ValueError):tile.setflags(write=True)
        with pytest.raises(ValueError):tile[0] = 7.


@pytest.mark.parametrize('positions', [(), tuple(range(33)), [0], (True,), (np.int64(0),),
                                     (0.,), (-1,), (40960,), (1,1), (3,2)])
def test_invalid_position_metadata(positions):
    with pytest.raises(ValueError):layout.place_embeddings(np.zeros((1,1,1024)), positions)


@pytest.mark.parametrize('shape', [(1,0,1024),(2,1,1024),(1,1,1023),(1,1,1025),
                                 (1,1024),(1,2,1024),(1,1,1024,1)])
def test_embedding_wrong_shapes(shape):
    with pytest.raises(ValueError):layout.place_embeddings(np.zeros(shape), (0,))


@pytest.mark.parametrize('dtype', ['float32','int64','complex128','object','>f8'])
def test_embedding_rejects_implicit_dtype_conversion(dtype):
    with pytest.raises(TypeError):layout.place_embeddings(np.zeros((1,1,1024), dtype=dtype),(0,))


@pytest.mark.parametrize('value', [float('nan'),float('inf'),-float('inf')])
def test_embedding_nonfinite(value):
    source=np.zeros((1,1,1024));source[0,0,1023]=value
    with pytest.raises(ValueError):layout.place_embeddings(source,(0,))


def test_no_array_protocol_or_subclass_execution():
    class Poison:
        def __array__(self,*args,**kwargs):raise AssertionError('must not invoke array conversion')
    class Subclass(np.ndarray):pass
    with pytest.raises(TypeError):layout.place_embeddings(Poison(),(0,))
    with pytest.raises(TypeError):layout.place_embeddings(np.zeros((1,1,1024)).view(Subclass),(0,))


def test_vocabulary_independent_global_index_oracle_and_final384(decoded):
    result=layout.read_vocabulary(decoded,descriptors())
    index=np.arange(151936)
    expected=(index//4096)*1000000 + index%4096
    np.testing.assert_array_equal(result,expected.astype(np.float64)[None,:])
    assert result.shape==(1,151936) and result.dtype==np.dtype('float64')
    assert result[0,-1]==37000383. and not result.flags.writeable
    assert not any(np.shares_memory(result,value)for value in decoded)
    with pytest.raises(ValueError):result.setflags(write=True)
    with pytest.raises(ValueError):result[0,0]=1.


def test_readout_no_alias_averaging_and_input_ownership(decoded):
    mutable=decoded[0].copy();reference=weakref.ref(mutable)
    blocks=(mutable,)+decoded[1:]
    out=layout.read_vocabulary(blocks,descriptors())
    old=out.copy();mutable.fill(-101.)
    np.testing.assert_array_equal(out,old)
    del mutable,blocks;gc.collect();assert reference()is None
    # A shared zero/vector input is legal host data, not authority for a CT
    # identity. Repeated inputs are read literally; no storage is borrowed.
    same=np.arange(65536,dtype=np.float64)
    result=layout.read_vocabulary((same,)*38,descriptors())
    expected=np.arange(151936)%4096
    np.testing.assert_array_equal(result,expected[None,:])


def test_readout_noncontiguous_vectors_and_exact_bits(decoded):
    array=np.zeros(131072,dtype=np.float64);view=array[::2]
    view[:4]=[-0.,0.,np.nextafter(0.,1.),np.finfo(np.float64).max]
    result=layout.read_vocabulary((view,)+decoded[1:],descriptors())
    np.testing.assert_array_equal(result[0,:4].view(np.uint64),view[:4].view(np.uint64))


@pytest.mark.parametrize('block,slot', [(0,0),(37,383),(37,384),(0,4096),(37,65535)])
@pytest.mark.parametrize('value',[float('nan'),float('inf'),-float('inf')])
def test_nonfinite_rejected_in_every_physical_region(decoded,block,slot,value):
    bad=decoded[block].copy();bad[slot]=value;rows=list(decoded);rows[block]=bad
    with pytest.raises(ValueError):layout.read_vocabulary(tuple(rows),descriptors())


@pytest.mark.parametrize('shape',[(1,65536),(65535,),(65537,),(16,4096),(65536,1)])
def test_readout_wrong_shape_no_implicit_batch_squeeze(decoded,shape):
    with pytest.raises(ValueError):layout.read_vocabulary((np.zeros(shape),)+decoded[1:],descriptors())


@pytest.mark.parametrize('dtype',['float32','complex128','int64','object','>f8'])
def test_readout_wrong_dtype(decoded,dtype):
    with pytest.raises(TypeError):layout.read_vocabulary((np.zeros(65536,dtype=dtype),)+decoded[1:],descriptors())


@pytest.mark.parametrize('change', ['short','long','list','none','subclass'])
def test_complete_exact_block_tuple_required(decoded,change):
    class TupleSubclass(tuple):pass
    wrong={'short':decoded[:-1],'long':decoded+(decoded[0],),'list':list(decoded),
           'none':None,'subclass':TupleSubclass(decoded)}[change]
    with pytest.raises(ValueError):layout.read_vocabulary(wrong,descriptors())


@pytest.mark.parametrize('change',['short','long','list','reversed','duplicate','missing','extra',
                                   'lastfull','lastshort','row','level','bool','numpyint','float','dictsubclass'])
def test_exact_ordered_descriptor_contract(decoded,change):
    class DictSubclass(dict):pass
    rows=list(descriptors())
    if change=='short':rows=rows[:-1]
    elif change=='long':rows.append(rows[-1])
    elif change=='list':
        with pytest.raises(ValueError):layout.read_vocabulary(decoded,rows)
        return
    elif change=='reversed':rows.reverse()
    elif change=='duplicate':rows[-1]=rows[0]
    elif change=='missing':del rows[0]['level']
    elif change=='extra':rows[0]['other']=0
    elif change=='lastfull':rows[-1]['stop']=38*4096
    elif change=='lastshort':rows[-1]['stop']=151935
    elif change=='row':rows[0]['physical_row']=1
    elif change=='level':rows[0]['level']=6
    elif change=='bool':rows[0]['index']=False
    elif change=='numpyint':rows[0]['index']=np.int64(0)
    elif change=='float':rows[0]['index']=0.
    elif change=='dictsubclass':rows[0]=DictSubclass(rows[0])
    with pytest.raises(ValueError):layout.read_vocabulary(decoded,tuple(rows))


def test_public_provenance_and_actual_descriptor_expression():
    provenance=json.loads((HERE/'source_provenance.json').read_bytes())
    for name,row in provenance['sources'].items():
        body=(HERE/'provenance'/name).read_bytes()
        assert len(body)==row['bytes'] and hashlib.sha256(body).hexdigest()==row['sha256']
    # Execute only the public pure literal descriptor function extracted as
    # AST; never import the adapter or a backend module.
    tree=ast.parse((HERE/'provenance/qwen_file_contract.py').read_text())
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='vocabulary')
    namespace={};exec(compile(ast.Module(body=[node],type_ignores=[]),'public-vocabulary-AST','exec'),namespace)
    assert tuple(namespace['vocabulary']())==descriptors()
    head=ast.parse((HERE/'provenance/encrypted_qwen_head_routing.py').read_text())
    cls=next(n for n in head.body if isinstance(n,ast.ClassDef)and n.name=='QwenHeadLayout')
    defaults={n.target.id:ast.literal_eval(n.value)for n in cls.body if isinstance(n,ast.AnnAssign)and n.value is not None}
    assert defaults['slots']==65536 and defaults['row_width']==4096


def test_decoder_float64_vector_facade_is_source_bound():
    tree=ast.parse((HERE/'provenance/ckks_ctx.py').read_text())
    methods={n.name:n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)}
    facade=methods['decrypt_slots'];ret=next(n for n in facade.body if isinstance(n,ast.Return)).value
    assert isinstance(ret,ast.Subscript)and ast.literal_eval(ret.slice)==0
    assert ret.value.func.id=='_fast_decrypt_decode'
    assert any(k.arg=='require_single_batch'and ast.literal_eval(k.value)is True for k in ret.value.keywords)
    batch=methods['_fast_decrypt_decode'];ret=next(n for n in batch.body if isinstance(n,ast.Return)).value
    assert ast.unparse(ret.func)=='np.stack'
    single=methods['_fast_decode']
    assert any(isinstance(n,ast.keyword)and n.arg=='dtype'and ast.unparse(n.value)=='np.float64'for n in ast.walk(single))
    assert '(batch, vocab_size) float64 logits' in (HERE/'provenance/qwen_he.py').read_text()
