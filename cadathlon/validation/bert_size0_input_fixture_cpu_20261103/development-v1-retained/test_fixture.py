from pathlib import Path
import json
import random
import sys
import pytest

import build_fixture as f


def rows(n=408):
    return [dict(idx=7000+i, label=i%2, sentence1='First '+str(i), sentence2='Second '+str(i)) for i in range(n)]


def test_size0_and_exact_derived_seed(tmp_path):
    import numpy as np
    params=f.instance_params(tmp_path)
    assert params.get_batch_size()==1
    assert f.upstream_seed(42)==191664963
    assert f.upstream_seed(42)==f.upstream_seed(42)  # RNG is recreated in each upstream run.
    assert f.upstream_seed(42)==int(np.random.Generator(np.random.PCG64(42)).integers(0,0x7fffffff))


@pytest.mark.parametrize('count',[1,100,200,408,-1,500])
def test_upstream_selection_target_ids_order_and_labels(tmp_path,count):
    seed=f.upstream_seed(42); data=rows()
    samples=tmp_path/'samples';labels=tmp_path/'labels'
    f.export(data,samples,labels,count,seed)
    indices=list(range(408)) if count==-1 or count>=408 else sorted(random.Random(seed).sample(range(408),count))
    records=[json.loads(x) for x in samples.read_text().splitlines()]
    assert [r['target_idx'] for r in records]==[7000+i for i in indices]
    assert [int(x) for x in labels.read_text().splitlines()]==[i%2 for i in indices]
    assert samples.read_text()==''.join(json.dumps({'target_idx':7000+i,'sentence1':'First '+str(i),'sentence2':'Second '+str(i)})+'\n' for i in indices)


def test_seed_source_substitution_rejected(tmp_path):
    p=tmp_path/'run.py'
    p.write_text((f.ROOT/'upstream/harness/run_submission.py').read_text().replace('0x7fffffff','0x7ffffffe'))
    with pytest.raises(ValueError,match='derivation changed'):f.upstream_seed(42,p)


def test_exact_preprocess_main_preserves_order_and_fixed_call(tmp_path):
    params=f.instance_params(tmp_path);params.dataset_intermediate_dir().mkdir(parents=True)
    records=[dict(target_idx=i,sentence1='first',sentence2='second') for i in (99,12)]
    params.get_test_input_file().write_text('\n'+''.join(json.dumps(r)+'\n\n' for r in records))
    calls=[]
    def tokenizer(a,b,**kw):
        calls.append((a,b,kw));return dict(input_ids=[101]+[0]*127,token_type_ids=[0]*128,attention_mask=[1]+[0]*127)
    f.preprocess(tmp_path,tokenizer)
    output=[json.loads(x) for x in (params.io_intermediate_dir()/'client_preprocessed_input').read_text().splitlines()]
    assert [x['target_idx'] for x in output]==[99,12]
    assert calls==[('first','second',dict(max_length=128,padding='max_length',truncation=True))]*2
    for item in output:f.validate_tokens(item)


@pytest.mark.parametrize('field,value',[('target_idx',True),('target_idx',-1),('input_ids',[101]*127),('input_ids',[True]*128),('attention_mask',[2]*128),('token_type_ids',[2]*128)])
def test_token_validation_rejects_wrong_schema(field,value):
    record=dict(target_idx=3,input_ids=[101]+[0]*127,token_type_ids=[0]*128,attention_mask=[1]+[0]*127)
    record[field]=value
    with pytest.raises(ValueError):f.validate_tokens(record)


def test_no_model_guard_and_restoration():
    import torch
    from transformers import PreTrainedModel
    old=PreTrainedModel.__init__; call=torch.nn.Module.__call__
    with f.no_model_execution():
        with pytest.raises(RuntimeError,match='forbidden'):PreTrainedModel.__init__(None)
        with pytest.raises(RuntimeError,match='forbidden'):torch.nn.Module.__call__(None)
    assert PreTrainedModel.__init__ is old and torch.nn.Module.__call__ is call


def test_failed_prepare_preserves_error_and_removes_owned_private(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('x=1\n')
    monkeypatch.setattr(f,'source_paths',lambda:{'source.py':source})
    original=f.prepare
    def fail(work):
        (work/'private.txt').write_text('client private fixture')
        raise KeyboardInterrupt('primary fixture cancellation')
    monkeypatch.setattr(f,'prepare',fail)
    with pytest.raises(KeyboardInterrupt,match='primary fixture cancellation'):f.collect(tmp_path/'out')
    report=json.loads((tmp_path/'out/report.json').read_text())
    assert report['passed'] is False and report['private_workspace_removed_on_failure']
    assert not Path(report['private_workspace']).exists()
    assert 'client private fixture' not in (tmp_path/'out/report.json').read_text()


def test_report_publication_failure_does_not_mask_primary(tmp_path,monkeypatch):
    source=tmp_path/'source.py';source.write_text('x=1\n')
    monkeypatch.setattr(f,'source_paths',lambda:{'source.py':source})
    def fail(work):raise RuntimeError('primary failure')
    monkeypatch.setattr(f,'prepare',fail)
    original=f.write_json; count=[0]
    def publish(path,value):
        count[0]+=1
        if count[0]>1:raise OSError('final publication')
        return original(path,value)
    monkeypatch.setattr(f,'write_json',publish)
    with pytest.raises(RuntimeError,match='primary failure') as caught:f.collect(tmp_path/'out')
    assert any('final publication' in x for x in caught.value.__notes__)


def test_replacement_directory_is_not_deleted(tmp_path):
    original=tmp_path/'owned';original.mkdir()
    fd,identity=f.open_owned_directory(original)
    original.rename(tmp_path/'moved');original.mkdir();(original/'keep').write_text('keep')
    try:
        with pytest.raises(RuntimeError,match='replaced'):f.remove_owned_directory(original,fd,identity)
        assert (original/'keep').read_text()=='keep'
    finally:f.os.close(fd)


def test_archive_exact_inventory_bytes(tmp_path):
    source=tmp_path/'source.py';source.write_text('x=1\n')
    assert f.archive_sources({'source.py':source},tmp_path/'sources.tar.gz')=={'source.py':f.sha(source)}
