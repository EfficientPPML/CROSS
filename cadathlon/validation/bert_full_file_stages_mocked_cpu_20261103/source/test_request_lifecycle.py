"""Completed-request cleanup uses tiny synthetic artifact files only."""
from pathlib import Path
import json
import shutil
import pytest
import request_lifecycle as lifecycle
from transport_contract import canonical


@pytest.fixture
def artifacts(tmp_path):
    paths={name:tmp_path/name for name in ('upload','download','server_report','decrypted_results','client_private')}
    paths['client_private'].mkdir();(paths['client_private']/'secret.npy').write_bytes(b'private sentinel')
    nonce='c'*32;batch=dict(public=dict(context_id='a'*64),requests=[dict(contract=dict(request_id=nonce))])
    for kind in ('upload','download'):
        path=paths[kind];(path/nonce).mkdir(parents=True)
        tiles=[path/nonce/f'tile{i}' for i in range(8)] if kind=='upload' else [path/nonce]
        for tile in tiles:
            tile.mkdir(exist_ok=True)
            for name in ('bundle.json','ct_0.npz','transport.json'):(tile/name).write_bytes(b'placeholder')
        (path/'batch.json').write_bytes(canonical(batch if kind=='upload' else dict(response='test')))
    paths['server_report'].write_bytes(canonical({'seconds':1.}))
    paths['decrypted_results'].write_bytes(canonical({'target_idx':7,'he_logits':[0.,1.]}))
    (paths['client_private']/'requests.json').write_bytes(canonical({'private_request':'test'}))
    return paths,batch


def record(artifacts):
    paths,batch=artifacts
    return lifecycle.record_completed(paths,context_id='a'*64,batch=batch)


def test_completed_only_retirement_and_private_key_retained(artifacts):
    paths,batch=artifacts
    assert lifecycle.retire_completed(paths,context_id='a'*64) is False
    assert paths['upload'].exists()
    record(artifacts)
    assert lifecycle.retire_completed(paths,context_id='a'*64) is True
    assert all(not paths[n].exists() for n in ('upload','download','server_report','decrypted_results'))
    assert {p.name for p in paths['client_private'].iterdir()}=={'secret.npy'}
    assert (paths['client_private']/'secret.npy').read_bytes()==b'private sentinel'
    assert lifecycle.retire_completed(paths,context_id='a'*64) is False


@pytest.mark.parametrize('kind',['context','parent','directory','file','extra','manifest','receipt','nonce','symlink',
    'nested_file','nested_directory','same_inode_payload'])
def test_changed_completed_artifacts_retained_before_any_deletion(artifacts,kind):
    paths,batch=artifacts;record(artifacts)
    receipt=paths['client_private']/'completed.json';context='a'*64
    if kind=='context':context='b'*64
    elif kind=='parent':
        parent=paths['download'].parent;old=parent.with_name(parent.name+'-old');parent.rename(old);parent.mkdir()
        # Restore only receipt and paths needed to load it; ownership differs.
        shutil.copytree(old/'client_private',parent/'client_private')
        shutil.copytree(old/'upload',parent/'upload')
    elif kind=='directory':
        paths['download'].rename(paths['download'].with_name('original'));paths['download'].mkdir()
        (paths['download']/'sentinel').write_text('keep')
    elif kind=='file':paths['server_report'].write_bytes(canonical({'seconds':2.}))
    elif kind=='extra':(paths['download']/'sentinel').write_text('keep')
    elif kind=='manifest':(paths['download']/'batch.json').write_bytes(canonical({'changed':True}))
    elif kind=='symlink':
        p=paths['download']/'batch.json';p.unlink();p.symlink_to(paths['server_report'])
    elif kind in ('nested_file','same_inode_payload'):
        p=paths['upload']/('c'*32)/'tile0'/'ct_0.npz'
        if kind=='nested_file':p.rename(p.with_name('prior-payload'));p.write_bytes(b'replacement');p.with_name('prior-payload').unlink()
        else:p.write_bytes(b'new content')
    elif kind=='nested_directory':
        p=paths['upload']/('c'*32)/'tile0';old=p.with_name('prior-tile');p.rename(old)
        shutil.copytree(old,p);shutil.rmtree(old)
    else:
        value=json.loads(receipt.read_text())
        if kind=='nonce':value['nonces']=['../outside']
        else:value['extra']=True
        receipt.write_bytes(canonical(value))
    with pytest.raises((ValueError,OSError)):lifecycle.retire_completed(paths,context_id=context)
    assert paths['upload'].exists() and receipt.exists()


def test_partial_retirement_can_resume_without_deleting_replacement(artifacts,monkeypatch):
    paths,batch=artifacts;record(artifacts);original=lifecycle.shutil.rmtree
    def fail(name,**kw):
        if name=='download':raise OSError('interrupted cleanup')
        return original(name,**kw)
    with monkeypatch.context() as patch:
        patch.setattr(lifecycle.shutil,'rmtree',fail)
        with pytest.raises(OSError):lifecycle.retire_completed(paths,context_id='a'*64)
    assert not paths['upload'].exists() and paths['download'].exists()
    assert lifecycle.retire_completed(paths,context_id='a'*64)
    assert not paths['download'].exists()
