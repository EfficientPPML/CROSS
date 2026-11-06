"""Entrypoint and cancellation regressions; no CKKS or model execution."""
from types import SimpleNamespace as NS
import sys
import pytest
import boundaries


@pytest.fixture
def fake(monkeypatch):
    def original(*args,**kwargs):return 'original'
    class Context:pass
    for n in boundaries.DECODE_METHODS+('encrypt_slots',):setattr(Context,n,original)
    model=NS(**{n:type(n,(),{'__init__':original}) for n in boundaries.MODEL_CLASSES})
    model.BertEmbeddings=type('BertEmbeddings',(),{'__init__':original})
    modules={n:NS(__name__=n,**{m:original for m in boundaries.KEY_METHODS}) for n in ('key_gen','jaxite_word.key_gen')}
    imports=modules|{'transformers.models.bert.modeling_bert':model}
    monkeypatch.setitem(sys.modules,'jaxite_word',NS(CKKSContext=Context))
    monkeypatch.setattr(boundaries.importlib,'import_module',imports.__getitem__)
    return NS(original=original,Context=Context,model=model,modules=modules)


def test_all_forbidden_entrypoints_and_allowed_embedding_encryption(fake):
    calls=[]
    with boundaries.encryption_only(calls):
        for module in fake.modules.values():
            for name in boundaries.KEY_METHODS:
                with pytest.raises(RuntimeError):getattr(module,name)()
        for name in boundaries.DECODE_METHODS:
            with pytest.raises(RuntimeError):getattr(fake.Context,name)(None)
        for name in boundaries.MODEL_CLASSES:
            with pytest.raises(RuntimeError):getattr(fake.model,name).__init__(None)
        assert fake.Context.encrypt_slots(None)=='original'
        assert fake.model.BertEmbeddings.__init__(None)=='original'
    assert len(calls)==21
    assert all(getattr(m,n)is fake.original for m in fake.modules.values() for n in boundaries.KEY_METHODS)
    assert all(getattr(fake.Context,n)is fake.original for n in boundaries.DECODE_METHODS)
    assert all(getattr(fake.model,n).__init__ is fake.original for n in boundaries.MODEL_CLASSES)


@pytest.mark.parametrize('error',(ValueError('primary'),KeyboardInterrupt('cancelled'),SystemExit(7)))
def test_primary_preserved_and_guards_restored(fake,error):
    with pytest.raises(type(error)) as caught:
        with boundaries.encryption_only([]):raise error
    assert caught.value is error
    assert all(getattr(m,n)is fake.original for m in fake.modules.values() for n in boundaries.KEY_METHODS)
    assert all(getattr(fake.Context,n)is fake.original for n in boundaries.DECODE_METHODS)


def test_partial_setup_restores_earlier_patches(fake):
    delattr(fake.model,boundaries.MODEL_CLASSES[-1])
    with pytest.raises(AttributeError):
        with boundaries.encryption_only([]):pytest.fail('entered after incomplete guard setup')
    assert all(getattr(m,n)is fake.original for m in fake.modules.values() for n in boundaries.KEY_METHODS)
    assert all(getattr(fake.Context,n)is fake.original for n in boundaries.DECODE_METHODS)


def test_missing_alias_fails_before_patch(fake):
    fake.modules['key_gen'].__name__='wrong'
    with pytest.raises(ValueError):
        with boundaries.encryption_only([]):pytest.fail('unverified alias')
    assert all(getattr(m,n)is fake.original for m in fake.modules.values() for n in boundaries.KEY_METHODS)
