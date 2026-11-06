"""Mocked resource lifecycle only; no encrypted/numerical performance claim."""
from types import SimpleNamespace as NS
import pytest
import full_model_engine as engine


@pytest.fixture
def fake(monkeypatch):
    calls=[];state=NS(fail=None,cleanup_failure=False)
    ctx=NS(secret_key=None,_param_cache=NS(secret_key=None),parameters={})
    class Helper:
        def __init__(self,name):self.name=name
        def clear(self,*,executables=False):
            calls.append(('clear',self.name,executables))
            if state.cleanup_failure:raise OSError('cleanup failed')
        def clear_executables(self):self.clear(executables=True)
    class Arithmetic:
        def __init__(self,ctx,**kw):
            calls.append(('arithmetic',kw));self.ctx=ctx
            self.rotation=Helper('rotation');self._plaintext_multiply=Helper('plain')
            self._rescale=Helper('rescale');self._multiplication_compiler=Helper('multiply')
        def _clear_working_caches(self):calls.append(('working',))
    class Linear:
        def __init__(self,ctx,layout,**kw):
            if state.fail=='linear':raise RuntimeError('linear constructor')
            calls.append(('linear',layout,kw));self.arithmetic=Arithmetic(ctx)
    class Refresh:
        def __init__(self,ctx,**kw):
            if state.fail=='refresh':raise RuntimeError('refresh constructor')
            calls.append(('refresh',kw));self.evaluator=Arithmetic(ctx)
            self.compiled_primitives=tuple(Helper('refresh'+str(i)) for i in range(5))
    class Owner:
        def __init__(self,arithmetic,linear,refresh,weights,manifest_bytes,**kw):
            calls.append(('owner',weights,manifest_bytes,kw))
        def evaluate(self,tiles,**kw):
            calls.append(('evaluate',tiles,kw))
            return NS(output=NS(polynomial=NS(block_until_ready=lambda:calls.append(('sync',)))))
    def secure(ctx,value):calls.append(('secure',value))
    def enable(ctx):
        calls.append(('provider',))
        return NS(close=lambda:calls.append(('close_provider',)))
    monkeypatch.setattr(engine,'_dependencies',lambda:(enable,Arithmetic,Linear,lambda n:n,Refresh,Owner,secure))
    return ctx,calls,state


def test_real_graph_boundary_and_complete_cleanup(fake):
    ctx,calls,state=fake;tiles=tuple(object() for _ in range(8));weights=object()
    with engine.server_engine(ctx,manifest_bytes=b'fixed') as graph:
        result=graph.evaluate(tiles,weights,original_positions=[0,7],context_id='ctx',request_id='nonce')
        evaluate=next(c for c in calls if c[0]=='evaluate')
        assert evaluate[1] is tiles and evaluate[2]==dict(capture_checkpoints=False)
        owner=next(c for c in calls if c[0]=='owner')
        assert owner[1] is weights and owner[3]['original_positions']==(0,7)
        secure=[c for c in calls if c[0]=='secure']
        assert [c[1] for c in secure[:-1]]==list(tiles) and secure[-1][1] is result.output
        assert calls.index(('sync',))<calls.index(secure[-1])
    refresh=next(c[1] for c in calls if c[0]=='refresh')
    assert refresh==dict(period=65536,level_budget=(4,4),public_bound=2.,meta_precision=10,
                         compiled=True,low_memory=True,preserve_dispatch_cache=True)
    assert len([c for c in calls if c[0]=='clear'])==17
    assert all(c[2] is True for c in calls if c[0]=='clear')
    assert calls[-1]==('close_provider',)


@pytest.mark.parametrize('site',['ctx','cache','parameters'])
def test_secret_rejected_before_provider(fake,site):
    ctx,calls,state=fake
    if site=='ctx':ctx.secret_key=object()
    elif site=='cache':ctx._param_cache.secret_key=object()
    else:ctx.parameters['secret_key']=object()
    with pytest.raises(ValueError):
        with engine.server_engine(ctx,manifest_bytes=b'fixed'):pass
    assert calls==[]


@pytest.mark.parametrize('site',['linear','refresh','body'])
def test_partial_construction_and_cancellation_cleanup(fake,site):
    ctx,calls,state=fake;state.fail=site;error=KeyboardInterrupt('same primary')
    with pytest.raises(KeyboardInterrupt if site=='body' else RuntimeError) as caught:
        with engine.server_engine(ctx,manifest_bytes=b'fixed'):raise error
    if site=='body':assert caught.value is error
    assert calls[-1]==('close_provider',) and any(c[0]=='clear' for c in calls)


def test_all_cleanup_attempted_preserving_primary(fake):
    ctx,calls,state=fake;error=KeyboardInterrupt('same primary')
    with pytest.raises(KeyboardInterrupt) as caught:
        with engine.server_engine(ctx,manifest_bytes=b'fixed'):
            state.cleanup_failure=True;raise error
    assert caught.value is error and len(error.__notes__)==17
    assert calls[-1]==('close_provider',)


def test_cleanup_failure_without_primary_propagates(fake):
    ctx,calls,state=fake
    with pytest.raises(OSError):
        with engine.server_engine(ctx,manifest_bytes=b'fixed'):state.cleanup_failure=True
    assert calls[-1]==('close_provider',)
