"""Actual worker API orchestration with inert substitutes; no files/keys/backend."""
from types import SimpleNamespace as NS
from contextlib import contextmanager
from pathlib import Path
import sys
import pytest
import group_worker as w

@pytest.mark.parametrize('failure',[False,True])
def test_grouped_factory_scope_and_original_owner_retirement(tmp_path,monkeypatch,failure):
    events=[];ctx=object();v=object();manifest={};primary=KeyboardInterrupt('test cancellation')
    class Rotation:
        def clear(self,**kw):events.append(('rotation-clear',type(self).__name__))
    class GroupRotation(Rotation):pass
    class Compiled:
        def __init__(self,c,**kw):
            assert c is ctx and kw==dict(low_memory=True,preserve_dispatch_cache=True)
            self.rotation=Rotation();events.append('baseline')
        def _clear_working_caches(self):events.append(('owner-clear',type(self.rotation).__name__))
    class Grouped(Compiled):
        def __init__(self):self.rotation=GroupRotation()
    owner=Grouped()
    def create(**kw):
        assert kw==dict(max_results=16,max_result_bytes=2<<30,max_digit_bytes=512<<20,low_memory=True,preserve_dispatch_cache=True)
        events.append('factory');return owner
    @contextmanager
    def load(path,**kw):
        assert kw==dict(profile_bytes=b'p',plan_bytes=b'plan',dedicated_no_write=True)
        events.append('loader-enter')
        try:yield NS(ctx=ctx,grouped_arithmetic=create)
        finally:events.append('loader-exit')
    def ciphertext(*args,**kw):events.append('load-upload');return v,manifest
    monkeypatch.setitem(sys.modules,'transport',NS(load_ciphertext=ciphertext,manifest_digest=lambda m:'tilehash'))
    monkeypatch.setitem(sys.modules,'key_contexts',NS(load_server_grouped=load))
    monkeypatch.setitem(sys.modules,'encrypted_packed_qwen',NS(CompiledArithmetic=Compiled))
    monkeypatch.setitem(sys.modules,'encrypted_rotation',NS(CompiledRotation=Rotation))
    monkeypatch.setitem(sys.modules,'grouped_rotation',NS(GroupedCompiledRotation=GroupRotation))
    monkeypatch.setitem(sys.modules,'support',NS(require_secure_execution=lambda *a:None))
    monkeypatch.setattr(w.protocol,'imported_sources',lambda expected:{})
    monkeypatch.setattr(w,'write',lambda *a:None)
    def exercise(c,value,baseline,grouped,**kw):
        assert c is ctx and value is v and grouped is owner
        events.append('groups')
        if failure:raise primary
        return dict(passed=True)
    monkeypatch.setattr(w,'exercise',exercise)
    request=dict(batch=dict(requests=[dict(contract=dict(request_id='r'))]),tiles=[dict(manifest_sha256='tilehash')])
    report={};args=(request,dict(upload=tmp_path,public_keys=tmp_path),b'p',b'plan',report,tmp_path/'out',{})
    if failure:
        with pytest.raises(KeyboardInterrupt) as got:w.actual(*args)
        assert got.value is primary and not report.get('all_rotations_completed')
    else:
        w.actual(*args);assert report['all_rotations_completed'] and report['loader_cleanup_completed']
    assert events==['load-upload','loader-enter','baseline','factory','groups',('owner-clear','Rotation'),('rotation-clear','Rotation'),('owner-clear','GroupRotation'),('rotation-clear','GroupRotation'),'loader-exit']
