import pytest
from full_synthetic import build
import encrypted_qwen_full_model as runtime
@pytest.mark.parametrize("error",[OSError,KeyboardInterrupt,SystemExit,MemoryError])
def test_partial_setup_failure_always_retires_active(monkeypatch,error):
    base=runtime.QwenFullModel
    class FailingSetup(base):
        armed=False
        def __setattr__(self,name,value):
            if name=="_refs" and self.armed:
                object.__setattr__(self,"armed",False)
                raise error("chosen setup failure")
            super().__setattr__(name,value)
    monkeypatch.setattr(runtime,"QwenFullModel",FailingSetup)
    o,a,b,r,p,x=build(monkeypatch);o.armed=True
    with pytest.raises(error,match="chosen setup"):o.evaluate(x)
    assert not o._active and o._borrowed==() and not a.calls and not p.calls and not runtime._QUARANTINE
    o._full_guard()
