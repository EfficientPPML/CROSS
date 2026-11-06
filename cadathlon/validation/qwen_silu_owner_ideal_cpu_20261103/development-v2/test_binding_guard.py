import numpy as np,pytest
from silu_ideal import owner,Value
import encrypted_qwen_silu as runtime
@pytest.fixture(autouse=True)
def clear_quarantine():
    runtime._QUARANTINE.clear();yield;runtime._QUARANTINE.clear()
@pytest.mark.parametrize('name',['refresh','check'])
@pytest.mark.parametrize('phase',['entry','first','final'])
def test_binding_accessor_replacement_rejected_before_next_primitive(name,phase):
    marks=[]
    def replace():
        original=getattr(o.refresh_binding,name)
        setattr(o.refresh_binding,name,lambda *args,**kwargs:original(*args,**kwargs))
        marks.append(dict(a.calls))
    def progress(label,value,metadata):
        if (phase=='first' and label=='map') or (phase=='final' and label=='output'):replace()
    o,a,r=owner(3,progress=progress)
    if phase=='entry':replace()
    with pytest.raises(ValueError,match='accessor|primitive'):o.evaluate_tile(Value(np.zeros(65536),24),tile=0)
    assert marks and dict(a.calls)==marks[0] and o.last_run is None
