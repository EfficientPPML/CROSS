import pytest
import full_synthetic as f
@pytest.mark.parametrize("change",["dnum","ordered_q"])
def test_fixed_profile_rejected_before_any_body(monkeypatch,change):
    base=f.Arithmetic
    class Altered(base):
        def __init__(self):
            super().__init__()
            if change=="dnum":self.ctx.dnum=self.ctx._param_cache.dnum=4
            else:
                q=list(self.ctx.q_towers);q[0],q[1]=q[1],q[0]
                self.ctx.q_towers=self.ctx._param_cache.q_towers=tuple(q)
    monkeypatch.setattr(f,"Arithmetic",Altered)
    with pytest.raises(ValueError):f.build(monkeypatch)
