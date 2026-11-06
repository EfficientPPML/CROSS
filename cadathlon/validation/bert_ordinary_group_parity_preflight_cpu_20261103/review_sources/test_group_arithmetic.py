"""Independent tiny-array scheduling tests; no backend or ciphertext reads."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
import group_arithmetic as a
from test_parity_arithmetic import Value,fixture,Owner
from group_protocol import GROUPS

class GroupOwner(Owner):
    def rotate_many(self,value,indices):
        results=tuple(value if i%65536==0 else super(GroupOwner,self).rotate(value,i) for i in indices)
        q=len(value.moduli);parts=(q+12)//13
        self.rotation.last_group=dict(mode='digits',requests=len(indices),decompositions=1,
            digit_bytes=parts*8*(q+14)*4,result_bytes=sum(i%65536!=0 for i in indices)*value.polynomial.nbytes)
        return results
class Baseline(Owner):
    def rotate(self,value,index):return value if index%65536==0 else super().rotate(value,index)

def test_all_groups_calls_order_aliases_and_reduction():
    ctx,v=fixture();ctx.num_slots=65536;left=Baseline();right=GroupOwner();rows=[]
    result=a.exercise(ctx,v,left,right,progress=rows.append,security=lambda *args:None)
    assert result['passed'] and len(rows)==66 and result['actual_rotation_calls']==116
    assert result['shared_decompositions']==6 and result['zero_alias_comparisons']==8
    assert left.calls==right.calls==[i for _ in range(2) for _,indices in GROUPS for i in indices if i%65536]
    assert sum(r['compared_words'] for r in rows)==33*(1600+832)

@pytest.mark.parametrize('change',['omit','reverse','alias_zero','reuse_duplicate','fallback','count','digits','result_bytes','source'])
def test_real_group_contract_failures_stop(change):
    ctx,v=fixture();ctx.num_slots=65536;owner=GroupOwner();original=owner.rotate_many;calls=[]
    def bad(value,indices):
        result=original(value,indices);calls.append(1)
        if change=='omit':result=result[:-1]
        if change=='reverse':result=tuple(reversed(result))
        if change=='alias_zero':result=(Value(value.polynomial.copy(),value.moduli),)+result[1:]
        if change=='reuse_duplicate':result=result[:2]+(result[1],)+result[3:]
        if change=='fallback':owner.rotation.last_group['mode']='ordinary'
        if change=='count':owner.rotation.last_group['decompositions']=0
        if change=='digits':owner.rotation.last_group['digit_bytes']=1<<40
        if change=='result_bytes':owner.rotation.last_group['result_bytes']+=4
        if change=='source':value.polynomial.flat[-1]^=1
        return result
    owner.rotate_many=bad
    with pytest.raises(ValueError):a.exercise(ctx,v,Baseline(),owner,progress=lambda r:None,security=lambda *args:None)
    assert len(calls)==1
