import pytest
from q5_checkpoint_contract import numerical_plan
from q5_checkpoint_arithmetic import Ledger,metadata
from q5_test_fixtures import RecordingArray,RecordingCiphertext


def test_original_payload_barrier_precedes_callback_and_owner_is_held():
    profile=numerical_plan()['profile'];value=RecordingCiphertext(profile);payload=value.polynomial;seen=[]
    def callback(index,current,row):
        assert payload.barriers==1
        assert ledger.callback_held[0][0]is value and ledger.callback_held[0][1]is payload
        seen.append((index,current,row.copy()))
    ledger=Ledger(callback)
    assert ledger.note('input','original_embeddings',value)is value
    assert len(seen)==1 and ledger.trace==[seen[0][2]]
    assert ledger.callback_held==()and ledger.callback_controls==()


@pytest.mark.parametrize('case',['row','payload','scale','moduli','public_bound','working_cache'],ids=['row','payload','scale','moduli','public_bound','working_cache'])
def test_checkpoint_callback_cannot_silently_rebind_diagnostic_state(case):
    profile=numerical_plan()['profile'];value=RecordingCiphertext(profile);original=value.polynomial;cache={'epoch':0}
    def callback(index,current,row):
        if case=='row':row['label']='wrong'
        elif case=='payload':current.polynomial=RecordingArray(current.num_moduli)
        elif case=='scale':current._ckks_scale*=2.
        elif case=='moduli':current.moduli=tuple(reversed(current.moduli))
        elif case=='public_bound':current._public_test=4.
        else:cache['epoch']=1
    ledger=Ledger(callback);ledger.working=lambda:[cache]
    with pytest.raises(ValueError):ledger.note('input','original_embeddings',value)
    assert ledger.trace==[]
    assert ledger.callback_held[0][0]is value and ledger.callback_held[0][1]is original
    assert original.barriers==1


def test_callback_primary_identity_retains_original_payload():
    value=RecordingCiphertext(numerical_plan()['profile']);payload=value.polynomial;primary=KeyboardInterrupt('synthetic callback')
    def callback(*args):raise primary
    ledger=Ledger(callback)
    with pytest.raises(KeyboardInterrupt)as caught:ledger.note('input','original_embeddings',value)
    assert caught.value is primary and ledger.trace==[]
    assert ledger.callback_held[0][1]is payload and payload.barriers==1


@pytest.mark.parametrize('case',['kind','label','stack','level'],ids=['kind','label','stack','level'])
def test_wrong_trace_attribution_rejects_before_export_callback(case):
    profile=numerical_plan()['profile'];value=RecordingCiphertext(profile);called=[];ledger=Ledger(lambda *args:called.append(args))
    kind='input';label='original_embeddings'
    if case=='kind':kind='output'
    elif case=='label':label='logits'
    elif case=='stack':ledger.stack.append('attention:1')
    else:value=RecordingCiphertext(profile,48)
    with pytest.raises(ValueError,match='original trace order/contract differs'):
        ledger.note(kind,label,value)
    assert called==[] and ledger.trace==[]
