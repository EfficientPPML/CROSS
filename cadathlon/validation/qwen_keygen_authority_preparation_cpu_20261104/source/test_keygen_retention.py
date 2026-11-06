"""Retained byte evidence is reauthenticated; actual factory still denies."""
import json
import pytest
from qwen_file_contract import canonical
from keygen_authority import create_actual_keygen_process
from synthetic_keygen_chain import fake_hash


@pytest.mark.parametrize('case', ['launch_grant', 'model_report', 'ready_outer',
                                  'expected_ready_pin', 'public_contract'])
def test_retained_object_tampering_cannot_bypass_reauthentication(chain, case):
    ready = chain().ready()
    if case == 'launch_grant':
        object.__setattr__(ready.launch,'grant_bytes',b'{}\n')
    elif case == 'model_report':
        value = json.loads(ready.launch.model.report_bytes)
        value['checks']['cleanup_completed'] = False
        object.__setattr__(ready.launch.model,'report_bytes',canonical(value))
    elif case == 'ready_outer':
        value = json.loads(ready.outer_bytes);value['exit_code'] = 70
        object.__setattr__(ready,'outer_bytes',canonical(value))
    elif case == 'expected_ready_pin':
        value = json.loads(ready.expected_pins.raw);value['context_id'] = fake_hash('rebound expected context')
        object.__setattr__(ready.expected_pins,'raw',canonical(value))
    elif case == 'public_contract':
        original = ready.launch.intent.public.blobs
        bad = (original[0][0],b'altered public bytes')
        object.__setattr__(ready.launch.intent.public,'blobs',(bad,)+original[1:])
    else: raise AssertionError(case)
    with pytest.raises(ValueError): ready.expected_ids(role='client')


def test_actual_entry_denies_before_argument_access():
    class Poison:
        def __getattribute__(self,name):
            raise AssertionError('actual factory inspected argument')
        def __iter__(self):
            raise AssertionError('actual factory iterated argument')
        def __bool__(self):
            raise AssertionError('actual factory evaluated argument')
    with pytest.raises(PermissionError,match='actual key generation awaits'):
        create_actual_keygen_process(Poison(),grant=Poison(),model=Poison())
