"""Public byte-gate tests; every grant/proof is a synthetic fixture."""
import builtins
import json
import os
import pytest
from qwen_file_contract import canonical, sha
from keygen_authority import authenticate_launch
from model_admission import MODEL_ID
from synthetic_keygen_chain import fake_hash


def test_complete_chain_returns_distinct_external_role_ids(chain):
    fixture = chain()
    launch = fixture.launch()
    ready = fixture.ready()
    client = ready.expected_ids(role='client')
    server = ready.expected_ids(role='server')
    assert launch.execution_authorized is ready.execution_authorized is False
    assert fixture.intent.execution_authorized is False
    assert client.context_id == server.context_id == fixture.ready_pins['context_id']
    assert client.context_id != MODEL_ID
    assert client.model_id == server.model_id == MODEL_ID
    assert client.producer_source_manifest_sha256 != client.consumer_source_manifest_sha256
    assert client.private_manifest_sha256 == fixture.ready_pins['private_manifest_sha256']
    assert server.private_manifest_sha256 is None
    assert len(json.loads(fixture.public_raw)['arrays']) == 335
    assert len(fixture.profile.rotations) == 166
    assert 65535 in fixture.profile.rotations and 262143 in fixture.profile.rotations
    # A trusted common existing parent is legal; the three future leaves differ.
    assert len({row['path'] for row in fixture.grant['parents'].values()}) == 1


def test_public_readiness_gate_performs_no_io_or_rng(chain, monkeypatch):
    fixture = chain()
    calls = []
    def forbidden(*args, **kwargs):
        calls.append(True)
        raise AssertionError('pure evidence gate attempted I/O or RNG')
    monkeypatch.setattr(builtins,'open',forbidden)
    for name in ('open','stat','lstat','listdir','scandir','urandom'):
        monkeypatch.setattr(os,name,forbidden)
    result = fixture.ready()
    assert result.expected_ids(role='server').private_manifest_sha256 is None
    assert result.expected_ids(role='client').private_manifest_sha256 == fixture.ready_pins['private_manifest_sha256']
    assert calls == []


@pytest.mark.parametrize('case', ['missing_store', 'changed_core'])
def test_required_actual_source_is_not_replaceable(chain, case):
    def change(source):
        if case == 'missing_store': source.pop('qwen_key_store.py')
        else: source['runtime/jaxite_word/key_gen.py'] = fake_hash('changed core')
    with pytest.raises(ValueError,match='fixed actual prerequisite source'):
        chain(change_source=change).launch()


def test_source_map_digest_requires_canonical_encoding(chain):
    def change(pins):
        pins['source_manifest_sha256'] = sha(json.dumps(pins['source_map'],sort_keys=True,indent=2).encode())
    with pytest.raises(ValueError,match='canonical compact source-map digest'):
        chain(change_launch_pins=change).launch()


@pytest.mark.parametrize('case', ['repinned_model_scope', 'different_consumer_map'])
def test_model_is_reauthenticated_for_this_generation_source(chain, case):
    if case == 'repinned_model_scope':
        fixture = chain(change_model_report=lambda row:row['scope'].__setitem__('parameter_body_reads_performed',False))
        message = 'model-only readonly validation scope'
    else:
        fixture = chain(model_consumer=fake_hash('other consumer'))
        message = 'selected model consumer source'
    with pytest.raises(ValueError,match=message): fixture.launch()


@pytest.mark.parametrize('case', ['role_source', 'model_proof', 'resource_policy',
                                  'parent_path', 'overlapping_leaves',
                                  'missing_role_grant', 'execute_integer_alias'])
def test_repinned_launch_document_mismatch_is_rejected(chain, case):
    options = {}
    if case == 'role_source':
        options['change_intent'] = lambda row:row.__setitem__('request_producer_source_manifest_sha256',fake_hash('other role'))
        message = 'actual role source'
    elif case == 'model_proof':
        options['change_launch_pins'] = lambda row:row['model_evidence'].__setitem__('report_sha256',fake_hash('other model proof'))
        message = 'external ModelAdmission proof'
    elif case == 'resource_policy':
        options['change_grant'] = lambda row:row['policies'].__setitem__('resource_policy_sha256',fake_hash('other resource policy'))
        message = 'independently reviewed policies'
    elif case == 'parent_path':
        options['change_grant'] = lambda row:row['parents']['private_root'].__setitem__('path','/other-parent')
        message = 'granted parent path'
    elif case == 'overlapping_leaves':
        options['change_intent'] = lambda row:row['paths'].__setitem__('private_root',row['paths']['public_root']+'/nested')
        message = 'owned key/evidence leaves'
    elif case == 'missing_role_grant':
        options['change_launch_pins'] = lambda row:row['role_evidence']['response_consumer'].pop('execution_grant_sha256')
        message = 'independently reviewed actual role pins'
    else:
        options['change_grant'] = lambda row:row.__setitem__('execute',1)
        message = 'explicit actual keygen grant'
    with pytest.raises(ValueError,match=message): chain(**options).launch()


@pytest.mark.parametrize('case', ['unknown_intent_field', 'duplicate_grant_key', 'boolean_parent_inode'])
def test_closed_typed_launch_arguments(chain, case):
    if case == 'unknown_intent_field':
        with pytest.raises(ValueError,match='exact actual intent fields'):
            chain(change_intent=lambda row:row.__setitem__('extra_authority',True)).launch()
    elif case == 'boolean_parent_inode':
        with pytest.raises(ValueError,match='parent inode integer'):
            chain(change_grant=lambda row:row['parents']['public_root'].__setitem__('inode',True)).launch()
    else:
        fixture = chain()
        duplicate = b'{"schema":"duplicate",'+fixture.grant_raw[1:]
        with pytest.raises(ValueError,match='duplicate'):
            authenticate_launch(fixture.intent,duplicate,fixture.model,expected_pins=fixture.launch_expected)
