"""Actual orchestration code with recording owners; no parameter arrays/files."""
from contextlib import nullcontext
from types import SimpleNamespace as NS
import json
import pytest
from qwen_file_contract import canonical,sha
from qwen_public_parameters import ParameterCatalog
from cleanup_origin import is_cleanup
import model_store_validation as validation


class View:
    """Only a lifetime counter; owns no array, descriptor or body."""
    def __init__(self,env):self.env=env;env.live_views+=1
    def __del__(self):self.env.live_views-=1


class Owner:
    def __init__(self,env,public,stores):
        self.env=env;self.catalog=ParameterCatalog.from_public(public);self.expected_store=stores
        self.matrices={};self.gamma_values={};self.closed=False;self.close_count=0
    def check(self):
        assert not self.closed and self.env.live_views==0
        self.env.log.append(('check',))
    def matrix(self,name):
        assert not self.closed and self.env.live_views==0
        row=json.loads(self.catalog.matrices)[name];key=(row['family'],row['descriptor']['file'])
        self.matrices.setdefault(key,NS(recording=True));self.env.matrix_calls.append(name);self.env.log.append(('matrix',name))
        if self.env.on_matrix:self.env.on_matrix(self,name)
        return View(self.env)
    def gamma(self,name):
        assert not self.closed and self.env.live_views==0
        self.gamma_values={k:None for k in json.loads(self.catalog.gammas)}
        self.env.gamma_calls.append(name);self.env.log.append(('gamma',name))
        if self.env.gamma_error and len(self.env.gamma_calls)==3:raise self.env.gamma_error
        return View(self.env)
    def readback(self):
        assert not self.closed and self.env.live_views==0
        self.env.log.append(('readback',))
    def close(self,primary=None):
        assert not self.closed and self.env.live_views==0
        self.closed=True;self.close_count+=1;self.env.log.append(('close',))
        if self.env.close_error:raise self.env.close_error
        if primary is not None:raise primary


def environment(public,aggregate,stores):
    source_map={'recording_validator.py':'a'*64}
    intent=validation.RecordingValidationIntent(aggregate,stores,canonical(source_map))
    env=NS(intent=intent,log=[],matrix_calls=[],gamma_calls=[],owners=[],live_views=0,source_good=True,factory_error=None,gamma_error=None,close_error=None,terminal_source_error=None,on_matrix=None)
    def source(got):
        assert got is intent;env.log.append(('source',))
        if env.terminal_source_error and env.owners and env.owners[0].closed:raise env.terminal_source_error
        return dict(source_hashes=source_map,source_manifest_sha256=sha(intent.producer_source_map),source_unchanged=env.source_good,synthetic_only=True)
    def factory(got_public,got_stores):
        assert got_public is public and got_stores is stores;env.log.append(('factory',))
        if env.factory_error:raise env.factory_error
        owner=Owner(env,public,stores);env.owners.append(owner);return owner
    env.deps=validation.RecordingDependencies(source,factory,nullcontext)
    env.run=lambda:validation.validate_store_for_recording(intent,env.deps)
    return env


def test_all198_logical197_physical113_gamma_visits_drop_views_and_stay_unready(public,aggregate,stores,tmp_path):
    env=environment(public,aggregate,stores);result=env.run();value=json.loads(result.raw);owner=env.owners[0]
    assert env.matrix_calls==sorted(aggregate.catalog['matrices'])and len(env.matrix_calls)==198
    assert len(owner.matrices)==197
    assert env.gamma_calls==sorted(aggregate.catalog['gammas'])and len(env.gamma_calls)==113
    assert env.log[0]==('source',)and env.log[1]==('factory',)
    assert max(i for i,x in enumerate(env.log)if x[0]=='matrix')<min(i for i,x in enumerate(env.log)if x[0]=='gamma')
    assert env.log[-1]==('source',)and env.log.index(('close',))<len(env.log)-1
    assert owner.closed and owner.close_count==1 and env.live_views==0
    assert value['schema']=='qwen-model-store-recording-completion-v1'
    assert value['logical_matrix_calls']==198 and value['physical_descriptors']==197 and value['gamma_calls']==113
    assert value['cleanup_completed']is True and value['source_postcheck_completed']is True
    assert value['synthetic_only']is True and value['body_integrity_admitted']is False and value['actual_execution_enabled']is False
    assert value['terminal_ready']is False and value['independent_outer_receipt_sha256']is None
    assert 'parameter_body_reads_performed'not in value
    assert result.body_integrity_admitted is False and result.execution_authorized is False and result.terminal_ready is False
    assert list(tmp_path.iterdir())==[], 'recording full catalog must not create physical stores'


def test_source_rejection_precedes_parameter_factory(public,aggregate,stores):
    env=environment(public,aggregate,stores);env.source_good=False
    with pytest.raises(ValueError,match='source prerequisite'):env.run()
    assert env.owners==[]and ('factory',)not in env.log and env.matrix_calls==env.gamma_calls==[]


def test_factory_failure_preserves_primary_without_claiming_complete_validation(public,aggregate,stores):
    env=environment(public,aggregate,stores);error=ValueError('controlled factory failure');env.factory_error=error
    with pytest.raises(ValueError)as caught:env.run()
    assert caught.value is error and env.owners==[]and env.log[-1]==('source',)
    assert validation._QUARANTINE[-1][-1]is error


@pytest.mark.parametrize('method',['close','readback'],ids=['close','readback'])
def test_mutated_owner_callbacks_reject_but_captured_originals_retire(public,aggregate,stores,method):
    env=environment(public,aggregate,stores);replacement=[]
    def mutate(owner,name):setattr(owner,method,lambda *a,**kw:replacement.append(method))
    env.on_matrix=mutate
    with pytest.raises(ValueError,match='callback changed'):env.run()
    owner=env.owners[0]
    assert replacement==[]and owner.closed and owner.close_count==1 and env.live_views==0
    # Terminal source callback is not entered after its binding guard rejects;
    # all original cleanup callbacks still run and no completion is returned.
    assert len(env.matrix_calls)==1 and env.gamma_calls==[]and env.log[-1]==('close',)


def test_partial_gamma_body_failure_retains_original_owner_and_drops_views(public,aggregate,stores):
    env=environment(public,aggregate,stores);error=ValueError('controlled gamma failure');env.gamma_error=error
    with pytest.raises(ValueError)as caught:env.run()
    owner=env.owners[0]
    assert caught.value is error and len(env.matrix_calls)==198 and len(env.gamma_calls)==3
    assert owner.closed and env.live_views==0 and validation._QUARANTINE[-1][6]is owner
    assert validation._QUARANTINE[-1][-1]is error


@pytest.mark.parametrize('cancel_type',[KeyboardInterrupt,SystemExit],ids=['keyboardinterrupt','systemexit'])
def test_cleanup_only_cancellation_outranks_ordinary_close(public,aggregate,stores,cancel_type):
    env=environment(public,aggregate,stores);ordinary=OSError('controlled close');cancel=cancel_type('controlled source cleanup')
    env.close_error=ordinary;env.terminal_source_error=cancel
    with pytest.raises(cancel_type)as caught:env.run()
    assert caught.value is cancel and is_cleanup(cancel)and env.owners[0].closed
    assert validation._QUARANTINE[-1][-1]is cancel


def test_genuine_body_survives_cleanup_cancellation(public,aggregate,stores):
    env=environment(public,aggregate,stores);primary=ValueError('controlled body');later=KeyboardInterrupt('controlled close')
    env.gamma_error=primary;env.close_error=later
    with pytest.raises(ValueError)as caught:env.run()
    assert caught.value is primary and env.owners[0].closed and env.live_views==0
    assert validation._QUARANTINE[-1][-1]is primary


def test_shield_construction_failure_still_retires_original_owner(public,aggregate,stores):
    env=environment(public,aggregate,stores);error=OSError('controlled shield construction')
    def fail_shield():raise error
    env.deps=validation.RecordingDependencies(env.deps.source_check,env.deps.parameter_factory,fail_shield)
    with pytest.raises(OSError)as caught:env.run()
    assert caught.value is error and env.owners[0].closed and env.owners[0].close_count==1
    assert env.log[-1]==('source',)and env.live_views==0
