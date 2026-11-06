"""Exact compose terminal body with recording output/runtime, plus CLI routing."""
from types import SimpleNamespace
import json
import case_fixtures as f

def test_actual_setup_cli_denies_before_helper_body_or_backend():raise AssertionError('owned exact CLI branch must handle this case before fixture import')
def test_actual_server_cli_denies_before_helper_body_or_backend():raise AssertionError('owned exact CLI branch must handle this case before fixture import')
def test_actual_client_cli_denies_before_helper_body_or_backend():raise AssertionError('owned exact CLI branch must handle this case before fixture import')
def test_help_and_usage_keep_zero_and_two_without_dispatch():raise AssertionError('two owned exact CLI branches must handle help/usage before fixture import')

def _compose(passed,close_error=None):
    import q5_actual_process as m,q5_evidence_output as output
    records=[];owners=[]
    g=dict(role='client',source_manifest_sha256=f.h('source'),source_archive_sha256=f.h('archive'),model_id=f.h('model'),outputs={'evidence':{'path':str(f.OUT/'unused-evidence-path'),'parent':f.root_meta('unused-parent')}})
    runtime=SimpleNamespace(check=lambda:f.event('record_runtime_check'),source_check=lambda:f.event('record_source_check'),phase=SimpleNamespace(check=lambda:g,setup=SimpleNamespace(check=lambda:{'context_id':f.h('context')}),expected=SimpleNamespace(check=lambda:{'grant_sha256':f.h('grant')})))
    class Evidence:
        def __init__(self,*args):self.closed=False;self.closes=0;self.files={};owners.append(self)
        def write(self,name,raw):assert not self.closed;self.files[name]=raw;records.append(name);f.event('evidence_write',name=name)
        def check(self):assert not self.closed
        def close(self,primary=None):
            self.closed=True;self.closes+=1;f.event('evidence_close')
            if close_error is not None:raise close_error
    def role(record_runtime,*,failure_note=None):assert record_runtime is runtime;f.event('recorded_client_result');return dict(original_final_gate={'passed':passed},synthetic_only=True)
    def run():
        with f.patch(output,'EvidenceOutput',Evidence),f.patch(m,'client_bound',role):return m.compose_bound(runtime)
    return SimpleNamespace(run=run,owners=owners,records=records,module=m)

def test_recorded_compose_normal_zero_one_follow_original_evidence_close():
    for passed,expected in ((True,0),(False,1)):
        start=len(f.EVENTS);x=_compose(passed);status=x.run();assert type(status)is int and status==expected
        assert x.records==['client_result.json','report.json']and x.owners[0].closes==1
        labels=[r['event']for r in f.EVENTS[start:]];assert labels.index('evidence_close')<len(labels)-1 and labels[-1]=='record_source_check'
        report=json.loads(x.owners[0].files['report.json']);assert report['passed']is passed and report['complete']is True
        assert x.module._QUARANTINE==[]

def test_recorded_compose_late_close_is_infrastructure_even_after_report():
    from q5_actual_entry import terminal_status
    error=OSError('SYNTHETIC late evidence close');x=_compose(True,error);status=terminal_status(x.run)
    assert status==70 and x.records==['client_result.json','report.json']and x.owners[0].closes==1
    assert f.labels()[-1]=='record_source_check'and x.module._QUARANTINE
    assert any(row[-1]is error for row in x.module._QUARANTINE)
