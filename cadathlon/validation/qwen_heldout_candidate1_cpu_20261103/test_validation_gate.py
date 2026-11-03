from pathlib import Path
from types import SimpleNamespace
import copy,hashlib,io,json,tarfile
import pytest
import calibration_gate as gate
import validate_heldout as wrapper
P=Path(__file__).resolve().parent
REQ=json.loads((P/'calibration_requirement.json').read_text())
REPORT=Path(REQ['calibration_report_reference']);ARCHIVE=Path(REQ['calibration_source_archive_reference'])

@pytest.fixture(scope='module')
def full_report():return json.loads(REPORT.read_text())

def verify(req=None,report=REPORT):return gate.verify(report,ARCHIVE,P,REQ if req is None else req)

def test_actual_completed_calibration_passes_without_reading_validation_or_model_assets(monkeypatch):
    forbidden=wrapper.VALIDATION_INPUTS
    original=Path.open
    def guarded(self,*args,**kwargs):
        assert self!=forbidden and self.name!='model.safetensors'
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Path,'open',guarded)
    assert verify()['calibration_contexts']==416

def test_pending_requirement_fails_before_any_evidence_read(monkeypatch):
    def fail(*args):raise AssertionError('must not open input')
    monkeypatch.setattr(gate,'sha',fail)
    with pytest.raises(ValueError,match='not ready'):verify(dict(REQ,ready=False))

@pytest.mark.parametrize('field',['full_report_sha256','full_source_archive_sha256'])
def test_wrong_prerequisite_hash_rejected(field):
    req=dict(REQ);req[field]='0'*64
    with pytest.raises(ValueError,match='hash'):verify(req)

@pytest.mark.parametrize('mutation,match',[
    ('failed','successful complete'),('partial','successful complete'),('smoke','successful complete'),
    ('case_missing','counts'),('operator_failure','operator/block gate'),('stopped_suffix','no unmeasured suffix'),('wrong_free_prefix','same-prefix free')])
def test_self_declared_pass_does_not_override_counts_gates_or_histories(tmp_path,full_report,mutation,match):
    r=copy.deepcopy(full_report)
    if mutation=='failed':r['passed']=False
    elif mutation=='partial':r['complete']=False
    elif mutation=='smoke':r['smoke']=True
    elif mutation=='case_missing':r['cases'].pop()
    elif mutation=='operator_failure':next(iter(r['cases'][0]['operator_metrics'].values()))['nrmse']=.06
    elif mutation=='stopped_suffix':r['decode'][0]['free_suffix_stopped']=True
    else:r['decode'][0]['free_records'][1]['token_ids'][-1]+=1
    path=tmp_path/'report.json';path.write_text(json.dumps(r));req=dict(REQ,full_report_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match=match):verify(req,report=path)

def test_current_runtime_drift_is_rejected(monkeypatch):
    original=gate.sha
    def changed(path):return '0'*64 if Path(path)==P/'arithmetic.py' else original(path)
    monkeypatch.setattr(gate,'sha',changed)
    with pytest.raises(ValueError,match='current runtime source arithmetic.py'):verify()

def test_wrapper_checks_prerequisite_before_validation_loader(tmp_path,monkeypatch):
    def reject(*args):raise ValueError('incomplete calibration')
    def forbidden(*args):raise AssertionError('validation inputs were touched')
    monkeypatch.setattr(gate,'verify',reject);monkeypatch.setattr(wrapper,'load_validation_inputs',forbidden)
    args=SimpleNamespace(output=tmp_path/'attempt',calibration_report=REPORT,calibration_source_archive=ARCHIVE)
    with pytest.raises(ValueError,match='incomplete calibration'):wrapper.collect(args)
    r=json.loads((args.output/'report.json').read_text());assert not r['passed'] and not r['validation_data_read']

def test_wrapper_records_successful_gate_before_attempting_validation_input_read(tmp_path,monkeypatch):
    called=[]
    def stop_before_open(expected_sha):called.append(expected_sha);raise RuntimeError('stop before validation data')
    monkeypatch.setattr(wrapper,'load_validation_inputs',stop_before_open)
    args=SimpleNamespace(output=tmp_path/'guard-only',calibration_report=REPORT,calibration_source_archive=ARCHIVE)
    with pytest.raises(RuntimeError,match='stop before validation data'):wrapper.collect(args)
    r=json.loads((args.output/'report.json').read_text());assert r['calibration_prerequisite']['verified'] and not r['validation_data_read'] and called==[REQ['validation_input_sha256']]

def test_original_validation_counts_and_published_gates_are_unmodified():
    # Read code/metadata only; the held-out fixture file remains unopened.
    text=(P/'validate_heldout.py').read_text()
    assert "len(rows)!=56" in text and "len(group)!=16" in text and "len(decode)!=8" in text
    assert 'args.smoke' not in text
    assert wrapper.math.ceil(.9*48)==44 and wrapper.math.ceil(.9*32)==29 and wrapper.math.ceil(.75*8)==6

@pytest.mark.parametrize('replace_member',[False,True])
def test_archive_duplicate_members_and_length_rejected(tmp_path,full_report,replace_member):
    with tarfile.open(ARCHIVE) as source:
        entries=[(m.name,source.extractfile(m).read()) for m in source.getmembers()]
    if replace_member:entries[-1]=entries[0]
    else:entries.append(entries[0])
    archive=tmp_path/'duplicate.tar.gz'
    with tarfile.open(archive,'w:gz') as target:
        for name,payload in entries:
            member=tarfile.TarInfo(name);member.size=len(payload);target.addfile(member,io.BytesIO(payload))
    archive_hash=gate.sha(archive)
    report=dict(full_report,source_archive_sha256=archive_hash)
    report_path=tmp_path/'report.json';report_path.write_text(json.dumps(report))
    req=dict(REQ,full_report_sha256=gate.sha(report_path),full_source_archive_sha256=archive_hash)
    with pytest.raises(ValueError,match='duplicate members' if replace_member else 'member count'):
        gate.verify(report_path,archive,P,req)
