from pathlib import Path
from types import SimpleNamespace
import importlib.util,json
import pytest

def test_frozen_wrapper_original_cleanup_interruption_gap(tmp_path,monkeypatch):
    path=Path(__file__).resolve().parent/'before_validate_heldout.py'
    spec=importlib.util.spec_from_file_location('old_frozen_wrapper',path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    primary=ValueError('original prerequisite failure');secondary=KeyboardInterrupt('cleanup interruption')
    def reject(*args):raise primary
    def cleanup():raise secondary
    def forbidden(*args):raise AssertionError('validation data must not be read')
    monkeypatch.setattr(old.calibration_gate,'verify',reject);monkeypatch.setattr(old.gc,'collect',cleanup);monkeypatch.setattr(old,'load_validation_inputs',forbidden)
    args=SimpleNamespace(output=tmp_path/'old',calibration_report=tmp_path/'unused',calibration_source_archive=tmp_path/'unused.tar')
    with pytest.raises(KeyboardInterrupt) as caught:old.collect(args)
    assert caught.value is secondary
    report=json.loads((args.output/'report.json').read_text())
    assert report['errors']==[] and not report['passed']
    (path.parent/'original_cleanup_reproducer.json').write_text(json.dumps({'reproduced':True,'original_primary':'ValueError','observed_secondary':'KeyboardInterrupt','initial_report_preserved_but_final_failure_missing':True,'validation_inputs_read':False,'original_source_bytes_unchanged':True},indent=2)+'\n')
