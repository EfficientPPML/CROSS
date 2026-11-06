from pathlib import Path
from types import SimpleNamespace
import json
import pytest
import validate_heldout as wrapper

@pytest.mark.parametrize('primary,cleanup',[(ValueError('original prerequisite failure'),KeyboardInterrupt('cleanup interruption')),(KeyboardInterrupt('original cancellation'),RuntimeError('cleanup error'))])
def test_cleanup_error_preserves_primary_and_publishes_failed_report(tmp_path,monkeypatch,primary,cleanup):
    def reject(*args):raise primary
    def cleanup_failure():raise cleanup
    def forbidden(*args):raise AssertionError('validation inputs must not be read')
    monkeypatch.setattr(wrapper.calibration_gate,'verify',reject)
    monkeypatch.setattr(wrapper.gc,'collect',cleanup_failure)
    monkeypatch.setattr(wrapper,'load_validation_inputs',forbidden)
    args=SimpleNamespace(output=tmp_path/'attempt',calibration_report=tmp_path/'unused',calibration_source_archive=tmp_path/'unused.tar')
    with pytest.raises(type(primary)) as caught:wrapper.collect(args)
    assert caught.value is primary
    report=json.loads((args.output/'report.json').read_text())
    assert not report['passed'] and not report['validation_data_read']
    assert any(e.get('phase')=='model_cleanup' and e['type']==type(cleanup).__name__ for e in report['errors'])
    assert any('Model cleanup failed' in note for note in primary.__notes__)
