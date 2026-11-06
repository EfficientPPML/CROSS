"""Actual public-report schema regression; no archive/token body reads."""
from pathlib import Path
import json
import pytest
from trajectory_assets import completed_report
P=Path(__file__).resolve().parent

def metadata():
    r=json.loads((P/'calibration_count_metadata.json').read_bytes())
    assert r['source_report_sha256']=='1e827000cc0368853f2e8932ec17ee5f318e973e3c80f67652790491fcf7c628'
    return r['fields']

def test_exact_actual_public_count_schema():
    r=metadata();assert type(r['all_workload_entries_accounted'])is int and r['all_workload_entries_accounted']==416
    completed_report(r)

@pytest.mark.parametrize('value',[True,False,416.,'416',415,417,None])
def test_count_type_and_value_are_fail_closed(value):
    r=metadata();r['all_workload_entries_accounted']=value
    with pytest.raises(ValueError,match='prerequisite'):completed_report(r)
