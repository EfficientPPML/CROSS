import json
from pathlib import Path
import pytest
from plan_public import derive
from verify_silu_v2 import verify
P=Path(__file__).resolve().parent

def test_frozen_actual_plan_needs_json_shape_normalization():
    memory=derive();parsed=json.loads((P/'source-bound-owner-v1/public_plan.json').read_bytes())
    assert isinstance(memory['sites'][3]['events'][0],tuple)
    assert isinstance(parsed['sites'][3]['events'][0],list)
    assert memory!=parsed
    assert json.loads(json.dumps(memory,allow_nan=False))==parsed

def test_normalization_does_not_discard_numeric_change():
    parsed=json.loads((P/'source-bound-owner-v1/public_plan.json').read_bytes());parsed['sites'][3]['output_level']=5
    assert json.loads(json.dumps(derive(),allow_nan=False))!=parsed

def test_complete_current_original34_archive_and_metrics():
    verify(P/'source-bound-owner-v1',current=True)
