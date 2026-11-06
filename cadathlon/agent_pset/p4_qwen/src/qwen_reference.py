"""Experimental real encrypted toy reference, not a calibrated scoring baseline.

Deeper graphs require the pending range-aware homomorphic refresh integration.
See common/encrypted_reference.py; full public-case TPU validation is pending.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'common'))
from encrypted_reference import create_toy_session

IMPLEMENTATION_STATUS = 'experimental-unvalidated'


def create_session(config, weights, *, batch, sequence_length, devices):
    return create_toy_session('qwen', config, weights, batch=batch, sequence_length=sequence_length, devices=devices)
