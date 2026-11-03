"""Experimental real encrypted toy reference, not a calibrated scoring baseline.

Deeper graphs require the pending range-aware homomorphic refresh integration.
See common/encrypted_reference.py; full public-case TPU validation is pending.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_runtime import install_common_path
install_common_path(__file__)
from encrypted_reference import create_toy_session

IMPLEMENTATION_STATUS = 'experimental-unvalidated'


def create_session(config, weights, *, batch, sequence_length, devices):
    return create_toy_session('bert', config, weights, batch=batch, sequence_length=sequence_length, devices=devices)
