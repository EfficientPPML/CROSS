"""Opt-in experimental bounded packed toy qwen; no full-model or calibration claim.

Only a real encrypted complete-case gate validates this candidate. Declared
polynomial input domains are conditions, and refresh accuracy is period-specific.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'common'))
from encrypted_reference import create_toy_session

IMPLEMENTATION_STATUS = 'experimental-unvalidated'

def create_session(config, weights, *, batch, sequence_length, devices):
    return create_toy_session('qwen', config, weights, batch=batch,
        sequence_length=sequence_length, devices=devices, packed_pairs=True, bounded_refresh=True)
