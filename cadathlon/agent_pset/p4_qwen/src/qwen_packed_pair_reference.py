"""Experimental packed pair-feature Qwen; full TPU validation pending.

Graphs needing bounded homomorphic refresh are rejected before key creation
until that integration is validated. No pretrained-model claim is made.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'common'))
from encrypted_reference import create_toy_session

IMPLEMENTATION_STATUS = 'experimental-unvalidated'


def create_session(config, weights, *, batch, sequence_length, devices):
    return create_toy_session('qwen', config, weights, batch=batch,
                              sequence_length=sequence_length, devices=devices, packed_pairs=True)
