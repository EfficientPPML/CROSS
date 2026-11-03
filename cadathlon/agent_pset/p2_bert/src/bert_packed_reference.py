"""Experimental packed encrypted toy BERT; full TPU validation pending.

Graphs needing bounded homomorphic refresh are rejected before key creation
until that integration is validated. No pretrained-model claim is made.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from p2_runtime import install_common_path
install_common_path(__file__)
from encrypted_reference import create_toy_session

IMPLEMENTATION_STATUS = 'experimental-unvalidated'


def create_session(config, weights, *, batch, sequence_length, devices):
    return create_toy_session('bert', config, weights, batch=batch,
                              sequence_length=sequence_length, devices=devices, packed_pairs=True)
