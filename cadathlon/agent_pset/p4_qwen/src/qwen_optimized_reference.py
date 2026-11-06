"""Explicit experimental packed toy reference with reviewed optional controls.

Enables scoped public NTT reuse and guarded dispatch-cache retention. Default
adapters and ring profiles are unchanged. Bootstrap still clears its own stage
caches. This adapter is unvalidated as a complete encrypted model, full-model
file-stage implementation, accelerator-memory result or calibrated baseline.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'common'))
from encrypted_reference import create_toy_session

IMPLEMENTATION_STATUS = 'experimental-unvalidated'

def create_session(config, weights, *, batch, sequence_length, devices):
    return create_toy_session('qwen', config, weights, batch=batch,
        sequence_length=sequence_length, devices=devices, packed_pairs=True, bounded_refresh=True,
        ntt_subsets=True, preserve_dispatch_cache=True)
