"""Repeated guard-only timing; no model/HE or end-to-end speedup claim."""
import json,time,statistics
import numpy as np
from test_mask_guards import before,candidate,owner

def bench(fn,repeats=5,iterations=100):
    fn();rows=[]
    for _ in range(repeats):
        start=time.perf_counter()
        for __ in range(iterations):fn()
        rows.append((time.perf_counter()-start)/iterations)
    return {'seconds_per_call':rows,'median_seconds_per_call':statistics.median(rows),'iterations_per_sample':iterations}

def main():
    old=object.__new__(before.ScoreReduction);old.layout=before.ScoreReductionLayout(np.ones(128,bool))
    new=owner(candidate.ScoreReductionLayout(np.ones(128,bool)))
    print(json.dumps({'scope':'Guard-only local CPU timings, excluding construction and arithmetic. Not fullmodel/HE speedup.',
        'array_bytes_per_complete_hash':sum(getattr(old.layout,name).nbytes for name in candidate._LAYOUT_ARRAYS),
        'numpy_version':np.__version__,'original_full_hash':bench(old._layout_fingerprint),
        'immutable_storage_guard':bench(new._layout_fingerprint)},indent=2))
if __name__=='__main__':main()
