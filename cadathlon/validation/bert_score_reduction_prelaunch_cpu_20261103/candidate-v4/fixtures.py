"""Deterministic public fixture contracts; plaintext values/oracles are client-only calls."""
from pathlib import Path
import json
import numpy as np
from public_schedule import PublicSchedule
SLOTS,ROWS,PITCH,KEYS,HEADS,PAD=65536,16,4096,128,12,16


def public_fixtures():
    baseline=json.loads((Path(__file__).with_name('baseline_manifest.json')).read_text())
    schedule=PublicSchedule(baseline['power_coefficients_in_t_2x'])
    keys15=[0,1,7,15,16,31,48,63,64,79,95,111,112,126,127]
    result=[]
    for name,k,rows,keys,seed in (('K128-full16',128,16,list(range(128)),20261103128),
                                ('K15-partial7',15,7,keys15,2026110315)):
        stage=schedule.stage(k,0)
        result.append(dict(name=name,K=k,valid_rows=rows,valid_key_indices=keys,
            domain=[stage['a'],stage['b']],inactive_dummy=stage['inactive_dummy'],
            public_seed=seed,squared_denominator_stage=0))
    return result


def make_client_values(public):
    rng=np.random.default_rng(public['public_seed'])
    values=rng.uniform(-4.,4.,SLOTS)
    indices=np.asarray(public['valid_key_indices'])
    for row in range(public['valid_rows']):
        for head in range(HEADS):
            positive=rng.uniform(.5,1.5,len(indices))
            values[row*PITCH+indices*PAD+head]=positive/positive.sum()
    return values


def make_client_scores(public):
    rng=np.random.default_rng(public['public_seed']+97)
    values=rng.uniform(-4.,4.,SLOTS)
    indices=np.asarray(public['valid_key_indices'])
    for row in range(public['valid_rows']):
        for head in range(HEADS):
            values[row*PITCH+indices*PAD+head]=rng.uniform(-1.,1.,len(indices))+.125*row-.25*head
    return values


def client_oracles(score_values,probability_values,public):
    mean=np.zeros(SLOTS)
    squared=np.full(SLOTS,public['inactive_dummy'])
    indices=np.asarray(public['valid_key_indices']);a,b=public['domain']
    for row in range(public['valid_rows']):
        for head in range(HEADS):
            selected=probability_values[row*PITCH+indices*PAD+head]
            assert abs(selected.sum()-1)<1e-14 and np.all(selected>0)
            denominator=float(np.dot(selected,selected))
            assert a<=denominator<=b
            positions=row*PITCH+np.arange(KEYS)*PAD+head
            mean[positions]=score_values[row*PITCH+indices*PAD+head].mean();squared[positions]=denominator
    active_means=np.asarray([[mean[row*PITCH+head] for head in range(HEADS)] for row in range(public['valid_rows'])])
    assert len(np.unique(active_means))==public['valid_rows']*HEADS
    assert np.all(np.ptp(active_means,axis=1)>.1) and np.all(np.ptp(active_means,axis=0)>.1)
    return {'mean':mean,'squared_denominator':squared}


def mean_variation(reference,public):
    data=np.asarray([[reference[row*PITCH+head] for head in range(HEADS)] for row in range(public['valid_rows'])])
    return dict(active_mean_count=int(data.size),unique_active_means=len(np.unique(data)),
        minimum=float(data.min()),maximum=float(data.max()),
        minimum_within_row_range=float(np.ptp(data,axis=1).min()),
        minimum_across_rows_per_head_range=float(np.ptp(data,axis=0).min()))
