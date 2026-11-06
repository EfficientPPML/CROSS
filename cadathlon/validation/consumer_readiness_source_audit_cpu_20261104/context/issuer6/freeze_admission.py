"""Pure first-pass freeze receipt contract; no IO and no execution authority.

The expected receipt digest must come from an independently reviewed caller.
This component checks that attestation's structure and causal candidate history;
it does not reproduce the underlying source/metric/journal verification.
"""
from dataclasses import dataclass
import hashlib
import json
import math
import re

PROTOCOL = 'eea8e4b38b4c2719e63a0aa13028150564901305e9e5f1f9d0f2985c821cb70f'
ARITHMETIC = '9689b843b4ec2d0a0ef724dbdca7a516e9a4f23da83f83af5571da13380e9a6f'
REFERENCE = '667c8483648adcf583212b0f828347614423c1c1401807f4f9512612f74aa147'
SCHEMA = 'qwen-first-complete-calibration-pass-freeze-v2'

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def sha(raw):return hashlib.sha256(raw).hexdigest()

def digest(value):
    if type(value)is not str or re.fullmatch('[0-9a-f]{64}',value)is None:
        raise ValueError('Exact external digest required')
    return value

def parse(raw):
    if type(raw)is not bytes or not 0<len(raw)<=1<<20:raise ValueError('Bounded immutable public receipt required')
    def pairs(rows):
        out={}
        for name,value in rows:
            if name in out:raise ValueError('Duplicate public receipt field')
            out[name]=value
        return out
    def invalid(value):raise ValueError('Nonfinite public receipt value')
    def number(value):
        x=float(value)
        if not math.isfinite(x):invalid(value)
        return x
    return json.loads(raw,object_pairs_hook=pairs,parse_constant=invalid,parse_float=number)

def fields(value,names):
    if type(value)is not dict or set(value)!=set(names):raise ValueError('Exact receipt schema required')

def integer(value,low,high=None):
    high=low if high is None else high
    if type(value)is not int or not low<=value<=high:raise ValueError('Exact bounded integer required')
    return value

def same(a,b):return canonical(a)==canonical(b)

def attempt(row,ordinal):
    pins=('manifest_sha256','parent_report_sha256','worker_report_sha256','source_manifest_sha256',
          'source_archive_sha256','grant_sha256','producer_outer_receipt_sha256',
          'reconstruction_outer_receipt_sha256','reconstruction_script_sha256','independent_review_sha256',
          'reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_grant_sha256')
    fields(row,('ordinal','outcome','terminal','coverage','gates',*pins))
    integer(row['ordinal'],ordinal)
    for name in pins:digest(row[name])
    if row['outcome']not in('complete_pass','complete_scientific_failure'):raise ValueError('Complete scientific attempt required')
    passed=row['outcome']=='complete_pass'
    expected_code=0 if passed else 1
    terminal_flags=('parent_reaped','parent_group_absent','worker_reaped','worker_group_absent',
        'source_verified','assets_verified','cleanup_completed','signal_restoration_completed',
        'ordered_ledger_independently_verified','all_metric_and_domain_records_independently_verified')
    t=row['terminal'];fields(t,(*terminal_flags,'parent_exit_code','worker_exit_code'))
    if any(t[n]is not True for n in terminal_flags):raise ValueError('Independently completed terminal proof required')
    integer(t['parent_exit_code'],expected_code);integer(t['worker_exit_code'],expected_code)
    c=row['coverage'];fields(c,('ledger_entries','measured_contexts','candidate_complete','candidate_invalid',
        'unmeasured_free_suffixes','comparison_invalid','semantic_passes','native_forward_started',
        'native_forward_completed','all_length_invariance','phase_counts'))
    integer(c['ledger_entries'],3200);measured=integer(c['measured_contexts'],3152,3200)
    integer(c['unmeasured_free_suffixes'],3200-measured)
    integer(c['comparison_invalid'],0);integer(c['semantic_passes'],measured)
    complete=integer(c['candidate_complete'],0,measured)
    integer(c['candidate_invalid'],measured-complete)
    for name in('native_forward_started','native_forward_completed'):integer(c[name],measured+32)
    if c['all_length_invariance']is not True or not same(c['phase_counts'],dict(native=measured,exact=measured,candidate=measured)):
        raise ValueError('Complete own-context comparison coverage required')
    g=row['gates'];fields(g,('all_numeric_semantic_domains','trajectories_complete','prefill_matches','teacher_matches','whole_free_sequences'))
    if type(g['all_numeric_semantic_domains'])is not bool or type(g['trajectories_complete'])is not bool:
        raise ValueError('Exact gate booleans required')
    if g['trajectories_complete']is not(complete==3200)or g['all_numeric_semantic_domains']and complete!=3200:
        raise ValueError('Stopped suffixes cannot claim complete gates')
    integer(g['prefill_matches'],0,3072);integer(g['teacher_matches'],0,64);integer(g['whole_free_sequences'],0,16)
    computed=(g['all_numeric_semantic_domains']and g['trajectories_complete']and
              g['prefill_matches']>=2765 and g['teacher_matches']>=58 and g['whole_free_sequences']>=12)
    if passed is not computed:raise ValueError('Declared outcome differs from unchanged calibration gates')
    return passed

@dataclass(frozen=True,slots=True)
class FrozenCandidate:
    receipt_bytes:bytes
    receipt_sha256:str
    manifest_sha256:str
    ordinal:int
    validation_input_sha256:str
    model_pins_sha256:str
    execution_authorized:bool=False

def authenticate(raw,expected_sha256):
    """Admit an externally pinned public freeze attestation; open nothing."""
    digest(expected_sha256)
    if type(raw)is not bytes or sha(raw)!=expected_sha256:raise ValueError('Independent freeze receipt digest differs')
    row=parse(raw)
    fixed=dict(schema=SCHEMA,protocol_sha256=PROTOCOL,arithmetic_sha256=ARITHMETIC,reference_sha256=REFERENCE,
        first_complete_calibration_pass=True,selection_permanently_closed=True,
        recorded_statistic_reconstruction=True,fresh_array_recomputation=False,
        argmax_recomputed_from_logits=False,invariance_recomputed_from_arrays=False,
        validation_previously_read=False,row0_previously_evaluated=False,
        validation_execution_authorized=False,row0_execution_authorized=False,
        metric_contract=dict(operators=421,blocks=28,logits=151936,normalization_inputs=113,
            operator_block_nrmse=.05,full_logit_nrmse=.10,semantic_nrmse=1e-5,
            semantic_max_abs=.001,semantic_top1=True,nrmse_floor=.01,all_propagated_domains=True),
        row0=dict(token_id=0,original_position=0,sequence_length=1,excluded_from_selection=True))
    extra=('attempts','manifest_sha256','ordinal','validation_input_sha256','tokenizer_report_sha256',
           'tokenizer_review_sha256','model_pins_sha256')
    fields(row,(*fixed,*extra))
    if any(not same(row[k],v)for k,v in fixed.items()):raise ValueError('Exact first-pass/no-feedback freeze contract required')
    for name in('manifest_sha256','validation_input_sha256','tokenizer_report_sha256','tokenizer_review_sha256','model_pins_sha256'):digest(row[name])
    ordinal=integer(row['ordinal'],1,3);history=row['attempts']
    if type(history)is not list or len(history)!=ordinal:raise ValueError('Complete ordered candidate history required')
    outcomes=[attempt(value,i+1)for i,value in enumerate(history)]
    if outcomes!=[False]*(ordinal-1)+[True]:raise ValueError('First complete PASS must freeze immediately')
    if len({a['manifest_sha256']for a in history})!=ordinal or history[-1]['manifest_sha256']!=row['manifest_sha256']:
        raise ValueError('Distinct candidate history and frozen last identity required')
    return FrozenCandidate(raw,expected_sha256,row['manifest_sha256'],ordinal,row['validation_input_sha256'],row['model_pins_sha256'])

def require_frozen(value):
    if type(value)is not FrozenCandidate:raise ValueError('Externally authenticated first-pass freeze required')
    expected=authenticate(value.receipt_bytes,value.receipt_sha256)
    if any(type(getattr(value,n))is not type(getattr(expected,n))or getattr(value,n)!=getattr(expected,n)
           for n in FrozenCandidate.__dataclass_fields__):raise ValueError('Freeze binding changed')
    return expected


def authenticate_producer_outer(raw,expected_sha256,*,attempt_row):
    """Verify separately pinned producer terminal attestation; open nothing."""
    digest(expected_sha256)
    if type(raw)is not bytes or sha(raw)!=expected_sha256 or expected_sha256!=attempt_row['producer_outer_receipt_sha256']:
        raise ValueError('Independent producer outer receipt differs')
    passed=attempt(attempt_row,attempt_row['ordinal'])
    fixed=dict(schema='qwen-candidate3200-producer-terminal-v1',
        report_sha256=attempt_row['parent_report_sha256'],worker_report_sha256=attempt_row['worker_report_sha256'],
        source_manifest_sha256=attempt_row['source_manifest_sha256'],source_archive_sha256=attempt_row['source_archive_sha256'],
        candidate_manifest_sha256=attempt_row['manifest_sha256'],grant_sha256=attempt_row['grant_sha256'],
        exit_code=0 if passed else 1,reaped=True,process_group_absent=True,
        worker_exit_code=0 if passed else 1,worker_reaped=True,worker_group_absent=True)
    value=parse(raw)
    if not same(value,fixed):raise ValueError('Exact producer terminal binding required')
    return value


def authenticate_reconstruction_outer(raw,expected_sha256,*,attempt_row,reviewed_script_sha256,reviewed_reconstruction_pins):
    """Bind root's successful recorded-statistic reconstruction and final IO."""
    digest(expected_sha256);digest(reviewed_script_sha256)
    names=('reconstruction_source_manifest_sha256','reconstruction_source_archive_sha256','reconstruction_grant_sha256')
    fields(reviewed_reconstruction_pins,names)
    for name in names:
        digest(reviewed_reconstruction_pins[name])
        if reviewed_reconstruction_pins[name]!=attempt_row[name]:raise ValueError('Independently reviewed reconstruction closure/grant differs')
    if type(raw)is not bytes or sha(raw)!=expected_sha256 or expected_sha256!=attempt_row['reconstruction_outer_receipt_sha256']:
        raise ValueError('Independent reconstruction outer receipt differs')
    attempt(attempt_row,attempt_row['ordinal'])
    if reviewed_script_sha256!=attempt_row['reconstruction_script_sha256']:
        raise ValueError('Externally reviewed reconstruction script differs')
    fixed=dict(schema='qwen-candidate-reconstruction-outer-v2',review_sha256=attempt_row['independent_review_sha256'],
        script_sha256=reviewed_script_sha256,producer_source_manifest_sha256=attempt_row['source_manifest_sha256'],
        reconstruction_source_manifest_sha256=attempt_row['reconstruction_source_manifest_sha256'],
        reconstruction_source_archive_sha256=attempt_row['reconstruction_source_archive_sha256'],
        reconstruction_grant_sha256=attempt_row['reconstruction_grant_sha256'],
        candidate_manifest_sha256=attempt_row['manifest_sha256'],producer_outer_receipt_sha256=attempt_row['producer_outer_receipt_sha256'],
        returncode=0,direct_child_reaped=True,owned_group_absent=True,publication_completed=True)
    value=parse(raw)
    if not same(value,fixed):raise ValueError('Exact successful reconstruction terminal binding required')
    return value


def authenticate_complete_freeze(raw,expected_sha256,*,producer_outers,reconstruction_outers,reviewed_script_sha256,reviewed_reconstruction_pins):
    """Public byte-only attestation chain; external caller must pin every input.

    Supplied byte tuples confer no execution authority. The separately hashed
    actual grant binds receipt/outer/script identities before any body reader.
    """
    frozen=authenticate(raw,expected_sha256);history=parse(raw)['attempts']
    for values in(producer_outers,reconstruction_outers,reviewed_script_sha256,reviewed_reconstruction_pins):
        if type(values)is not tuple or len(values)!=frozen.ordinal:
            raise ValueError('Every ordered attempt requires independent outer/script evidence')
    for row,p,r,script,pins in zip(history,producer_outers,reconstruction_outers,reviewed_script_sha256,reviewed_reconstruction_pins):
        authenticate_producer_outer(p,row['producer_outer_receipt_sha256'],attempt_row=row)
        authenticate_reconstruction_outer(r,row['reconstruction_outer_receipt_sha256'],attempt_row=row,reviewed_script_sha256=script,reviewed_reconstruction_pins=pins)
    return frozen
