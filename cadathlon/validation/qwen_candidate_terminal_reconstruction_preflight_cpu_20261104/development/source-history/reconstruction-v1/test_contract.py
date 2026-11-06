"""Synthetic scalar-record tests; no real producer/outcome/input reads."""
import copy
import pytest
import contract as c

def metric(kind='operator',value=0.):
    row=dict(passed=True,finite=True,shape_matches=True,real_numeric=True,count=1,
        max_abs=value,nrmse=value,limit={'operator':.05,'block':.05,'logit':.1,'semantic':1e-5}[kind],floor=.01)
    if kind=='semantic':row.update(top1_match=True,max_abs_limit=.001)
    return row

def test_independent_public_workload_and_full_pass_inventory():
    entries=c.entries();assert len(entries)==3200 and sum(r['length']for r in entries)==52928
    assert entries[0]['length']==1 and entries[3071]['length']==32
    assert entries[3072]['kind']=='teacher'and entries[3136]['kind']=='free'
    assert [r['length']for r in entries[3136:3140]]==[16,17,18,19]
    for s in(1,8,32):
        counts=c.operator_counts(s);assert len(counts)==421 and sum(counts.values())==716800*s+1024
    p=c.full_pass_inventory();assert p['original416_members']==416 and p['private_files_with_ledger']==9601
    assert p['operator_metric_rows']==1347200 and p['block_metric_rows']==89600
    assert p['logit_elements']==p['semantic_elements']==486195200

def test_journal_names_distinguish_full_pass_stopped_suffix_and_comparison_failure():
    rows=[('complete',None)]*3200
    assert len(c.expected_files(tuple(rows)))==9601
    rows[3136]=('candidate_invalid','candidate');rows[3137:3140]=[('unmeasured_suffix',None)]*3
    names=c.expected_files(tuple(rows));assert len(names)==9592 and '3136-candidate.json'in names and '3137-native.json'not in names
    rows[0]=('comparison_invalid','native');rows[1]=('comparison_invalid','exact')
    names=c.expected_files(tuple(rows));assert '0000-exact.json'not in names and '0001-candidate.json'not in names
    assert not any('baseline'in name or'started'in name for name in names)

@pytest.mark.parametrize('row',[('unknown',None),('candidate_invalid',None),('complete','candidate'),('unmeasured_suffix','native')])
def test_unknown_journal_status_never_generates_inventory(row):
    rows=[('complete',None)]*3200;rows[1]=row
    with pytest.raises(ValueError):c.expected_files(tuple(rows))

@pytest.mark.parametrize('kind',['operator','block','logit','semantic'])
def test_stored_gate_recomputed_from_unchanged_threshold(kind):
    row=metric(kind);assert c.metric(row,kind,1,top1_match=True if kind=='semantic'else None)
    row['nrmse']=row['limit']*2;row['passed']=False
    assert not c.metric(row,kind,1,top1_match=True if kind=='semantic'else None)
    row['passed']=True
    with pytest.raises(ValueError):c.metric(row,kind,1,top1_match=True if kind=='semantic'else None)

@pytest.mark.parametrize('name,value',[('count',True),('finite',1),('limit',.051),('floor',.1),('real_numeric',False),
    ('shape_matches',False),('nrmse',float('nan')),('max_abs',float('inf')),('passed',1)])
def test_bad_stored_metric_cannot_freeze(name,value):
    row=metric();row[name]=value
    with pytest.raises(ValueError):c.metric(row,'operator',1)

def test_finite_extreme_failure_remains_strict_json_failure():
    row=metric();row.update(finite=False,passed=False,max_abs=1e308,nrmse=None)
    assert not c.metric(row,'operator',1)

def test_semantic_record_uses_actual_reported_argmax_and_maxabs():
    row=metric('semantic');row.update(top1_match=False,passed=False)
    assert not c.metric(row,'semantic',1,top1_match=False)
    with pytest.raises(ValueError):c.metric(row,'semantic',1,top1_match=True)
    row=metric('semantic');row.update(max_abs=.002,passed=False)
    assert not c.metric(row,'semantic',1,top1_match=True)

def test_no_actual_reconstruction_or_private_read_api():
    with pytest.raises(PermissionError):c.activate_verification()
