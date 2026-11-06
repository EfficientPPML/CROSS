"""Independent public reconstruction contract; no IO/model/backend execution.

These functions validate declared metadata and recorded scalar gates. They do
not reconstruct the underlying model arrays or authorize opening a journal.
"""
import math

SOURCE_MANIFEST_SHA256='ccd5834b83b9dfec2d3ea988fb395a61ec4b05ce3f2f54621f29200cbbb8e06a'
PROTOCOL_SHA256='eea8e4b38b4c2719e63a0aa13028150564901305e9e5f1f9d0f2985c821cb70f'

def entries():
    rows=[]
    for p in range(96):
        for s in range(1,33):
            rows.append(dict(index=len(rows),kind='prefill',prompt_ordinal=p,seed_ordinal=None,
                step=None,length=s,original416_member=s in(8,16,32)))
    for kind in('teacher','free'):
        for seed in range(16):
            for step in range(4):
                rows.append(dict(index=len(rows),kind=kind,prompt_ordinal=None,seed_ordinal=seed,
                    step=step,length=16+step,original416_member=True))
    return tuple(rows)

def operator_counts(sequence):
    if type(sequence)is not int or not 1<=sequence<=32:raise ValueError('Exact S1..32 required')
    widths=(('input.inverse_sqrt',1024),('q_linear',2048),('k_linear',1024),('v_linear',1024),
        ('q.inverse_sqrt',2048),('q_rope',2048),('k.inverse_sqrt',1024),('k_rope',1024),
        ('gqa',2048),('output_linear',1024),('post.inverse_sqrt',1024),('gate_linear',3072),
        ('up_linear',3072),('silu',3072),('gated_mlp',1024))
    result={f'{layer}.{name}':sequence*width for layer in range(28)for name,width in widths}
    result['final.inverse_sqrt']=1024
    return result

def expected_files(statuses):
    """Every successful/scientific-invalid measured context has3 phase files.

    Comparison-invalid tuples identify the failed phase explicitly. Incomplete
    infrastructure runs can be audited but cannot satisfy this complete ledger.
    """
    if type(statuses)is not tuple or len(statuses)!=3200:raise ValueError('Complete typed3200 ledger required')
    names={'ledger.json'}
    for index,row in enumerate(statuses):
        if type(row)is not tuple or len(row)!=2:raise ValueError('Typed status/stage pair required')
        status,phase=row
        if status=='complete'and phase is None or status=='candidate_invalid'and phase=='candidate':
            phases=('native','exact','candidate')
        elif status=='comparison_invalid'and phase in('native','exact'):
            phases=('native',)if phase=='native'else('native','exact')
        elif status=='unmeasured_suffix'and phase is None:phases=()
        else:raise ValueError('Declared status/phase pair required')
        names.update(f'{index:04d}-{p}.json'for p in phases)
    return names

def metric(row,kind,count,*,top1_match=None):
    """Recompute a stored metric's acceptance, not its array-derived value."""
    limits={'operator':.05,'block':.05,'logit':.10,'semantic':1e-5}
    if kind not in limits or type(count)is not int or count<=0:raise ValueError('Fixed metric role/count required')
    schema={'passed','finite','shape_matches','real_numeric','count','max_abs','nrmse','limit','floor'}
    if kind=='semantic':schema|={'top1_match','max_abs_limit'}
    if type(row)is not dict or set(row)!=schema:raise ValueError('Exact stored metric schema required')
    if type(row['count'])is not int or row['count']!=count:raise ValueError('Declared metric element count differs')
    if row['shape_matches']is not True or row['real_numeric']is not True:raise ValueError('Structural metric mismatch cannot be scientific evidence')
    if type(row['passed'])is not bool or type(row['finite'])is not bool:raise ValueError('Exact metric flags required')
    for key,wanted in [('limit',limits[kind]),('floor',.01)]:
        if type(row[key])is not float or row[key]!=wanted:raise ValueError('Frozen metric limit differs')
    for name in('max_abs','nrmse'):
        value=row[name]
        if value is not None and(type(value)is not float or not math.isfinite(value)or value<0):
            raise ValueError('Finite nonnegative recorded metric required')
    if row['finite']:
        if row['max_abs']is None or row['nrmse']is None:raise ValueError('Finite metric values missing')
        passed=row['nrmse']<=limits[kind]
    else:
        if row['nrmse']is not None:raise ValueError('Nonfinite metric cannot report finite nRMSE')
        passed=False
    if kind=='semantic':
        if type(top1_match)is not bool or row['top1_match']is not top1_match or type(row['max_abs_limit'])is not float or row['max_abs_limit']!=.001:
            raise ValueError('Exact semantic argmax/limit binding required')
        passed=passed and row['max_abs']<=.001 and top1_match
    if row['passed']is not passed:raise ValueError('Stored metric gate differs from recomputed threshold')
    return passed

def full_pass_inventory():
    rows=entries();operators=sum(sum(operator_counts(r['length']).values())for r in rows)
    blocks=sum(28*1024*r['length']for r in rows)
    return dict(contexts=3200,prefills=3072,teacher=64,free=64,original416_members=sum(r['original416_member']for r in rows),
        private_phase_journals=9600,private_files_with_ledger=9601,native_forwards=3232,
        operator_metric_rows=3200*421,block_metric_rows=3200*28,logit_metric_rows=3200,semantic_metric_rows=3200,
        operator_elements=operators,block_elements=blocks,logit_elements=3200*151936,semantic_elements=3200*151936,
        normalization_input_records_per_origin=3200*113,per_origin_score_query_rows=sum(28*r['length']for r in rows),
        scope='Full complete PASS metadata only; no actual journal/model read or array recomputation')

def activate_verification(*args,**kwargs):
    raise PermissionError('Actual private reconstruction requires separate reviewed root execution and immutable terminal pins')
