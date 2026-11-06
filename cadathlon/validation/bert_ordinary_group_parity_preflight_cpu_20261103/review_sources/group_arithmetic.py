"""Bounded ordinary/group representation comparison, with no decoder/oracle."""
import time
from parity_arithmetic import metadata,fingerprint,check_level,compare,verify_reduction,OwnerCleanup
from group_protocol import LEVELS,GROUPS,MAX_RESULTS,MAX_RESULT_BYTES,MAX_DIGIT_BYTES

def exercise(ctx,value,baseline,grouped,*,progress,security):
    initial=fingerprint(value);security(baseline,value);security(grouped,value)
    result=dict(passed=False,comparisons=[],groups=[],input=initial)
    current=value
    for level in LEVELS:
        if level==25:
            current=ctx.he_level_reduce[49,25].level_reduce(value)
            result['level_reduction']=verify_reduction(value,current,ctx)
        check_level(current,ctx,level);security(grouped,current);source=fingerprint(current)
        for name,indices in GROUPS:
            started=time.monotonic();right=grouped.rotate_many(current,indices)
            if type(right)is not tuple or len(right)!=len(indices):raise ValueError('complete ordered group result required')
            stats=dict(grouped.rotation.last_group)
            nonzero=sum(i%ctx.num_slots!=0 for i in indices)
            expected_bytes=nonzero*int(current.polynomial.size)*int(current.polynomial.dtype.itemsize)
            if (stats.get('mode')!='digits' or stats.get('requests')!=len(indices) or stats.get('decompositions')!=1
                    or stats.get('result_bytes')!=expected_bytes or not 0<stats.get('digit_bytes',0)<=MAX_DIGIT_BYTES
                    or expected_bytes>MAX_RESULT_BYTES or len(indices)>MAX_RESULTS):
                raise ValueError('actual same-input digits path/count/budget differs')
            # Baseline one result at a time; the bounded group tuple is retained
            # only within this group. Duplicate nonzero outputs are independent.
            seen={}
            for sequence,(index,answer) in enumerate(zip(indices,right)):
                normalized=index%ctx.num_slots
                if normalized==0 and answer is not current:raise ValueError('zero rotation must alias the original input')
                if normalized and any(answer is old for old in seen.values()):raise ValueError('duplicate nonzero results must be separate objects')
                if normalized:seen[sequence]=answer
                left=baseline.rotate(current,index)
                if normalized==0 and left is not current:raise ValueError('ordinary zero rotation must alias input')
                security(baseline,left);security(grouped,answer)
                row=compare(left,answer,current,ctx,level)
                row.update(level=level,group=name,index=index,normalized_index=normalized,sequence=sequence,
                    zero_alias=normalized==0 and left is answer is current)
                result['comparisons'].append(row);progress(row)
                if not row['passed']:raise ValueError('ordinary/group payload or metadata parity failed')
                if fingerprint(current)!=source or fingerprint(value)!=initial:raise ValueError('original input changed')
                del left,answer
            result['groups'].append(dict(level=level,group=name,stats=stats,elapsed_group_s=time.monotonic()-started))
            del right,seen
    nonzero=sum(r['normalized_index']!=0 for r in result['comparisons'])
    result.update(passed=True,actual_rotation_calls=2*nonzero,nonzero_pair_comparisons=nonzero,
        zero_alias_comparisons=len(result['comparisons'])-nonzero,shared_decompositions=sum(g['stats']['decompositions'] for g in result['groups']),
        input_unchanged=fingerprint(value)==initial)
    return result
