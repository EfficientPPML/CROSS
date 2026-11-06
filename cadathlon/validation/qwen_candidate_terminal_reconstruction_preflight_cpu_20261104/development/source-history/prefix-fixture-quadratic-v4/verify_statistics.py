"""Independent recorded-statistic reconstruction. Standard library, no IO.

No producer modules are imported. Source-bound callers must authenticate the
manifest and journal bytes before passing parsed objects here.
"""
import copy
import hashlib
import json
import math
import struct

INPUT_SCOPE='Input telemetry only; no inverse/softmax/model evaluation or output-substitution claim.'
DOMAIN_SCOPE='Candidate propagated inputs only; weighted normalization outputs are separate operator metrics'

def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def same(x,y):
    if canonical(x)!=canonical(y):raise ValueError('Typed recorded binding differs')
def fields(x,names):
    if type(x)is not dict or set(x)!=set(names):raise ValueError('Exact recorded schema required')
def integer(x,low,high=None):
    if type(x)is not int or not low<=x<=(low if high is None else high):raise ValueError('Exact bounded integer required')
    return x
def digest(x):
    if type(x)is not str or len(x)!=64 or any(c not in '0123456789abcdef'for c in x):raise ValueError('SHA256 required')
    return x
def token(x):return integer(x,0,151935)
def tokens(x,n):
    if type(x)is not list or len(x)!=n:raise ValueError('Typed token prefix required')
    for v in x:token(v)
    return tuple(x)
def context_hash(prefix):return hashlib.sha256(struct.pack('<'+'q'*len(prefix),*prefix)).hexdigest()
def bound_hash(value,pin):
    if hashlib.sha256(canonical(value)).hexdigest()!=digest(pin):raise ValueError('Recorded content digest differs')
def finite(x):
    if type(x)is not float or not math.isfinite(x):raise ValueError('Finite binary64 statistic required')
    return x
def interval(x):
    if type(x)is not list or len(x)!=2 or any(type(v)not in(int,float)or not math.isfinite(v)for v in x)or x[0]>=x[1]:raise ValueError('Finite ordered interval required')
    return x

def merge(left,right,counters):
    if left is None:return copy.deepcopy(right)
    for k in counters:left[k]+=right[k]
    for key,fn in [('minimum',min),('maximum',max)]:
        values=[v for v in(left[key],right[key])if v is not None]
        left[key]=fn(values)if values else None
    return left

def stat(row,count,positive=False):
    fields(row,('count','minimum','maximum','nonfinite','nonpositive'))
    integer(row['count'],count);integer(row['nonfinite'],0);integer(row['nonpositive'],0,count)
    low,high=finite(row['minimum']),finite(row['maximum'])
    if low>high or positive and(low<=0 or row['nonpositive']):raise ValueError('Input extrema inconsistent')
    if low>0 and row['nonpositive'] or high<=0 and row['nonpositive']!=count:raise ValueError('Input sign count inconsistent')

def input_counts(s):
    out={f'{i}.{part}.inverse_sqrt':s*width for i in range(28)for part,width in [('input',1),('post',1),('q',16),('k',8)]}
    out['final.inverse_sqrt']=1
    out.update({f'{i}.silu':3072*s for i in range(28)})
    out.update({f'{i}.softmax.{p}':16*s*(s+1)//2 for i in range(28)for p in('raw_valid','centered')})
    return out

def verify_inputs(row,s,origin):
    fields(row,('sequence_length','ranges','score_rows','ignored_exact_output_callbacks','complete','invalid','norm_input_sites','silu_gate_sites','scope'))
    integer(s,1,32)
    if origin not in('native','exact'):raise ValueError('Recorded input origin required')
    for k,v in dict(sequence_length=s,ignored_exact_output_callbacks=0 if origin=='native'else 169,complete=True,invalid=False,norm_input_sites=113,silu_gate_sites=28,scope=INPUT_SCOPE).items():same(row[k],v)
    counts=input_counts(s);fields(row['ranges'],counts)
    for name,n in counts.items():stat(row['ranges'][name],n,name.endswith('.inverse_sqrt'))
    fields(row['score_rows'],map(str,range(28)))
    for layer,queries in row['score_rows'].items():
        fields(queries,map(str,range(1,s+1)))
        totals={}
        for k in range(1,s+1):
            q=queries[str(k)];fields(q,('query','key_count','shape','k1_bypass','reciprocal_observations','raw_valid','centered'))
            for key,value in dict(query=k-1,key_count=k,shape=[1,16,k],k1_bypass=k==1,reciprocal_observations=0).items():same(q[key],value)
            for phase in('raw_valid','centered'):
                stat(q[phase],16*k)
                totals[phase]=merge(totals.get(phase),q[phase],('count','nonfinite','nonpositive'))
            if k==1 and(q['centered']['minimum']!=0. or q['centered']['maximum']!=0.):raise ValueError('K1 centered diagnostic differs')
        for phase in totals:same(row['ranges'][f'{layer}.softmax.{phase}'],totals[phase])
    return row

class InputAggregate:
    def __init__(self):self.contexts=0;self.ranges={};self.rows={}
    def add(self,row):
        self.contexts+=1
        for label,value in row['ranges'].items():self.ranges[label]=merge(self.ranges.get(label),value,('count','nonfinite','nonpositive'))
        for layer,queries in row['score_rows'].items():
            target=self.rows.setdefault(layer,{})
            for k,value in queries.items():
                if k not in target:target[k]=copy.deepcopy(value);target[k]['contexts']=1
                else:
                    q=target[k];q['contexts']+=1
                    for phase in('raw_valid','centered'):merge(q[phase],value[phase],('count','nonfinite','nonpositive'))
    def report(self):
        return dict(contexts=self.contexts,ranges=self.ranges,score_rows=self.rows,reciprocal_inputs_observed=False,scope='Native/exact input observations only; K1 is an explicit bypass.')

def domain_plan(manifest,s):
    """Independent literal input event inventory, not polynomial evaluation."""
    integer(s,1,32);rules=manifest['nonlinear'];out=[]
    names={f'{l}.{p}'for l in range(28)for p in('input.inverse_sqrt','q.inverse_sqrt','k.inverse_sqrt','post.inverse_sqrt','silu')}|{'final.inverse_sqrt'}
    fields(rules,names);fields(manifest['softmax'],map(str,range(28)))
    def add(label,rule,count):out.append((label,interval(rule['domain']),count))
    def norm(label,count):
        rule=rules[label]
        if rule['method']not in('newton','polynomial'):raise ValueError('Norm method differs')
        if rule['method']=='newton':integer(rule['iterations'],1,64)
        add(label,rule,count)
    for layer in range(28):
        p=str(layer)
        norm(p+'.input.inverse_sqrt',s);norm(p+'.q.inverse_sqrt',16*s);norm(p+'.k.inverse_sqrt',8*s)
        soft=manifest['softmax'][p];rounds=integer(soft['squarings'],1,64)
        for k in range(2,s+1):
            base=f'{p}.attention.query{k-1}'
            add(base+'.centered',{'domain':soft['centered_domain']},16*k)
            add(base+'.exp',manifest['exp'],16*k)
            initial=manifest['initial_reciprocal'];integer(initial['iterations'],1,64)
            add(base+'.initial_inverse',initial,16)
            reciprocal=manifest['reciprocal_by_keys'][str(k)];integer(reciprocal['iterations'],1,64)
            for j in range(rounds):add(base+f'.round{j}.inverse',reciprocal,16)
        norm(p+'.post.inverse_sqrt',s)
        rule=rules[p+'.silu'];add(p+'.silu',rule,3072*s)
        if rule['method']=='rational_tanh':
            add(p+'.silu.tanh_base',rule['base'],3072*s);integer(rule['reciprocal']['iterations'],1,64)
            for j in range(integer(rule['doublings'],1,64)):add(p+f'.silu.inverse{j}',rule['reciprocal'],3072*s)
        elif rule['method']!='polynomial':raise ValueError('SiLU method differs')
    norm('final.inverse_sqrt',1)
    return tuple(out)

def domain_stat(row,domain,count):
    fields(row,('domain','count','minimum','maximum','below','above','nonfinite'));same(row['domain'],domain);integer(row['count'],count)
    for key in('below','above','nonfinite'):integer(row[key],0,count)
    if row['below']+row['above']>count:raise ValueError('Disjoint below/above counts exceed population')
    lo,hi=row['minimum'],row['maximum'];nf=row['nonfinite']
    if nf==count:
        if lo is not None or hi is not None:raise ValueError('All nonfinite population has finite extrema')
    else:
        finite(lo);finite(hi)
        if lo>hi:raise ValueError('Domain extrema reversed')
        # Infinities may contribute to both side and nonfinite counters.
        for n,violates in [('below',lo<domain[0]),('above',hi>domain[1])]:
            if violates and not row[n]or not violates and row[n]>nf:raise ValueError('Domain side count/extrema inconsistent')
    return bool(row['below']or row['above']or nf)

def verify_domains(observed,plan,complete,failure):
    fields(observed,('expected_observations','observations','domains','normalization_variance','scope'))
    same(observed['scope'],DOMAIN_SCOPE);integer(observed['expected_observations'],len(plan));n=integer(observed['observations'],0,len(plan))
    expected=plan[:n];fields(observed['domains'],[x[0]for x in expected]);bad=[]
    for label,domain,count in expected:
        if domain_stat(observed['domains'][label],domain,count):bad.append(label)
    same(observed['normalization_variance'],{k:v for k,v in observed['domains'].items()if k.endswith('.inverse_sqrt')})
    if complete:
        if n!=len(plan)or bad or failure is not None:raise ValueError('Complete candidate domain inventory differs')
    else:
        fields(failure,('type','label','record'))
        if failure['type']=='InvalidCircuit':
            if type(failure['label'])is not str or not failure['label']:raise ValueError('Typed stopped operation required')
            if bad:
                if bad!=[expected[-1][0]]or failure['label']!=bad[0]:raise ValueError('First rejected input must be final observed event')
                same(failure['record'],observed['domains'][bad[0]])
            else:
                fields(failure['record'],('nonfinite',));integer(failure['record']['nonfinite'],1,1<<50)
        elif failure['type']=='NonfiniteOutput':
            if bad or failure['label']is not None or failure['record']is not None:raise ValueError('Recorded nonfinite output failure differs')
        else:raise ValueError('Unknown candidate stop cannot authorize scientific revision')
    return observed['domains']

def partial_capture_consistency(observed,plan,operator_count,block_count):
    """Cross-bind stored prefixes to the literal forward's observation points.

    An input event is reached only after a fixed preceding operator prefix. Its
    dependent output is unavailable until every required input event succeeds.
    Residual addition can fail after a down output but before block publication.
    """
    def before(label):
        if label=='final.inverse_sqrt':return 420
        layer=int(label.split('.',1)[0]);suffix=label.split('.',1)[1]
        if suffix=='input.inverse_sqrt':offset=0
        elif suffix=='q.inverse_sqrt':offset=4
        elif suffix=='k.inverse_sqrt':offset=6
        elif suffix.startswith('attention.'):offset=8
        elif suffix=='post.inverse_sqrt':offset=10
        elif suffix=='silu'or suffix.startswith('silu.'):offset=13
        else:raise ValueError('Unknown independent event stage')
        return 15*layer+offset
    n=observed['observations'];rows=observed['domains'];completed=set()
    for label,domain,count in plan[:n]:
        if operator_count<before(label):raise ValueError('Observed input precedes missing required operator outputs')
        row=rows[label]
        if not(row['below']or row['above']or row['nonfinite']):completed.add(label)
    # The event with largest before-index relevant to each output is required;
    # multiple tanh/softmax events at the same index must all have completed.
    for label,domain,count in plan:
        if before(label)<operator_count and label not in completed:raise ValueError('Output appears after an unobserved or rejected dependency')
    if n<len(plan)and operator_count>before(plan[n][0]):raise ValueError('Output passed the next unobserved input')
    floor=max(0,(operator_count-1)//15)
    observed_layers=[28 if label=='final.inverse_sqrt'else int(label.split('.',1)[0])for label,_,_ in plan[:n]]
    minimum=max([floor,*observed_layers])
    maximum=min(28,operator_count//15)
    if not minimum<=block_count<=maximum:raise ValueError('Block prefix is inconsistent with actual operator/input dependencies')
