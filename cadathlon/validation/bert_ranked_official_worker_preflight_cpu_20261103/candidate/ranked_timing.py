"""Typed arithmetic for BERT stages 5–9; never a timer or evidence authenticator.

Only unchanged upstream ``Timing`` values can have kind ``upstream_harness``.
Collector, sideband validation and monotonic worker measurements must have kind
``component_diagnostic``. Callers authenticate source, hardware/cache policy,
case/input and successful official integer counts before constructing records.
This module neither reads payloads nor proves such authentication itself.

The pinned harness resets the seeded RNG inside each repeated-run iteration;
num_runs>1 shares setup and selection. Its repeat median is diagnostic for the
ranked recipe, which instead requires three fresh one-run invocations per size.
Failed/partial original evidence must be retained externally, never filtered
out or replaced to make one of these complete aggregates pass.
"""
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
import re

STAGES=((5,'Input preprocessing'),(6,'Input encryption'),(7,'Encrypted computation'),
        (8,'Result decryption'),(9,'Result postprocessing'))
SIZES=((1,100),(2,200),(3,408))
SEEDS=(42,43,44)
DEFAULT_NUM_RUNS=1
SOURCE_PINS=(
 ('run_submission.py','997fcf575999c6c0a35fa0561dd1a90d3f5565303bd1b6800c078511139f4590'),
 ('utils.py','e73b7ba164e18da00b57a53f8245a1df29cbfc3fc5424a7e734c0a4c7e2dabae'),
 ('params.py','e1e12436a385e1ca6f59f3c7b4ec0f4a4d3dc38c6ab9669125a66b4df6b9cd3b'))
HANDOUT_SHA256='650ff90e473fc6386aad7cfe28aedbb820476daad9ce2b5eb426cabe17a0abae'
_SHA=re.compile(r'[0-9a-f]{64}\Z')
_NONCE=re.compile(r'[0-9a-f]{32}\Z')


def verify_harness_sources(sources):
    """Verify explicit public source bytes; does no I/O or source execution."""
    if type(sources)is not dict or set(sources)!={n for n,_ in SOURCE_PINS}:
        raise ValueError('exact three upstream public source files required')
    if any(type(sources[n])is not bytes or hashlib.sha256(sources[n]).hexdigest()!=h for n,h in SOURCE_PINS):
        raise ValueError('unchanged upstream harness bytes required')
    return True


def _sha(value):
    if type(value)is not str or not _SHA.fullmatch(value):raise ValueError('exact caller-authenticated SHA256 required')


def _seconds(text):
    if type(text)is not str or not 2<=len(text)<=64 or not text.endswith('s') or text.startswith('-'):
        raise ValueError('original finite nonnegative harness seconds string required')
    try:value=Decimal(text[:-1]);numeric=float(value)
    except (ValueError,ArithmeticError):raise ValueError('finite seconds required') from None
    # Upstream uses str(round(seconds,4)), NOT four fixed decimal places.
    if not value.is_finite() or not math.isfinite(numeric) or value<0 or f'{round(numeric,4)}s'!=text:
        raise ValueError('original round(seconds,4) representation required')
    return value


@dataclass(frozen=True)
class TimingPolicy:
    kind:str
    source_inventory_sha256:str
    hardware_policy_sha256:str
    cache_policy_sha256:str
    fresh_setup_per_invocation:bool
    def __post_init__(self):
        if type(self.kind)is not str or self.kind not in ('upstream_harness','component_diagnostic'):
            raise ValueError('explicit upstream or diagnostic timer origin required')
        for name in ('source_inventory_sha256','hardware_policy_sha256','cache_policy_sha256'):_sha(getattr(self,name))
        if type(self.fresh_setup_per_invocation)is not bool:raise ValueError('exact setup-policy boolean required')


@dataclass(frozen=True)
class StageTiming:
    number:int
    name:str
    seconds_text:str
    passed:bool=True
    def __post_init__(self):
        if type(self.number)is not int or type(self.name)is not str or (self.number,self.name) not in STAGES:
            raise ValueError('one exact original stage5–9 number/name required')
        _seconds(self.seconds_text)
        if type(self.passed)is not bool:raise ValueError('exact stage status required')


def stages_from_harness_json(raw):
    """Select five labels from an original result; reject duplicate JSON keys.

    Setup/Total/Server Reported fields are preserved by the caller, never added
    to the online sum here. Parsing a result is not successful-run acceptance.
    """
    if type(raw)is not bytes or len(raw)>1<<20:raise ValueError('bounded original public timing JSON bytes required')
    def pairs(items):
        result={}
        for k,v in items:
            if k in result:raise ValueError('duplicate harness field')
            result[k]=v
        return result
    value=json.loads(raw,object_pairs_hook=pairs,parse_constant=lambda v:(_ for _ in ()).throw(ValueError('nonfinite JSON')))
    if type(value)is not dict or value.get('model_name')!='bert-base' or value.get('dataset_name')!='mrpc' or type(value.get('Timing'))is not dict:
        raise ValueError('original BERT/MRPC timing object required')
    if any(name not in value['Timing'] for _,name in STAGES):raise ValueError('all five online stage fields required')
    return tuple(StageTiming(number,name,value['Timing'][name]) for number,name in STAGES)


@dataclass(frozen=True)
class RunTiming:
    size:int
    seed:int
    invocation_id:str
    key_context_id:str
    input_sha256:str
    policy:TimingPolicy
    stages:tuple
    he_correct:int
    official_correct:int
    passed:bool=True
    run_number:int=1
    num_runs:int=DEFAULT_NUM_RUNS
    def __post_init__(self):
        if type(self.size)is not int or self.size not in dict(SIZES) or type(self.seed)is not int or self.seed<0:
            raise ValueError('exact ranked size and nonnegative seed required')
        if type(self.invocation_id)is not str or not _NONCE.fullmatch(self.invocation_id):raise ValueError('opaque invocation ID required')
        for name in ('key_context_id','input_sha256'):_sha(getattr(self,name))
        if type(self.policy)is not TimingPolicy or type(self.stages)is not tuple or len(self.stages)!=5:
            raise ValueError('typed policy and five ordered stages required')
        if any(type(s)is not StageTiming for s in self.stages) or tuple((s.number,s.name) for s in self.stages)!=STAGES:
            raise ValueError('missing/duplicate/reordered stage inventory')
        if type(self.num_runs)is not int or self.num_runs<1 or type(self.run_number)is not int or not 1<=self.run_number<=self.num_runs:
            raise ValueError('configured repeat count and one-based run number required')
        count=dict(SIZES)[self.size]
        if any(type(n)is not int or not 0<=n<=count for n in (self.he_correct,self.official_correct)) or type(self.passed)is not bool:
            raise ValueError('complete official integer accuracy counts/status required')

    @property
    def online_seconds(self):
        with localcontext() as ctx:
            ctx.prec=400
            return sum((_seconds(s.seconds_text) for s in self.stages),Decimal(0))


def _accepted(run):
    if type(run)is not RunTiming:raise ValueError('typed complete run required')
    count=dict(SIZES)[run.size]
    if run.passed is not True or any(s.passed is not True for s in run.stages):raise ValueError('failed run/stage cannot enter aggregate')
    if not (4*run.he_correct>=3*count and 100*(run.official_correct-run.he_correct)<=2*count):
        raise ValueError('original integer ranked accuracy gates required')


def _median(values):
    values=sorted(values);n=len(values)
    with localcontext() as ctx:
        ctx.prec=400
        return values[n//2] if n%2 else (values[n//2-1]+values[n//2])/2


def repeat_summary(runs,*,num_runs=DEFAULT_NUM_RUNS):
    """Median of per-run stage sums, for one configured harness invocation."""
    if type(num_runs)is not int or num_runs<1 or type(runs)not in (tuple,list) or len(runs)!=num_runs:
        raise ValueError('every configured repeat must be present')
    for run in runs:_accepted(run)
    first=runs[0]
    for i,run in enumerate(runs,1):
        if (run.run_number!=i or run.num_runs!=num_runs or
                any(getattr(run,n)!=getattr(first,n) for n in ('size','seed','invocation_id','key_context_id','input_sha256','policy'))):
            raise ValueError('mixed case/selection/policy or incomplete/reordered repeat run')
    values=[run.online_seconds for run in runs]
    return dict(size=first.size,seed=first.seed,num_runs=num_runs,timer_kind=first.policy.kind,
        per_run_online_seconds=[str(v) for v in values],median_online_seconds=str(_median(values)),
        source='sum of each complete run stages5–9, then median; never sum of stage medians',
        ranked_recipe_complete=False,setup_shared_within_repeats=num_runs>1,
        seed_selection_repeated_by_upstream=num_runs>1)


def seed_summary(runs,*,seeds=SEEDS):
    """Required three fresh one-run invocations for ONE ranked size."""
    if type(seeds)is not tuple or seeds!=SEEDS or any(type(n)is not int for n in seeds):
        raise ValueError('fixed published three-seed schedule required')
    if type(runs)not in (tuple,list) or len(runs)!=3:raise ValueError('all three invocations required')
    for run in runs:_accepted(run)
    first=runs[0]
    if tuple(run.seed for run in runs)!=seeds:raise ValueError('exact ordered seed schedule required')
    if any(run.size!=first.size or run.policy!=first.policy or run.num_runs!=1 or run.run_number!=1 for run in runs):
        raise ValueError('mixed size/policy or repeated-run records cannot substitute fresh invocations')
    if first.policy.kind!='upstream_harness' or first.policy.fresh_setup_per_invocation is not True:
        raise ValueError('diagnostic/retained-setup timers cannot become official harness timing')
    if len({r.invocation_id for r in runs})!=3 or len({r.key_context_id for r in runs})!=3:
        raise ValueError('three independent fresh invocation/key identities required')
    values=[r.online_seconds for r in runs];median=_median(values)
    if median<=0:raise ValueError('positive measured latency required for throughput')
    with localcontext() as ctx:
        ctx.prec=40;throughput=Decimal(dict(SIZES)[first.size])/median
    return dict(size=first.size,count=dict(SIZES)[first.size],seeds=list(seeds),num_runs_each=1,
        per_invocation_online_seconds=[str(v) for v in values],median_online_seconds=str(median),
        throughput_samples_per_second=str(throughput),timer_kind='upstream_harness',
        ranked_timing_recipe_complete=True,accuracy_counts_validated=True,
        authentication_is_external=True,full_benchmark_acceptance_not_implied=True)
