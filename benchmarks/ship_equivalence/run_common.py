"""Run full bootstrap correctness under an explicit common parameter manifest.

CROSS sources are selected through PYTHONPATH; Jaxite is loaded only for its
variant. Timing is allowed only after both bootstrap and square gates pass.
"""
import argparse
import copy
import hashlib
import importlib.metadata
import resource
import json
import math
import os
from pathlib import Path
import platform
import random as std_random
import subprocess
import time
import traceback

import jax
import jaxlib
import jax.numpy as jnp
import numpy as np
import bootstrapping
import ckks_ctx
import key_gen
from ship import SHIPConfig, ntt_cpu, random

jax.config.update('jax_enable_x64', True)
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--variant', choices=['cross-native', 'cross-ship', 'jaxite-patched', 'jaxite-optimized'], required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--iterations', type=int, default=0)
parser.add_argument('--profile-dir', type=Path)
args = parser.parse_args()
if args.iterations < 0:
    parser.error('iterations must be nonnegative')
if args.profile_dir:
    if jax.default_backend() != 'tpu':
        parser.error('Trace collection requires TPU')
    options = jax.profiler.ProfileOptions()
    options.host_tracer_level = 0
    options.python_tracer_level = 0
    options.advanced_configuration = {'tpu_trace_mode': 'TRACE_ONLY_XLA'}
m = json.loads(args.manifest.read_text())
n, q, p = m['degree'], m['q'], m['p']
scale = m['scale']
r = 1 << ((n.bit_length()-1)//2)
c = n//r
report = dict(variant=args.variant, manifest=m,
              manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
              jax_version=jax.__version__, numpy_version=np.__version__,
              jaxlib_version=jaxlib.__version__, python_version=platform.python_version(),
              host=platform.node(), visible_chips=os.environ.get('TPU_VISIBLE_CHIPS'),
              harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              backend=jax.default_backend(), device_kind=jax.devices()[0].device_kind,
              scope='full encrypted slot-to-slot bootstrap, output normalization included',
              security_qualified=False, errors={}, post_square_errors={}, passed=False)
def provenance(module):
    path = Path(module.__file__).resolve()
    root = subprocess.check_output(['git','-C',str(path.parent),'rev-parse','--show-toplevel'],text=True).strip()
    sha = subprocess.check_output(['git','-C',root,'rev-parse','HEAD'],text=True).strip()
    patch = subprocess.check_output(['git','-C',root,'diff','HEAD','--'])
    return dict(commit=sha, patch_sha256=hashlib.sha256(patch).hexdigest(), file=str(path))
report['cross_source'] = provenance(bootstrapping)
report['libtpu_version'] = (importlib.metadata.version('libtpu')
                            if report['backend'] == 'tpu' else None)
report['xla_flags'] = os.environ.get('XLA_FLAGS', '')
report['stack_limit_bytes'] = resource.getrlimit(resource.RLIMIT_STACK)[0]
report['persistent_cache_enabled'] = bool(os.environ.get('JAX_COMPILATION_CACHE_DIR'))
def save():
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
def stage(name):
    report['stage'] = name
    save()
    print(name, flush=True)
try:
    stage('context setup')
    keys = key_gen.gen_pke_pair(q, [], n)
    ctx = ckks_ctx.CKKSContext(dict(degree=n,num_slots=n//2,q_towers=q,p_towers=p,
        scaling_factor=scale,output_scale=scale,p=round(math.log2(scale)),
        CKKS_M_FACTOR=1,max_bits_in_word=31,composite_degree=1,noise_scale_degree=1,**keys))
    ctx.program_initialization(total_rotation_indices=[],dnum=m['dnum'],r=r,c=c,batch=1)
    if args.variant == 'cross-native':
        engine = bootstrapping.Bootstrap(ctx)
        engine.control_gen(level_budget=m['level_budget'])
        if 'native_correction' in m:
            engine._correctionFactor = m['native_correction']
            engine._correction = engine._correctionFactor - engine._deg
            engine._corFactor = float(2**engine._correction)
        engine.setup_key()
        if m.get('native_dc_subtract', False):
            stage('native encrypted-zero template setup')
            engine._dc_subtract = True
            engine.precompute_dc_template()
        old_checkpoint = engine._stage_checkpoint
        def checkpoint(tag, ct=None):
            old_checkpoint(tag,ct)
            print('native stage: '+tag, flush=True)
        engine._stage_checkpoint = checkpoint
        def apply(ct):
            target = copy.deepcopy(ct)
            if m['meta_precision']:
                out = engine.meta_bootstrap(ct,precision=m['meta_precision'])
            else:
                out = engine.bootstrap(ct)
            # Scale alignment requires equal noise-scale degrees. Normalize
            # the S2C nsd=2 result before aligning to the nsd=1 input contract.
            out, _ = engine._ensure_nsd1(out,engine._infer_level(out))
            out, _ = engine._adjust_ct_to_match_scale(out,engine._infer_level(out),target)
            lev = engine._infer_level(out)
            return out if lev == 1 else ctx.he_level_reduce[lev,1].level_reduce(out)
    else:
        factor = scale*m['gamma']/q[0]
        config = SHIPConfig(weight=m['weight'],theta=m['theta'],gamma=m['gamma'],input_scale_factor=factor)
        plan = config.plan(ctx)
        if not math.isclose(plan.input_scale,scale,rel_tol=1e-12) or plan.output_scale != scale:
            raise ValueError('SHIP boundary scales do not match the manifest')
        stage('native SHIP setup and common input adapter')
        accessor = ctx.he_bootstrap.configure(scheme='ship',ship_config=config).setup(random_source=random.TestRandomSource(m['seed']))
        engine = accessor._engine
        if args.variant.startswith('jaxite'):
            from jaxite.jaxite_ckks import bootstrapping as jb, types as jt, random as jr
            report['jaxite_source'] = provenance(jb)
            stage('Jaxite half-bootstrap setup')
            sparse = np.zeros(n,np.int64)
            support = std_random.Random(m['seed']).sample(range(n),m['weight'])
            sparse[support] = 1
            sparse[support[:m['weight']//2]] = -1
            sparse_data = ntt_cpu.ntt_negacyclic_poly(np.column_stack([sparse%v for v in q]).astype(np.uint64),q)
            sk = jt.SecretKey(sparse_data,np.array(q,np.uint32))
            dense = jt.SecretKey(np.asarray(ctx.secret_key,np.uint64).T,np.array(q,np.uint32))
            k = jb.SHIP()
            k.precompute_constants(q,p,n,m['dnum'],sk,m['theta'],jr.TestRandomSource(m['seed']),output_sk=dense)
            def half(arr, kernel):
                ct = jt.Ciphertext(arr.reshape(2,n,1),jnp.array(q[:1],jnp.uint32))
                out = kernel.half_bootstrap(ct,m['theta'],scale,m['gamma'])
                return out.data.reshape(1,2,r,c,plan.expected_output_level+1)
            engine.keys = k
            engine._half = jax.jit(half)
            engine.kernel = None
            # Native half setup is reused only to construct the existing input
            # adapter. Its unused half keys/executable are replaced above; all
            # setup and warmup costs remain outside reported bootstrap timings.
            jax.clear_caches()
        def apply(ct):
            out = engine.bootstrap(ct)
            return ctx.he_level_reduce[plan.expected_output_level,1].level_reduce(out)
    fixtures = dict(ramp=np.linspace(-m['magnitude'],m['magnitude'],n//2),
                    random=np.random.default_rng(m['seed']).uniform(-m['magnitude'],m['magnitude'],n//2))
    for name,values in fixtures.items():
        stage('bootstrap correctness: '+name)
        ct = ctx.encrypt(ctx.encode_at_level(values.tolist(),1,scale=scale))
        out = apply(copy.deepcopy(ct))
        jax.block_until_ready(out.polynomial)
        if list(map(int,out.moduli)) != q[:2]:
            raise ValueError('Output Q subset mismatch')
        actual_scale = float(out._ckks_scale)
        if not math.isclose(actual_scale,scale,rel_tol=1e-10) or out._ckks_nsd != 1:
            raise ValueError(f'Output scale/NSD mismatch: {actual_scale}, {out._ckks_nsd}')
        error = float(np.max(np.abs(ctx.decrypt_slots(out)-values)))
        report['errors'][name]=error
        save()
        if not math.isfinite(error) or error>m['error_gate']:
            raise ValueError(f'{name} bootstrap error {error} exceeds {m["error_gate"]}')
        # Exercise the supported public operation on every refreshed result.
        squared = ctx.he_mul[0].mul(out,out)
        square_error = float(np.max(np.abs(ctx.decrypt_slots(squared)-values**2)))
        report['post_square_errors'][name]=square_error
        save()
        if not math.isfinite(square_error) or square_error>m['error_gate']:
            raise ValueError(f'{name} square error {square_error} exceeds gate')
    report['passed']=True
    report['output_q']=q[:2]
    report['output_scale']=actual_scale
    report['output_level']=1
    report['output_nsd']=1
    report['slots']=n//2
    report['retained_multiplication_levels']=1
    stage('warm benchmark')
    samples=[]
    if args.iterations and jax.default_backend()!='tpu':
        raise ValueError('Comparative latency collection requires TPU; CPU runs are correctness only')
    if args.variant=='cross-native':
        engine._stage_checkpoint=old_checkpoint
    for index in range(args.iterations):
        fresh=copy.deepcopy(ct)
        jax.block_until_ready(fresh.polynomial)
        started=time.perf_counter()
        result=apply(fresh)
        jax.block_until_ready(result.polynomial)
        samples.append(1000*(time.perf_counter()-started))
    report['warm_wall_ms']=samples
    if args.profile_dir:
        if jax.default_backend()!='tpu':
            raise ValueError('Trace collection requires TPU')
        from device_profile import capture
        report['device_profiles']=[]
        for index in range(args.iterations):
            fresh=copy.deepcopy(ct)
            jax.block_until_ready(fresh.polynomial)
            report['device_profiles'].append(capture(
                lambda: apply(fresh), args.profile_dir/str(index),
                checkpoint_owner=engine if args.variant == 'cross-native' else None))
            save()
    stage('complete')
except Exception as exc:
    report.update(passed=False,error_type=type(exc).__name__,error=str(exc),traceback=traceback.format_exc())
    save()
    print(report['traceback'],flush=True)
    raise SystemExit(1)
save()
print(json.dumps({k:v for k,v in report.items() if k!='manifest'},indent=2))
