"""Public common-ring/scale/key-storage plan only; no contexts or encrypted operations."""
from pathlib import Path
import hashlib,json,os,shutil,sys
sys.path.insert(0,str(Path(__file__).parent))
import proof

def prepare():
    import jaxite_word
    from jaxite_word import key_gen
    from jaxite_word.bootstrapping import Bootstrap
    def forbidden(*args,**kwargs):raise AssertionError('context/key generation forbidden in public preflight')
    jaxite_word.CKKSContext.__init__=forbidden;Bootstrap.__init__=forbidden
    for module in {sys.modules[n] for n in ('key_gen','jaxite_word.key_gen')}:
        for name in proof.support.KEYGEN_METHODS:setattr(module,name,forbidden)
    params,plan=proof.public_parameters();usage=shutil.disk_usage('/dev/shm')
    report={'passed':True,'scope':__doc__,'plan':plan,'key_generation_context_ciphertext_or_HE_launched':False,
        'CPU_backend_may_initialize_through_public_package_import':True,'cpu_affinity':sorted(os.sched_getaffinity(0)),
        'tmpfs_observation':dict(total=usage.total,used=usage.used,free=usage.free,fits_raw_keys_plus_margin=usage.free>=plan['required_free_bytes']),
        'input_pattern_sha256':hashlib.sha256(proof.fixtures.values().astype('<f8').tobytes()).hexdigest()}
    (proof.LAB/'public_plan.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'routekeys':plan['rotation_count'],'raw_key_GiB':plan['raw_rotation_GiB'],'QP':plan['actual_log2_QP'],
        'noncanonical_scale':plan['noncanonical_actual_scale'],'stage_scales':plan['exact_stage_scales'],'tmpfs_free':usage.free},indent=2))
if __name__=='__main__':prepare()
