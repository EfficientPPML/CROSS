"""Pure external readonly-role binding. No filesystem or NumPy access.

The caller's source/grant digests are trust inputs, not report-derived values.
Constructing this object does not bypass the fresh CLI's source/path/audit gate.
"""
from dataclasses import dataclass
from model_admission import (Aggregate, ExpectedStoreIdentity, MODEL_ID,
    AGGREGATE_SHA256, CATALOG_SHA256, _json, _object, _source_map, _path)
from qwen_file_contract import PublicContract, canonical, sha, hash_value, same
from qwen_public_parse import parse
from public_source_pins import PUBLIC_PINS

ROLE = 'validate_existing_model_stores'
FORBIDDEN = ('jax', 'jaxlib', 'jaxite_word', 'torch', 'transformers',
             'tensorflow', 'flax', 'torch_xla', 'safetensors', 'tokenizers', 'libtpu')
POLICY_CONSTANTS = dict(schema='qwen-model-store-readonly-policy-v1',
    default_execution_authorized=False, external_readonly_grant_required=True,
    role=ROLE, cores=[120,121,122,123], host_numeric_threads=1,
    minimum_soft_nofile=512, memory_margin_bytes=512<<20,
    evidence_reserve_bytes=128<<20, planning_body_fds=398,
    runtime_fd_allowance=32, body_copy_bytes=0,
    forbidden_imports=list(FORBIDDEN), maximum_source_files=512,
    maximum_source_file_bytes=8<<20, maximum_source_bytes=32<<20,
    maximum_source_map_bytes=1<<20, maximum_grant_bytes=1<<16,
    maximum_review_bytes=1<<16, maximum_store_metadata_bytes=2<<20,
    maximum_report_bytes=2<<20)
GRANT_CONSTANTS = dict(schema='qwen-model-store-readonly-grant-v1', role=ROLE,
    execute=True, model_id=MODEL_ID, aggregate_manifest_sha256=AGGREGATE_SHA256,
    catalog_sha256=CATALOG_SHA256, cores=[120,121,122,123],
    host_numeric_threads=1, minimum_soft_nofile=512,
    memory_margin_bytes=512<<20, evidence_reserve_bytes=128<<20,
    parameter_body_read_only=True, new_export_authorized=False,
    model_execution_authorized=False, key_or_ciphertext_access_authorized=False,
    token_or_calibration_access_authorized=False, selection_or_revision_authorized=False)


def _descriptor(value):
    _object(value, ('path','sha256'), 'public role descriptor')
    _path(value['path']); hash_value(value['sha256'])
    return value


def public_requests(grant_bytes):
    """Bounded parse only. Returned paths are not yet authorized to be opened."""
    v = _json(grant_bytes, 1<<16, 'readonly grant')
    fields = set(GRANT_CONSTANTS) | {'source_root','source_manifest_sha256',
        'source_archive_sha256','expected_stores','prelaunch_review',
        'evidence_output','dependency_policy_sha256','readonly_policy_sha256'}
    _object(v, fields, 'readonly grant')
    same({k:v[k] for k in GRANT_CONSTANTS}, GRANT_CONSTANTS, 'readonly grant policy differs')
    for name in ('source_root','evidence_output'): _path(v[name])
    for name in ('source_manifest_sha256','source_archive_sha256',
                 'dependency_policy_sha256','readonly_policy_sha256'): hash_value(v[name])
    for name in ('expected_stores','prelaunch_review'): _descriptor(v[name])
    return {name:dict(v[name]) for name in ('expected_stores','prelaunch_review')}


def policies(blobs):
    policy = parse(blobs['model_readonly_policy.json'])
    _object(policy, set(POLICY_CONSTANTS)|{'protected_paths','bootstrap_borrowed_roots'}, 'readonly source policy')
    same({k:policy[k] for k in POLICY_CONSTANTS}, POLICY_CONSTANTS, 'exact readonly source policy differs')
    for name in ('protected_paths','bootstrap_borrowed_roots'):
        rows=policy[name]
        if type(rows)is not list or not rows or len(rows)!=len(set(rows)):
            raise ValueError('explicit unique protected paths required')
        for row in rows: _path(row)
    dep = parse(blobs['model_dependencies.json'])
    _object(dep, ('schema','interpreter','python_major_minor','numpy_sources',
        'numpy_origins','library_roots','environment','complete_binary_dependency_closure_claimed'), 'dependency policy')
    same({n:dep[n] for n in ('schema','python_major_minor','complete_binary_dependency_closure_claimed')},
         dict(schema='qwen-model-store-dependencies-v1',python_major_minor=[3,13],
              complete_binary_dependency_closure_claimed=False), 'dependency scope differs')
    _path(dep['interpreter'])
    _object(dep['numpy_origins'], ('numpy','numpy.lib._format_impl','numpy.lib._npyio_impl'), 'NumPy origins')
    if type(dep['numpy_sources'])is not dict or len(dep['numpy_sources'])!=4:
        raise ValueError('four selected NumPy sources required')
    for path,digest in dep['numpy_sources'].items(): _path(path); hash_value(digest)
    for path in dep['numpy_origins'].values():
        _path(path)
        if path not in dep['numpy_sources']: raise ValueError('NumPy origin lacks a source pin')
    if type(dep['library_roots'])is not list or not dep['library_roots']:
        raise ValueError('explicit installed library roots required')
    for path in dep['library_roots']: _path(path)
    same(dep['environment'], dict(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',
        MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1'), 'host thread environment differs')
    return policy,dep


@dataclass(frozen=True)
class ReadonlyLaunch:
    source_root:str
    output_path:str
    source_map_bytes:bytes
    source_manifest_sha256:str
    blobs:tuple
    grant_bytes:bytes
    grant_sha256:str
    store_bytes:bytes
    review_bytes:bytes
    def check(self):
        if type(self)is not ReadonlyLaunch: raise TypeError('exact retained readonly launch required')
        return authenticate(self.grant_bytes,self.grant_sha256,self.store_bytes,
            self.review_bytes,source_root=self.source_root,output_path=self.output_path,
            source_map_bytes=self.source_map_bytes,
            source_manifest_sha256=self.source_manifest_sha256,blobs=dict(self.blobs))
    @property
    def stores(self):
        self.check(); values=dict(self.blobs)
        public=PublicContract.from_blobs({n:values['public_inputs/'+n] for n in PUBLIC_PINS})
        return ExpectedStoreIdentity(Aggregate(public,values['unadmitted_aggregate_manifest.json']),self.store_bytes)


def authenticate(grant_bytes,grant_sha256,store_bytes,review_bytes,*,source_root,
                 output_path,source_map_bytes,source_manifest_sha256,blobs):
    """Reauthenticate exact bytes; caller separately enforces path/resource IO gates."""
    hash_value(grant_sha256); hash_value(source_manifest_sha256)
    _path(source_root); _path(output_path)
    if sha(grant_bytes)!=grant_sha256: raise ValueError('external readonly grant digest differs')
    requests=public_requests(grant_bytes); grant=parse(grant_bytes)
    source=_json(source_map_bytes,1<<20,'canonical source map'); _source_map(source)
    if sha(source_map_bytes)!=source_manifest_sha256 or grant['source_manifest_sha256']!=source_manifest_sha256:
        raise ValueError('canonical external source map digest differs')
    if type(blobs)is not dict or set(blobs)!=set(source) or any(type(v)is not bytes for v in blobs.values()):
        raise ValueError('complete immutable source bytes required')
    if {n:sha(v) for n,v in blobs.items()}!=source: raise ValueError('retained executable source bytes differ')
    inventory=parse(blobs['model_source_inventory.json'])
    _object(inventory,('schema','files'),'selected source inventory')
    if inventory['schema']!='qwen-model-store-source-inventory-v1': raise ValueError('source inventory schema differs')
    same(inventory['files'],sorted(source),'exact selected source inventory differs')
    policy,dep=policies(blobs)
    if not 0<len(source)<=policy['maximum_source_files'] or any(not 0<len(v)<=policy['maximum_source_file_bytes'] for v in blobs.values()) or sum(map(len,blobs.values()))>policy['maximum_source_bytes']:
        raise ValueError('bounded selected source closure required')
    same([grant['source_root'],grant['evidence_output']],[source_root,output_path],'exact launch paths differ')
    if grant['dependency_policy_sha256']!=sha(blobs['model_dependencies.json']) or grant['readonly_policy_sha256']!=sha(blobs['model_readonly_policy.json']):
        raise ValueError('grant source policy binding differs')
    for name,raw in (('expected_stores',store_bytes),('prelaunch_review',review_bytes)):
        if type(raw)is not bytes or sha(raw)!=requests[name]['sha256']:
            raise ValueError('externally granted public role bytes differ')
    # The trusted grant authenticates the existing root review's bytes. Do not
    # invent a historical schema or infer archive members from its digest.
    review=_json(review_bytes,1<<16,'root prelaunch review')
    if review.get('passed')is not True: raise ValueError('root prelaunch review must pass')
    public=PublicContract.from_blobs({n:blobs['public_inputs/'+n] for n in PUBLIC_PINS})
    aggregate=Aggregate(public,blobs['unadmitted_aggregate_manifest.json']); aggregate.check()
    ExpectedStoreIdentity(aggregate,store_bytes).check()
    return ReadonlyLaunch(source_root,output_path,source_map_bytes,source_manifest_sha256,
        tuple(sorted(blobs.items())),grant_bytes,grant_sha256,store_bytes,review_bytes)
