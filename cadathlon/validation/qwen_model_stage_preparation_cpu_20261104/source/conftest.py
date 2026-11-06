"""Synthetic public proofs, recording owners, and owned small files only."""
import importlib.abc,sys,json
from pathlib import Path
from types import SimpleNamespace as NS
import pytest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
FORBIDDEN={'jax','jaxlib','jaxite_word','torch','transformers','tensorflow','safetensors','libtpu'}
class DenyBackend(importlib.abc.MetaPathFinder):
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0]in FORBIDDEN:raise RuntimeError('model-stage preparation forbids backend/model imports: '+fullname)
sys.meta_path.insert(0,DenyBackend())
if FORBIDDEN.intersection(sys.modules):raise RuntimeError('backend/model preloaded before focused preparation')
@pytest.fixture(autouse=True)
def no_backend():
    assert not FORBIDDEN.intersection(sys.modules)
    yield
    assert not FORBIDDEN.intersection(sys.modules)
@pytest.fixture(scope='session')
def public():
    from qwen_file_contract import PublicContract
    from public_source_pins import PUBLIC_PINS
    return PublicContract.from_blobs({n:(HERE/'public_inputs'/n).read_bytes()for n in PUBLIC_PINS})
@pytest.fixture(scope='session')
def aggregate(public):
    from model_admission import Aggregate
    return Aggregate(public,(HERE/'unadmitted_aggregate_manifest.json').read_bytes())


def synthetic_stores(aggregate,tmp_path,*,real_roots=False):
    """Metadata claims only. No197 physical matrix bodies are created."""
    from qwen_file_contract import canonical
    from model_admission import ExpectedStoreIdentity
    roots={}
    for i,family in enumerate(('qkv','decoder','gamma')):
        path=tmp_path/family
        if real_roots:
            path.mkdir(mode=0o700);path.chmod(0o700);st=path.stat()
            roots[family]=dict(path=str(path),device=st.st_dev,inode=st.st_ino,mode=st.st_mode&0o7777)
        else:roots[family]=dict(path=str(path),device=7,inode=100+i,mode=0o700)
    physical={}
    for row in aggregate.catalog['matrices'].values():physical[(row['family'],row['descriptor']['file'])]=row['descriptor']
    files=[]
    for i,((family,name),d)in enumerate(sorted(physical.items())):
        size=8
        for x in d['shape']:size*=x
        files.append(dict(family=family,name=name,state=dict(device=7,inode=1000+i,size=size+128,mtime_ns=1,ctime_ns=2,mode=0o644),file_sha256=d['file_sha256'],raw_sha256=d['sha256']))
    files.append(dict(family='gamma',name='synthetic-gamma.npz',state=dict(device=7,inode=2000,size=1024,mtime_ns=1,ctime_ns=2,mode=0o644),file_sha256=aggregate.catalog['gamma_container_sha256'],raw_sha256=None))
    value=dict(roots=roots,files=sorted(files,key=lambda x:(x['family'],x['name'])))
    return ExpectedStoreIdentity(aggregate,canonical(value))
@pytest.fixture
def stores(aggregate,tmp_path):return synthetic_stores(aggregate,tmp_path)

@pytest.fixture
def chain(public,aggregate,stores):
    """Fake trusted expectations authorize only these supplied synthetic bytes.

    Re-pin mutated proof bytes deliberately to exercise semantic rejection,
    rather than merely tripping a changed digest. This is not real admission.
    """
    from model_admission import ExpectedModelPins,authenticate_model
    from qwen_file_contract import canonical,sha
    def build(*,change_report=None,change_outer=None,change_review=None,change_expected=None,source_map=None,consumer='c'*64):
        source_map={'entry.py':'a'*64}if source_map is None else source_map
        source_digest=sha(canonical(source_map));archive='b'*64;grant='d'*64
        report=dict(schema='qwen-model-store-validation-v1',mode='reuse_existing_exports',passed=True,complete=True,error_type=None,
            model_id='ee3daf8b07141134466d02a6e1c141365ac4e0f0117d62f887c416a050b04bfb',aggregate_manifest_sha256=sha(aggregate.raw),catalog_sha256=aggregate.check()['catalog_sha256'],execution_grant_sha256=grant,source_manifest_sha256=source_digest,source_archive_sha256=archive,source_hashes_before=json.loads(canonical(source_map)),source_hashes_after=json.loads(canonical(source_map)),process=dict(pid=101,pgid=100),stores=json.loads(stores.raw),counts=dict(logical_matrices=198,physical_matrices=197,gamma_vectors=113,physical_matrix_raw_bytes=4767875072,gamma_raw_bytes=524288),
            checks={n:True for n in ('public_provenance_authenticated','complete_catalog_rederived','all_matrix_headers_extents_hashes_finite','all_gamma_members_headers_hashes_finite','original_held_identity_unchanged','readback_completed','source_postcheck_completed','cleanup_completed','signals_restored','borrowed_stores_retained')},
            scope=dict(parameter_body_reads_performed=True,new_export_performed=False,model_forward=False,backend_context_constructed=False,keys_read=False,ciphertexts_read=False,token_inputs_read=False,coefficient_selection=False,scientific_calibration_pass_claimed=False),audit_denials=[])
        if change_report:change_report(report)
        report_raw=canonical(report)
        outer=dict(schema='qwen-model-store-validation-outer-v1',report_sha256=sha(report_raw),aggregate_manifest_sha256=sha(aggregate.raw),model_id=aggregate.check()['model_id'],source_manifest_sha256=source_digest,source_archive_sha256=archive,execution_grant_sha256=grant,pid=101,pgid=100,exit_code=0,normal_return=True,reaped=True,process_group_absent=True)
        if change_outer:change_outer(outer)
        outer_raw=canonical(outer)
        review=dict(schema='qwen-model-stage-root-admission-v1',passed=True,model_id=aggregate.check()['model_id'],aggregate_manifest_sha256=sha(aggregate.raw),catalog_sha256=aggregate.check()['catalog_sha256'],validation_report_sha256=sha(report_raw),validation_outer_sha256=sha(outer_raw),source_manifest_sha256=source_digest,source_archive_sha256=archive,execution_grant_sha256=grant,current_store_binding_sha256=sha(canonical(report['stores'])),all_required_links_verified=True,source_archive_members_verified=True,actual_server_execution_authorized=False)
        if change_review:change_review(review)
        review_raw=canonical(review)
        expected=dict(schema='qwen-model-expected-pins-v1',model_id=aggregate.check()['model_id'],aggregate_manifest_sha256=sha(aggregate.raw),catalog_sha256=aggregate.check()['catalog_sha256'],producer_source_manifest_sha256=source_digest,producer_source_archive_sha256=archive,producer_source_map=source_map,execution_grant_sha256=grant,validation_report_sha256=sha(report_raw),validation_outer_sha256=sha(outer_raw),root_review_sha256=sha(review_raw),consumer_source_manifest_sha256=consumer)
        if change_expected:change_expected(expected)
        pins=ExpectedModelPins(canonical(expected))
        invoke=lambda:authenticate_model(public,aggregate.raw,report_raw,outer_raw,review_raw,expected_pins=pins)
        return NS(report=report,outer=outer,review=review,expected=expected,pins=pins,report_raw=report_raw,outer_raw=outer_raw,review_raw=review_raw,call=invoke)
    return build
