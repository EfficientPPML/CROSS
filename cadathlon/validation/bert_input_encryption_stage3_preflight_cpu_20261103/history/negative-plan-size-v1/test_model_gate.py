"""Completed-model prerequisite regressions; no real key/model payload reads."""
from pathlib import Path
from types import SimpleNamespace as NS
import io,json,sys,tarfile
import pytest
HERE=Path(__file__).resolve().parent
sys.path[:0]=[str(HERE/'runtime/stage'),str(HERE/'runtime')]
import model_gate,key_gate,server_public_model as model


@pytest.fixture
def case(tmp_path):
    evidence=tmp_path/'evidence';evidence.mkdir();(evidence/'worker').mkdir()
    workspace=tmp_path/'workspace';workspace.mkdir();context='a'*64;model_id='b'*64
    cache=workspace/'server_data'/context;cache.mkdir(parents=True)
    manifest=dict(context_id=context,model_id=model_id,checkpoint_sha256=model.CHECKPOINT_SHA256,
        config_sha256=model.CONFIG_SHA256,candidate_sha256=model.CANDIDATE_SHA256,profile_sha256=model.PROFILE_SHA256)
    raw=key_gate.canonical(manifest);(cache/'model_manifest.json').write_bytes(raw);(evidence/'model_manifest.json').write_bytes(raw)
    sources={'source.py':key_gate.digest(b'# frozen source\n')}
    with tarfile.open(evidence/'sources.tar.gz','w:gz') as t:
        m=tarfile.TarInfo('source.py');m.size=len(b'# frozen source\n');t.addfile(m,io.BytesIO(b'# frozen source\n'))
    keys=dict(context_id=context,key_report_sha256='c'*64,key_source_archive_sha256='d'*64)
    assets={'model.safetensors':model.CHECKPOINT_SHA256,'config.json':model.CONFIG_SHA256}
    worker=dict(passed=True,actual_server_preprocess_called=True,readback_complete=True,all_readback_fp64_immutable=True,
        source_files_unchanged=True,keygen_and_codecs_forbidden=True,context_construction_forbidden=True,
        forbidden_attempts=[],source_hashes_start=sources,source_hashes_end=sources,tensor_count=196,matrix_count=74,
        public_model_manifest=manifest,prerequisite=keys,asset_hashes_start=assets,asset_hashes_end=assets)
    report=dict(passed=True,execute=True,actual_preprocessing=True,model_workspace_retained=True,
        all_processes_reaped=True,cleanup_completed=True,source_files_unchanged=True,source_archive_verified=True,
        source_hashes_start=sources,source_hashes_end=sources,source_archive_sha256=key_gate.digest((evidence/'sources.tar.gz').read_bytes()),
        prerequisite=keys,model_workspace=str(workspace),model_workspace_identity=[workspace.stat().st_dev,workspace.stat().st_ino],
        model_cache=str(cache),model_cache_identity=[cache.stat().st_dev,cache.stat().st_ino],model_manifest_sha256=key_gate.digest(raw),
        asset_hashes_start=assets,asset_hashes_end=assets,context_id=context,model_id=model_id,
        worker_process=dict(returncode=0,process_reaped=True,public_key_store_removed=True,owned_directory_handle_closed=True,
            result=dict(passed=True,context_id=context,model_id=model_id)))
    def publish():
        raw=key_gate.canonical(worker);(evidence/'worker/worker_report.json').write_bytes(raw)
        report['worker_process']['result']['worker_report_sha256']=key_gate.digest(raw)
        raw=key_gate.canonical(report);(evidence/'report.json').write_bytes(raw)
        return key_gate.digest(raw)
    return NS(evidence=evidence,workspace=workspace,cache=cache,keys=keys,report=report,worker=worker,publish=publish)


def check(c):return model_gate.authenticate(c.evidence,c.publish(),c.workspace,keys=c.keys)


def test_complete_model_prerequisite(case):
    result=check(case)
    assert result['server_data']==case.workspace/'server_data'
    assert result['context_id']==case.keys['context_id'] and result['model_id']=='b'*64


def test_external_hash_checked_before_other_evidence(case):
    case.publish();(case.evidence/'sources.tar.gz').unlink()
    with pytest.raises(ValueError,match='externally authenticated'):
        model_gate.authenticate(case.evidence,'0'*64,case.workspace,keys=case.keys)


@pytest.mark.parametrize('flag',('passed','execute','actual_preprocessing','model_workspace_retained',
    'all_processes_reaped','cleanup_completed','source_files_unchanged','source_archive_verified'))
def test_pending_or_incomplete_model_report_rejected(case,flag):
    case.report[flag]=False
    with pytest.raises(ValueError,match='completed actual'):check(case)


@pytest.mark.parametrize('bad',('bool_exit','unreaped','cleanup','worker_digest','source_digest','worker_source',
    'tensor_count','matrix_count','forbidden','key_context','workspace_inode','cache_inode','cache_manifest','asset','model_id'))
def test_wrong_terminal_binding_rejected(case,bad):
    c=case
    if bad=='bool_exit':c.report['worker_process']['returncode']=False
    elif bad=='unreaped':c.report['worker_process']['process_reaped']=False
    elif bad=='cleanup':c.report['cleanup_errors']=[{'error':'cleanup failed'}]
    elif bad=='worker_digest':
        sha=c.publish();(c.evidence/'worker/worker_report.json').write_bytes(b'{}')
        with pytest.raises(ValueError,match='worker report digest'):
            model_gate.authenticate(c.evidence,sha,c.workspace,keys=c.keys)
        return
    elif bad=='source_digest':c.report['source_archive_sha256']='0'*64
    elif bad=='worker_source':c.worker['source_hashes_end']={}
    elif bad=='tensor_count':c.worker['tensor_count']=195
    elif bad=='matrix_count':c.worker['matrix_count']=True
    elif bad=='forbidden':c.worker['forbidden_attempts']=['codec']
    elif bad=='key_context':c.keys=c.keys|{'context_id':'e'*64}
    elif bad=='workspace_inode':c.report['model_workspace_identity']=[1,2]
    elif bad=='cache_inode':c.report['model_cache_identity']=[1,2]
    elif bad=='cache_manifest':(c.cache/'model_manifest.json').write_bytes(b'{}\n')
    elif bad=='asset':c.report['asset_hashes_end']={}
    else:c.report['model_id']='e'*64
    with pytest.raises(ValueError):check(c)


def test_same_hash_source_archive_with_duplicate_members_rejected(case):
    p=case.evidence/'sources.tar.gz'
    with tarfile.open(p,'w:gz') as t:
        for _ in range(2):
            m=tarfile.TarInfo('source.py');m.size=1;t.addfile(m,io.BytesIO(b'x'))
    case.report['source_archive_sha256']=key_gate.digest(p.read_bytes())
    with pytest.raises(ValueError,match='bounded complete'):check(case)


def test_actual_inherited_public_plan_member_fits_the_completed_model_gate(case):
    raw=(HERE/'runtime/stage/server_inputs/twelve_layer_plan.json').read_bytes()
    assert len(raw)==6461050 and key_gate.digest(raw)=='bdc527179b9d1082d3dc0d90f1203523089aa8a24831941ad42a2513d20f7e04'
    name='checkout/stage/server_inputs/twelve_layer_plan.json'
    case.report['source_hashes_start'][name]=key_gate.digest(raw)
    p=case.evidence/'sources.tar.gz'
    with tarfile.open(p,'w:gz') as t:
        for n,b in [('source.py',b'# frozen source\n'),(name,raw)]:
            m=tarfile.TarInfo(n);m.size=len(b);t.addfile(m,io.BytesIO(b))
    case.report['source_archive_sha256']=key_gate.digest(p.read_bytes())
    assert check(case)['model_id']=='b'*64
