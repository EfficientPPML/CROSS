from pathlib import Path
import importlib.util,hashlib,json,shutil,tempfile
LAB=Path('/tmp/bert-full-model-file-stages-20261103')
HERE=Path(__file__).parent
SOURCE=LAB/'server_public_model.py'
spec=importlib.util.spec_from_file_location('review_public_model',SOURCE);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
(HERE/'server_public_model_before.py').write_bytes(SOURCE.read_bytes())
kwargs=dict(context_id='a'*64,candidate_bytes=Path('/tmp/bert-pooler-head-owner-20261103/candidate_manifest.json').read_bytes(),profile_bytes=Path('/tmp/bert-pooler-head-owner-20261103/p60_profile_and_refresh.json').read_bytes())
SNAP=Path('/home/jianming_gatech/cross_validation_assets/bert/f150c1d609d1e50dd5e2e5408661cfac8339277c')
original_save=m.np.save;original_rmtree=m.shutil.rmtree
records=[]
with tempfile.TemporaryDirectory(prefix='public-model-cleanup-repro-') as root:
    root=Path(root);output=root/'cache'
    def fail_save(*args,**kw):raise KeyboardInterrupt('primary-cancellation')
    def fail_cleanup(*args,**kw):raise OSError('secondary-cleanup-error')
    m.np.save=fail_save;m.shutil.rmtree=fail_cleanup
    try:m.preprocess(SNAP,output,**kwargs)
    except BaseException as exc:records.append(dict(case='cleanup masks cancellation',raised=type(exc).__name__,message=str(exc),primary_preserved=isinstance(exc,KeyboardInterrupt)))
    finally:m.np.save=original_save;m.shutil.rmtree=original_rmtree
    original_rmtree(output)
    moved=root/'original-owned'
    def replace_then_fail(*args,**kw):
        output.rename(moved);output.mkdir();(output/'unowned-sentinel').write_text('replacement')
        raise RuntimeError('primary-write-error')
    m.np.save=replace_then_fail
    try:m.preprocess(SNAP,output,**kwargs)
    except BaseException as exc:records.append(dict(case='path replacement',raised=type(exc).__name__,replacement_survives=(output/'unowned-sentinel').exists(),original_owned_survives=moved.exists()))
    finally:m.np.save=original_save
report=dict(source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),source_unchanged=(HERE/'server_public_model_before.py').read_bytes()==SOURCE.read_bytes(),cases=records,scope='Injected public-writer/cleanup failures only; no private inputs or crypto.')
(HERE/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
