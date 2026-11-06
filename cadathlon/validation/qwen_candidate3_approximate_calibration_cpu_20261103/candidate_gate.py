"""Verify frozen public candidate3 selection before original calibration/model reads."""
from pathlib import Path
import hashlib,json,tarfile
EXPECTED_REQUIREMENT='6db84ccfc566a0aa4104f27c81f12865b232aaf55778e7b3ad0c8e262a8180ff'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_archive(path,expected):
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        if (len(members)!=len(expected) or len({m.name for m in members})!=len(members) or {m.name for m in members}!=set(expected)
                or any(not m.isfile() or m.size<0 or m.size>8<<20 for m in members) or sum(m.size for m in members)>64<<20):raise ValueError('Selection source archive membership mismatch')
        actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
    if actual!=expected:raise ValueError('Selection source archive bytes mismatch')
    return actual

def verify(root):
    root=Path(root)
    if sha(root/'collector_requirement.json')!=EXPECTED_REQUIREMENT:raise ValueError('Collector requirement mismatch')
    req=json.loads((root/'collector_requirement.json').read_text())
    if sha(root/'candidate-v3/manifest.json')!=req['manifest_sha256']:raise ValueError('Frozen candidate3 manifest mismatch')
    manifest=json.loads((root/'candidate-v3/manifest.json').read_text())
    if manifest['source_input_sha256']!=req['selection_input_sha256'] or len(req['selection_input_sha256'])!=12:raise ValueError('Frozen selection input inventory mismatch')
    verified={}
    for group in ('selection_artifact_sha256','selection_input_sha256','runtime_dependency_sha256'):
        expected={('selection/'+n if group=='selection_input_sha256' else n):h for n,h in req[group].items()}
        actual={n:sha(root/n) for n in expected}
        if actual!=expected:raise ValueError('Frozen '+group+' mismatch')
        verified[group]=actual
    selection=json.loads((root/'candidate-v3/report.json').read_text());wrapper=json.loads((root/'selection_wrapper/report.json').read_text())
    for name in ('passed','complete','protected_other_rules_unchanged','source_files_unchanged','source_archive_verified','calibration_or_validation_not_executed'):
        if selection.get(name)is not True:raise ValueError('Selection prerequisite incomplete: '+name)
    if any(selection.get(n)is not False for n in ('model_evaluated','HE_executed','heldout_or_fresh56_read')) or selection.get('error'):raise ValueError('Selection scope differs')
    expected=req['selection_input_sha256']
    for record in (selection,wrapper):
        if record.get('passed')is not True or record.get('source_hashes_before')!=expected or record.get('source_hashes_after')!=expected:raise ValueError('Selection before/after provenance differs')
    if (selection.get('manifest_sha256')!=req['manifest_sha256'] or wrapper.get('manifest_sha256')!=req['manifest_sha256']
            or wrapper.get('selection_report_sha256')!=req['selection_artifact_sha256']['candidate-v3/report.json']):raise ValueError('Selection manifest/report linkage differs')
    for directory,record in (('candidate-v3',selection),('selection_wrapper',wrapper)):
        if record.get('source_archive_sha256')!=req['selection_artifact_sha256'][directory+'/sources.tar.gz']:raise ValueError('Selection archive digest differs')
        verify_archive(root/directory/'sources.tar.gz',expected)
    if wrapper.get('tests')!=dict(tests=28,failures=0,errors=0,skipped=0):raise ValueError('Frozen selection tests incomplete')
    review=json.loads((root/'independent_review/report.json').read_text())
    if (review.get('passed')is not True or review.get('source_files_unchanged')is not True
            or review.get('selected_manifest_sha256')!=req['manifest_sha256'] or review.get('source_hashes_before')!=expected or review.get('source_hashes_after')!=expected
            or review.get('parent_source_archive_sha256')!=wrapper['source_archive_sha256'] or review.get('review_script_sha256')!=sha(root/'independent_review/review.py')
            or review.get('polynomial_certificates_independently_recomputed')!=36 or review.get('factor4_domains_checked')!=113
            or review.get('unsuccessful_polynomial_attempts_retained')!=610 or review.get('public_downstream_radius_checks')!=28):raise ValueError('Independent mathematical review incomplete')
    actual={n:sha(root/n) for n in req['calibration_input_sha256']}
    if actual!=req['calibration_input_sha256']:raise ValueError('Frozen calibration_input_sha256 mismatch')
    verified['calibration_input_sha256']=actual
    base=json.loads((root/'selection/inputs/candidate2_manifest.json').read_text())
    norms=lambda m:{k:v for k,v in m['nonlinear'].items() if k.endswith('.inverse_sqrt')}
    protected=lambda m:({k:v for k,v in m['nonlinear'].items() if not k.endswith('.inverse_sqrt')},{k:m[k] for k in ('softmax','exp','initial_reciprocal','reciprocal_by_keys')})
    if len(norms(manifest))!=113 or set(norms(manifest))!=set(norms(base)) or protected(manifest)!=protected(base):raise ValueError('Protected other rules changed')
    if req['runtime_dependency_sha256']['numeric.py']!=expected['numeric.py']:raise ValueError('Selected numeric semantics changed')
    if req['runtime']!=dict(cores=[124,125,126,127],torch_threads=4,inter_op_threads=1):raise ValueError('Declared calibration CPU policy differs')
    if req['calibration_input_sha256']['calibration/protocol.json']==expected['protocol.json']:raise ValueError('Selection protocol cannot replace calibration protocol')
    return dict(passed=True,manifest_sha256=req['manifest_sha256'],requirement_sha256=EXPECTED_REQUIREMENT,verified=verified,
        selection_archive_members=len(expected),selection_model_evaluated=False,calibration_protocol_distinct_from_selection_protocol=True,runtime=req['runtime'])
