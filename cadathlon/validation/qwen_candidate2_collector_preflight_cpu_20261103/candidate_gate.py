"""Pure-stdlib, fail-closed prerequisite checks before model or calibration reads."""
from pathlib import Path
import hashlib,json,tarfile
EXPECTED_REQUIREMENT='25aebcccf1e456fb2f566eb170e275459c3fca23adea8ece7de0d1e45b459636'

def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def verify_archive(path,expected):
    with tarfile.open(path,'r:gz') as archive:
        members=archive.getmembers()
        if len(members)!=len(expected) or {m.name for m in members}!=set(expected) or any(not m.isfile() for m in members):
            raise ValueError('Selection source archive membership mismatch')
        actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
    if actual!=expected:raise ValueError('Selection source archive bytes mismatch')
    return actual

def verify(root):
    root=Path(root)
    if sha(root/'collector_requirement.json')!=EXPECTED_REQUIREMENT:raise ValueError('Collector requirement mismatch')
    req=json.loads((root/'collector_requirement.json').read_text())
    if sha(root/'candidate-v2/manifest.json')!=req['manifest_sha256']:raise ValueError('Frozen candidate2 manifest mismatch')
    manifest=json.loads((root/'candidate-v2/manifest.json').read_text())
    if manifest['source_input_sha256']!=req['selection_input_sha256'] or len(req['selection_input_sha256'])!=20:
        raise ValueError('Frozen selection input inventory mismatch')
    verified={}
    for group in ('selection_artifact_sha256','selection_input_sha256','calibration_input_sha256','runtime_dependency_sha256'):
        actual={name:sha(root/name) for name in req[group]}
        for name,expected in req[group].items():
            if actual[name]!=expected:raise ValueError('Frozen '+group+' mismatch: '+name)
        verified[group]=actual
    selection=json.loads((root/'candidate-v2/report.json').read_text())
    for name in ('passed','complete','all28_supported','unchanged_norms_verified','unchanged_softmax_verified','source_files_unchanged'):
        if selection.get(name) is not True:raise ValueError('Selection prerequisite is incomplete: '+name)
    if selection.get('errors')!=[] or selection.get('unsupported')!=[] or selection.get('model_evaluated') is not False or selection.get('validation_outputs_read') is not False:
        raise ValueError('Selection prerequisite scope/failure mismatch')
    expected=req['selection_input_sha256']
    if selection.get('source_hashes_before')!=expected or selection.get('source_hashes_after')!=expected:
        raise ValueError('Selection before/after provenance mismatch')
    if selection.get('manifest_sha256')!=req['manifest_sha256'] or selection.get('source_archive_sha256')!=req['selection_artifact_sha256']['candidate-v2/sources.tar.gz']:
        raise ValueError('Selection output binding mismatch')
    verify_archive(root/'candidate-v2/sources.tar.gz',expected)
    pairs={'calibration/corpus.json':'inputs/calibration_corpus.json','calibration/tokenized_calibration_inputs.json':'inputs/calibration_inputs.json','calibration/model_pins.json':'inputs/model_pins.json'}
    if any(req['calibration_input_sha256'][a]!=expected[b] for a,b in pairs.items()):raise ValueError('Calibration and selection input identities differ')
    if req['calibration_input_sha256']['calibration/protocol.json']==expected['protocol.json']:
        raise ValueError('Selection protocol must not replace calibration protocol')
    for name in ('arithmetic.py','numeric.py'):
        if req['runtime_dependency_sha256'][name]!=expected[name]:raise ValueError('Selected arithmetic dependency changed')
    return {'passed':True,'manifest_sha256':req['manifest_sha256'],'requirement_sha256':EXPECTED_REQUIREMENT,
        'verified':verified,'selection_archive_members':len(expected),'selection_model_evaluated':False,
        'calibration_protocol_distinct_from_selection_protocol':True}
