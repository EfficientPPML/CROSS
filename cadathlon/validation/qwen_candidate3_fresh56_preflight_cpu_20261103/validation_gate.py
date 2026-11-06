"""Pinned validation wrapper contract; does not open validation inputs/model assets."""
from pathlib import Path
import hashlib,json,tarfile
EXPECTED_REQUIREMENT='129cb79c13c14270f8b892fc15697eaed3a9a290824b50c0290772577ce1a137'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify_archive(path,expected):
    """Exact wrapper closure; only the authenticated calibration report is large."""
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        if (len(members)!=len(expected) or len({m.name for m in members})!=len(members)
                or {m.name for m in members}!=set(expected)
                or any(not m.isfile() or m.size<0 or m.size>(64 if m.name=='calibration_prerequisite/report.json' else 8)*1024**2 for m in members)
                or sum(m.size for m in members)>96*1024**2):
            raise ValueError('Validation source archive membership mismatch')
        actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}
    if actual!=expected:raise ValueError('Validation source archive bytes mismatch')
    return actual
def verify(root):
    root=Path(root)
    if sha(root/'validation_requirement.json')!=EXPECTED_REQUIREMENT:raise ValueError('Frozen validation requirement mismatch')
    req=json.loads((root/'validation_requirement.json').read_text())
    if req.get('ready') is not True:raise ValueError('Successful terminal calibration required')
    if sha(root/'validation_protocol.json')!=req['validation_protocol_sha256']:raise ValueError('Frozen validation protocol mismatch')
    for name,digest in req['validation_code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen validation wrapper mismatch: '+name)
    return req
