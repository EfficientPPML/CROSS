"""Pinned validation wrapper contract; does not open validation inputs/model assets."""
from pathlib import Path
import hashlib,json
EXPECTED_REQUIREMENT='254af804a0cff8cc0d5fba85f953132da1f05b4e57997c8aa52ee47eea5aea19'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(root):
    root=Path(root)
    if sha(root/'validation_requirement.json')!=EXPECTED_REQUIREMENT:raise ValueError('Frozen validation requirement mismatch')
    req=json.loads((root/'validation_requirement.json').read_text())
    if req.get('ready') is not True:raise ValueError('Successful terminal calibration required')
    if sha(root/'validation_protocol.json')!=req['validation_protocol_sha256']:raise ValueError('Frozen validation protocol mismatch')
    for name,digest in req['validation_code_sha256'].items():
        if sha(root/name)!=digest:raise ValueError('Frozen validation wrapper mismatch: '+name)
    return req
