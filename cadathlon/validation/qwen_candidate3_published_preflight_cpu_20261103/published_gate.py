"""Original published token identity; no validation outcomes are loaded."""
import hashlib
from pathlib import Path

EXPECTED='5caad37526dfda5a379f029ddfc75fb10879b3313d3533ec4cbf60d4192ef359'


def verify_metadata(root,requirements):
    if requirements['validation_input_sha256']!=EXPECTED:
        raise ValueError('Original published input identity changed')


def verify_inputs(root,requirements,path,*,on_read=lambda:None):
    verify_metadata(root,requirements)
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk:=stream.read(1024**2):
            on_read();digest.update(chunk)
    if digest.hexdigest()!=requirements['validation_input_sha256']:
        raise ValueError('Original published input bytes differ')
    return dict(verified=True,tokenized_sha256=digest.hexdigest(),tokenized_records=56,planned_contexts=112)
