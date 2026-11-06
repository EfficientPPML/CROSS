"""Public source inventory and evidence publication, without task asset reads."""
from pathlib import Path
import hashlib,json,os,sys,tarfile,tempfile
ROOT=Path(__file__).resolve().parent
def source_paths():
    return {p.name:p for p in ROOT.iterdir() if p.suffix in ('.py','.json')}
def write_json(path,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    fd,name=tempfile.mkstemp(prefix='.'+path.name+'-',dir=path.parent);temporary=Path(name)
    try:
        with os.fdopen(fd,'wb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        primary=sys.exception()
        try:temporary.unlink(missing_ok=True)
        except BaseException as error:
            if primary is None:raise
            primary.add_note('metadata temporary cleanup: '+type(error).__name__)
def verify_archive(path,expected):
    with tarfile.open(path) as archive:
        members=archive.getmembers()
        if (len(members)!=len(expected) or len({m.name for m in members})!=len(members)
                or any(not m.isfile() or Path(m.name).is_absolute() or '..' in Path(m.name).parts for m in members)):
            raise ValueError('exact source archive inventory required')
        if {m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in members}!=expected:
            raise ValueError('source archive final bytes differ')
    return True
