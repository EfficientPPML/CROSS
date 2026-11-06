"""Read public source/proof artifacts only; no runtime imports or test reruns."""
from pathlib import Path
import ast, datetime, hashlib, json, os, tarfile
import xml.etree.ElementTree as ET
root=Path('/tmp/qwen-organizer-file-engine-integration-20261104')
sha=lambda b:hashlib.sha256(b).hexdigest()
load=lambda p:json.loads(p.read_bytes())
selection=root/'integration_selection.json'
assert sha(selection.read_bytes())=='ff28797f6e0dfc5edf554aadf24f662602a08eb3b6425f40649f8e0910b65b38'
d=load(selection)
assert len(d['copied_sources'])==337 and len(d['overlays'])==4
for name,row in d['copied_sources'].items():
    rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
    raw=Path(row['origin']).read_bytes();assert sha(raw)==row['sha256']
    if name in d['overlays']:
        overlay=d['overlays'][name];assert sha(raw)==overlay['before_sha256']
        text=raw.decode()
        for change in overlay['replacements']:
            assert text.count(change['before'])==1
            text=text.replace(change['before'],change['after'])
        raw=text.encode();assert sha(raw)==overlay['after_sha256']
    assert (root/name).read_bytes()==raw,name
assert sha((root/'cleanup_origin.py').read_bytes())==d['shared_cleanup_origin_sha256']=='4543b26bf1168e29bc21375c2f80a9aa99648205a9fd9aea6d5a8059c6d18175'
assert sha((root/'qwen_public_parse.py').read_bytes())=='d29919969c9950a86d2e663f5392c401742e6306a0ed7c9ad7ab3ea2f4e830f0'
assert sha((root/'qwen_finalizer.py').read_bytes())=='46e6566a275a84b4ae4596e066c592a4aa3a60e41b3afb4ec30bb96b72551240'
def node(path,kind,name):
    return ast.dump(next(n for n in ast.parse(path.read_bytes()).body if isinstance(n,kind) and n.name==name),include_attributes=False)
assert node(root/'qwen_public_parse.py',ast.FunctionDef,'parse')==node(root/'reused_client_io.py',ast.FunctionDef,'parse')
records=[]
for label,path,failures,exit_code,pid in (
    ('negative',Path('/tmp/qwen-cleanup-alignment-negative-20261104'),2,1,3880712),
    ('successor',Path('/tmp/qwen-cleanup-alignment-successor-20261104'),0,0,3883329),
):
    report=load(path/'report.json');before=load(path/'sources-before.json');after=load(path/'sources-after.json')
    assert before==after and len(before)==3
    for name,digest in before.items():assert sha((path/name).read_bytes())==digest
    assert sha((path/'sources.tar.gz').read_bytes())==report['source_archive_sha256']
    with tarfile.open(path/'sources.tar.gz')as archive:
        members=archive.getmembers();assert len(members)==3 and all(m.isfile()for m in members)
        assert {m.name:sha(archive.extractfile(m).read())for m in members}==before
    cases=list(ET.parse(path/'tests.xml').getroot().iter('testcase'))
    assert len(cases)==5 and sum(x.find('failure')is not None for x in cases)==failures
    assert all(x.find('error')is None and x.find('skipped')is None for x in cases)
    assert report['exit_code']==exit_code and report['reaped']is True and report['group_absent']is True
    assert report['child_pid']==report['child_pgid']==pid and not Path('/proc',str(pid)).exists()
    try:os.killpg(pid,0)
    except ProcessLookupError:pass
    else:raise AssertionError(('group present',pid))
    expected=(root/'reused_client_io.py') if label=='negative' else (root/'qwen_finalizer.py')
    assert node(path/'finalizer_source.py',ast.ClassDef,'Finalizer')==node(expected,ast.ClassDef,'Finalizer')
    assert (path/'cleanup_origin.py').read_bytes()==(root/'cleanup_origin.py').read_bytes()
    records.append(dict(label=label,report_sha256=sha((path/'report.json').read_bytes()),source_archive_sha256=report['source_archive_sha256'],tests=5,failures=failures,process_absent=True))
negative=Path('/tmp/qwen-cleanup-alignment-negative-20261104/test_alignment.py')
successor=Path('/tmp/qwen-cleanup-alignment-successor-20261104/test_alignment.py')
assert negative.read_bytes()==successor.read_bytes()
peer=Path('/tmp/qwen-integration-selection-cnn-review-20261104/review.json')
assert sha(peer.read_bytes())=='1726b549403defadb8c1e852b158d29f14186b47c38c88c730d60db2843ab4dc'
out=Path(__file__).with_name('selection-review.json');assert not out.exists()
out.write_text(json.dumps(dict(schema='qwen-integration-selection-root-review-v1',passed=True,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),copied_sources=337,overlays=4,selection_sha256=sha(selection.read_bytes()),peer_sha256=sha(peer.read_bytes()),cleanup_alignment=records,identical_probe_sha256=sha(negative.read_bytes()),script_sha256=sha(Path(__file__).read_bytes()),scope='Exact source selection and five-case cleanup-policy evidence only. Role-stage glue and broader integration unadmitted; root identified possible callback-replaced close and late handle mutation gaps, currently under focused development. Actual factories disabled; no body/backend/context access.'),sort_keys=True,indent=2)+'\n')
print(json.dumps(dict(passed=True,receipt=str(out),sha256=sha(out.read_bytes()),copied=337,overlays=4,cleanup_cases=5)))
