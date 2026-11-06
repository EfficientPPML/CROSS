"""Recorded public evidence verification only; imports no producer code."""
from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
SOURCE=Path('/tmp/qwen-organizer-file-engine-integration-v2-20261104')
EXECUTION=Path('/tmp/qwen-stage-integration-successor-proof-20261104/development-v1/execution')
def sha(b):return hashlib.sha256(b).hexdigest()
def read(n):return json.loads((P/n).read_bytes())
def xml(n,total,failures=0):
    x=ET.parse(P/n);assert len(x.findall('.//testcase'))==total
    assert len(x.findall('.//failure'))==failures
    assert not x.findall('.//error')and not x.findall('.//skipped')
def archive(n,expected):
    with tarfile.open(P/n,'r:gz')as t:
        ms=t.getmembers();assert len(ms)==len(expected)and len({m.name for m in ms})==len(ms)
        assert all(m.isfile()and not Path(m.name).is_absolute()and '..'not in Path(m.name).parts for m in ms)
        blobs={m.name:t.extractfile(m).read()for m in ms}
        assert {n:sha(b)for n,b in blobs.items()}==expected
        return blobs
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args()
    index=read('files.sha256.json')
    assert not any(p.is_symlink()for p in P.rglob('*'))
    observed={p.relative_to(P).as_posix():sha(p.read_bytes())for p in P.rglob('*')if p.is_file()and p.name!='files.sha256.json'}
    assert observed==index,'capsule inventory mismatch'
    expected=read('measured/source_manifest.json');assert len(expected)==366
    assert expected==read('measured/sources-before.json')==read('measured/sources-after.json')==read('measured/execution-after.json')
    blobs=archive('measured/sources.tar.gz',expected)
    assert all(Path(n).suffix in ('.py','.json','.md')for n in expected)
    for f in (P/'source').iterdir():assert f.is_file()and sha(f.read_bytes())==expected[f.name]
    r=read('measured/report.json');owner=read('measured/owner-terminal.json')
    assert r['passed']is True and r['exit_code']==0 and r['source_count']==366 and r['source_unchanged']is True
    assert r['test_count']==12 and r['test_failures_errors_skips']==0 and r['child_reaped']is True and r['child_group_absent']is True
    assert r['source_archive_sha256']==sha((P/'measured/sources.tar.gz').read_bytes())
    assert owner['report_sha256']==sha((P/'measured/report.json').read_bytes())
    assert owner['source_manifest_sha256']==sha((P/'measured/source_manifest.json').read_bytes())
    assert owner['source_archive_sha256']==r['source_archive_sha256']
    assert owner['driver_sha256']==sha((P/'measured/run.py').read_bytes())==read('measured/driver-before.json')['sha256']
    assert owner['xml_sha256']==sha((P/'measured/tests.xml').read_bytes())
    assert owner['session_id']==68008 and owner['terminal_tool_chunk']=='17d040'and owner['exit_code']==0 and owner['reaped']is True
    assert all(v is True for v in owner['process_absence'].values())
    xml('measured/tests.xml',12)
    old=read('history/stage-negative/sources-before.json');assert len(old)==365
    assert old==read('history/stage-negative/sources-after.json')==read('history/stage-negative/execution-after.json')
    oldblobs=archive('history/stage-negative/sources.tar.gz',old)
    nr=read('history/stage-negative/report.json');assert nr['passed']is False and nr['exit_code']==1 and nr['test_failures_errors_skips']==3
    assert nr['source_archive_sha256']==sha((P/'history/stage-negative/sources.tar.gz').read_bytes())
    xml('history/stage-negative/tests.xml',12,3)
    assert oldblobs['test_stage_boundaries.py']==blobs['test_stage_boundaries.py']
    delta=read('source/stage_ownership_successor.json')
    assert delta['predecessor_archive_sha256']==nr['source_archive_sha256']
    assert {n for n in old if expected[n]!=old[n]}==set(delta['overlays'])
    assert set(expected)-set(old)=={'stage_ownership_successor.json'}
    for n,d in delta['overlays'].items():assert old[n]==d['before']and expected[n]==d['after']
    for label,fail in [('negative',2),('successor',0)]:
        stem='history/cleanup-'+label
        m=read(stem+'/sources-before.json');assert len(m)==3 and m==read(stem+'/sources-after.json')
        archive(stem+'/sources.tar.gz',m);xml(stem+'/tests.xml',5,fail)
        rr=read(stem+'/report.json');assert rr['reaped']is True and rr['group_absent']is True
        assert rr['exit_code']==(1 if fail else 0)and rr['source_archive_sha256']==sha((P/stem/'sources.tar.gz').read_bytes())
    assert (P/'history/cleanup-negative/test_alignment.py').read_bytes()==(P/'history/cleanup-successor/test_alignment.py').read_bytes()
    selection=read('source/integration_selection.json');assert len(selection['copied_sources'])==337 and len(selection['overlays'])==4
    for n,row in selection['copied_sources'].items():
        if n not in selection['overlays']:assert expected[n]==row['sha256']
        else:
            overlay=selection['overlays'][n];assert expected[n]==overlay['after_sha256']
            raw=blobs[n].decode()
            for replacement in reversed(overlay['replacements']):
                assert raw.count(replacement['after'])==1;raw=raw.replace(replacement['after'],replacement['before'])
            assert sha(raw.encode())==overlay['before_sha256']==row['sha256']
    assert expected['cleanup_origin.py']==selection['shared_cleanup_origin_sha256']
    auth=read('source/execution_authorization.json')
    assert all(auth[n]is False for n in ('actual_execution_enabled','key_generation_enabled','server_he_enabled','private_client_enabled'))
    refs=read('parent-proof-references.json')
    for label,ref in refs.items():
        stem='parents/'+label;pm=read(stem+'/sources-before.json');assert len(pm)==ref['source_count']
        assert pm==read(stem+'/sources-after.json')==read(stem+'/execution-after.json')
        assert read(stem+'/report.json')['passed']is True
        xml(stem+'/tests.xml',ref['test_count'])
        for name,d in ref['files'].items():
            if name!='sources.tar.gz':assert sha((P/stem/name).read_bytes())==d['sha256']
            if args.current:assert sha(Path(d['path']).read_bytes())==d['sha256']
        assert read(stem+'/report.json')['source_archive_sha256']==ref['files']['sources.tar.gz']['sha256']
    review=read('reviews/root/stage-successor-review.json');assert review['passed']is True
    assert review['report_sha256']==owner['report_sha256']and review['source_archive_sha256']==owner['source_archive_sha256']
    assert review['source_manifest_sha256']==owner['source_manifest_sha256']
    assert review['script_sha256']==sha((P/'reviews/root/verify_stage_successor.py').read_bytes())
    for field,name in [('selection_review_sha256','reviews/root/selection-review.json'),('prior_negative_review_sha256','reviews/root/stage-negative-review.json'),('security_receipt_sha256','reviews/security/review.json')]:assert review[field]==sha((P/name).read_bytes())
    if args.current:
        for root in (SOURCE,EXECUTION):
            current={p.relative_to(root).as_posix():sha(p.read_bytes())for p in root.rglob('*')if p.is_file()};assert current==expected
    print(json.dumps(dict(passed=True,source_count=366,joined_tests=12,negative_stage_failures=3,cleanup_negative_failures=2,current=args.current,scope='Recorded public evidence only; no producer imports or test/backend execution')))
if __name__=='__main__':main()
