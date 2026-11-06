"""Portable recorded-source/XML verification; never imports producer code."""
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET
from pathlib import Path
P=Path(__file__).resolve().parent
SOURCE=Path('/tmp/qwen-organizer-file-adapter-v4-20261104')
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(name):return json.loads((P/name).read_bytes())
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--current',action='store_true');args=parser.parse_args()
    index=read('files.sha256.json')
    actual={p.relative_to(P).as_posix():sha(p.read_bytes())for p in P.rglob('*')if p.is_file()and p.name!='files.sha256.json'}
    assert actual==index,'public capsule bytes differ'
    before=read('measured/sources-before.json');assert len(before)==238
    assert before==read('measured/sources-after.json')==read('measured/execution-after.json')
    with tarfile.open(P/'measured/sources.tar.gz')as archive:
        members=archive.getmembers();assert len(members)==238 and len({m.name for m in members})==238 and all(m.isfile()for m in members)
        assert {m.name:sha(archive.extractfile(m).read())for m in members}==before
    for name,pin in before.items():
        if not name.startswith('parent208/'):assert sha((P/'source'/name).read_bytes())==pin
        if args.current:assert sha((SOURCE/name).read_bytes())==pin
    report=read('measured/report.json');owner=read('measured/owner-terminal.json')
    assert report['passed']is True and report['exit_code']==0 and report['test_count']==191 and report['test_failures_errors_skips']==0
    assert report['source_count']==238 and report['source_unchanged']is True and report['child_reaped']is True and report['child_group_absent']is True
    assert sha((P/'measured/sources.tar.gz').read_bytes())==report['source_archive_sha256']
    assert sha((P/'measured/run.py').read_bytes())==report['driver_sha256']
    assert owner['report_sha256']==sha((P/'measured/report.json').read_bytes())and owner['session_id']==46227
    assert owner['tool_exit_code']==0 and owner['tool_reaped']is True and all(r['pid_absent']is True and r['group_absent']is True for r in owner['processes'])
    xml=ET.parse(P/'measured/tests.xml').getroot()
    assert len(xml.findall('.//testcase'))==191 and not any(xml.findall('.//'+n)for n in ('failure','error','skipped'))
    old=read('history/focused-v2/report.json');assert old['exit_code']==0 and old['source_unchanged']is False
    old_xml=ET.parse(P/'history/focused-v2/tests.xml').getroot();assert len(old_xml.findall('.//testcase'))==71 and not old_xml.findall('.//failure')
    neg=ET.parse(P/'history/held-v1/negative.xml').getroot();assert len(neg.findall('.//testcase'))==1 and len(neg.findall('.//failure'))==1
    assert read('source/execution_authorization.json')['actual_execution_enabled']is False
    print(json.dumps(dict(passed=True,sources=238,tests=191,current=args.current,scope='recorded source/terminal evidence only; no backend or test execution')))
if __name__=='__main__':main()
