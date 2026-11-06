"""Portable public artifact verification; never opens model/input bodies."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def verify(root):
    proof=root/'proof';r=json.loads((proof/'report.json').read_text());outer=json.loads((proof/'outer_receipt.json').read_text())
    assert all(r.get(n)is True for n in('passed','complete','cleanup_completed','termination_scope_completed','source_files_unchanged','acceptance_requires_independent_outer_exit0_and_reap'))
    assert outer['returncode']==0 and type(outer['returncode'])is int and outer['reaped']is True and outer['report_sha256']==sha(proof/'report.json')
    expected=r['source_hashes_before'];assert expected==r['source_hashes_after']==r['execution_hashes_before']==r['execution_hashes_after']
    assert sha(proof/'sources.tar.gz')==r['source_archive_sha256']
    with tarfile.open(proof/'sources.tar.gz')as t:
        ms=t.getmembers();assert len(ms)==len(expected)==len({m.name for m in ms})
        assert all(m.isfile()for m in ms)
        assert {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest()for m in ms}==expected
    assert len(r['children'])==2
    for child in r['children']:
        assert all(child.get(n)is True for n in('direct_child_reaped','group_empty_verified','cleanup_completed'))and child['direct_returncode']==0
    for mode,n in [('component',79),('harness',7)]:
        audit=json.loads((proof/(mode+'-audit.json')).read_text());assert audit['passed']is True and not audit['denied_attempts']and not audit['backend_modules']
        ss=list(ET.parse(proof/(mode+'-audit.xml')).getroot().iter('testsuite'))
        counts={k:sum(int(s.get(k,0))for s in ss)for k in('tests','errors','failures','skipped')}
        assert counts==r['tests'][mode]==dict(tests=n,errors=0,failures=0,skipped=0)
    assert r['actual_candidate_execution']is False and r['model_or_HE_execution']is False
    print(json.dumps({'verified_sources':len(expected),'synthetic_component_tests':79,'harness_tests':7,'scope':'No actual candidate/model/HE result'}))
if __name__=='__main__':verify(P)
