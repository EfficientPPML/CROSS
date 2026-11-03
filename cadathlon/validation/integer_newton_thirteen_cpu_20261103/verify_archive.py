"""Verify retained source/test/process evidence without importing the runtime."""
from pathlib import Path,PurePosixPath
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def check(value,message):
    if not value:raise ValueError(message)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def archive(name,expected):
    with tarfile.open(ROOT/name) as t:
        members=t.getmembers()
        check(len(members)==len(expected)==len({m.name for m in members}),'source inventory mismatch')
        check(set(expected)=={m.name for m in members},'source names mismatch')
        for m in members:
            p=PurePosixPath(m.name)
            check(m.isfile() and not p.is_absolute() and '..' not in p.parts,'unsafe archive member')
            check(p.suffix in {'.py','.json'},'unexpected source artifact')
            check(hashlib.sha256(t.extractfile(m).read()).hexdigest()==expected[m.name],'source hash mismatch')
def verify():
    r=json.loads((ROOT/'actual/report.json').read_text());a=json.loads((ROOT/'audit.json').read_text());p=json.loads((ROOT/'prelaunch/ready.json').read_text())
    check(r['passed'] is True and r['cleanup_errors']==[] and r['input_immutability'] is True,'execution failed')
    check(a['observed_terminal_exit_code']==0 and a['reaped'] is True,'terminal outcome not proven')
    before=r['source_hashes_before']
    check(len(before)==99 and before==r['source_hashes_after']==p['source_hashes_before']==p['source_hashes_after'],'source drift')
    archive('actual/sources.tar.gz',before)
    pre=json.loads((ROOT/'prelaunch/public-preflight.json').read_text())
    check(pre['passed'] is True and pre['preflight_only'] is True,'preflight failed')
    check(pre['source_hashes_before']==pre['source_hashes_after']==before,'preflight source differs')
    archive('prelaunch/public-preflight-sources.tar.gz',before)
    check(p['passed'] is True and p['returncode']==0,'contract tests failed')
    suites=ET.parse(ROOT/'prelaunch/junit.xml').getroot().findall('testsuite')
    check(sum(int(s.attrib['tests']) for s in suites)==14,'test count differs')
    check(all(int(s.attrib[k])==0 for s in suites for k in ('errors','failures','skipped')),'nonpassing tests')
    checks=r['checks'];stages=r['declared_stage_outputs']
    check(len(checks)==13 and [c['update'] for c in checks]==list(range(1,14)),'missing checkpoints')
    check(all(c['passed'] is True and c['metadata_ok'] is True and c['all_slots_checked']==32768 and c['max_abs_error_against_exact_float_circuit']<1e-6 for c in checks),'checkpoint gate differs')
    check([c['update'] for c in stages]==[9,12,13,13] and all(c['passed'] is True and c['all_segment_slots_checked']==8192 and c['max_abs_error_against_true_inverse']<1e-6 for c in stages),'declared stage gate differs')
    for name,h in a['file_sha256'].items():check(sha(ROOT/name)==h,'archived artifact changed: '+name)
    for path in ROOT.glob('*.py'):
        if path.name in before:check(sha(path)==before[path.name],'loose source differs')
    check(not any(p.suffix in {'.npy','.npz','.safetensors','.bin'} for p in ROOT.rglob('*') if p.is_file()),'unexpected data artifact')
    return dict(passed=True,source_files=99,tests=14,checkpoints=13,declared_outputs=4,observed_exit_code=0,scope='independent conditional inverse components only; no full softmax or model')
if __name__=='__main__':print(json.dumps(verify(),indent=2))
