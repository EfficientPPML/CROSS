"""Source-bound preparation only; no genuine rotation or payload reads."""
from pathlib import Path
import hashlib,io,json,os,subprocess,sys,tarfile,time,xml.etree.ElementTree as ET
ROOT=Path('/tmp/bert-prepared-rotation-parity-20261103')
OUT=ROOT/'source-bound-preflight-v1'
PYTHON=Path('/tmp/cross-pretrained-validation-env/bin/python')
GATE=Path('/tmp/check_prepared_rotation_public_gates_20261103.py')
sys.path.insert(0,str(ROOT));import parity_protocol as protocol

def source_bytes():
    return protocol.sources()|{'external/preflight.py':Path(__file__).read_bytes(),'external/public_gate.py':GATE.read_bytes()}
def sha(x):return hashlib.sha256(x).hexdigest()
def write(x): (OUT/'report.json').write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def main():
    started=time.monotonic();OUT.mkdir(exist_ok=False);raw=source_bytes();before={n:sha(b) for n,b in raw.items()}
    report=dict(passed=False,scope='Mocked lifecycle/representation checks plus installed import boundary and public prerequisite gates only; no context, keys, ciphertext or model payload reads and no HE execution.',
        source_hashes_before=before,source_count=len(before),candidate_source_count=len(protocol.sources()),pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)))
    write(report)
    with tarfile.open(OUT/'sources.tar.gz','w:gz') as tar:
        for n,b in raw.items():m=tarfile.TarInfo(n);m.size=len(b);tar.addfile(m,io.BytesIO(b))
    tests=['test_parity_arithmetic.py','test_parity_protocol.py','test_parity_collector.py','test_parity_worker.py','test_parity_boundaries.py']
    commands=[('tests',[str(PYTHON),'-m','pytest','-q',*tests,'--junitxml='+str(OUT/'tests.xml')]),
        ('installed',[str(PYTHON),str(ROOT/'installed_probe.py'),'--output',str(OUT/'installed')]),
        ('public-gates',[str(PYTHON),str(GATE),str(OUT/'public-gates')])]
    records=[]
    for name,command in commands:
        start=time.monotonic()
        with (OUT/(name+'.log')).open('wb') as log:
            child=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            code=child.wait()
        records.append(dict(name=name,command=command,pid=child.pid,returncode=code,reaped=True,elapsed_s=time.monotonic()-start))
        report['processes']=records;write(report)
        if code:raise RuntimeError(name+' failed')
    suites=ET.parse(OUT/'tests.xml').getroot();count=sum(int(s.attrib.get('tests','0')) for s in suites)
    assert count==92 and all(int(s.attrib.get('failures','0'))==int(s.attrib.get('errors','0'))==0 for s in suites)
    installed=json.loads((OUT/'installed/report.json').read_text());gates=json.loads((OUT/'public-gates/report.json').read_text())
    assert installed['passed'] and installed['denied_count']==28 and installed['all28_restored']
    assert gates['passed'] and gates['execute']is False and gates['actual_execution_attempted']is False
    after={n:sha(b) for n,b in source_bytes().items()};assert before==after
    with tarfile.open(OUT/'sources.tar.gz') as tar:
        members=tar.getmembers();assert len(members)==len(before)==len({m.name for m in members})
        assert all(m.isfile() for m in members)
        assert {m.name:sha(tar.extractfile(m).read()) for m in members}==before
    report.update(passed=True,tests_passed=count,installed_denials=28,public_prerequisite_gates_passed=True,
        source_hashes_after=after,source_files_unchanged=True,source_archive_verified=True,
        source_archive_sha256=sha((OUT/'sources.tar.gz').read_bytes()),
        installed_report_sha256=sha((OUT/'installed/report.json').read_bytes()),public_gate_report_sha256=sha((OUT/'public-gates/report.json').read_bytes()),
        elapsed_s=time.monotonic()-started)
    write(report);print(json.dumps({k:v for k,v in report.items() if not isinstance(v,(dict,list))}))
if __name__=='__main__':main()
