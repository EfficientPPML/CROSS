"""Portable public source/proof verifier. Never imports archived runtime."""
from pathlib import Path
import argparse,hashlib,io,json,tarfile,xml.etree.ElementTree as ET
HERE=Path(__file__).resolve().parent
REPORT='31f9120534e1d23843b3f7a5ec7e68c77963ac7456e5e0b492557aedd26b4315'
ARCHIVE='fd46274e075565bcec545cca13f03470d4b9506b6327ffd7f403a756acbcaa94'
def sha(raw):return hashlib.sha256(raw).hexdigest()
def read_json(path):return json.loads(path.read_bytes())
def source_tar(raw,expected=None,depth=0):
    assert depth<=5
    with tarfile.open(fileobj=io.BytesIO(raw))as t:
        rows=t.getmembers();assert len(rows)==len({x.name for x in rows})
        assert sum(x.size for x in rows)<=32<<20
        actual={}
        for x in rows:
            path=Path(x.name)
            assert x.isfile()and not path.is_absolute()and '..'not in path.parts and 0<=x.size<=16<<20
            data=t.extractfile(x).read();assert len(data)==x.size;actual[x.name]=sha(data)
            if x.name.endswith('.tar.gz'):source_tar(data,depth=depth+1)
            else:assert path.suffix in ('.py','.json','.md')or x.name.endswith('.py.txt')
    if expected is not None:assert actual==expected
    return actual

def verify(current=None):
    manifest=read_json(HERE/'files.sha256.json')
    files={p.relative_to(HERE).as_posix():p for p in HERE.rglob('*')if p.is_file()and p!=HERE/'files.sha256.json'and '__pycache__'not in p.parts}
    assert set(files)==set(manifest)
    for name,path in files.items():assert not path.is_symlink()and sha(path.read_bytes())==manifest[name],name
    p=HERE/'measured';r=read_json(p/'report.json');i=read_json(p/'installed.json');terminal=read_json(p/'terminal_observation.json')
    assert sha((p/'report.json').read_bytes())==REPORT and sha((p/'sources.tar.gz').read_bytes())==ARCHIVE
    expected=r['source_hashes_start'];assert len(expected)==286 and expected==r['source_hashes_end']==read_json(p/'source_manifest.json')
    source_tar((p/'sources.tar.gz').read_bytes(),expected)
    assert all(r[k]is True for k in ('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','source_archive_verified','all_started_children_reaped'))
    assert r['actual_HE']is False and r['actual_server_launch']is False
    assert r['dependency_readiness']['ready']is True and r['dependency_readiness']['execution_authorized']is False
    cases=list(ET.parse(p/'tests.xml').iter('testcase'));assert len(cases)==r['test_count']==157
    assert not any(x.findall(k)for x in cases for k in ('failure','error','skipped'))
    assert all(i[k]is True for k in ('passed','body_complete','termination_scope_completed','cleanup_completed','source_files_unchanged','all_forbidden_entries_restored','allowed_server_entries_unchanged'))
    assert all(i[k]is False for k in ('actual_HE','actual_contexts','actual_keys','payload_reads'))
    assert i['source_hashes_start']==i['source_hashes_end']==expected and i['denied_count']==len(i['denied_entries'])==25
    assert i['source_guard_start']['fresh_runtime_gate']is True
    for records in (i['source_guard_end']['loaded_local_modules'],i['source_guard_end']['entry_helpers']):
        for row in records.values():assert expected[row['path']]==row['sha256']
    assert terminal['session_id']==69570 and terminal['observed_exit_code']==0 and terminal['session_reaped']is True and terminal['collector_absent']is True
    assert terminal['report_sha256']==REPORT and terminal['source_archive_sha256']==ARCHIVE
    for row in terminal['children']:
        assert row['absent']is True
        process=row['process'];assert process['direct_returncode']==0 and not process['cleanup_errors']
        assert all(process[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
    failed=HERE/'history/failed-installed-v1';old=read_json(failed/'report.json')
    assert old['passed']is False and old['source_hashes_start']==old['source_hashes_end']
    source_tar((failed/'sources.tar.gz').read_bytes(),old['source_hashes_start'])
    assert read_json(failed/'terminal_observation.json')['observed_exit_code']==1
    old_cases=list(ET.parse(failed/'tests.xml').iter('testcase'));assert len(old_cases)==157 and not any(x.findall(k)for x in old_cases for k in ('failure','error','skipped'))
    pending=HERE/'history/development/pending-dependencies-v1';snapshot=read_json(pending/'snapshot.json')
    assert snapshot['source_count']==283 and snapshot['tests']==124
    source_tar((pending/'sources.tar.gz').read_bytes(),snapshot['source_hashes'])
    for version in (1,2):
        base=HERE/'history/development';hashes=read_json(base/f'worker-collector-v{version}-sources.json')
        assert len(hashes)==6;source_tar((base/f'worker-collector-v{version}-sources.tar.gz').read_bytes(),hashes)
    proposal=read_json(HERE/'launch-proposal/proposal.json')
    grant=read_json(HERE/'root-review/launch.json');prelaunch=read_json(HERE/'root-review/prelaunch.json')
    for name,digest in read_json(HERE/'launch-proposal/files.sha256.json').items():
        assert Path(name).name==name and sha((HERE/'launch-proposal'/name).read_bytes())==digest
    assert sha((HERE/'launch-proposal/proposal.json').read_bytes())==prelaunch['proposal_sha256']
    assert sha((HERE/'root-review/launch.json').read_bytes())==prelaunch['root_launch_receipt_sha256']
    assert grant['command']==proposal['command'] and grant['actual_client_authorized']is False
    assert proposal['source_count']==286 and proposal['source_archive_sha256']==ARCHIVE
    assert proposal['source_report_sha256']==REPORT and prelaunch['passed']is True
    if current is not None:
        current=Path(current)
        assert {n:sha((current/n).read_bytes())for n in expected}==expected
    return dict(passed=True,files=len(files),sources=286,tests=157,installed_denials=25,current_checked=current is not None,actual_HE=False)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--current',type=Path);a=p.parse_args();print(json.dumps(verify(a.current)))
