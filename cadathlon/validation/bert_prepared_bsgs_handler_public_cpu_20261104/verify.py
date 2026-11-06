"""Portable public evidence verification; imports no backend or runtime."""
from pathlib import Path
import argparse,hashlib,json,tarfile,xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_bytes())
def source_hashes(path):
    with tarfile.open(path)as tar:
        members=tar.getmembers()
        assert len(members)==len({m.name for m in members})
        assert all(m.isfile()and not m.name.startswith('/')and '..'not in Path(m.name).parts and m.name.endswith(('.py','.json','.md'))for m in members)
        return {m.name:hashlib.sha256(tar.extractfile(m).read()).hexdigest()for m in members}
def cases(path,count,failed=0):
    rows=list(ET.parse(path).iter('testcase'));assert len(rows)==count
    assert sum(c.find('failure')is not None for c in rows)==failed
    assert not any(c.find(t)is not None for c in rows for t in ('error','skipped'))
def verify(root,current=None):
    root=Path(root);manifest=read(root/'artifact_sha256.json')
    actual={str(p.relative_to(root)):sha(p)for p in root.rglob('*')if p.is_file()and p.name!='artifact_sha256.json'}
    assert actual==manifest
    r=read(root/'measured/report.json');h=r['source_hashes_before']
    assert len(h)==265 and h==r['source_hashes_after']==source_hashes(root/'measured/sources.tar.gz')
    assert sha(root/'measured/sources.tar.gz')==r['source_archive_sha256']=='1218e06b8458996f669392276ef1e76ed2bb96618079b01c8818e2520f2e9834'
    assert sha(root/'measured/report.json')=='4bdf99941c6782f27e163e694d230f570ec07a1f2355da952f7852eea8c5e02d'
    for flag in ('passed','body_passed','complete','source_files_unchanged','source_archive_verified','source_archive_unchanged','all_started_children_reaped','termination_scope_completed'):assert r[flag]is True
    assert r['actual_HE']is False
    expected=[('engine',37),('prepared',48),('native-wrapper',9),('combined',26),('collector',13),('source-guard',36)]
    assert [(g['name'],g['tests'])for g in r['groups']]==expected
    assert all(g['passed']and g['bad']==0 and g['tests']==g['expected']for g in r['groups'])
    for name,count in expected:cases(root/'measured'/(name+'.xml'),count)
    assert len(r['started_children'])==6
    for child in r['started_children']:
        p=child['process'];assert p['direct_returncode']==0 and not p['cleanup_errors']and not p['signals_sent']
        assert all(p[k]is True for k in ('direct_child_reaped','group_empty_verified','cleanup_completed'))
    for n in ('initial_source_guard','final_source_guard'):
        g=r[n];assert g['passed']and g['source_file_count']==265 and len(g['critical_resolution'])==54 and g['combined_HE_or_full_numeric_pass']is False
    signal=list((root/'measured/signal-cases').glob('*.json'));assert len(signal)==9
    for path in signal:
        row=read(path);assert row['reaped']and row['returncode']in(129,130,143)
        for field in ('validation_passed','handler_state_restored','main_mask_restored','loader_lease_still_owned','original_prepared_owners_closed','provider_closed'):assert row['report'][field]is True
        assert row['report']['refresh_clear_count']==5 and row['report']['native_thread_count']>=4
    terminal=read(root/'measured/terminal_observation.json');assert terminal['session']==41112 and terminal['exit_code']==0 and terminal['reaped']and terminal['tests']==169 and terminal['all_producer_pids_and_child_groups_absent']
    failed=read(root/'failed-parent-closure/report.json');assert failed['passed']is False and failed['source_count']==265
    assert failed['source_hashes_before']==failed['source_hashes_after']==source_hashes(root/'failed-parent-closure/sources.tar.gz')
    assert sha(root/'failed-parent-closure/sources.tar.gz')==failed['source_archive_sha256']
    cases(root/'failed-parent-closure/engine.xml',37);cases(root/'failed-parent-closure/prepared.xml',49,1)
    for name in ('loader-export','collector-publication'):
        directory=root/'history'/name;row=read(directory/'report.json');archived=source_hashes(directory/'sources.tar.gz')
        assert sha(directory/'sources.tar.gz')==row['source_archive_sha256']
        assert row['source_hashes_before']==row['source_hashes_after']
        assert all(archived[n]==h for n,h in row['source_hashes_before'].items())
    cases(root/'history/collector-publication/tests.xml',3,3)
    if current:assert all(sha(Path(current)/name)==digest for name,digest in h.items())
    return dict(passed=True,sources=265,tests=169,current_checked=current is not None,scope='synthetic handler/owner/source construction only; no HE/full-model numeric result')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--current');a=p.parse_args();print(json.dumps(verify(a.root,a.current),sort_keys=True))
