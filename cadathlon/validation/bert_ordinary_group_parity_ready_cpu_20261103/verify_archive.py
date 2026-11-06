"""Portable, read-only verification; no runtime, model or key imports."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def archive(path,expected):
    with tarfile.open(path) as tar:
        members=tar.getmembers();assert len(members)==len(expected)==len({m.name for m in members})
        assert all(m.isfile() and m.size<=8<<20 and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        raw={m.name:tar.extractfile(m).read() for m in members}
    assert {n:sha(b) for n,b in raw.items()}==expected
    return raw

def main():
    inv=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    assert inv=={p.relative_to(ROOT).as_posix():sha(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
    b=(ROOT/'report.json').read_bytes();assert sha(b)=='d9674ca058d4209e3f79cde67ba97974ad8b09507ae252cd4033a95b00a89f22';r=json.loads(b)
    assert r['passed']is True and r['tests_passed']==214 and r['installed_denials']==28 and r['actual_execution_attempted']is False
    assert all(r[n]is True for n in ('source_files_unchanged','source_archive_verified','all_processes_reaped','cleanup_completed','terminal_scope_completed'))
    expected=r['source_hashes_before'];assert len(expected)==259 and expected==r['source_hashes_after']
    assert sha((ROOT/'sources.tar.gz').read_bytes())==r['source_archive_sha256']=='055c8731d6de348f981b9c865fa6d9f92da95ca8c841f719bcca6484aac9c9a2'
    raw=archive(ROOT/'sources.tar.gz',expected);origin=json.loads(raw['readiness_origin.json'])
    oldbytes=(ROOT/'parent/report.json').read_bytes();assert sha(oldbytes)==origin['parent_report_sha256'];old=json.loads(oldbytes)
    assert sha((ROOT/'parent/sources.tar.gz').read_bytes())==origin['parent_archive_sha256'];parent=archive(ROOT/'parent/sources.tar.gz',old['source_hashes_before'])
    assert origin['parent_source_hashes']==old['source_hashes_before'] and len(parent)==258
    assert {n for n in parent if raw[n]!=parent[n]}=={'group_dependencies.json','test_group_protocol.py'}
    assert set(raw)-set(parent)=={'readiness_origin.json'}
    assert json.loads(raw['group_dependencies.json'])['ready']is True and json.loads(parent['group_dependencies.json'])['ready']is False
    suites=ET.parse(ROOT/'tests.xml').getroot();assert sum(int(s.attrib.get('tests',0)) for s in suites)==214
    assert all(int(s.attrib.get('errors',0))==int(s.attrib.get('failures',0))==int(s.attrib.get('skipped',0))==0 for s in suites)
    assert [x['kind'] for x in r['processes']]==['tests','installed','public']
    for x in r['processes']:
        row=x['record'];assert row['returncode']==0 and not row.get('error') and row['result']['passed']is True
        assert all(row[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
        assert row==json.loads((ROOT/('process-'+x['kind'])/'process.json').read_bytes())
    installed=json.loads((ROOT/'installed/report.json').read_bytes());assert installed['passed']is True and installed['denied_count']==28 and installed['all28_restored']is True and installed['forbidden_payload_reads']==[]
    assert installed['source_hashes_before']==installed['source_hashes_after']==expected;archive(ROOT/'installed/sources.tar.gz',expected)
    p=json.loads((ROOT/'public-gates/report.json').read_bytes());assert p['passed']is True and p['execute']is False and p['actual_execution_attempted']is False
    assert p['reviewed_sources']['ready']is True and p['terminal_scope_completed']is True and p['external_workspaces_never_removed']is True
    assert p['source_hashes_start']==p['source_hashes_end']==p['execution_source_hashes_end']==expected;archive(ROOT/'public-gates/sources.tar.gz',expected)
    plan=p['public_plan'];assert plan['actual_rotation_calls']==116 and plan['exact_pair_comparisons']==66 and plan['shared_decompositions']==6
    assert sum(x['nonzero_completions'] for x in plan['rows'])==58 and sum(x['zero_aliases'] for x in plan['rows'])==8
    t=json.loads((ROOT/'terminal.json').read_bytes());assert t['session_id']==72899 and t['authoritative_tool_exit_code']==0 and t['authoritative_tool_reaped']is True
    print(json.dumps(dict(passed=True,sources=259,tests=214,installed_denials=28,runtime_unchanged=True,actual_execution=False)))
if __name__=='__main__':main()
