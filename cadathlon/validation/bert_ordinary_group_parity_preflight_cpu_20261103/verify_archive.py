"""Portable source/evidence verification only; no runtime imports or writes."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent
SHA=lambda b:hashlib.sha256(b).hexdigest()

def archive(path,expected):
    with tarfile.open(path) as tar:
        members=tar.getmembers()
        assert len(members)==len(expected)==len({m.name for m in members})
        assert all(m.isfile() and not m.name.startswith('/') and '..' not in Path(m.name).parts for m in members)
        assert all(m.name.endswith(('.py','.json')) and m.size<=8<<20 for m in members)
        data={m.name:tar.extractfile(m).read() for m in members}
    assert {n:SHA(b) for n,b in data.items()}==expected
    return data

def main():
    inventory=json.loads((ROOT/'artifact_hashes.json').read_bytes())
    actual={p.relative_to(ROOT).as_posix():SHA(p.read_bytes()) for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_hashes.json' and '__pycache__' not in p.parts}
    assert actual==inventory
    path=ROOT/'final-v2';raw=(path/'report.json').read_bytes();report=json.loads(raw)
    assert SHA(raw)=='67aaba20f6810663159858e74ec51414bbf5420bc566e207099f133a4ee9c350'
    assert report['passed']is True and report['tests_passed']==214 and report['installed_denials']==28
    assert report['actual_execution_attempted']is False and report['terminal_scope_completed']is True
    assert report['source_files_unchanged']is True and report['source_archive_verified']is True and report['all_processes_reaped']is True
    expected=report['source_hashes_before'];assert len(expected)==258 and expected==report['source_hashes_after']
    assert SHA((path/'sources.tar.gz').read_bytes())==report['source_archive_sha256']=='251a8bc6d8f555d0cb54539158475440d9d4f750649545daa4aa9e933817b5fa'
    sources=archive(path/'sources.tar.gz',expected)
    deps=json.loads(sources['group_dependencies.json']);origin=json.loads(sources['group_origin.json'])
    assert deps['ready']is False and origin['parent_source_count']==231
    changed={n for n,h in origin['parent_source_hashes'].items() if expected[n]!=h}
    assert changed==set(deps['overlays'])=={'runtime/jaxite_word/herot.py','runtime/stage/key_contexts.py'}
    assert all(expected[n]==h for n,h in deps['overlays'].items())
    assert expected['grouped_rotation.py']==origin['group_module_sha256']=='7e042d5bbb2e10cff2e223cbab19542ee1d95e1f4b21b8c3f44a3d7aeb3b594b'
    assert expected['runtime/jaxite_word/herot.py']==origin['raw_herot_sha256']=='d70926d7b6d934016a87fa106ce7c31a414f401877a520d1f0a81d8d781ac333'
    for p in (ROOT/'review_sources').iterdir():assert p.read_bytes()==sources[p.name]
    suites=ET.parse(path/'tests.xml').getroot()
    assert sum(int(s.attrib.get('tests',0)) for s in suites)==214
    assert all(int(s.attrib.get('errors',0))==int(s.attrib.get('failures',0))==0 for s in suites)
    assert [r['kind'] for r in report['processes']]==['tests','installed','public']
    for row in report['processes']:
        r=row['record'];assert r['returncode']==0 and not r.get('error') and r['result']['passed']is True
        assert all(r[n]is True for n in ('process_reaped','public_key_store_removed','owned_directory_handle_closed'))
        assert r==json.loads((path/('process-'+row['kind'])/'process.json').read_bytes())
    installed=json.loads((path/'installed/report.json').read_bytes())
    assert installed['passed']is True and installed['denied_count']==28 and installed['all28_restored']is True
    assert installed['forbidden_payload_reads']==[] and installed['source_hashes_before']==installed['source_hashes_after']==expected
    archive(path/'installed/sources.tar.gz',expected)
    public=json.loads((path/'public-gates/report.json').read_bytes())
    assert public['passed']is True and public['execute']is False and public['actual_execution_attempted']is False
    assert public['reviewed_sources']['ready']is False and public['terminal_scope_completed']is True
    assert public['source_hashes_start']==public['source_hashes_end']==public['execution_source_hashes_end']==expected
    assert public['io_workspace_retained']is True and public['external_workspaces_never_removed']is True
    archive(path/'public-gates/sources.tar.gz',expected)
    plan=public['public_plan'];assert plan['actual_rotation_calls']==116 and plan['exact_pair_comparisons']==66 and plan['shared_decompositions']==6
    assert len(plan['required_rotation_indices'])==26 and set(plan['required_rotation_indices'])<=set(plan['rotation_inventory'])
    assert sum(r['requests'] for r in plan['rows'])==66 and sum(r['nonzero_completions'] for r in plan['rows'])==58
    assert max(r['result_bytes'] for r in plan['rows'])==1572864000 and max(r['digit_bytes'] for r in plan['rows'])==478150656
    terminal=json.loads((path/'terminal.json').read_bytes());assert terminal['session_id']==42797 and terminal['authoritative_tool_exit_code']==0 and terminal['authoritative_tool_reaped']is True
    old=json.loads((ROOT/'failed-preflight-v1/report.json').read_bytes());assert old['passed']is False and old['actual_execution_attempted']is False
    archive(ROOT/'failed-preflight-v1/sources.tar.gz',old['source_hashes_before'])
    assert old['processes'][0]['record']['returncode']==1 and old['processes'][0]['record']['process_reaped']is True
    development=json.loads((ROOT/'development/harness-development-v1/source_hashes.json').read_bytes())
    archive(ROOT/'development/harness-development-v1/sources.tar.gz',development)
    print(json.dumps(dict(passed=True,source_count=258,tests=214,installed_denials=28,public_prerequisites=True,actual_execution=False)))
if __name__=='__main__':main()
