"""Offline focused proof verifier; no model, parameter, input or backend reads."""
from pathlib import Path
import hashlib,json,tarfile,xml.etree.ElementTree as ET

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def verify(directory):
    p=Path(directory);r=json.loads((p/'report.json').read_bytes())
    for n in ('passed','cleanup_completed','termination_scope_completed','terminal_scope_completed','predecessor_verified','source_files_unchanged','zero_audited_parameter_input_reads'):
        if r.get(n)is not True:raise ValueError('missing true terminal gate: '+n)
    if r.get('encrypted_execution')is not False or r.get('backend_imported')is not False:raise ValueError('unexpected execution scope')
    expected=r['source_hashes_before']
    if expected!=r['source_hashes_after'] or len(expected)!=87:raise ValueError('exact87 unchanged sources required')
    if r['source_archive_sha256']!=sha(p/'sources.tar.gz'):raise ValueError('archive digest differs')
    with tarfile.open(p/'sources.tar.gz') as t:
        members=t.getmembers()
        if len(members)!=87 or len({m.name for m in members})!=87 or any(not m.isfile() or m.name.startswith('/') or '..' in Path(m.name).parts for m in members):raise ValueError('exact regular source inventory required')
        actual={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in members}
    if actual!=expected:raise ValueError('archive source bytes differ')
    if r['tests']!=dict(tests=150,failures=0,errors=0,skipped=0):raise ValueError('complete150 focused checks required')
    cases=ET.parse(p/'tests.xml').getroot().findall('.//testcase')
    if len(cases)!=150 or any(c.find(n)is not None for c in cases for n in ('failure','error','skipped')):raise ValueError('XML test result differs')
    if r['test_exit_code']!=0 or r['plan_exit_code']!=0:raise ValueError('both child exit0 required')
    if len(r['children'])!=2 or any(row.get(n)is not True for row in r['children'] for n in ('direct_child_reaped','group_empty_verified','cleanup_completed')):raise ValueError('direct wait plus owned-group absence required')
    if any(row.get('direct_returncode')!=0 for row in r['children']):raise ValueError('child nonzero')
    for n in ('tests-audit.json','plan-audit.json'):
        a=json.loads((p/n).read_bytes())
        if a['passed']is not True or a['denied_attempts'] or a['backend_modules']:raise ValueError('zero data/backend attempt audit required')
    if sha(p/'public_plan.json')!=r['public_plan_sha256'] or r['public_plan_sha256']!='c3bd5385c43234e83740a76c03f9ae91acaa7decb7c97b17e241db49436af980':raise ValueError('public continuation plan differs')
    plan=json.loads((p/'public_plan.json').read_bytes())
    if plan['reference']['executed']is not False or plan['vocabulary']['all_logits']!=151936:raise ValueError('public-only trajectory scope differs')
    return r
if __name__=='__main__':
    import sys
    r=verify(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'focused-v3')
    print('VERIFIED:87 sources/150 focused synthetic and tiny-reference checks; no fullmodel trajectory/HE.')
