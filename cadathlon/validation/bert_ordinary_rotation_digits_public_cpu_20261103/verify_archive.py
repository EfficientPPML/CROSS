"""Portable public evidence/source/AST readback; no external payload or JAX."""
from pathlib import Path
import ast,copy,hashlib,json,tarfile,xml.etree.ElementTree as ET
ROOT=Path(__file__).resolve().parent

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read_json(name):return json.loads((ROOT/name).read_bytes())
def archive(name,expected):
    with tarfile.open(ROOT/name) as handle:
        members=handle.getmembers()
        assert len(members)==len(expected)==len({m.name for m in members})
        assert all(m.isfile() and not Path(m.name).is_absolute() and '..' not in Path(m.name).parts and m.size<=8<<20 for m in members)
        assert sum(m.size for m in members)<=16<<20
        raw={m.name:handle.extractfile(m).read() for m in members}
    assert {n:sha(v) for n,v in raw.items()}==expected
    return raw

def methods(raw):
    tree=ast.parse(raw)
    return {n.name:n for c in tree.body if isinstance(c,ast.ClassDef) and c.name=='_HERotKernel' for n in c.body if isinstance(n,ast.FunctionDef)}

def main():
    pins=read_json('artifact_hashes.json')
    files={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='artifact_hashes.json'}
    assert files==set(pins)
    for name,digest in pins.items():assert sha((ROOT/name).read_bytes())==digest,name
    report=read_json('v2/report.json');before=report['source_hashes_before']
    assert report['passed'] is report['terminal_scope_completed'] is report['cleanup_completed'] is report['process_reaped'] is True
    assert report['source_count']==len(before)==40
    assert before==report['source_hashes_after']==report['snapshot_hashes']
    assert report['source_files_unchanged'] is report['source_archive_verified'] is True and report['cleanup_errors']==[]
    assert sha((ROOT/'v2/sources.tar.gz').read_bytes())==report['source_archive_sha256']=='2d8d4f9454f013d11c3ab2d7d2bf288834abf009b6b86c6616bebb9aa7c0c081'
    raw=archive('v2/sources.tar.gz',before)
    for name in ('candidate.patch','check_public.py','collect_public.py','test_collector.py','source_origin.json'):assert raw[name]==(ROOT/'v2'/name).read_bytes()
    origin=json.loads(raw['source_origin.json'])
    for name,digest in origin['before'].items():assert sha(raw['baseline_herot.py'] if name=='herot.py' else raw['checkout/jaxite_word/'+name])==digest
    old,new=methods(raw['baseline_herot.py']),methods(raw['checkout/jaxite_word/herot.py'])
    assert set(new)-set(old)=={'_rotate_from_digits_array'}
    assert all(ast.dump(old[n])==ast.dump(new[n]) for n in old)
    class Transform(ast.NodeTransformer):
        def visit_Assign(self,node):
            if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Attribute) and node.value.func.attr=='_prepare_decomposition_inputs':return None
            return self.generic_visit(node)
        def visit_Call(self,node):
            if isinstance(node.func,ast.Attribute) and node.func.attr=='_decompose_part':return ast.Subscript(value=ast.Name(id='digits',ctx=ast.Load()),slice=copy.deepcopy(node.args[2]),ctx=ast.Load())
            return self.generic_visit(node)
    expected=Transform().visit(copy.deepcopy(old['_rotate_barrett_fused'])).body[1:]
    actual=new['_rotate_from_digits_array'].body
    start=next(i for i,n in enumerate(actual) if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple) and [v.id for v in n.targets[0].elts]==['eval_a','eval_b'])
    assert ast.dump(ast.Module(body=expected,type_ignores=[]))==ast.dump(ast.Module(body=actual[start:],type_ignores=[]))
    parity=read_json('v2/public-parity/report.json');assert parity==report['public_result'] and parity['passed'] is True
    assert parity['no_context_keygen_modules_loaded'] is True and parity['cpu_affinity']==[168,169,170,171]
    assert [(r['N'],r['Q']) for r in parity['fixtures']]==[(n,q) for n in (16,256) for q in (12,14,26,28,52,100)]
    assert parity['total_comparisons']==sum(len(r['comparisons']) for r in parity['fixtures'])==372
    assert parity['total_negative_guards']==sum(r['negative_guards'] for r in parity['fixtures'])==180
    for row in parity['fixtures']:
        assert row['P']==14 and row['alpha']==13 and row['effective_parts']==(row['Q']+12)//13
        assert row['negative_guards']==15 and row['source_and_digits_unchanged'] is row['keys_unchanged'] is True
        assert all(c['equal'] is True and c['dtype']=='uint32' for c in row['comparisons'])
    counts=report['test_counts'];assert counts==dict(tests=9,errors=0,failures=0,skipped=0)
    suites=ET.parse(ROOT/'v2/tests.xml').getroot().findall('testsuite')
    assert {k:sum(int(s.attrib.get(k,0)) for s in suites) for k in counts}==counts
    assert len(report['processes'])==2 and all(p['reaped'] is True and p['returncode']==0 for p in report['processes'])
    terminal=read_json('terminal.json');assert terminal['authoritative_session_id']==82669 and terminal['outer_exit_code']==0 and terminal['outer_reaped'] is True
    prior=read_json('historical/collector-v1/report.json');priorraw=archive('historical/collector-v1/sources.tar.gz',prior['source_hashes_before'])
    assert prior['passed'] is False and prior['source_files_unchanged'] is False
    assert prior['source_hashes_before']==prior['snapshot_hashes'] and prior['returncode']==0 and prior['process_reaped'] is True
    assert len(prior['source_hashes_before'])==37 and len(prior['source_hashes_after'])==67
    assert all(prior['source_hashes_after'][n]==h for n,h in prior['source_hashes_before'].items())
    extras=set(prior['source_hashes_after'])-set(prior['source_hashes_before']);assert len(extras)==30 and all(n.startswith('source-bound-v1/snapshot/checkout/') for n in extras)
    assert prior['public_result']['total_comparisons']==372 and prior['public_result']['total_negative_guards']==144
    assert terminal['historical_v1']['outer_exit_code'] is None
    print(json.dumps(dict(passed=True,local_sources=40,collector_tests=9,public_comparisons=372,rejected_inputs=180,historical_failed_collector_preserved=True,no_HE_execution=True)))
if __name__=='__main__':main()
