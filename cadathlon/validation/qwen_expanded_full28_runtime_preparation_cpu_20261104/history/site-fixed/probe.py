"""Exact old/new audit AST on fresh synthetic stdlib/site roots; no imports."""
from pathlib import Path
from types import SimpleNamespace
import argparse,ast,hashlib,json,os

def run(driver,output):
    output.mkdir(exist_ok=False);raw=driver.read_bytes();(output/'driver.py.txt').write_bytes(raw)
    (output/'probe.py').write_bytes(Path(__file__).read_bytes())
    stdlib=output/'synthetic-base/lib/python3.13';base=stdlib/'site-packages';venv=output/'synthetic-venv/lib/python3.13/site-packages';owned=output/'owned';owned.mkdir()
    for p in (base/'unrelated_package',base/'numpy',base/'pytest',venv,stdlib/'json'):p.mkdir(parents=True,exist_ok=True)
    targets=[('unrelated_base_json',base/'unrelated_package/ordinary.json',False),('required_numpy',base/'numpy/__init__.py',True),('required_pytest',base/'pytest/__init__.py',True),('stdlib_json',stdlib/'json/__init__.py',True)]
    for _,p,_ in targets:p.write_text('public synthetic path-policy fixture\n')
    tree=ast.parse(raw);nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('trusted_paths','PublicAudit')];assert len(nodes)==2
    ns=dict(Path=Path,os=os,sysconfig=SimpleNamespace(get_paths=lambda:dict(stdlib=str(stdlib),purelib=str(venv),platlib=str(venv))),
        site=SimpleNamespace(getsitepackages=lambda:[str(venv),str(base)]),sys=SimpleNamespace(path=[str(stdlib),str(venv),str(base)]),FORBIDDEN=(),HISTORICAL=())
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(driver),'exec'),ns)
    audit=ns['PublicAudit'](set(),owned,SimpleNamespace(check=lambda:None));rows=[]
    before={str(p.relative_to(output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in output.rglob('*') if p.is_file()}
    for name,path,allowed in targets:
        error=None
        try:audit('open',(str(path),'r',0));observed=True
        except PermissionError as e:observed=False;error=type(e).__name__
        rows.append(dict(case=name,expected_allowed=allowed,observed_allowed=observed,passed=allowed==observed,error=error))
    after={n:hashlib.sha256((output/n).read_bytes()).hexdigest() for n in before};assert before==after
    report=dict(scope='Exact source AST, synthetic path roots; no installed content modified or imported and no actual private/body read.',driver_sha256=hashlib.sha256(raw).hexdigest(),source_before=before,source_after=after,cases=rows,passed=all(r['passed'] for r in rows))
    (output/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report,sort_keys=True))
    return 0 if report['passed']else 1
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--driver',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args();raise SystemExit(run(a.driver,a.output))
