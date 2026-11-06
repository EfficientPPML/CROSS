from pathlib import Path
import ast,os,json
from types import SimpleNamespace
import pytest
import client_io as io

@pytest.mark.parametrize('kind',['fixture','reference'])
@pytest.mark.parametrize('failure',['bootstrap','services'])
def test_startup_never_publishes_in_borrowed_private_root(tmp_path,kind,failure):
    private={n:tmp_path/('private-'+n)for n in ('fixture','reference')}
    for p in private.values():p.mkdir(mode=0o700);(p/'sentinel').write_text('untouched synthetic fixture')
    fields=('key_workspace','model_workspace','io_workspace','response_root','key_evidence','model_evidence','stage3_evidence','stage4_evidence','fixture_evidence','reference_evidence')
    args={n:tmp_path/n for n in fields}
    for p in args.values():p.mkdir()
    for name in ('fixture','reference'):
        raw=json.dumps({'private_workspace':str(private[name])}).encode();(args[name+'_evidence']/'report.json').write_bytes(raw);args[name+'_report_sha256']=io.digest(raw)
    def bootstrap(*a):
        if failure=='bootstrap':raise ValueError('synthetic source failure')
        return {},{}
    def services():raise ValueError('synthetic import failure')
    scope=dict(Path=Path,io=io,HERE=tmp_path/'source',os=os,bootstrap=bootstrap,services=services)
    tree=ast.parse(Path(__file__).with_name('worker.py').read_text());selected=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in ('run','fixture_root_public','public_private_roots')]
    exec(compile(ast.Module(body=selected,type_ignores=[]),'<actual worker.run AST>','exec'),scope)
    output=private[kind]/'new-report.json'
    with pytest.raises(ValueError):scope['run'](source_manifest=tmp_path/'manifest',source_manifest_sha256='unused',result_workspace=tmp_path/'results',result_identity=[1,2],output=output,**args)
    assert not output.exists(),'standalone startup failure published inside borrowed private tree'
    assert(private[kind]/'sentinel').read_text()=='untouched synthetic fixture'
