"""Stage4 source assembly only. No runtime/library/model or crypto imports."""
from pathlib import Path
import ast
import hashlib
import importlib.machinery
import io
import json
import sys
import tarfile

HERE=Path(__file__).resolve().parent
STAGE3=Path('/tmp/bert-input-encryption-stage3-component-20261103/source-bound-preflight-v1')
JOINED=Path('/tmp/bert-full-model-immutable-masks-20261103/source-bound-v1')
ENGINE=Path('/tmp/bert-stage4-engine-cleanup-20261103/candidate-v2/full_model_engine.py')
STAGE3_TAR='87edee7a73d2f0c4309160c0e325aa1365aeae757d8072a22c7e3e189fc981df'
JOINED_TAR='eadb43c936ee82454e3716ae078e37ae45064483a5994c6235e146f2ff2fb5b1'
ENGINE_BASE='841fb5a5f635a10a5e2f3f84b04361816deda11ab1869f0472e1acc00507f009'
ENGINE_AFTER='2c68a0d7125bbdc314845553162cc0f9b07ea7fec283481d4b143f2130ddf129'


def digest(raw):return hashlib.sha256(raw).hexdigest()
def write(path,value):path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def source_archive(path,pinned,count):
    raw=path.read_bytes()
    if digest(raw)!=pinned:raise ValueError('externally pinned source archive differs')
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        members=archive.getmembers()
        if len(members)!=count or len({m.name for m in members})!=count:raise ValueError('exact source archive inventory required')
        result={}
        for member in members:
            name=Path(member.name)
            if not member.isfile() or name.is_absolute() or '..' in name.parts or member.size>4<<20:
                raise ValueError('ordinary bounded source member required')
            result[member.name]=archive.extractfile(member).read()
    return result


def copy_source_set(target,raw):
    target.mkdir(exist_ok=False)
    for name,data in raw.items():
        path=target/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)


def resolver_roots(root,*,after_package=True):
    roots=[root/'runtime/stage',root/'joined',root/'joined/checkout/cadathlon/common',root/'runtime']
    if after_package:roots.insert(0,root/'runtime/jaxite_word')
    return roots


def resolve_module(name,roots):
    """PathFinder reads module paths only; it never executes a loader."""
    parts=name.split('.');search=list(map(str,roots));spec=None
    for i in range(len(parts)):
        spec=importlib.machinery.PathFinder.find_spec('.'.join(parts[:i+1]),search)
        if spec is None:return None
        if i+1<len(parts):
            if spec.submodule_search_locations is None:return None
            search=list(spec.submodule_search_locations)
    return None if spec.origin is None else Path(spec.origin).resolve()


def audit_resolution(root):
    roots=resolver_roots(root);records={};collisions={}
    top={p.stem for directory in roots for p in directory.glob('*.py') if p.name!='__init__.py'}
    top.add('jaxite_word')
    for name in sorted(top):
        matches=[d/(name+'.py') for d in roots if (d/(name+'.py')).is_file()]
        if len(matches)>1:
            values={str(p.relative_to(root)):digest(p.read_bytes()) for p in matches}
            if len(set(values.values()))!=1:raise ValueError('conflicting import basename: '+name)
            collisions[name]=values
        chosen=resolve_module(name,roots)
        if chosen is None or not chosen.is_relative_to(root):raise ValueError('unresolved assembled module: '+name)
        records[name]={'path':str(chosen.relative_to(root)),'sha256':digest(chosen.read_bytes())}
    expected={'full_model_engine':'runtime/stage/full_model_engine.py','file_stages':'runtime/stage/file_stages.py',
        'key_contexts':'runtime/stage/key_contexts.py','server_public_model':'runtime/stage/server_public_model.py',
        'support':'runtime/stage/support.py','encrypted_bert_full_model':'joined/encrypted_bert_full_model.py',
        'encrypted_bert_layer_attention':'joined/encrypted_bert_layer_attention.py','layer_attention_contract':'joined/layer_attention_contract.py',
        'encrypted_score_reduction':'joined/checkout/cadathlon/common/encrypted_score_reduction.py',
        'jaxite_word':'runtime/jaxite_word/__init__.py','polynomial':'runtime/jaxite_word/polynomial.py',
        'ntt_mm':'runtime/jaxite_word/ntt_mm.py','ckks_ctx':'runtime/jaxite_word/ckks_ctx.py'}
    for name,path in expected.items():
        if records[name]['path']!=path:raise ValueError('critical module selection differs: '+name)
    for name in ('polynomial','ckks_ctx','mapping','key_gen','he_params','he_ops','bootstrapping'):
        p=resolve_module('jaxite_word.'+name,roots)
        if p!=resolve_module(name,roots):raise ValueError('flat/package source path differs: '+name)
        records['jaxite_word.'+name]={'path':str(p.relative_to(root)),'sha256':digest(p.read_bytes())}
    # Audit reachable literal imports, including those inside lazy functions.
    # Package __getattr__ aliases and type-derived NTT imports are explicitly
    #listed above; external library availability is not a hermeticity claim.
    pending=['file_stages','full_model_engine','encrypted_bert_full_model'];visited=set();external=set();dynamic=[]
    while pending:
        name=pending.pop()
        if name in visited:continue
        visited.add(name);path=resolve_module(name,roots)
        if path is None:external.add(name);continue
        tree=ast.parse(path.read_text(),filename=str(path))
        for node in ast.walk(tree):
            names=[]
            if isinstance(node,ast.Import):names=[a.name for a in node.names]
            elif isinstance(node,ast.ImportFrom) and node.level==0 and node.module:
                names=[node.module]
                for item in node.names:
                    nested=node.module+'.'+item.name
                    if item.name!='*' and resolve_module(nested,roots) is not None:names.append(nested)
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='import_module':
                if node.args and isinstance(node.args[0],ast.Constant) and isinstance(node.args[0].value,str):names=[node.args[0].value]
                else:dynamic.append({'file':str(path.relative_to(root)),'line':node.lineno,'expression':ast.unparse(node)})
            for target in names:
                if resolve_module(target,roots) is None:external.add(target)
                else:pending.append(target)
    return {'passed':True,'search_path_before_package':[str(p.relative_to(root)) for p in resolver_roots(root,after_package=False)],
        'search_path_after_package':[str(p.relative_to(root)) for p in roots],
        'critical_expected':expected,'module_resolution':records,'equal_basename_collisions':collisions,
        'reachable_local_modules':sorted(visited-external),'external_imports':sorted(external),'dynamic_imports':dynamic,
        'scope':'Static AST and PathFinder only; no loaders executed, no context or model/backend initialized.'}


def assemble(root=HERE):
    blocked=('numpy','jax','torch','transformers','jaxite_word')
    if any(n in sys.modules for n in blocked):raise RuntimeError('source assembler must run without numeric/runtime imports')
    stage=source_archive(STAGE3/'sources.tar.gz',STAGE3_TAR,76)
    graph=source_archive(JOINED/'sources.tar.gz',JOINED_TAR,137)
    sr=json.loads((STAGE3/'report.json').read_text());jr=json.loads((JOINED/'report.json').read_text())
    sh={n:digest(v) for n,v in stage.items()};jh={n:digest(v) for n,v in graph.items()}
    if not sr['passed'] or sh!=sr['source_hashes_start'] or sh!=sr['source_hashes_end'] or sh!=sr['archive_member_hashes']:
        raise ValueError('stage3 source preflight does not bind this archive')
    if not jr['passed'] or not jr['complete'] or jh!=jr['source_hashes_before'] or jh!=jr['source_hashes_after'] or jh!=jr['snapshot_source_hashes_after']:
        raise ValueError('completed joined ideal source proof required')
    if jr['tests']!=dict(tests=46,errors=0,failures=0,skipped=0) or jr['cleanup_errors'] or not jr['process_cleanup']['reaped'] or not jr['process_cleanup']['group_gone']:
        raise ValueError('joined ideal terminal checks incomplete')
    runtime={n[len('runtime/'):]:v for n,v in stage.items() if n.startswith('runtime/')}
    if len(runtime)!=59:raise ValueError('exact59-file copied runtime required')
    before_engine=runtime['stage/full_model_engine.py'];new_engine=ENGINE.read_bytes()
    if digest(before_engine)!=ENGINE_BASE or digest(new_engine)!=ENGINE_AFTER:raise ValueError('reviewed engine overlay identity differs')
    overlap={}
    for n,data in runtime.items():
        other='checkout/'+n
        if n.startswith('jaxite_word/') and other in graph:
            if data!=graph[other]:raise ValueError('shared core source differs: '+n)
            overlap[n]=digest(data)
    if len(overlap)!=30:raise ValueError('expected30 common core files')
    for a,b in [('stage/server_public_model.py','server_public_model.py'),('stage/support.py','checkout/cadathlon/common/support.py')]:
        if runtime[a]!=graph[b]:raise ValueError('shared public helper differs: '+a)
        overlap[a]=digest(runtime[a])
    copy_source_set(root/'runtime',runtime);copy_source_set(root/'joined',graph)
    (root/'runtime/stage/full_model_engine.py').write_bytes(new_engine)
    origins=root/'origins';origins.mkdir()
    for path,name in [(STAGE3/'sources.tar.gz','stage3_sources.tar.gz'),(STAGE3/'report.json','stage3_preflight_report.json'),
                      (JOINED/'sources.tar.gz','joined_sources.tar.gz'),(JOINED/'report.json','joined_ideal_report.json')]:
        (origins/name).write_bytes(path.read_bytes())
    (origins/'engine_before.py').write_bytes(before_engine)
    write(root/'import_resolution.json',audit_resolution(root))
    expected={'runtime/'+n:digest(v) for n,v in runtime.items()}|{'joined/'+n:digest(v) for n,v in graph.items()}
    expected['runtime/stage/full_model_engine.py']=ENGINE_AFTER
    actual={str(p.relative_to(root)):digest(p.read_bytes()) for d in ('runtime','joined') for p in (root/d).rglob('*') if p.is_file()}
    if actual!=expected or len(actual)!=196:raise ValueError('assembled bytes/inventory differ')
    if any(n in sys.modules for n in blocked):raise RuntimeError('assembler imported runtime unexpectedly')
    report={'passed':True,'scope':'Source assembly/import-resolution audit only; joined prerequisite is ideal-only, no actual server execution.',
        'source_file_count':196,'source_hashes':actual,'stage3_runtime_files':59,'joined_snapshot_files':137,
        'stage3_source_archive_sha256':STAGE3_TAR,'joined_source_archive_sha256':JOINED_TAR,
        'stage3_report_sha256':digest((STAGE3/'report.json').read_bytes()),'joined_report_sha256':digest((JOINED/'report.json').read_bytes()),
        'shared_core_file_count':30,'shared_equal_hashes':overlap,
        'sole_runtime_overlay':{'path':'runtime/stage/full_model_engine.py','before':ENGINE_BASE,'after':ENGINE_AFTER},
        'joined_ideal_proof_complete':True,'actual_server_compute_started':False,'contexts_constructed':False,
        'model_assets_keys_ciphertexts_read':False,'runtime_modules_imported':False}
    write(root/'assembly_report.json',report)
    return report


if __name__=='__main__':
    r=assemble();print(json.dumps({k:r[k] for k in ('passed','source_file_count','shared_core_file_count','joined_ideal_proof_complete','runtime_modules_imported')}))
