from pathlib import Path
import ast,json,tarfile,hashlib,shutil
root=Path('/tmp/bert-prepared-bsgs-client-component-v3-20261104');root.mkdir(exist_ok=False)
parent=Path('/tmp/bert-prepared-bsgs-client-v2-preflight-v1-20261104');original=Path('/tmp/bert-original-stage5-io-successor-20261104');sha=lambda b:hashlib.sha256(b).hexdigest()
expected=json.loads((parent/'report.json').read_bytes())['source_hashes_start']
assert sha((parent/'sources.tar.gz').read_bytes())=='f9265bea1194dd4d9c35fede0e6aad279e30eb8d3a5b34da2feda9b8fe515bb6'
with tarfile.open(parent/'sources.tar.gz')as t:
 assert len(t.getmembers())==117
 for m in t:
  raw=t.extractfile(m).read();assert sha(raw)==expected[m.name];p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw)
for n in ('key_gate.py','input_gate.py'):
 shutil.copyfile(root/n,root/'baseline'/n);shutil.copyfile(original/n,root/n)
shutil.copyfile(root/'runtime/stage/transport.py',root/'baseline/transport.py');shutil.copyfile(original/'runtime/stage/transport.py',root/'runtime/stage/transport.py')
for a,b in [('test_prerequisite_readers.py','test_prerequisite_readers.py'),('test_original_transport_io.py','test_client_transport_io.py'),('test_client_bootstrap.py','test_client_bootstrap.py'),('client_bootstrap.py','client_bootstrap.py'),('READERS.md','READERS.md')]:
 text=(original/a).read_text().replace('original_client_worker','separate_client_worker').replace('original_client_collect','separate_client_collect')
 (root/b).write_text(text)
# Exact reviewed inline bootstrap with role-specific entry names supplied dynamically.
src=(original/'original_client_worker.py').read_text();a=src.index('def _cli_bootstrap():');b=src.index("sys.path[:0]",a);inline=src[a:b]
for name in ('separate_client_worker.py','separate_client_collect.py'):
 p=root/name;s=p.read_text();needle="sys.path[:0]=[str(HERE),str(HERE/'runtime/stage'),str(HERE/'runtime')]";assert needle in s;s=s.replace(needle,inline+needle);p.write_text(s)
p=root/'separate_client_collect.py';s=p.read_text().replace('def body(report,*,output,result_root,execute,args,worker_python,s):','def body(report,*,output,result_root,execute,args,worker_python,s,expected_sources=None):')
s=s.replace('io.environment();raw=sources.raw_sources(HERE);before=io.hashes(raw)',"io.environment();raw=sources.raw_sources(HERE);before=io.hashes(raw)\n        if expected_sources is not None and before!=expected_sources:raise ValueError('collector body differs from external source pin')")
s=s.replace("command=[str(worker_python),str(execution/'separate_client_worker.py')]","command=[str(worker_python),'-I','-B',str(execution/'separate_client_worker.py')]")
s=s.replace("def run(output,*,result_workspace,execute=False,worker_python=Path('/tmp/cross-pretrained-validation-env/bin/python'),**args):","def run(output,*,result_workspace,source_manifest=None,source_manifest_sha256=None,execute=False,worker_python=Path('/tmp/cross-pretrained-validation-env/bin/python'),**args):")
start="    # Discover externally authenticated PUBLIC"
insert="""    if source_manifest is None or source_manifest_sha256 is None:raise ValueError('external current source manifest required before collector services')
    source_bytes=io.read(source_manifest)
    if io.digest(source_bytes)!=source_manifest_sha256:raise ValueError('external collector source digest changed')
    expected_sources=io.parse(source_bytes)
    if _CLI_BOOTSTRAP is not None and expected_sources!=_CLI_BOOTSTRAP['source_hashes']:raise ValueError('collector source pin differs from fresh bootstrap')
    sources.check_sources(expected_sources,HERE)
"""
assert start in s;s=s.replace(start,insert+start)
s=s.replace('worker_python=worker_python,s=s),s.termination_signals','worker_python=worker_python,s=s,expected_sources=expected_sources),s.termination_signals')
s=s.replace("    p.add_argument('--output',required=True,type=Path)","    p.add_argument('--source-manifest',required=True,type=Path);p.add_argument('--source-manifest-sha256',required=True)\n    p.add_argument('--output',required=True,type=Path)");p.write_text(s)
p=root/'test_separate_client_collect.py';s=p.read_text().replace("assert command[1]==str(output/'execution_source/separate_client_worker.py')","assert command[1:3]==['-I','-B']and command[3]==str(output/'execution_source/separate_client_worker.py')").replace('range(2,len(command),2)','range(4,len(command),2)')
s=s.replace(" monkeypatch.setattr(c,'services',lambda:fake)"," monkeypatch.setattr(c,'services',lambda:fake)\n monkeypatch.setattr(c.io,'read',lambda *a:b'{}');monkeypatch.setattr(c.io,'digest',lambda *a:'synthetic');monkeypatch.setattr(c.sources,'check_sources',lambda *a:None)")
s=s.replace("c.run(output,result_workspace=tmp_path/'results',**args)","c.run(output,result_workspace=tmp_path/'results',source_manifest=tmp_path/'manifest',source_manifest_sha256='synthetic',**args)")
s+='\n'+(original/'test_original_client_collect.py').read_text().split('def test_external_pin_stays_bound_through_body',1)[1].join(['def test_external_pin_stays_bound_through_body','']) if False else ''
# Append exact independently tested continuity regression, without role-specific names.
extra=(original/'test_original_client_collect.py').read_text();extra=extra[extra.index('def test_external_pin_stays_bound_through_body'):];s+='\n\n'+extra;p.write_text(s)
p=root/'client_source_guard.py';s=p.read_text();tree=ast.parse(s)
newnames=['baseline/key_gate.py','baseline/input_gate.py','baseline/transport.py','test_prerequisite_readers.py','test_client_transport_io.py','client_bootstrap.py','test_client_bootstrap.py','READERS.md']
for node in reversed(tree.body):
 if isinstance(node,ast.Assign)and isinstance(node.targets[0],ast.Name)and node.targets[0].id in ('NEW','OVERLAYS'):
  name=node.targets[0].id;value=ast.literal_eval(node.value)
  if name=='NEW':value=(*value,*newnames)
  else:
   for n in ('key_gate.py','input_gate.py','runtime/stage/transport.py'):value[n]=sha((root/n).read_bytes())
  lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=[name+'='+repr(value)+'\n'];s=''.join(lines)
p.write_text(s)
p=root/'client_bootstrap.py';s=p.read_text();tree=ast.parse(s);node=next(n for n in tree.body if isinstance(n,ast.Assign)and isinstance(n.targets[0],ast.Name)and n.targets[0].id=='EXPECTED_NAMES');lines=s.splitlines(keepends=True);lines[node.lineno-1:node.end_lineno]=['EXPECTED_NAMES='+repr(tuple(sorted(set(expected)|set(newnames))))+'\n'];p.write_text(''.join(lines))
p=root/'client_preflight.py';s=p.read_text().replace("'test_client_descriptor_io.py')","'test_client_descriptor_io.py','test_prerequisite_readers.py','test_client_transport_io.py','test_client_bootstrap.py')");p.write_text(s)
p=root/'test_client_source_guard.py';s=p.read_text().replace('five_overlays','eight_overlays').replace("len(r['overlays'])==5","len(r['overlays'])==8");p.write_text(s)
p=root/'READERS.md';s=p.read_text().replace('Selected original client reader map','Selected combined-response client reader map').replace('unchanged `file_stages.client_decrypt_decode`','frozen117 optional-receipt `file_stages.client_decrypt_decode`').replace('original217-stage4','combined286-stage4').replace('`record_completed` and `retire_completed` remain exact','`record_completed`, `completion_path` and `retire_completed` remain exact from frozen117').replace('default receipt writes','separate owned receipt writes').replace('exactly five runtime overlays plus two top-level consumer gates, with the other82 original89 sources exact','exactly six runtime overlays plus two top-level consumer gates against original89;81 original89 files remain exact. Relative to the frozen117 parent, only the three additional readers and explicit bootstrap/collector/test/provenance files change').replace('no combined286 gate or optional receipt API','exact unchanged combined286 gate and optional receipt API from117').replace('Original output','Separate output');p.write_text(s)
p=root/'README_CLIENT.md';s=p.read_text();s+='''\n## New separately frozen successor of117

This fresh lab preserves the exact117 parent archive and ports the reviewed original122 active-reader and actual CLI bootstrap fixes. Additional reader changes are only top-level key_gate/input_gate and runtime transport; all combined286 metadata gate, separate response ownership, optional completion_directory/default semantics, arithmetic and original P2 gates remain unchanged. Both actual CLIs require externally pinned source-manifest bytes before local imports and isolated no-bytecode startup (-I -B). Fresh trees reject undeclared importable source/package/bytecode/native shadows and preloaded local aliases. Collector source identity remains bound through archive/execution copy. Actual client still requires independently completed286 evidence and a distinct launch grant.

The explicit closure contains125 sources. Against original89 there are8 permitted overlays and81 unchanged files; all new changes relative to historical117 are separately reviewable. The selected reader map is READERS.md. Historical117 is not relabeled as launch-ready. No private/retained payload reads occur in preparation, and no original request/completion ledger is written by this component.\n''';p.write_text(s)
print(root,'sources',len(set(expected)|set(newnames)))
