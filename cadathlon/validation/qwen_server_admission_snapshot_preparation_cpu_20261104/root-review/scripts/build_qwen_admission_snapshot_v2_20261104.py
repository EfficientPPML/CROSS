from pathlib import Path
import ast,hashlib,json,tarfile,difflib
sha=lambda b:hashlib.sha256(b).hexdigest()
parent=Path('/tmp/qwen-server-admission-snapshot-source-20261104');out=Path('/tmp/qwen-server-admission-snapshot-source-v2-20261104');handoff=Path('/tmp/qwen-server-admission-snapshot-handoff-v2-20261104')
old=(parent/'qwen_admission_snapshot.py').read_text();assert sha(old.encode())=='69a9bc77612aa9e21f7afa178a43741b0bda9475de35e5bbc4e750e9fd1bf6d0'
out.mkdir();handoff.mkdir()
for p in sorted(parent.iterdir()):
 assert p.is_file()and not p.is_symlink()
 (out/p.name).write_bytes(p.read_bytes())
new=old.replace('_ISSUED=False\n','_ISSUED=False\n_CAPTURE_ATTEMPT=None\n',1)
new=new.replace("        if status!='active':raise PermissionError('failed or reentrant snapshot requires process exit')\n        _REGISTRY[id(self)]=(self,original,pid,thread,bindings,graph,checker,'checking')", "        if status!='active':\n            if status=='checking':\n                _REGISTRY[id(self)]=(self,original,pid,thread,bindings,graph,checker,'failed')\n            raise PermissionError('failed or reentrant snapshot requires process exit')\n        checking=(self,original,pid,thread,bindings,graph,checker,'checking')\n        _REGISTRY[id(self)]=checking",1)
new=new.replace('            _bindings_unchanged(bindings);_graph_unchanged(graph)\n        except BaseException:', "            _bindings_unchanged(bindings);_graph_unchanged(graph)\n            # A callback may catch the inner rejection; the outer check must\n            # still preserve the failure instead of restoring an active state.\n            if _REGISTRY.get(id(self))is not checking:\n                raise PermissionError('snapshot reentry remains a sticky failure')\n        except BaseException:",1)
new=new.replace('    global _ISSUED\n', '    global _ISSUED,_CAPTURE_ATTEMPT\n',1)
new=new.replace('    _ISSUED=True\n    checker=_callback', "    _ISSUED=True\n    # Retain the original packet before any fallible callback. A failed first\n    # capture leaves its packet and later captured graph in this process only.\n    _CAPTURE_ATTEMPT=(packet,None,None,None)\n    checker=_callback",1)
new=new.replace('    graph=_graph(packet)\n    _FULL_PACKET_CHECK(packet)', '    graph=_graph(packet)\n    _CAPTURE_ATTEMPT=(packet,bindings,graph,checker)\n    _FULL_PACKET_CHECK(packet)',1)
new=new.replace('    guard.check(packet)\n    return guard', '    guard.check(packet)\n    _CAPTURE_ATTEMPT=None  # The successfully checked registry now owns the graph.\n    return guard',1)
assert new!=old;ast.parse(new);(out/'qwen_admission_snapshot.py').write_text(new)
design=out/'DESIGN.md';design.write_text(design.read_text()+'\nThe v2 successor keeps the original first-capture packet before any callback and\nretains the completed graph before full validation; failures leave that attempt\nretained until process exit. A reentrant check marks the registry failed before\nraising, and the outer check verifies its original in-progress registry identity\nbefore restoring active state, even when the callback catches the inner error.\nThese two fixes are source-derived; no runtime failure or successful test is claimed.\n')
history=dict(schema='qwen-admission-snapshot-v2-source-correction',parent_source=str(parent),parent_guard_sha256=sha(old.encode()),successor_guard_sha256=sha(new.encode()),findings=['Pre-registry failed capture did not retain its original graph.','Caught reentry rejection could allow outer check to restore active state.'],runtime_failure_observed=False,imports_tests_or_integration=False,actual_authority=False)
(out/'successor_history.json').write_text(json.dumps(history,sort_keys=True,indent=2)+'\n')
(handoff/'guard.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='v1/qwen_admission_snapshot.py',tofile='v2/qwen_admission_snapshot.py')))
(handoff/'build_source.py').write_bytes(Path(__file__).read_bytes())
mapping={p.name:sha(p.read_bytes())for p in out.iterdir()if p.is_file()};raw=(json.dumps(mapping,sort_keys=True,separators=(',',':'))+'\n').encode();(handoff/'sources.json').write_bytes(raw)
with tarfile.open(handoff/'sources.tar.gz','x:gz')as tf:
 for name in sorted(mapping):tf.add(out/name,arcname=name,recursive=False)
record=dict(schema='qwen-admission-snapshot-v2-source-handoff',source_root=str(out),source_count=len(mapping),source_manifest_sha256=sha(raw),source_archive_sha256=sha((handoff/'sources.tar.gz').read_bytes()),guard_sha256=sha(new.encode()),parent_preserved=True,imports_tests_or_integration=False,actual_authority=False)
(handoff/'handoff.json').write_text(json.dumps(record,sort_keys=True,indent=2)+'\n')
print(json.dumps(record|dict(handoff_sha256=sha((handoff/'handoff.json').read_bytes()))))
