"""Preserve the failed261 proof; author only the semantic phase fixture fix."""
import ast
from collections import Counter
import difflib
import gzip
import hashlib
import io
import json
from pathlib import Path
import stat
import tarfile
import xml.etree.ElementTree as ET

OLD=Path('/tmp/qwen-toy5-checkpoint-tests-20261104')
NEW=Path('/tmp/qwen-toy5-checkpoint-tests-v2-20261104')
OLDMAP=Path('/tmp/qwen-toy5-checkpoint-test-sources-v1-20261104.json')
OUT=Path('/tmp/qwen-toy5-checkpoint-test-source-handoff-v2-20261104')
MAP=Path('/tmp/qwen-toy5-checkpoint-test-sources-v2-20261104.json')
PROOF=Path('/tmp/qwen-toy5-checkpoint-focused-proof-20261104/preparation-v1')

def sha(raw):return hashlib.sha256(raw).hexdigest()
def read(path):
    before=path.lstat();assert stat.S_ISREG(before.st_mode)and not path.is_symlink()
    raw=path.read_bytes();after=path.lstat()
    fields=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
    assert fields(before)==fields(after)and len(raw)==after.st_size
    return raw
def canonical(v):return(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def emit(name,value):
    raw=value if isinstance(value,bytes)else value.encode()if isinstance(value,str)else canonical(value)
    with(OUT/name).open('xb')as f:f.write(raw)

oldraw=read(OLDMAP);oldmap=json.loads(oldraw)
assert sha(oldraw)=='91a4bc8c1715f8900edb49e97cc9c2be696b30a5037b5efd4a136ac088664339'and len(oldmap)==261
NEW.mkdir()
for name,pin in oldmap.items():
    raw=read(OLD/name);assert sha(raw)==pin
    target=NEW/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
name='q5_test_fixtures.py';before=read(OLD/name).decode();after=before
anchor="    captures=[];original_record=arithmetic.record\n"
replacement=("    # Exact original _qwen phase labels, independent of expected trace indices.\n"
             "    # BoundedArithmetic also records refresh-start/complete phases; those\n"
             "    # are not the diagnostic layer ends and must not be exported twice.\n"
             "    layer_phases={f'packed Qwen layer {layer}: {phase} ready'\n"
             "                  for layer in range(config['num_hidden_layers'])for phase in ('attention','MLP')}\n"
             "    captures=[];original_record=arithmetic.record\n")
assert after.count(anchor)==1;after=after.replace(anchor,replacement)
anchor="        if kind in ('input','output','phase','refresh_before','refresh_after') or (kind=='normalize_after' and str(label)=='6'):\n"
replacement="        if kind in ('input','output','refresh_before','refresh_after') or (kind=='phase' and label in layer_phases) or (kind=='normalize_after' and str(label)=='6'):\n"
assert after.count(anchor)==1;after=after.replace(anchor,replacement)
(NEW/name).write_text(after)
ast.parse(after)
def expected_indices(text):
    return ast.dump(next(n for n in ast.parse(text).body if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='EXPECTED_SELECTED_TRACE_INDICES'for t in n.targets)),include_attributes=False)
assert expected_indices(before)==expected_indices(after)
assert 'EXPECTED_SELECTED_TRACE_INDICES'not in ast.unparse(next(n for n in ast.parse(after).body if isinstance(n,ast.FunctionDef)and n.name=='fixed_math'))
newmap={n:sha(read(NEW/n))for n in sorted(oldmap)}
assert [n for n in newmap if newmap[n]!=oldmap[n]]==[name]
assert all(newmap[n]==oldmap[n]for n in oldmap if n.startswith('runtime/'))
mapraw=canonical(newmap);MAP.write_bytes(mapraw);emit('sources.json',mapraw);emit('parent261-sources.json',oldraw)
emit('before_q5_test_fixtures.py',before)
emit('semantic_phase_filter.patch',''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='failed261/q5_test_fixtures.py',tofile='successor261/q5_test_fixtures.py')))

# Public failed-proof readback only, no fixture evaluation or candidate import.
report=json.loads(read(PROOF/'report.json'));xml=ET.fromstring(read(PROOF/'tests.xml'))
assert report['passed']is False and report['tests']==dict(tests=48,failures=6,errors=0,skipped=0)
failed=[c.attrib['name']for c in xml.iter('testcase')if c.find('failure')is not None]
assert len(list(xml.iter('testcase')))==48 and len(failed)==6
trace=json.loads(read(OLD/'runtime/public/replay-report.json'))['diagnostic']['trace']
phases=Counter(r['label']for r in trace if r['kind']=='phase')
assert phases['bounded homomorphic refresh starting']==phases['bounded homomorphic refresh complete']==19
assert sum(v for k,v in phases.items()if k.startswith('packed Qwen layer '))==6
correction=dict(schema='q5-focused-fixture-phase-correction-v1',scope='Test fixture selection only; original runtime250 and mathematical/decoder/gate definitions unchanged.',predecessor_source_manifest_sha256=sha(oldraw),successor_source_manifest_sha256=sha(mapraw),changed_files=[name],unchanged_files=260,runtime250_unchanged=True,selected48_nodes_unchanged=True,expected47_literal_indices_unchanged=True,selector_uses_expected_numeric_indices=False,
    source_cause='The predecessor captured every phase. Original BoundedArithmetic emits two bookkeeping phase records per19 refreshes; _qwen emits six distinct layer-end phase labels. The first failure observed85 selected records rather than47.',semantic_selection='Only phase labels exactly generated from original _qwen layer count and attention/MLP suffixes; input/output/refresh-before/refresh-after/final RMS selection unchanged.',
    original_phase_source_refs={n:oldmap[n]for n in ['runtime/reference/frozen/cadathlon/common/encrypted_bounded_arithmetic.py','runtime/reference/frozen/cadathlon/common/encrypted_packed_transformer.py']},
    public_nominal_trace_phase_counts=dict(phases),failed_proof=dict(path=str(PROOF),report_sha256=sha(read(PROOF/'report.json')),xml_sha256=sha(read(PROOF/'tests.xml')),log_sha256=sha(read(PROOF/'tests.log')),tests=48,passed=42,failed=6,failed_names=failed,owner='root',session_id=35774,terminal_tool_chunk='07788f',exit_code=1,reaped='parent-reported; root terminal receipt remains authoritative'),
    predecessor_proof_modified=False,successor_candidate_imports=False,successor_collection=False,successor_execution=False,actual_execution_authorized=False,first_encrypted_divergence_claim=False)
emit('correction.json',correction)
with(OUT/'sources.tar.gz').open('xb')as f:
    with gzip.GzipFile(filename='',fileobj=f,mode='wb',mtime=0)as gz:
        with tarfile.open(fileobj=gz,mode='w')as tar:
            for n in sorted(newmap):
                data=read(NEW/n);info=tarfile.TarInfo(n);info.mode=0o644;info.size=len(data);tar.addfile(info,io.BytesIO(data))
with tarfile.open(OUT/'sources.tar.gz')as tar:assert {m.name:sha(tar.extractfile(m).read())for m in tar}==newmap
for n,d in oldmap.items():assert sha(read(OLD/n))==d
selection=json.loads(read(NEW/'selected_tests.json'))
assert len(selection['ordered_nodes'])==48 and read(NEW/'selected_tests.json')==read(OLD/'selected_tests.json')
handoff=dict(schema='q5-focused-fixture-v2-held-handoff-v1',source_root=str(NEW),source_map=str(MAP),source_count=261,source_manifest_sha256=sha(mapraw),source_archive_sha256=sha(read(OUT/'sources.tar.gz')),parent_source_manifest_sha256=sha(oldraw),changed_files={name:dict(before=sha(before.encode()),after=sha(after.encode()))},unchanged_parent_files=260,runtime250_unchanged=True,static_cases=48,selection_sha256=sha(read(NEW/'selected_tests.json')),correction_sha256=sha(read(OUT/'correction.json')),patch_sha256=sha(read(OUT/'semantic_phase_filter.patch')),candidate_imports=False,collection=False,execution=False,ready_for_source_review_only=True,actual_grant=False)
emit('handoff.json',handoff)
print(json.dumps(handoff,sort_keys=True))
