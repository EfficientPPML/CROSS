from pathlib import Path
import hashlib,io,json,os,tarfile
import pytest
import candidate_gate as gate
P=Path(__file__).resolve().parent
REQ=json.loads((P/'collector_requirement.json').read_text())

@pytest.fixture
def copied(tmp_path):
    names={'collector_requirement.json','candidate-v2/manifest.json'}
    for group in ('selection_artifact_sha256','selection_input_sha256','calibration_input_sha256','runtime_dependency_sha256'):names.update(REQ[group])
    for name in names:
        target=tmp_path/name;target.parent.mkdir(parents=True,exist_ok=True);os.link(P/name,target)
    return tmp_path

def replace(path,data):
    path.unlink();path.write_bytes(data)

def rebind_requirement(root,monkeypatch,update):
    req=json.loads((root/'collector_requirement.json').read_text());update(req)
    replace(root/'collector_requirement.json',(json.dumps(req)+'\n').encode())
    monkeypatch.setattr(gate,'EXPECTED_REQUIREMENT',gate.sha(root/'collector_requirement.json'))


def test_success_verifies_twenty_selection_inputs_and_separate_original_contract():
    result=gate.verify(P)
    assert result['passed'] and result['selection_archive_members']==20
    assert len(result['verified']['calibration_input_sha256'])==5
    assert result['calibration_protocol_distinct_from_selection_protocol']
    assert REQ['calibration_input_sha256']['calibration/protocol.json']!=REQ['selection_input_sha256']['protocol.json']

@pytest.mark.parametrize('name',list(REQ['selection_input_sha256'])+list(REQ['calibration_input_sha256'])+['native_capture.py','reference.py'])
def test_every_selection_calibration_and_runtime_dependency_rejects_changed_bytes(copied,name):
    path=copied/name;replace(path,path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='Frozen .* mismatch'):gate.verify(copied)


def test_mutually_rehashed_corpus_tokens_and_pins_cannot_replace_original(copied):
    for name in ('calibration/corpus.json','calibration/tokenized_calibration_inputs.json','calibration/model_pins.json'):
        path=copied/name;replace(path,path.read_bytes()+b'\n')
    with pytest.raises(ValueError,match='calibration_input_sha256'):gate.verify(copied)


def test_selection_protocol_cannot_be_used_as_calibration_protocol(copied):
    replace(copied/'calibration/protocol.json',(copied/'protocol.json').read_bytes())
    with pytest.raises(ValueError,match='calibration_input_sha256'):gate.verify(copied)


@pytest.mark.parametrize('field',['passed','complete','all28_supported','unchanged_norms_verified','unchanged_softmax_verified','source_files_unchanged'])
def test_failed_selection_is_rejected_even_if_report_digest_is_explicitly_rebound(copied,monkeypatch,field):
    path=copied/'candidate-v2/report.json';report=json.loads(path.read_text());report[field]=False;replace(path,json.dumps(report).encode())
    rebind_requirement(copied,monkeypatch,lambda req:req['selection_artifact_sha256'].update({'candidate-v2/report.json':gate.sha(path)}))
    with pytest.raises(ValueError,match='incomplete'):gate.verify(copied)


def test_manifest_substitution_cannot_self_declare_new_input_hashes(copied):
    path=copied/'candidate-v2/manifest.json';manifest=json.loads(path.read_text());manifest['source_input_sha256']['arithmetic.py']='0'*64;replace(path,json.dumps(manifest).encode())
    with pytest.raises(ValueError,match='manifest mismatch'):gate.verify(copied)


def test_requirement_cannot_self_declare_a_new_runtime(copied):
    path=copied/'collector_requirement.json';req=json.loads(path.read_text());req['runtime_dependency_sha256']['native_capture.py']='0'*64;replace(path,json.dumps(req).encode())
    with pytest.raises(ValueError,match='requirement mismatch'):gate.verify(copied)


@pytest.mark.parametrize('kind',['duplicate','missing','extra','nonregular','changed'])
def test_source_archive_members_are_exact_and_unique(tmp_path,kind):
    p=tmp_path/'source.tar.gz';items=[('a',b'a'),('b',b'b')]
    if kind=='duplicate':items=[('a',b'a'),('a',b'a'),('b',b'b')]
    if kind=='missing':items=items[:1]
    if kind=='extra':items.append(('c',b'c'))
    if kind=='changed':items[0]=('a',b'wrong')
    with tarfile.open(p,'w:gz') as archive:
        for name,data in items:
            member=tarfile.TarInfo(name);member.size=len(data)
            if kind=='nonregular' and name=='a':member.type=tarfile.SYMTYPE;member.linkname='b';member.size=0
            archive.addfile(member,io.BytesIO(data))
    with pytest.raises(ValueError):gate.verify_archive(p,{n:hashlib.sha256(b).hexdigest() for n,b in [('a',b'a'),('b',b'b')]})
