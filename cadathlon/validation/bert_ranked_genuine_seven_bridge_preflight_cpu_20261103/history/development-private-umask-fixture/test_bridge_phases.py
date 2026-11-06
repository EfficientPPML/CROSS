"""Actual full-count original pre/post/handler loops; explicitly synthetic backend."""
from pathlib import Path
from types import SimpleNamespace as NS
from contextlib import nullcontext
from dataclasses import replace
import json,sys
import pytest
import bridge_common as common,bridge_phases as phases,bridge_protocol as protocol
import ranked_worker as parent,ranked_execution as execution,ranked_case_gate as cases
import ranked_private_reference,ranked_readback
import test_ranked_handlers as h
HERE=Path(__file__).resolve().parent

@pytest.mark.parametrize('size',(1,2,3))
def test_five_split_online_phases_fullN_literal_original_bodies(tmp_path,monkeypatch,size):
    tmp_path.chmod(0o700);paths,state,refs=h.flow(tmp_path,monkeypatch,size=size)
    paths.update(test_input=tmp_path/'test_input',predictions=tmp_path/'predictions')
    tokens=[json.loads(line) for line in paths['preprocessed_input'].read_text().splitlines()];token_bytes=paths['preprocessed_input'].read_bytes()
    raw=''.join(json.dumps(dict(target_idx=r['target_idx'],sentence1='synthetic',sentence2=str(i)))+'\n' for i,r in enumerate(tokens)).encode()
    paths['test_input'].write_bytes(raw);paths['test_input'].chmod(0o600)
    h.stages.client_key_generation(paths);h.stages.server_preprocess_model({k:v for k,v in paths.items() if k in execution.SERVER_FIELDS});paths['preprocessed_input'].unlink()
    calls=[]
    class Tokenizer:
        @classmethod
        def from_pretrained(cls,*a,**kw):calls.append('load');return cls()
        def __call__(self,a,b,**kw):
            calls.append('tokenize');r=tokens[int(b)];return {k:r[k] for k in ('input_ids','token_type_ids','attention_mask')}
    monkeypatch.setitem(sys.modules,'transformers',NS(AutoTokenizer=Tokenizer))
    authorized=cases.authenticate(HERE/'public_evidence/fixture',cases.FIXTURE_REPORT_SHA,HERE/'public_evidence/reference',cases.REFERENCE_REPORT_SHA)
    old=authorized.case(size,42);files=dict(old.input_files);folder=execution.NAMES[size]
    files[f'datasets/{folder}/intermediate/test_input.txt']=common.digest(raw);files[f'io/{folder}/intermediate/client_preprocessed_input']=common.digest(token_bytes)
    case=replace(old,input_sha256=common.digest(token_bytes),input_files=tuple(files.items()),official_correct=old.count)
    monkeypatch.setattr(parent,'guard',lambda *a:nullcontext());monkeypatch.setattr(ranked_private_reference,'load',lambda *a,**kw:NS(records=lambda:refs))
    partial=[None];monkeypatch.setattr(protocol,'authenticate_partial',lambda *a:partial[0])
    previous=None;reports=[];source_hashes=common.hashes(common.sources())
    for phase in common.NAMES[2:]:
        stage=protocol.STAGES[phase];report={'source_hashes_start':source_hashes,'bindings':{}}
        env=dict(phase=phase,request={'stage':stage},partial=None if phase not in protocol.REQUIRES else {'report_sha256':'f'*64})
        output=phases.execute_phase(env,authorized,case,paths,{'context_id':'a'*64},{'model_id':'b'*64},previous,report,[])
        assert report['invoked_entrypoints']==[phase] and report['actual_stage_completed']
        if phase in protocol.PARTIAL:partial[0]={'readback':output}
        if phase in ('client_encode_encrypt_input','server_encrypted_compute'):previous={'stage':stage,'readback':output}
        if phase=='client_decrypt_decode':assert not output['metrics']['final_ranked_acceptance'] and not paths['predictions'].exists()
        reports.append(report)
    assert calls.count('load')==1 and calls.count('tokenize')==case.count
    assert state.encrypts==8*case.count and state.computes==state.decrypts==case.count
    assert reports[-1]['readback']['metrics']['final_ranked_acceptance']
    assert paths['preprocessed_input'].read_bytes()==token_bytes
    assert (paths['client_private']/'secret.npy').exists()
