"""Synthetic public reports only; never open real private fixtures/references."""
from dataclasses import FrozenInstanceError
from pathlib import Path
import copy
import io
import json
import os
import tarfile
from types import SimpleNamespace
import pytest
import ranked_case_gate as gate


def digest(raw):return gate.key_gate.digest(raw)
def write(path,value):
    raw=gate.key_gate.canonical(value);path.write_bytes(raw);return digest(raw)


def archive(path,bodies,duplicate=None,kind=None):
    with tarfile.open(path,'w:gz') as tar:
        for name,body in bodies.items():
            item=tarfile.TarInfo(name);item.size=len(body)
            if kind is not None and name==next(iter(bodies)):item.type=kind;item.linkname='elsewhere'
            tar.addfile(item,io.BytesIO(body) if item.isfile() else None)
        if duplicate is not None:
            body=bodies[duplicate];item=tarfile.TarInfo(duplicate);item.size=len(body);tar.addfile(item,io.BytesIO(body))
    return digest(path.read_bytes())


@pytest.fixture
def public(tmp_path,monkeypatch):
    fixture=tmp_path/'fixture';reference=tmp_path/'reference';fixture.mkdir();reference.mkdir()
    pins=tmp_path/'pins';pins.mkdir();assets={'/nonexistent/public/model':digest(b'asset')}
    monkeypatch.setattr(gate.input_gate,'ASSETS',assets)
    monkeypatch.setattr(gate,'HERE',pins);monkeypatch.setattr(gate.ranked_input_gate,'HERE',pins)
    upstream={'harness.py':{'sha256':digest(b'upstream')}}
    fb={f'fixture{i}.py':str(i).encode() for i in range(26)};fb['upstream_origin.json']=gate.key_gate.canonical(upstream)
    rb={f'reference{i}.py':str(i).encode() for i in range(41)}
    fp={n:digest(b) for n,b in fb.items()};rp={n:digest(b) for n,b in rb.items()}
    write(pins/'producer_source_manifest.json',fp);write(pins/'reference_source_manifest.json',rp)
    f={'schema':'bert-ranked-tokenizer-fixtures-v1','passed':True,'private_workspace_retained':True,
        'sources_unchanged':True,'assets_unchanged':True,'upstream_originals_unchanged':True,
        'source_archive_verified':True,'cleanup_completed':True,'terminal_scope_completed':True,
        'model_construction_and_forward_guards':True,'model_forward':False,'encrypted_execution':False,
        'ranked_cases_expected':9,'selected_records_expected':2124,'source_hashes_before':fp,'source_hashes_after':fp,
        'source_archive_sha256':archive(fixture/'sources.tar.gz',fb),'asset_file_sha256_before':assets,
        'asset_file_sha256_after':assets,'upstream_original_sha256_before':{'harness.py':digest(b'upstream')},
        'upstream_original_sha256_after':{'harness.py':digest(b'upstream')},
        'private_workspace':'/does-not-exist-fixture','private_workspace_identity':{'device':1,'inode':2},'cases':[]}
    rows=[]
    for size,count,seed in gate.ranked_input_gate.SCHEDULE:
        files={n:digest(f'{size}:{seed}:{n}'.encode()) for n in gate.ranked_input_gate.expected_files(size)}
        name=f'io/{gate.ranked_input_gate.NAMES[size]}/intermediate/client_preprocessed_input'
        order=digest(f'order:{size}:{seed}'.encode())
        f['cases'].append(dict(size=size,count=count,base_seed=seed,record_count=count,fixed_token_length=128,
            generator_seed=seed,selection_byte_parity=True,tokenization_byte_parity=True,labels_private=True,
            all_private_directories_0700=True,all_private_files_0600=True,input_file_sha256=files,
            private_target_order_sha256=order,private_row_order_sha256=digest(f'rows:{size}:{seed}'.encode()),
            preprocessed_input_path=f'/does-not-exist-fixture/size{size}-seed{seed}/{name}',valid_token_count_histogram={'32':count}))
        rows.append(dict(size=size,seed=seed,count=count,official_reference_complete=True,
            input_sha256=files[name],private_target_order_sha256=order,official_correct=count-1,
            official_accuracy=(count-1)/count,approximation_complete=True,prediction_agreement=True,
            approximate_numeric_gate_passed=True))
    fsha=write(fixture/'report.json',f);monkeypatch.setattr(gate,'FIXTURE_REPORT_SHA',fsha)
    private={n:dict(bytes=100,sha256=digest(n.encode())) for n in
        {'master_references.jsonl','private_manifest.json'}|{f'size{s}-seed{seed}-references.jsonl' for s,c,seed in gate.ranked_input_gate.SCHEDULE}}
    for seed in (42,43,44):private[f'size3-seed{seed}-references.jsonl']=dict(private['master_references.jsonl'])
    r={n:True for n in gate.REFERENCE_FLAGS}
    r.update(execute=True,actual_native_started=True,official_evaluation_complete=True,
        all_nine_private_inputs_authenticated=True,cleanup_errors=[],raw_arrays_archived=False,
        reference_acceptance_requires=list(gate.REFERENCE_FLAGS),fixture_report_sha256=fsha,
        candidate_sha256=gate.CANDIDATE_SHA,unique_records=408,selected_records=2124,
        native_policy=dict(official_dtype='float32',official_attention='sdpa',official_attention_classes=['BertSdpaSelfAttention'],
            native_forward_original_tokens=128,encoder_layers=12),asset_hashes_before=assets,asset_hashes_after=assets,
        source_hashes_before=rp,source_hashes_after=rp,source_archive_sha256=archive(reference/'sources.tar.gz',rb),
        private_workspace='/does-not-exist-reference',private_workspace_identity={'device':1,'inode':3},
        private_reference_files=private,cases=rows,passed=True,approximation_validation_passed=True,
        exact_reference_parity_passed=True)
    rsha=write(reference/'report.json',r);monkeypatch.setattr(gate,'REFERENCE_REPORT_SHA',rsha)
    def repin():
        # These synthetic publications are explicitly re-trusted only by tests;
        # production constants remain fixed to the reviewed real artifacts.
        value=write(reference/'report.json',r);monkeypatch.setattr(gate,'REFERENCE_REPORT_SHA',value);return value
    return SimpleNamespace(fixture=fixture,reference=reference,f=f,r=r,fsha=fsha,rsha=rsha,repin=repin,rb=rb,pins=pins)


def authenticate(p):return gate.authenticate(p.fixture,p.fsha,p.reference,gate.REFERENCE_REPORT_SHA)


def test_complete_all_nine_public_chain_and_no_private_reads(public,monkeypatch):
    opened=[];original=os.open
    def safe(path,*args,**kwargs):
        opened.append(str(path));assert not str(path).startswith('/does-not-exist');return original(path,*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(os,'open',safe);result=authenticate(public)
    assert len(result.cases)==9 and len(result.reference_files)==11
    assert [(r.size,r.count,r.seed) for r in result.cases]==list(gate.ranked_input_gate.SCHEDULE)
    assert result.case(3,44).reference_file.sha256==result.case(3,42).reference_file.sha256
    assert result.private_manifest.name=='private_manifest.json'
    assert all(Path(p).name in ('report.json','sources.tar.gz','reference_source_manifest.json') for p in opened)
    assert result.case(1,42).official_correct==99
    with pytest.raises(FrozenInstanceError):result.case(1,42).count=2
    with pytest.raises(FrozenInstanceError):result.reference_workspace.inode=4
    with pytest.raises(TypeError):result.assets[0]=('changed','value')


@pytest.mark.parametrize('size,seed',[(True,42),(1.,42),(1,42.),(1,True),(0,42),(1,45)])
def test_strict_case_lookup(public,size,seed):
    with pytest.raises(ValueError):authenticate(public).case(size,seed)


def test_wrong_external_pin_fails_before_any_open(public,monkeypatch):
    monkeypatch.setattr(gate.key_gate,'read',lambda *a,**k:pytest.fail('evidence read before external pins'))
    with pytest.raises(ValueError):gate.authenticate(public.fixture,'a'*64,public.reference,public.rsha)


def test_report_substitution_without_external_repin_rejected(public):
    public.r['official_reference_complete']=False;write(public.reference/'report.json',public.r)
    with pytest.raises(ValueError,match='external reference'):authenticate(public)


@pytest.mark.parametrize('flag',gate.REFERENCE_FLAGS)
def test_every_mandatory_reference_flag(public,flag):
    public.r[flag]=False;public.repin()
    with pytest.raises(ValueError):authenticate(public)


@pytest.mark.parametrize('field,value',[('cleanup_errors',['failed']),('error',{'type':'error'}),
    ('unique_records',True),('selected_records',2123),('candidate_sha256','a'*64),
    ('fixture_report_sha256','a'*64),('raw_arrays_archived',True),('official_evaluation_complete',False),
    ('execute',False),('private_workspace','relative'),('private_workspace','/does-not-exist-fixture/child')])
def test_reference_publication_identity_and_phase_failures(public,field,value):
    public.r[field]=value;public.repin()
    with pytest.raises(ValueError):authenticate(public)


def test_official_availability_not_approximation_acceptance(public):
    public.r.update(passed=False,approximation_validation_passed=False,exact_reference_parity_passed=False)
    for r in public.r['cases']:r.update(approximation_complete=False,prediction_agreement=False,approximate_numeric_gate_passed=False)
    public.repin();assert len(authenticate(public).cases)==9


@pytest.mark.parametrize('field,value',[('official_dtype','float64'),('official_attention','eager'),
    ('official_attention_classes',['BertSelfAttention']),('encoder_layers',12.),('native_forward_original_tokens',True)])
def test_official_native_policy_is_original_fp32(public,field,value):
    public.r['native_policy'][field]=value;public.repin()
    with pytest.raises(ValueError):authenticate(public)


@pytest.mark.parametrize('mutation',['swap','last_input','last_order','last_correct_bool','last_count','last_accuracy','missing','extra'])
def test_all_rows_order_count_input_binding(public,mutation):
    rows=public.r['cases']
    if mutation=='swap':rows[0],rows[1]=rows[1],rows[0]
    elif mutation=='last_input':rows[-1]['input_sha256']='b'*64
    elif mutation=='last_order':rows[-1]['private_target_order_sha256']='b'*64
    elif mutation=='last_correct_bool':rows[-1]['official_correct']=True
    elif mutation=='last_count':rows[-1]['count']=407
    elif mutation=='last_accuracy':rows[-1]['official_accuracy']=.5
    elif mutation=='missing':rows.pop()
    else:rows.append(dict(rows[-1]))
    public.repin()
    with pytest.raises(ValueError):authenticate(public)


@pytest.mark.parametrize('mutation',['missing','extra','large','negative','bool','master','digest'])
def test_all11_private_descriptors_are_bound_without_open(public,mutation):
    files=public.r['private_reference_files'];row=files['size3-seed44-references.jsonl']
    if mutation=='missing':files.pop('private_manifest.json')
    elif mutation=='extra':files['unexpected.json']=dict(row)
    elif mutation=='large':row['bytes']=(16<<20)+1
    elif mutation=='negative':row['bytes']=-1
    elif mutation=='bool':row['bytes']=True
    elif mutation=='master':row['sha256']='b'*64
    else:row['sha256']='not a digest'
    public.repin()
    with pytest.raises(ValueError):authenticate(public)


@pytest.mark.parametrize('mutation',['duplicate','nonregular','changed_bytes','source_after','assets_after'])
def test_exact_reference_archive_and_asset_identity(public,mutation):
    if mutation=='duplicate':public.r['source_archive_sha256']=archive(public.reference/'sources.tar.gz',public.rb,duplicate=next(iter(public.rb)))
    elif mutation=='nonregular':public.r['source_archive_sha256']=archive(public.reference/'sources.tar.gz',public.rb,kind=tarfile.SYMTYPE)
    elif mutation=='changed_bytes':
        b=dict(public.rb);b[next(iter(b))]=b'changed';public.r['source_archive_sha256']=archive(public.reference/'sources.tar.gz',b)
    elif mutation=='source_after':public.r['source_hashes_after']={}
    else:public.r['asset_hashes_after']={}
    public.repin()
    with pytest.raises(ValueError):authenticate(public)


def test_fixture_must_finish_before_reference_read(public,monkeypatch):
    public.f['terminal_scope_completed']=False
    public.fsha=write(public.fixture/'report.json',public.f);monkeypatch.setattr(gate,'FIXTURE_REPORT_SHA',public.fsha)
    original=gate.key_gate.read
    def read(path,*a,**k):
        assert public.reference not in Path(path).parents;return original(path,*a,**k)
    monkeypatch.setattr(gate.key_gate,'read',read)
    with pytest.raises(ValueError):authenticate(public)
