"""Original serial role bodies; opaque readers/codecs/graph are recorded."""
from contextlib import contextmanager,ExitStack
from types import SimpleNamespace
import case_fixtures as f

def _client(mode='clean'):
    import numpy as np
    import q5_checkpoint_roles as m,q5_checkpoint_oracle as oracle
    s=f.setup_bundle();public_original=f.server_bundle(s);manifest=public_original.check();rows=manifest['rows'];reads=[];decodes=[];owners=[];snapshots=[];readers=[];notes=[];private=[];replacements=[]
    class ApproximationErrorTooHigh(ValueError):pass
    error=(ApproximationErrorTooHigh if mode=='decoder_failure'else ValueError)('SYNTHETIC '+mode);ctx=object();scratch=np.zeros(65536,dtype=np.float64);pattern=np.arange(65536,dtype=np.float64)/1024.;handle=SimpleNamespace(ctx=ctx)
    class Public:
        def check(self):f.event('public_check');return public_original.check()
    class Reader:
        def __init__(self,ignored,raw):self.raw=raw;self.closed=False;readers.append(self);f.event('reader_open',ordinal=len(readers)-1)
        def read(self,artifact,*,retain_payload=False):
            i=artifact['metadata']['index'];reads.append((i,retain_payload));f.event('reader_payload'if retain_payload else'reader_first',index=i)
            if mode=='late_artifact'and i==46 and not retain_payload:raise error
            return dict(index=i)if retain_payload else dict(artifact)
        def check(self):assert not self.closed
        def close(self,primary=None):assert not self.closed;self.closed=True;f.event('reader_close',ordinal=readers.index(self))
    class Owner:
        def __init__(self,index,payload,row):self.index=index;self.payload=payload;self.row=row;self.value=None;self.closes=0;owners.append(self)
        def load(self):
            f.event('ct_load',index=self.index)
            if mode=='load_failure':raise error
            self.value=object();return self.value
        def check(self):return self.value
        def close(self,primary=None):self.closes+=1;f.event('ct_close',index=self.index);self.value=None
    def factory(payload,row):f.event('owner_factory',index=row['index']);return Owner(row['index'],payload,row)
    def check():assert handle.ctx is ctx
    def decode(original_ctx,value,*,scale,slots_to_decode,validate_approximation):
        i=len(decodes);assert original_ctx is ctx and value is owners[-1].value and scale==rows[i]['scale'] and type(scale)is float
        assert slots_to_decode==65536 and validate_approximation is True;decodes.append((i,id(original_ctx),id(value),scale,slots_to_decode,validate_approximation));f.event('decode',index=i)
        if mode=='decoder_failure'and i==7:raise error
        scratch[:]=pattern+i
        if mode=='nonfinite':scratch[-1]=np.nan
        if mode=='codec_replacement':handle.decrypt_slots=lambda *a,**kw:replacements.append(True)
        return scratch
    handle.check=check;handle.decrypt_slots=decode;handle.new_ciphertext_owner=factory;handle.decoder_source_sha256=m.DEFAULT_DECODER_SHA256
    @contextmanager
    def private_loader():
        private.append('factory');f.event('private_factory');assert reads==[(i,False)for i in range(47)]
        private.append('entered');f.event('private_enter')
        try:yield handle
        except BaseException as e:private.append(e);f.event('private_failed_exit',error=type(e).__name__);raise
        else:private.append('clean');f.event('private_clean_exit')
    def replay(*args):f.event('recorded_replay');return dict(synthetic_only=True)
    def references(*args):f.event('recorded_dense');return (None,)*47,dict(synthetic_only=True)
    def diagnostics(ignored,snapshot,reference,index):
        assert reference is None and snapshot.shape==(65536,)and snapshot.dtype==np.dtype('float64')and not snapshot.flags.writeable
        assert not np.shares_memory(snapshot,scratch)and np.array_equal(snapshot,pattern+index)
        for old_i,old in enumerate(snapshots):assert np.array_equal(old,pattern+old_i)and not np.shares_memory(old,snapshot)
        snapshots.append(snapshot);f.event('diagnostics',index=index);return dict(index=index,synthetic_only=True)
    def gate(*args):f.event('recorded_final_gate');return dict(passed=True,synthetic_only=True)
    def note(i,t):notes.append((i,t));f.event('failure_note',index=i,error=t)
    def run():
        with ExitStack()as stack:
            for obj,name,value in [(m,'ArtifactReader',Reader),(m,'client_replay',replay),(oracle,'references',references),(oracle,'diagnostics',diagnostics),(oracle,'original_final_gate',gate)]:stack.enter_context(f.patch(obj,name,value))
            return m.client_bound(np,Public(),private_loader,object(),object(),object(),f.RUNTIME/'reference',admission_check=lambda:None,failure_note=note)
    return SimpleNamespace(run=run,error=error,reads=reads,decodes=decodes,owners=owners,snapshots=snapshots,readers=readers,notes=notes,private=private,replacements=replacements,scratch=scratch,pattern=pattern,np=np)

def test_all47_recorded_public_reads_precede_private_factory():
    x=_client();result=x.run();assert result['decoded_records']==47 and result['complete']is True
    assert x.reads==[(i,False)for i in range(47)]+[(i,True)for i in range(47)]
    labels=f.labels();first=[i for i,v in enumerate(f.EVENTS)if v['event']=='reader_first'];priv=labels.index('private_factory')
    assert len(first)==47 and max(first)<priv and any(v=='public_check'for v in labels[max(first)+1:priv])
    assert len(x.decodes)==47 and len(x.owners)==47 and all(o.closes==1 for o in x.owners)and all(r.closed for r in x.readers)

def test_last_public_artifact_rejection_keeps_private_unentered():
    x=_client('late_artifact');f.expect(ValueError,x.run,same=x.error)
    assert x.reads==[(i,False)for i in range(47)]and x.private==[]and x.owners==[]and x.decodes==[]and all(r.closed for r in x.readers)

def test_default_decoder_abi_and_full_scratch_snapshots_are_original():
    x=_client();x.run();assert len(x.snapshots)==47
    for i,array in enumerate(x.snapshots):assert x.np.array_equal(array,x.pattern+i)and not array.flags.writeable and not x.np.shares_memory(array,x.scratch)
    assert all(v[4:]==(65536,True)for v in x.decodes)and len({id(v)for v in x.snapshots})==47

def test_decoder_failure_stops_without_relaxed_retry_and_reaches_key_scope():
    x=_client('decoder_failure');f.expect(ValueError,x.run,same=x.error)
    assert len(x.decodes)==8 and len(x.owners)==8 and x.notes==[(7,'ApproximationErrorTooHigh')]
    assert all(o.closes==1 for o in x.owners)and x.private[-1]is x.error
    close=max(i for i,r in enumerate(f.EVENTS)if r['event']=='ct_close');assert close<f.labels().index('private_failed_exit')
    assert'private_clean_exit'not in f.labels()and'recorded_final_gate'not in f.labels()

def test_nonfinite_unused_slot_fails_full_decode_gate():
    x=_client('nonfinite');f.expect(ValueError,x.run,'full finite default decode required')
    assert len(x.decodes)==1 and x.snapshots==[]and x.owners[0].closes==1 and'private_failed_exit'in f.labels()

def test_returned_consumer_load_failure_retires_inside_private_scope():
    x=_client('load_failure');f.expect(ValueError,x.run,same=x.error)
    assert len(x.owners)==1 and x.decodes==[]and x.owners[0].closes==1
    assert f.labels().index('ct_close')<f.labels().index('private_failed_exit')

def test_codec_or_context_replacement_is_rejected_before_next_decode():
    x=_client('codec_replacement');f.expect(ValueError,x.run,'original private context/codec/factory changed')
    assert len(x.decodes)==1 and x.replacements==[]and x.owners[0].closes==1
    assert f.labels().index('ct_close')<f.labels().index('private_failed_exit')

def _server(mode='clean'):
    import q5_checkpoint_roles as m,q5_checkpoint_arithmetic as arithmetic
    s=f.setup_bundle();raw=f.server_bundle(s).manifest_bytes;rows,obs=f.checkpoint_fixture();ctx=object();handle=SimpleNamespace(ctx=ctx,check=lambda:None);owners=[];closes=[];foreign=[];pubs=[];error=ValueError('SYNTHETIC partial input accessor');values=object();weights={'SYNTHETIC':object()}
    @contextmanager
    def loader():
        f.event('server_key_enter')
        try:yield handle
        except BaseException as e:f.event('server_key_failed_exit',error=type(e).__name__);raise
        else:f.event('server_key_clean_exit')
    class Weight:
        def __init__(self):self.weights=None;owners.append(self)
        def load(self):self.weights=weights;f.event('weight_load');return weights
        def check(self):return self.weights
        def close(self,primary=None):closes.append('weight');f.event('weight_close');self.weights=None
    class Input:
        def __init__(self):self.value=None;owners.append(self)
        def load(self):self.value=values;f.event('input_load');return values
        def check(self):return self.value
        def close(self,primary=None):closes.append('input');f.event('input_close');self.value=None
    class PartialInput(Input):
        @property
        def load(self):raise error
    class Publisher:
        def __init__(self):self.indices=[];pubs.append(self);owners.append(self)
        def check(self):pass
        def checkpoint(self,index,value,row):assert value is values and row==rows[index];self.indices.append(index);f.event('checkpoint',index=index)
        def finish(self,result):
            assert result==dict(complete=True,checkpoints=rows,refresh_observations=obs)and self.indices==list(range(47));f.event('publisher_finish')
            if mode=='finish_mutation':self.close=lambda *a,**kw:foreign.append(True)
            return raw
        def close(self,primary=None):closes.append('publisher');f.event('publisher_close')
    def exercise(original_ctx,original_input,original_weights,*,checkpoint,dependencies):
        assert original_ctx is ctx and original_input is values and original_weights is weights;f.event('recorded_exercise')
        for index,row in enumerate(rows):checkpoint(index,values,row)
        return dict(complete=True,checkpoints=rows,refresh_observations=obs)
    def run():
        with f.patch(arithmetic,'exercise',exercise):return m.server_bound(object(),loader,lambda h:Weight(),lambda h:(PartialInput()if mode=='partial_input'else Input()),lambda h:object(),Publisher,admission_check=lambda:None)
    return SimpleNamespace(run=run,closes=closes,pubs=pubs,error=error,foreign=foreign,raw=raw,owners=owners)

def test_server_recorded47_finish_and_all_consumers_retire_before_keys():
    x=_server();assert x.run()==x.raw and x.pubs[0].indices==list(range(47))and x.closes==['publisher','input','weight']
    assert f.labels().index('publisher_finish')<f.labels().index('publisher_close')<f.labels().index('input_close')<f.labels().index('weight_close')<f.labels().index('server_key_clean_exit')

def test_server_finish_callback_mutation_reaches_original_cleanup_and_key_failure():
    x=_server('finish_mutation');f.expect(ValueError,x.run,'original role owner callback changed')
    assert x.closes==['publisher','input','weight']and x.foreign==[]and'server_key_failed_exit'in f.labels()
    assert f.labels().index('weight_close')<f.labels().index('server_key_failed_exit')

def test_server_partial_owner_acquisition_preserves_prior_cleanup():
    x=_server('partial_input');f.expect(ValueError,x.run,same=x.error)
    assert x.closes==['input','weight']and x.pubs==[]and'recorded_exercise'not in f.labels()
    assert f.labels().index('weight_close')<f.labels().index('server_key_failed_exit')
