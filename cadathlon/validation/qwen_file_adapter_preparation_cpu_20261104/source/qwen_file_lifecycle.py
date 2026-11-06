"""Qwen response admission and client ordering; actual process entry is disabled.

External grants/outer receipts remain caller-owned. These checks reconstruct
public declarations; they do not independently observe a producer process.
Only owned synthetic files and callback doubles are used in this preparation.
"""
from dataclasses import dataclass
import json,os
import numpy as np
from reused_client_io import HeldDirectory,Finalizer
from reused_rns_payload import _payload
from qwen_public_parameters import _HeldFile,_FdReader
from qwen_file_contract import (canonical,sha,digest,hash_value,same,validate_request,
    validate_response,validate_request_manifest,ciphertext_binding,request,vocabulary)
from qwen_layout import place_embeddings,read_vocabulary

_QUARANTINE=[]

@dataclass(frozen=True)
class ProducerTerminal:
    report:bytes
    outer:bytes
    @classmethod
    def authenticate(cls,public,contract,input_digest,response_bytes,*,report_bytes,report_sha256,outer_bytes,outer_sha256):
        """Pins must come from an independent role grant, not these documents."""
        for raw,pin in ((report_bytes,report_sha256),(outer_bytes,outer_sha256)):
            if type(raw)is not bytes or sha(raw)!=hash_value(pin):raise ValueError('externally pinned producer proof differs')
        c=validate_request(public,contract);hash_value(input_digest)
        expected=dict(schema='qwen-organizer-server-report-v1',
            producer_source_manifest_sha256=c['public']['producer_source_manifest_sha256'],
            consumer_source_manifest_sha256=c['public']['consumer_source_manifest_sha256'],
            context_id=c['public']['context_id'],model_id=c['public']['model_id'],request_contract_sha256=digest(c),
            input_digest=input_digest,response_manifest_sha256=sha(response_bytes),output_count=38,
            body_completed=True,source_unchanged=True,parameters_rechecked=True,cleanup_completed=True,termination_scope_completed=True,passed=True)
        if report_bytes!=canonical(expected):raise ValueError('complete producer report binding differs')
        outer=json.loads(outer_bytes)
        if type(outer)is not dict:raise ValueError('outer receipt dictionary required')
        for field in ('server_pid','server_pgid'):
            if type(outer.get(field))is not int or outer[field]<=0:raise ValueError('observed producer process identities required')
        expected_outer=dict(schema='qwen-organizer-server-outer-v1',report_sha256=report_sha256,
            producer_source_manifest_sha256=expected['producer_source_manifest_sha256'],request_contract_sha256=digest(c),
            server_pid=outer['server_pid'],server_pgid=outer['server_pgid'],exit_code=0,reaped=True,process_group_absent=True)
        if outer_bytes!=canonical(expected_outer):raise ValueError('independent successful outer receipt required')
        return cls(report_bytes,outer_bytes)

@dataclass(frozen=True)
class AdmittedPayload:
    binding:bytes
    residues:bytes
    def array(self):
        b=json.loads(self.binding)
        return np.frombuffer(self.residues,dtype=np.uint32).reshape(tuple(b['shape']))

@dataclass(frozen=True)
class PreparedEmbeddings:
    contract_bytes:bytes
    tiles:tuple
    original_positions:tuple

def prepare_client_embeddings(public,identity,request_id,embeddings,positions):
    """Pure trusted-client placement, never an encoder or execution grant."""
    if type(positions)is not tuple:raise TypeError('exact original position tuple required')
    contract=request(public,identity,request_id,list(positions))
    tiles=place_embeddings(embeddings,positions)
    return PreparedEmbeddings(canonical(contract),tiles,positions)

class RequestFiles:
    """Held freshL49 requests; this does not perform the required49→25 step."""
    def __init__(self,public,contract,manifest_bytes,root,*,request_manifest_sha256):
        self.directory=None;self.files=[];self.records=();self.closed=False;self.failed=None;self.entered=False;self.retired=False
        if type(manifest_bytes)is not bytes or sha(manifest_bytes)!=hash_value(request_manifest_sha256):raise ValueError('externally bound request bytes differ')
        manifest=json.loads(manifest_bytes);validate_request_manifest(public,contract,manifest)
        if manifest_bytes!=canonical(manifest):raise ValueError('canonical request metadata required')
        self.public=public;self.contract=canonical(contract);self.manifest=manifest_bytes;self.input_digest=manifest['input_digest']
        self._fixed=(public,self.contract,manifest_bytes,self.input_digest)
        try:
            self.directory=HeldDirectory(root);count=contract['input']['input_tile_count']
            if set(os.listdir(self.directory.fd))!={f'tile{i:02}.npz'for i in range(count)}|{'request.json'}:raise ValueError('exact variable input tile inventory required')
            metadata=_HeldFile(self.directory,'request.json',1<<20,request_manifest_sha256);self.files.append(metadata)
            if _FdReader(metadata.fd,metadata.size).read()!=manifest_bytes:raise ValueError('held request metadata differs')
            records=[]
            for index,row in enumerate(manifest['artifacts']):
                binding=row['binding'];maximum=int(np.prod(binding['shape']))*4+65536
                held=_HeldFile(self.directory,f'tile{index:02}.npz',maximum,row['payload_sha256']);self.files.append(held)
                raw=_FdReader(held.fd,held.size).read()
                payload,_=_payload(raw,dict(shape=tuple(binding['shape']),moduli=binding['moduli'],layout=binding['degree_layout'],scale=binding['scale']))
                held.check();records.append(AdmittedPayload(canonical(binding),payload.tobytes(order='C')))
            self.records=tuple(records);self._records=self.records;self._files=tuple(self.files);self._directory=self.directory;self.check()
        except BaseException as primary:
            try:self.close(primary=primary)
            except BaseException as secondary:
                if secondary is not primary:primary.add_note('request setup cleanup: '+repr(secondary))
            raise
    def check(self,*,readback=False):
        if self.failed is not None:raise self.failed
        if self.closed:raise ValueError('request backing closed')
        if self.public is not self._fixed[0]or (self.contract,self.manifest,self.input_digest)!=self._fixed[1:]:raise ValueError('request binding state changed')
        self.public.check();self.directory.check()
        if self.records and (self.records is not self._records or self.directory is not self._directory or len(self.files)!=len(self._files)or any(a is not b for a,b in zip(self.files,self._files))):raise ValueError('retained request ownership changed')
        for held in self.files:held.readback()if readback else held.check()
    def consumer_entered(self):
        self.check(readback=True)
        if self.entered or self.retired:raise ValueError('one server consumer lifetime required')
        self.entered=True
    def consumers_retired(self):
        self.check(readback=True)
        if not self.entered or self.retired:raise ValueError('live server consumer lifetime required')
        self.retired=True
    def close(self,primary=None):
        if self.failed is not None:raise primary if primary is not None else self.failed
        if self.closed:return
        if self.entered and not self.retired:
            error=primary if primary is not None else RuntimeError('unretired server requires dedicated-process quarantine')
            self.failed=error;_QUARANTINE.append((self,error));raise error
        final=Finalizer({},primary)
        if self.directory is not None:
            for held in self.files:final.attempt('request final readback',held.readback)
            for held in reversed(self.files):final.attempt('request descriptor retirement',held.close)
            final.attempt('request directory retirement',self.directory.close)
        self.closed=True
        try:final.raise_if_failed()
        except BaseException as error:self.failed=error;_QUARANTINE.append((self,error));raise

class ResponseFiles:
    """Authenticate all38 before any private loader callback is invoked."""
    def __init__(self,public,contract,input_digest,response_bytes,root,*,terminal,producer_report_sha256,producer_outer_sha256):
        self.directory=None;self.files=[];self.records=();self.closed=False;self.failed=None;self.used=False;self.client=None
        if type(terminal)is not ProducerTerminal:raise TypeError('externally authenticated producer terminal required')
        if type(response_bytes)is not bytes:raise TypeError('exact response bytes required')
        response=json.loads(response_bytes)
        validate_response(public,contract,input_digest,response)
        if response_bytes!=canonical(response):raise ValueError('canonical response metadata required')
        # A direct dataclass construction cannot bypass the same public checks.
        ProducerTerminal.authenticate(public,contract,input_digest,response_bytes,report_bytes=terminal.report,
            report_sha256=producer_report_sha256,outer_bytes=terminal.outer,outer_sha256=producer_outer_sha256)
        self.public=public;self.contract=canonical(contract);self.response=response_bytes;self.input_digest=input_digest;self.terminal=terminal
        self._fixed=(public,self.contract,response_bytes,input_digest,terminal)
        self._terminal_state=(terminal.report,terminal.outer)
        try:
            self.directory=HeldDirectory(root)
            names={f'block{i:02}.npz'for i in range(38)}|{'response.json'}
            if set(os.listdir(self.directory.fd))!=names:raise ValueError('exact38-output directory inventory required')
            metadata=_HeldFile(self.directory,'response.json',1<<20,sha(response_bytes));self.files.append(metadata)
            if _FdReader(metadata.fd,metadata.size).read()!=response_bytes:raise ValueError('response file differs from admitted bytes')
            records=[]
            for index,row in enumerate(response['artifacts']):
                binding=row['binding'];maximum=int(np.prod(binding['shape']))*4+65536
                held=_HeldFile(self.directory,f'block{index:02}.npz',maximum,row['payload_sha256']);self.files.append(held)
                raw=_FdReader(held.fd,held.size).read()
                # Same six-field payload decoder as reviewed default125, with
                # its exact tuple/layout interface supplied by the Qwen schema.
                checked=dict(shape=tuple(binding['shape']),moduli=binding['moduli'],layout=binding['degree_layout'],scale=binding['scale'])
                payload,_=_payload(raw,checked);held.check()
                records.append(AdmittedPayload(canonical(binding),payload.tobytes(order='C')))
            self.records=tuple(records);self._records=self.records;self._files=tuple(self.files);self._directory=self.directory;self.check()
        except BaseException as primary:
            try:self.close(primary=primary)
            except BaseException as secondary:
                if secondary is not primary:primary.add_note('response setup cleanup: '+repr(secondary))
            raise
    def check(self,*,readback=False):
        if self.failed is not None:raise self.failed
        if self.closed:raise ValueError('response backing closed')
        if (self.public is not self._fixed[0]or self.contract!=self._fixed[1]or self.response!=self._fixed[2]or self.input_digest!=self._fixed[3]or self.terminal is not self._fixed[4]):raise ValueError('admitted response state changed')
        if (self.terminal.report,self.terminal.outer)!=self._terminal_state:raise ValueError('producer terminal state changed')
        self.public.check();self.directory.check()
        if self.records and (self.records is not self._records or len(self.records)!=38):raise ValueError('retained complete payload inventory changed')
        if self.records and (self.directory is not self._directory or len(self.files)!=len(self._files)or any(a is not b for a,b in zip(self.files,self._files))):raise ValueError('retained response ownership changed')
        for held in self.files:held.readback()if readback else held.check()
    def run_client(self,private_loader,decode_block):
        """Ordinary trusted callbacks; an actual role needs a separate grant.

        The future private loader must quarantine keys on an entered failure;
        no callback is permitted to relax metadata or numerical error gates.
        """
        self.check(readback=True)
        if self.used or len(self.records)!=38:raise ValueError('one complete client consumption required')
        if not callable(private_loader)or not callable(decode_block):raise TypeError('client loader and decoder callbacks required')
        self.used=True;manager=None;entered=False;primary=None;result=None
        try:
            manager=private_loader();self.client=manager.__enter__();entered=True
            outputs=[]
            for record in self.records:
                binding=json.loads(record.binding);self.check()
                value=decode_block(self.client,record.array(),binding)
                if type(value)is not np.ndarray or value.dtype!=np.dtype('float64')or value.shape!=(65536,)or not np.isfinite(value).all():raise ValueError('all-slot finite real float64 decode required')
                # A legitimate decoder can reuse its scratch array. Retain an
                # independent immutable full-slot snapshot before calling it
                # again; the pure layout helper cannot recover overwritten data.
                outputs.append(np.frombuffer(value.tobytes(order='C'),dtype=np.float64));self.check()
            result=read_vocabulary(tuple(outputs),tuple(vocabulary()))
            self.check(readback=True)
        except BaseException as error:
            primary=error;self.failed=error;_QUARANTINE.append((self,self.client,manager,error))
        final=Finalizer({},primary)
        if entered:
            final.attempt('private loader retirement',lambda:manager.__exit__(type(primary)if primary is not None else None,primary,primary.__traceback__ if primary is not None else None))
        try:final.raise_if_failed()
        except BaseException as error:
            if self.failed is None:self.failed=error;_QUARANTINE.append((self,self.client,manager,error))
            raise
        self.client=None;return result
    def close(self,primary=None):
        if self.failed is not None:raise primary if primary is not None else self.failed
        if self.closed:return
        final=Finalizer({},primary)
        if self.directory is not None:
            for held in self.files:final.attempt('response final readback',held.readback)
            for held in reversed(self.files):final.attempt('response descriptor retirement',held.close)
            final.attempt('response directory retirement',self.directory.close)
        self.closed=True
        try:final.raise_if_failed()
        except BaseException as error:self.failed=error;_QUARANTINE.append((self,error));raise

def create_reference_session(*args,**kwargs):
    """Reserved organizer/reference seam; never edits the contestant starter."""
    raise PermissionError('actual Qwen session construction is disabled in this public/synthetic preparation')
