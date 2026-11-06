"""Q5 serial role composition with source-fixed adapter and explicit recording seams.

Actual entries still deny. The retained owner and pure envelope contract
and bodies below are for source review and later focused recording proof only.
No constructor or self-consistent bytes here constitute execution authority.
"""
from dataclasses import dataclass
import hashlib
from cleanup_origin import protect_body
from q5_checkpoint_contract import canonical,parse,all_checkpoints,DEFAULT_DECODER_SHA256
from q5_checkpoint_transport import ArtifactReader,Exporter,finish,path
from q5_actual_metadata_replay import check_refreshes,client_replay
_ACTIVE=None
_QUARANTINE=[]


def digest(raw):return hashlib.sha256(raw).hexdigest()
def accessor(value):return (getattr(value,'__self__',None),getattr(value,'__func__',value))
def same_accessor(a,b):return all(x is y for x,y in zip(accessor(a),accessor(b)))

def _object(raw,keys):
    value=parse(raw)
    if type(value)is not dict or set(value)!=set(keys)or canonical(value)!=raw:raise ValueError('closed canonical role envelope required')
    return value

@dataclass(frozen=True)
class PublicServer:
    """Authenticated public bytes only; not an actual execution capability."""
    manifest_bytes:bytes
    pins_bytes:bytes
    evidence:tuple
    def check(self):
        new=authenticate_completed_server(self.manifest_bytes,dict(self.evidence),parse(self.pins_bytes))
        if new!=self:raise ValueError('retained public proof changed')
        return parse(self.manifest_bytes)

def authenticate_completed_server(manifest_bytes,evidence,pins):
    """Proposed future envelope, always coupled to independent external pins.

    Names/fields are new Q5 proposal fields, not claims that such producers or
    receipts already exist. Exact outer0 rejects report-only/late-close success.
    """
    names={'setup_report','setup_outer','server_report','server_outer','server_root_review'}
    if type(evidence)is not dict or set(evidence)!=names or type(pins)is not dict or set(pins)!=names|{'manifest','context_id','model_id','setup_source','server_source','server_grant','output_root','input_root','original_input'}:raise ValueError('independent complete role pins required')
    for key in pins.keys()-{'output_root','input_root','original_input'}:
        if type(pins[key])is not str or len(pins[key])!=64 or any(c not in '0123456789abcdef' for c in pins[key]):raise ValueError('typed external digest pin required')
    from q5_checkpoint_transport import descriptor
    for name in ('output_root','input_root'):descriptor(canonical(pins[name]))
    for name,raw in [('manifest',manifest_bytes),*evidence.items()]:
        if type(raw)is not bytes or not 0<len(raw)<=1<<20 or digest(raw)!=pins[name]:raise ValueError('independent public proof digest differs')
    report_fields={'schema','role','complete','passed','context_id','model_id','source_manifest_sha256','source_archive_sha256','grant_sha256','manifest_sha256','cleanup_completed','audit_denials'}
    outer_fields={'schema','role','report_sha256','source_manifest_sha256','source_archive_sha256','grant_sha256','exit_code','normal_return','reaped','process_group_absent','publication_completed'}
    reports={}
    for role in ('setup','server'):
        report=_object(evidence[role+'_report'],report_fields);outer=_object(evidence[role+'_outer'],outer_fields)
        if report['schema']!='qwen-toy5-role-report-v1' or outer['schema']!='qwen-toy5-role-outer-v1' or report['role']!=role or outer['role']!=role:raise ValueError('separate exact role schemas required')
        for field in ('complete','passed','cleanup_completed'):
            if report[field]is not True:raise ValueError('complete successful producer required; partial artifacts remain inadmissible')
        if report['audit_denials']!=[] or type(outer['exit_code'])is not int or outer['exit_code']!=0 or any(outer[k]is not True for k in ('normal_return','reaped','process_group_absent','publication_completed')):raise ValueError('independent normal0/reap/publication required')
        if outer['report_sha256']!=pins[role+'_report']:raise ValueError('outer report digest differs')
        for k in ('source_manifest_sha256','source_archive_sha256','grant_sha256'):
            if outer[k]!=report[k]:raise ValueError('outer/source/grant linkage differs')
        if report['context_id']!=pins['context_id'] or report['model_id']!=pins['model_id'] or report['source_manifest_sha256']!=pins[role+'_source']:raise ValueError('independently pinned role identity differs')
        reports[role]=report
    if reports['server']['grant_sha256']!=pins['server_grant']or reports['server']['manifest_sha256']!=pins['manifest']:raise ValueError('actual server grant/manifest binding differs')
    root=_object(evidence['server_root_review'],{'schema','passed','server_report_sha256','server_outer_sha256','manifest_sha256','setup_report_sha256','setup_outer_sha256'})
    if root['schema']!='qwen-toy5-server-root-review-v1' or root['passed']is not True:raise ValueError('independent completed-server review required')
    for name in ('server_report','server_outer','manifest','setup_report','setup_outer'):
        if root[name+'_sha256']!=pins[name]:raise ValueError('independent root proof link differs')
    m=_object(manifest_bytes,{'schema','bindings','original_input','output_root','artifacts','rows','refresh_observations','complete','logical_checkpoints','new_exports','terminal_authority'})
    if m['schema']!='qwen-toy5-checkpoint-manifest-v1' or m['complete']is not True or type(m['logical_checkpoints'])is not int or m['logical_checkpoints']!=47 or type(m['new_exports'])is not int or m['new_exports']!=46 or m['terminal_authority']is not False:raise ValueError('complete47/46 manifest required')
    wanted=dict(setup_report_sha256=pins['setup_report'],setup_outer_sha256=pins['setup_outer'],context_id=pins['context_id'],model_id=pins['model_id'],source_manifest_sha256=pins['server_source'],grant_sha256=pins['server_grant'])
    if canonical(m['original_input'])!=canonical(pins['original_input']):raise ValueError('independently admitted original request artifact required')
    if canonical(m['bindings'])!=canonical(wanted)or canonical(m['output_root'])!=canonical(pins['output_root'])or canonical(m['original_input']['root'])!=canonical(pins['input_root']):raise ValueError('external original input/output roots and role bindings required')
    roots=[path(pins[n]['path'])for n in ('input_root','output_root')]
    if roots[0]==roots[1]or roots[0]in roots[1].parents or roots[1]in roots[0].parents:raise ValueError('serial roles require disjoint original request/response roots')
    all_checkpoints(m['rows']);check_refreshes(m['refresh_observations'],m['rows'])
    if type(m['artifacts'])is not list or len(m['artifacts'])!=46:raise ValueError('46 ordered actual exports required')
    records=[m['original_input']['artifact'],*m['artifacts']]
    if [r['metadata']for r in records]!=m['rows'] or [r['name']for r in records[1:]]!=[f'checkpoint_{i:02d}.npz'for i in range(1,47)]:raise ValueError('ordered artifact metadata differs')
    return PublicServer(manifest_bytes,canonical(pins),tuple(sorted(evidence.items())))

class SerialRole:
    def __init__(self,name):self.name=name
    def __enter__(self):
        global _ACTIVE
        if _ACTIVE is not None or _QUARANTINE:raise RuntimeError('nonoverlapping clean dedicated role required')
        _ACTIVE=self;return self
    def __exit__(self,kind,value,tb):
        global _ACTIVE
        if _ACTIVE is not self:raise ValueError('serial role owner changed')
        _ACTIVE=None
        return False

def server_bound(np,loader,weight_owner_factory,input_owner_factory,dependencies,publisher_factory,*,admission_check):
    """Concrete adapters select this ownership body; actual factory remains denied.

    All factories are trusted recording seams unless separately source-selected
    by the exact runtime. Partial returned owners are retained before accessors.
    Original cleanup and late guards execute inside the entered key scope.
    """
    from q5_checkpoint_arithmetic import exercise
    from qwen_finalizer import Finalizer
    admission_check();captured=(loader,weight_owner_factory,input_owner_factory,publisher_factory,admission_check)
    owners=[];methods=[];retired=set();primary=None;result=None;raw=None;weights=None;original_input=None
    with SerialRole('server'):
        with loader()as handle:
            original_ctx=handle.ctx;handle_check=handle.check;handle_check()
            def acquire(owner,names):
                owners.append(owner);row={};methods.append(row);row['close']=owner.close
                for name in names:row[name]=getattr(owner,name)
                if any(not callable(fn)for fn in row.values()):raise TypeError('original owner callbacks required')
                return row
            def guard(*,live=True):
                admission_check();handle_check()
                if handle.ctx is not original_ctx or not same_accessor(handle.check,handle_check):raise ValueError('original borrowed key context changed')
                for i,(owner,row)in enumerate(zip(owners,methods)):
                    for name,method in row.items():
                        if not same_accessor(getattr(owner,name),method):raise ValueError('original role owner callback changed')
                    if live and i not in retired and 'check'in row:row['check']()
                if live and original_input is not None and input_owner.value is not original_input:raise ValueError('original input ciphertext changed')
                if live and weights is not None and weight_owner.weights is not weights:raise ValueError('original public weight owner changed')
            try:
                weight_owner=weight_owner_factory(handle);weight_methods=acquire(weight_owner,('load','check'))
                weights=weight_methods['load']();weight_methods['check']()
                input_owner=input_owner_factory(handle);input_methods=acquire(input_owner,('load','check'))
                original_input=input_methods['load']();input_methods['check']()
                publisher=publisher_factory();pub=acquire(publisher,('check','finish','checkpoint'));guard()
                def checkpoint(index,value,row):guard();pub['checkpoint'](index,value,row);guard()
                result=exercise(original_ctx,original_input,weights,checkpoint=checkpoint,dependencies=dependencies)
                guard();raw=pub['finish'](result);guard()
            except BaseException as error:primary=protect_body(error)
            finally:
                originals=tuple(owners);callbacks=tuple(dict(row)for row in methods);final=Finalizer({},primary)
                final.attempt('original owner binding before cleanup',lambda:guard(live=False))
                for i in reversed(range(len(callbacks))):
                    close=callbacks[i].get('close')
                    if close is not None:final.attempt('original role consumer close',lambda close=close:close(primary))
                    retired.add(i);final.attempt('original bindings after cleanup',lambda:guard(live=False))
                error=final.selected()
                if error is not None:
                    _QUARANTINE.append((captured,handle,original_ctx,originals,callbacks,weights,original_input,result,raw,error));raise error
            if raw is None:raise RuntimeError('server did not complete original47 checkpoint manifest')
            guard(live=False);return raw

def client_bound(np,public,private_loader,weights,embeddings,rules,reference_root,*,admission_check):
    """Full public auth before private; fail-fast decoding with whole-owner quarantine.

    Expected private-handle API is a source proposal: check(), ctx,
    new_ciphertext_owner(payload,row), decrypt_slots, decoder_source_sha256.
    The owner is acquired BEFORE load() allocates: value, load(), check(), close().
    Its constructor must not allocate unreturned runtime owners. No real adapter
    for this API is selected yet. No scientific decoder continuation is enabled.
    """
    from q5_checkpoint_oracle import references,diagnostics,original_final_gate
    admission_check();manifest=public.check();root_readers=[];retained=[];primary=None
    with SerialRole('client'):
        try:
            first=ArtifactReader(np,canonical(manifest['original_input']['root']));root_readers.append(first)
            output=ArtifactReader(np,canonical(manifest['output_root']));root_readers.append(output)
            records=[manifest['original_input']['artifact'],*manifest['artifacts']]
            # Every entire opaque body, residue/header/metadata and original
            #root identity is checked BEFORE private_loader is even called.
            for index,artifact in enumerate(records):(first if index==0 else output).read(artifact)
            public.check();admission_check()
            replay=client_replay(np,reference_root,weights,embeddings,manifest['rows'],manifest['refresh_observations'])
            dense,dense_info=references(np,numerical_config(),weights,embeddings,rules)
            results=[]
            with private_loader()as handle:
                ctx=handle.ctx;check=handle.check;decode=handle.decrypt_slots;factory=handle.new_ciphertext_owner
                check()
                if handle.decoder_source_sha256!=DEFAULT_DECODER_SHA256:raise ValueError('exact default decoder source required')
                def guard():
                    public.check();admission_check();check()
                    if handle.ctx is not ctx or not same_accessor(handle.check,check)or not same_accessor(handle.decrypt_slots,decode)or not same_accessor(handle.new_ciphertext_owner,factory):raise ValueError('original private context/codec/factory changed')
                    for reader in root_readers:reader.check()
                for index,artifact in enumerate(records):
                    guard();reader=first if index==0 else output;payload=reader.read(artifact,retain_payload=True)
                    owner=factory(payload,artifact['metadata']);retained.append((owner,payload));close=owner.close;load=owner.load;owner_check=owner.check;body=None
                    try:
                        load();owner_check();value=owner.value
                        # Original actual scale, ALL slots, mandatory default
                        #approximation validation. No relaxed retry/averaging.
                        decoded=decode(ctx,value,scale=artifact['metadata']['scale'],slots_to_decode=65536,validate_approximation=True)
                        if type(decoded)is not np.ndarray or decoded.dtype!=np.dtype('float64')or decoded.shape!=(65536,):raise ValueError('exact default decoder float64 convention required')
                        snapshot=np.array(decoded,copy=True)
                        if snapshot.shape!=(65536,)or not np.isfinite(snapshot).all():raise ValueError('full finite default decode required')
                        snapshot.flags.writeable=False
                        guard();owner_check()
                        if owner.value is not value or not same_accessor(owner.close,close)or not same_accessor(owner.load,load)or not same_accessor(owner.check,owner_check):raise ValueError('original ciphertext owner changed')
                        results.append(diagnostics(np,snapshot,dense[index],index))
                    except BaseException as error:
                        body=protect_body(error)
                        _QUARANTINE.append((handle,ctx,decode,factory,tuple(retained),tuple(results),index,body))
                        # Includes known ApproximationErrorTooHigh: stop, retain,
                        #and cross the key scope. No continue/normal-failure
                        #relabeling before an explicit future cleanup policy.
                        raise
                    finally:
                        def retired_binding():
                            guard()
                            if not same_accessor(owner.close,close)or not same_accessor(owner.load,load)or not same_accessor(owner.check,owner_check):raise ValueError('original consumer callback changed during retirement')
                        finish(body,[('original ciphertext retirement before keys',lambda:close(body),owner),
                                     ('original bindings after ciphertext retirement',retired_binding,(handle,ctx,owner))])
                    retained.pop();guard()
                # A final callback failure must cross the still-entered key loader.
                guard()
            gate=original_final_gate(np,reference_root,snapshot)
            return dict(complete=True,decoded_records=47,metadata_replay=replay,dense=dense_info,diagnostics=results,
                        original_numeric_gate_evaluated=True,original_final_gate=gate,actual_execution_authorized=False)
        except BaseException as error:
            primary=protect_body(error);_QUARANTINE.append((public,tuple(root_readers),tuple(retained),primary));raise
        finally:finish(primary,[('original public reader close',lambda reader=r:reader.close(primary),r)for r in reversed(root_readers)])

def numerical_config():
    from q5_checkpoint_contract import numerical_plan
    return numerical_plan()['case']['config']

def setup_actual(*args,**kwargs):raise PermissionError('fresh key generation/resource/source bootstrap is not selected in Q5 source preparation')
def server_actual(*args,**kwargs):raise PermissionError('actual Q5 server needs a reviewed source/bootstrap, key-loader and separate root grant')
def client_actual(*args,**kwargs):raise PermissionError('actual Q5 client needs a reviewed private-owner adapter and separate root grant')
