"""Trusted test seams, not real store admission, body integrity, or execution authority.

All data files made here are new tiny fixtures. The complete fixed catalog is
public metadata. RecordingRun replaces admission, resource observation, signals,
audit installation, catalog construction, reader factory and observed-store
reconstruction; production traversal, callback guards, Finalizer, publication,
and terminal-status code are selected unchanged. observed_stores has a separate
real-small-FD test. No existing parameter or private body is opened here.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace as NS,ModuleType
import weakref

ROOT=Path(__file__).resolve().parents[1]
RUNTIME=ROOT/'runtime'
MAP_SHA='b73e669b0cc2bc34d7930f2a8a4c620be719fe663dfa6ae7b12df7cae22c2bc0'
def canonical(v):return (json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n').encode()
def sha(raw):return hashlib.sha256(raw).hexdigest()
def source_bundle():
    raw=(ROOT/'runtime_sources.json').read_bytes();assert sha(raw)==MAP_SHA
    expected=json.loads(raw);blobs={n:(RUNTIME/n).read_bytes()for n in expected}
    assert len(expected)==42 and {n:sha(b)for n,b in blobs.items()}==expected
    assert set(p.relative_to(RUNTIME).as_posix()for p in RUNTIME.rglob('*')if p.is_file())==set(expected)
    return NS(raw=raw,expected=expected,blobs=blobs)

def synthetic_stores(aggregate,base):
    """Public catalog hashes plus invented metadata: no197 matrix bodies."""
    from model_admission import ExpectedStoreIdentity
    roots={f:dict(path=str(base/f),device=7,inode=100+i,mode=0o700)
           for i,f in enumerate(('qkv','decoder','gamma'))}
    physical={(r['family'],r['descriptor']['file']):r['descriptor']for r in aggregate.catalog['matrices'].values()}
    files=[]
    for i,((family,name),d)in enumerate(sorted(physical.items())):
        size=8
        for x in d['shape']:size*=x
        files.append(dict(family=family,name=name,state=dict(device=7,inode=1000+i,size=size+128,
            mtime_ns=1,ctime_ns=2,mode=0o644),file_sha256=d['file_sha256'],raw_sha256=d['sha256']))
    files.append(dict(family='gamma',name='synthetic-gamma.npz',state=dict(device=7,inode=2000,
        size=1024,mtime_ns=1,ctime_ns=2,mode=0o644),file_sha256=aggregate.catalog['gamma_container_sha256'],raw_sha256=None))
    return ExpectedStoreIdentity(aggregate,canonical(dict(roots=roots,files=sorted(files,key=lambda r:(r['family'],r['name'])))))

def grant_fixture(source,aggregate,base):
    from model_validation_grant import GRANT_CONSTANTS,authenticate
    stores=synthetic_stores(aggregate,base/'borrowed')
    review=canonical(dict(passed=True,scope='invented focused fixture; no actual body integrity'))
    value=dict(GRANT_CONSTANTS,source_root=str(RUNTIME),evidence_output=str(base/'output'),
        source_manifest_sha256=sha(source.raw),source_archive_sha256='a'*64,
        expected_stores=dict(path=str(base/'expected.json'),sha256=sha(stores.raw)),
        prelaunch_review=dict(path=str(base/'review.json'),sha256=sha(review)),
        dependency_policy_sha256=sha(source.blobs['model_dependencies.json']),
        readonly_policy_sha256=sha(source.blobs['model_readonly_policy.json']))
    def call(value_override=None,*,raw_override=None,store_override=None,review_override=None,external=None):
        raw=canonical(value if value_override is None else value_override)if raw_override is None else raw_override
        return authenticate(raw,sha(raw)if external is None else external,
            stores.raw if store_override is None else store_override,review if review_override is None else review_override,
            source_root=str(RUNTIME),output_path=str(base/'output'),source_map_bytes=source.raw,
            source_manifest_sha256=sha(source.raw),blobs=source.blobs)
    return NS(value=value,stores=stores,review=review,call=call,base=base)

def copy_runtime(source,target):
    target.mkdir()
    for n,b in source.blobs.items():
        p=target/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
    return target

def child(argv,work):
    """Own one fresh test child without a healthy-job deadline; direct wait/reap."""
    work.mkdir()
    env={k:v for k,v in os.environ.items()if k in('PATH','LANG','LC_ALL','HOME')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    with (work/'stdout.log').open('wb')as out,(work/'stderr.log').open('wb')as err:
        proc=subprocess.Popen(argv,cwd=work,env=env,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
        pgid=proc.pid
        try:
            observed=os.getpgid(proc.pid);assert observed==pgid
            code=proc.wait()
        except BaseException:
            try:os.killpg(pgid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();raise
        finally:
            if proc.returncode is None:
                try:os.killpg(pgid,signal.SIGKILL)
                except ProcessLookupError:pass
                proc.wait()
    try:os.killpg(pgid,0)
    except ProcessLookupError:absent=True
    else:absent=False
    receipt=dict(pid=proc.pid,pgid=pgid,exit_code=code,reaped=True,process_group_absent=absent,argv=argv)
    (work/'terminal.json').write_bytes(canonical(receipt));assert absent
    return NS(code=code,stdout=(work/'stdout.log').read_bytes(),stderr=(work/'stderr.log').read_bytes(),receipt=receipt)

@contextmanager
def inert_signals():yield

class RecordingRun:
    """Explicit recorded reader/resource/audit seams; real run()/Output/Finalizer."""
    def __init__(self,monkeypatch,source,aggregate,tmp_path,*,matrix_error=None,open_error=None,
                 dependency_error=None,catalog_error=None,close_error=None,swap_callback=False,
                 publish_denial=False,publish_error=None,shield_error=None,sticky_body=False):
        import model_validation_process as process
        from model_validation_grant import policies
        self.process=process;self.events=[];self.source=source
        self.stores=synthetic_stores(aggregate,tmp_path/'borrowed');self.catalog=aggregate.catalog
        self.output=tmp_path/'evidence';self.finder=NS(allow_numpy=False)
        self.matrix_names=[];self.gamma_names=[];self.close_primary=[];self.views=[]
        self.snapshot_calls=0;self.origin_calls=0;self.observed_calls=0
        launch=NS(stores=self.stores,source_root=str(RUNTIME),output_path=str(self.output),
            blobs=tuple(sorted(source.blobs.items())),grant_bytes=canonical({'source_archive_sha256':'a'*64}),
            grant_sha256='b'*64,source_manifest_sha256=sha(source.raw),check=lambda:None)
        self.launch=launch;policy,dep=policies(source.blobs)
        monkeypatch.setattr(process,'admit',lambda *args:(launch,policy,dep,[]))
        monkeypatch.setattr(process,'capacity',lambda *args:self.events.append('capacity')or {'recording':True})
        outer=self
        class Audit:
            def __init__(self,*args):self.denials=[];self.installed=False;self.closed=False;self.body_enabled=False;outer.audit=self
            def install(self):self.installed=True;outer.events.append('audit.install')
            def check(self):
                if self.denials:raise PermissionError('recorded sticky denial')
            def close(self):self.closed=True;outer.events.append('audit.close')
        monkeypatch.setattr(process,'ReadScope',Audit)
        monkeypatch.setattr(process,'termination_signals',inert_signals)
        if shield_error is None:monkeypatch.setattr(process,'cleanup_signals',inert_signals)
        else:
            def failed_shield():raise shield_error
            monkeypatch.setattr(process,'cleanup_signals',failed_shield)
        def dependencies(dep,*,loaded):
            self.events.append('dependency.loaded'if loaded else'dependency.before')
            if dependency_error is not None and not loaded:raise dependency_error
        monkeypatch.setattr(process,'dependency_check',dependencies)
        @dataclass(frozen=True)
        class Catalog:
            matrices:bytes
            gammas:bytes
            gamma_container_sha256:str
            provenance:bytes
            @classmethod
            def from_public(cls,public):
                if catalog_error is not None:raise catalog_error
                assert public is aggregate.public
                return cls(canonical(outer.catalog['matrices']),canonical(outer.catalog['gammas']),
                    outer.catalog['gamma_container_sha256'],canonical(outer.catalog['provenance']))
        class View:pass
        class Owner:
            def __init__(self):
                self.catalog=Catalog.from_public(aggregate.public);self.expected_store=outer.stores
                self.matrices={};self.gamma_values={};self.closed=False
            def check(self):outer.events.append('owner.check')
            def matrix(self,name):
                assert all(ref()is None for ref in outer.views),'previous transient view retained'
                if matrix_error is not None:raise matrix_error
                outer.matrix_names.append(name);r=outer.catalog['matrices'][name]
                self.matrices[(r['family'],r['descriptor']['file'])]=object()
                value=View();outer.views.append(weakref.ref(value))
                if swap_callback:self.close=lambda **kw:outer.events.append('replacement.close')
                return value
            def gamma(self,name):
                assert all(ref()is None for ref in outer.views)
                outer.gamma_names.append(name);self.gamma_values[name]=object()
                value=View();outer.views.append(weakref.ref(value));return value
            def readback(self):
                outer.events.append('owner.readback')
                if sticky_body and matrix_error is not None:raise matrix_error
            def close(self,primary=None):
                outer.events.append('owner.close');outer.close_primary.append(primary);self.closed=True
                if close_error is not None:raise close_error
        def opened(public,stores):
            self.events.append('reader.open');assert public is aggregate.public and stores is self.stores
            assert self.audit.body_enabled and self.finder.allow_numpy
            if open_error is not None:raise open_error
            self.owner=Owner();return self.owner
        module=ModuleType('qwen_public_parameters');module.ParameterCatalog=Catalog;module.open_identity_bound_parameters=opened
        monkeypatch.setitem(sys.modules,'qwen_public_parameters',module)
        def observed(owner,stores):
            self.observed_calls+=1;assert owner is self.owner and stores is self.stores
            assert len(owner.matrices)==197 and len(owner.gamma_values)==113
            return stores.check()
        monkeypatch.setattr(process,'observed_stores',observed)
        OriginalOutput=process.Output
        class RecordedOutput(OriginalOutput):
            def write(self,name,raw):
                self_outer.events.append('publication.write');super().write(name,raw)
                if publish_denial:self_outer.audit.denials.append('recorded denied operation swallowed after publication')
            def close(self,primary=None):
                self_outer.events.append('output.close');super().close(primary=primary)
                if publish_error is not None and 'publication.write'in self_outer.events:raise publish_error
        self_outer=self
        monkeypatch.setattr(process,'Output',RecordedOutput)
    def snapshot(self,expected):self.snapshot_calls+=1;assert expected==self.source.expected;return self.source.blobs
    def origins(self,expected):self.origin_calls+=1;assert expected==self.source.expected
    def run(self):return self.process.run(NS(),self.source.raw,self.source.expected,self.source.blobs,self.finder,self.snapshot,self.origins)
    def report(self):
        paths=[p for p in (self.output/'report.json',self.output/'failure.json')if p.exists()]
        assert len(paths)==1;return json.loads(paths[0].read_bytes())
