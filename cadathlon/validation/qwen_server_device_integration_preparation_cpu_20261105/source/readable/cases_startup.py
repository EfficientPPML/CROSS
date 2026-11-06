"""B2 startup ordering with explicit unadmitted inspection/observation seams.

The false-status branch uses original entry definitions before helper loading.
The public-memory branch runs original admit and original subset checker, but
its scientific packet inspection, public reads, path resolution and kernel
observations are recordings. This never supplies an actual-ready source.
"""
import json
from pathlib import Path
from types import SimpleNamespace


def entry_denial(entry,expected,blobs,loader):
    original_auth,original_public=entry.authenticate,entry.early_public
    calls=[]
    before=set(loader.loaded)
    def authenticate(manifest,digest):
        calls.append('source-recording')
        return entry.canonical(expected),expected,blobs
    def forbidden(value):
        calls.append('public-read')
        raise AssertionError('status denial must precede public paths')
    entry.authenticate,entry.early_public=authenticate,forbidden
    try:
        try:entry.dispatch(SimpleNamespace(manifest='UNOPENED',manifest_sha256='0'*64))
        except PermissionError as error:
            assert 'source-only pending' in str(error)
        else:raise AssertionError('false actual status was accepted')
        assert calls==['source-recording'] and set(loader.loaded)==before
        assert 'qwen_server_process'not in loader.loaded
    finally:entry.authenticate,entry.early_public=original_auth,original_public
    return dict(passed=True,case='entry_status_denial',group='B',checks=[
        'original dispatch rejects held false status before public-path or helper use'],
        recording_only=True,actual_authority=False,source_authentication_recorded=True,
        genuine_cli_bootstrap=False)


def run(f,case):
    assert case=='public_memory_before_sidecars'
    archive=b'SYNTHETIC ARCHIVE MARKER; no actual source archive admission\n'
    archive_sha=f.sha(archive)
    parent=f.owned/'public-memory';parent.mkdir(mode=0o700)
    paths={n:parent/n for n in('packet.json','archive.bin','device.json','proof.json')}
    public_keys=parent/'unopened-public-keys';private_keys=parent/'unopened-private-keys'
    def mutate(role,value):
        if role=='resource_review':value['source_archive_sha256']=archive_sha
    b=f.new_packet(mutate_document=mutate,grant_extra={
        'source_archive_sha256':archive_sha,'public_key_store':{'path':str(public_keys)}})
    pins=b['phase_pins'];pins['source_archive_sha256']=archive_sha
    pin_bytes=f.canonical(pins)
    packet=f.canonical({'phase_grant':b['phase'].grant_bytes.decode()})
    proof=SimpleNamespace(name='synthetic-proof',path=str(paths['proof.json']),sha256='8'*64,limit=1024)
    inspection=SimpleNamespace(phase_pins_bytes=pin_bytes,source_root=str(f.root),
        dependency_policy=b['documents']['dependency_policy'],role_policy=b'synthetic role marker',
        model_roots=tuple(f.process.MODEL_STORAGE_SCOPES),key_roots=(str(public_keys),str(private_keys)),
        request_root=str(parent/'unopened-request'),output_roots=(str(parent/'response'),str(parent/'evidence')),
        forbidden_proof_roots=(),proof_descriptors=(proof,),public=object(),
        resource_policy=b['documents']['resource_policy'],resource_review=b['documents']['resource_review'])
    # This is an explicit per-call unadmitted metadata overlay. Runtime files,
    # map and immutable false status remain exact and are never rewritten.
    blobs=dict(f.blobs)
    blobs.update({
        'qwen_server_entry_status.json':f.canonical(dict(schema='qwen-server-entry-status-v1',
            actual_entry_ready=True,actual_grant_present=False,closure_preflight_passed=True,
            dependency_policy_instance_supplied=True,resource_policy_instance_supplied=False,source_only=False)),
        'qwen_server_static_policy.json':f.canonical(dict(schema='qwen-server-static-policy-v1',
            protected_paths=list(f.process.MODEL_STORAGE_SCOPES))),
        'qwen_server_dependencies.json':inspection.dependency_policy,
        'qwen_server_role_policy.json':inspection.role_policy})
    observations=[];reads=[];marker=RuntimeError('stop after genuine public-memory subset, before device/proof bytes')
    def pin_check(value):
        assert value.raw==pin_bytes
        return pins
    def lexical(value):
        p=Path(value)
        assert p.is_absolute()and str(p)==value and '..'not in p.parts
        return p  # Recording only: no borrowed-name lstat/resolve.
    def public_from_sources(value):assert value is blobs;return inspection.public
    def inspect(public,raw,*,expected_sha256):
        assert public is inspection.public and raw is packet and expected_sha256==f.sha(packet)
        return inspection
    def read(path,*,limit,expected):
        path=Path(path);reads.append(path.name)
        if path==paths['packet.json']:raw=packet
        elif path==paths['archive.bin']:raw=archive
        elif path==paths['device.json']:
            assert observations[-1]=='memory.complete'
            raise marker
        else:raise AssertionError('larger scientific proof read before controlled stop')
        assert len(raw)<=limit and f.sha(raw)==expected
        return raw
    memory=f.module('qwen_server_public_memory');original_memory=memory.check_public_memory
    kernel_values={'/proc/meminfo':'MemAvailable: 1000000000 kB\n',
        '/proc/self/cgroup':'0::/synthetic-device-fixture\n',
        '/sys/fs/cgroup/synthetic-device-fixture/memory.current':'0\n',
        '/sys/fs/cgroup/synthetic-device-fixture/memory.max':'max\n'}
    def kernel(path):observations.append(path);return kernel_values[path]
    def affinity(pid):assert pid==0;return {0}
    def memory_call(*args,**kwargs):
        observations.append('memory.begin')
        result=original_memory(*args,**kwargs)
        assert result['public_metadata_only']is True and result['actual_execution_authorized']is False
        observations.append('memory.complete');return result
    for owner,name,value,label in (
        (f.phase_module.ExpectedServerPhasePins,'check',pin_check,'B: typed independent-pin recording for fabricated inspection'),
        (f.process,'canonical_path',lexical,'B: lexical metadata paths; no borrowed root observation'),
        (f.process,'public_from_sources',public_from_sources,'B: deliberately unadmitted source/public-contract recording'),
        (f.process,'inspect_packet',inspect,'B: fabricated bounded inspection; not scientific admission'),
        (f.process,'read',read,'B: packet/archive byte fixtures and first-sidecar stop'),
        (memory,'_path',lexical,'B: lexical synthetic cgroup path'),
        (memory,'_kernel',kernel,'B: fixed kernel text; no current capacity observation'),
        (memory.os,'sched_getaffinity',affinity,'B: fixed synthetic affinity'),
        (f.process,'check_public_memory',memory_call,'B: wrapper delegates original complete subset checker')):
        f.patch(owner,name,value,label)
    raw=f.canonical(f.expected)
    args=SimpleNamespace(packet=str(paths['packet.json']),packet_sha256=f.sha(packet),
        archive=str(paths['archive.bin']),archive_sha256=archive_sha,manifest_sha256=f.sha(raw),
        device_packet=str(paths['device.json']),device_packet_sha256='7'*64,role='full28_server')
    error=f.expect(RuntimeError,lambda:f.process.admit(args,raw,f.expected,blobs))
    assert error is marker
    assert reads==['packet.json','archive.bin','device.json']
    assert observations==['memory.begin',*kernel_values,'memory.complete']
    f.no_device_work()
    return dict(passed=True,case=case,group='B',recording_only=True,actual_authority=False,
        checks=['real public-memory subset authenticates twice before first sidecar read',
                'original admit stops before scientific proof/body or native initialization'],
        ordered_public_reads=reads,recorded_kernel_paths=list(kernel_values),
        observed_current_capacity=False,unadmitted_call_metadata_overlay=True)
