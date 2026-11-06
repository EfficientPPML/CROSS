"""Source-copy collector: public synthetic raw-array parity, never encrypted HE."""
from pathlib import Path
import argparse,hashlib,io,json,os,signal,subprocess,sys,tarfile,time
from cleanup import cleanup_signals,termination_signals
from terminal import terminal_scope
HERE=Path(__file__).resolve().parent
SOURCE_NAMES = ('DESIGN.md', 'base/encrypted_compilation.py', 'base/encrypted_packed_qwen.py', 'base/encrypted_rotation.py', 'check_public_group.py', 'checkout/jaxite_word/__init__.py', 'checkout/jaxite_word/bconv.py', 'checkout/jaxite_word/bootstrapping.py', 'checkout/jaxite_word/bootstrapping_openfhe_crosscheck.py', 'checkout/jaxite_word/bsgs.py', 'checkout/jaxite_word/ckks_ctx.py', 'checkout/jaxite_word/composite_prime_gen.py', 'checkout/jaxite_word/finite_field.py', 'checkout/jaxite_word/he_ops.py', 'checkout/jaxite_word/he_params.py', 'checkout/jaxite_word/headd.py', 'checkout/jaxite_word/hemul.py', 'checkout/jaxite_word/herot.py', 'checkout/jaxite_word/hesub.py', 'checkout/jaxite_word/key_gen.py', 'checkout/jaxite_word/mapping.py', 'checkout/jaxite_word/nn.py', 'checkout/jaxite_word/ntt_mm.py', 'checkout/jaxite_word/ntt_sm.py', 'checkout/jaxite_word/packing.py', 'checkout/jaxite_word/polynomial.py', 'checkout/jaxite_word/profiler.py', 'checkout/jaxite_word/ptct_mul.py', 'checkout/jaxite_word/rescale.py', 'checkout/jaxite_word/rns.py', 'checkout/jaxite_word/tabIX.py', 'checkout/jaxite_word/tabV.py', 'checkout/jaxite_word/tabVI.py', 'checkout/jaxite_word/tabVII.py', 'checkout/jaxite_word/util.py', 'cleanup.py', 'collect_public.py', 'grouped_rotation.py', 'lease_test_helpers.py', 'origins.json', 'p60_profile_and_refresh.json', 'prepared_rotation.py', 'terminal.py', 'test_backing_successor.py', 'test_collector.py', 'test_grouped_rotation.py')

def sources():
    return {name:(HERE/name).read_bytes() for name in SOURCE_NAMES}
def hashes(raw):return {n:hashlib.sha256(v).hexdigest() for n,v in raw.items()}
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def run(output):
    output=Path(output);output.mkdir(exist_ok=False);start=time.monotonic()
    primary=None;cleanup=[];child=None;snapshot=None;before={}
    report=dict(passed=False,scope='Mocked group lifecycle plus actual JAX public residue/synthetic-key parity under explicit mocked security metadata. No CKKSContext, secret, keygen, encryption or decryption; no secure-HE/performance/peak-memory claim.',cpu_affinity=sorted(os.sched_getaffinity(0)),pid=os.getpid(),processes=[])
    def cleanup_error():return next((e for e in cleanup if not isinstance(e,Exception)),cleanup[0] if cleanup else None)
    def attempt(label,fn):
        try:return fn()
        except BaseException as e:
            cleanup.append(e)
            if primary is not None:primary.add_note(label+': '+repr(e))
    def read_archive():
        with tarfile.open(output/'sources.tar.gz') as t:
            ms=t.getmembers()
            if len(ms)!=len(before) or len({m.name for m in ms})!=len(ms) or not all(m.isfile() for m in ms):raise ValueError('source archive inventory mismatch')
            if {m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in ms}!=before:raise ValueError('source archive byte mismatch')
        return True
    try:
        write(output/'report.json',report);raw=sources();before=hashes(raw);report['source_hashes_before']=before
        with tarfile.open(output/'sources.tar.gz','w:gz') as t:
            for n,v in raw.items():
                m=tarfile.TarInfo(n);m.size=len(v);t.addfile(m,io.BytesIO(v))
        read_archive();snapshot=output/'snapshot';snapshot.mkdir()
        for n,v in raw.items():
            p=snapshot/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(v)
        env=os.environ.copy();env.update(JAX_PLATFORMS='cpu',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')
        commands=[('collector-tests',[sys.executable,'-m','pytest','-q','test_collector.py','test_grouped_rotation.py','test_backing_successor.py','--junitxml='+str(output/'tests.xml')]),('public-parity',[sys.executable,str(snapshot/'check_public_group.py'),'--output',str(output/'public-parity')])]
        for phase,cmd in commands:
            with (output/(phase+'.log')).open('wb') as stream:
                child=subprocess.Popen(cmd,cwd=snapshot,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                record=dict(phase=phase,pid=child.pid,command=cmd,reaped=False);report['processes'].append(record);write(output/'report.json',report)
                code=child.wait();record.update(returncode=code,reaped=True)
            if code!=0:raise RuntimeError(phase+' failed; failure preserved')
        import xml.etree.ElementTree as ET
        suites=ET.parse(output/'tests.xml').getroot().findall('testsuite')
        counts={k:sum(int(s.attrib.get(k,0)) for s in suites) for k in ('tests','errors','failures','skipped')}
        if counts['tests']<7 or any(counts[k] for k in ('errors','failures','skipped')):raise ValueError('complete collector tests required')
        result=json.loads((output/'public-parity/report.json').read_bytes())
        if result['passed']is not True or len(result['fixtures'])!=12 or result['total_comparisons']!=372 or result['no_context_keygen_modules_loaded']is not True:raise ValueError('complete public proof required')
        report.update(operation_complete=True,public_result=result,test_counts=counts)
    except BaseException as e:
        primary=e;report['error']=dict(type=type(e).__name__,message=str(e))
    finally:
        with terminal_scope(report,output/'report.json',write,cleanup_signals,lambda:(primary,cleanup_error())):
            if child is not None and child.poll()is None:
                attempt('stop owned child',lambda:os.killpg(child.pid,signal.SIGKILL));attempt('reap owned child',child.wait)
            if child is not None and report['processes']:
                report['processes'][-1].update(reaped=child.poll()is not None,returncode=child.poll())
            report['process_reaped']=all(p['reaped'] for p in report['processes'])
            after=attempt('source readback',lambda:hashes(sources()))
            copied=attempt('snapshot readback',lambda:{n:hashlib.sha256((snapshot/n).read_bytes()).hexdigest() for n in before}) if snapshot else None
            verified=attempt('archive readback',read_archive) if before else False
            report.update(source_hashes_after=after,snapshot_hashes=copied,source_files_unchanged=bool(before) and before==after==copied,source_archive_verified=verified is True,source_count=len(before),cleanup_errors=[repr(e) for e in cleanup],elapsed_s=time.monotonic()-start,cleanup_completed=not cleanup and report['process_reaped'])
            if (output/'sources.tar.gz').is_file():report['source_archive_sha256']=hashlib.sha256((output/'sources.tar.gz').read_bytes()).hexdigest()
            report['passed']=bool(report.get('operation_complete') and primary is None and not cleanup and report['process_reaped'] and report['source_files_unchanged'] and verified)
            write(output/'report.json',report)
            if primary is not None:raise primary
            if cleanup:raise cleanup_error()
    if report['passed'] and report['terminal_scope_completed'] is not True:raise ValueError('terminal scope incomplete')
    print(json.dumps(dict(passed=report['passed'],source_count=report['source_count'],comparisons=report.get('public_result',{}).get('total_comparisons'),source_archive_sha256=report.get('source_archive_sha256'))))
    return report

def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    with termination_signals():report=run(parser.parse_args(argv).output)
    return 0 if report['passed'] and report['terminal_scope_completed'] is True else 1
if __name__=='__main__':raise SystemExit(main())
