from pathlib import Path
import hashlib,importlib.util,json,sys,types
root=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('reviewed_collector',root/'before.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
sys.modules['client_keys']=types.SimpleNamespace(_OwnedDirectory=object)
m.sources=lambda:{'unit.py':b'# public-only fixture\n'}
original=Path.mkdir;primary=OSError('original snapshot construction failure')
def fail(path,*args,**kwargs):
    if path.name=='snapshot':raise primary
    return original(path,*args,**kwargs)
Path.mkdir=fail;sys.argv=['reproduce','--output',str(root/'failed-attempt')]
try:m.main()
except BaseException as error:
    result=dict(primary_preserved=error is primary,caught_type=type(error).__name__,caught_message=str(error),report_written=(root/'failed-attempt/report.json').exists(),source_sha256=hashlib.sha256((root/'before.py').read_bytes()).hexdigest())
finally:Path.mkdir=original
(root/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
