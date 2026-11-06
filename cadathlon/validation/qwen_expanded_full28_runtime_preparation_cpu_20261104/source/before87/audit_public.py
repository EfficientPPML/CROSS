"""Ordinary-Python public-plan audit, not an OS sandbox."""
from pathlib import Path
import json,os,runpy,sys
P=Path(__file__).resolve().parent
class Audit:
    def __init__(self):self.denied=[]
    def __call__(self,event,args):
        if event=='import' and args and str(args[0]).split('.')[0] in ('jax','jaxlib','torch','transformers','safetensors'):
            self.denied.append(dict(kind='backend_import',name=str(args[0])));raise RuntimeError('public plan forbids backend/model imports')
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=os.fsdecode(args[0]);absolute=os.path.abspath(path)
        if (Path(path).suffix.lower() in ('.npy','.npz','.safetensors','.parquet') or
            absolute.startswith('/home/jianming_gatech/cross_validation_assets/') or
            any(piece in absolute for piece in ('tokenized_fresh_inputs','validation_inputs','calibration_inputs','client_gamma_fixtures'))):
            self.denied.append(dict(kind='parameter_or_input_body',path=absolute));raise RuntimeError('public plan forbids parameter/input bodies')
def main():
    if len(sys.argv)<3 or sys.argv[2] not in ('pytest','plan'):raise ValueError('audit output and fixed mode required')
    output=Path(sys.argv[1]);audit=Audit();sys.addaudithook(audit);primary=None;code=1
    try:
        if sys.argv[2]=='pytest':
            import pytest
            code=pytest.main(sys.argv[3:])
        else:runpy.run_path(str(P/'plan_full.py'),run_name='__main__');code=0
    except BaseException as error:primary=error
    finally:
        loaded=[n for n in ('jax','jaxlib','torch','transformers','safetensors') if n in sys.modules]
        report=dict(denied_attempts=audit.denied,backend_modules=loaded,passed=not audit.denied and not loaded,scope='ordinary-Python audit; no OS isolation claim')
        try:
            tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');tmp.replace(output)
        except BaseException as error:
            if primary is None:primary=error
            else:primary.add_note(repr(error))
    if primary is not None:raise primary
    return code if report['passed'] else 1
if __name__=='__main__':raise SystemExit(main())
