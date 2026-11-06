import sys,json,pytest
from pathlib import Path
source=Path(sys.argv[1]);out=Path(sys.argv[2]);sys.path.insert(0,str(source))
denied=[]
def audit(event,args):
 if event=='import'and args and str(args[0]).split('.')[0]in ('torch','transformers','jax','jaxlib','jaxite_word'):
  denied.append(str(args[0]));raise PermissionError('backend import disabled in synthetic layout proof')
sys.addaudithook(audit)
code=pytest.main(['-q','-p','no:cacheprovider','--rootdir='+str(source),str(source/'test_qwen_layout.py'),'--junitxml='+str(out/'tests.xml')])
(out/'worker-audit.json').write_text(json.dumps(dict(denied=denied,loaded_backend=[n for n in sys.modules if n.split('.')[0]in ('torch','transformers','jax','jaxlib','jaxite_word')],exit_code=int(code)),sort_keys=True)+'\n')
raise SystemExit(int(code)if not denied else 70)
