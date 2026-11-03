"""Compare public eager/lazy LevelParams serialization in one fixed checkout."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
root=args.root.resolve()
sys.path[:0]=[str(root),str(root/'cadathlon/tests')]
from test_lazy_bconv import make_cache,assert_public_equal
files=[root/'jaxite_word/he_params.py',root/'cadathlon/tests/test_lazy_bconv.py',Path(__file__)]
def hashes():
    return {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
before=hashes()
report={'scope':__doc__,'source_hashes_before':before,'cases':{}}
for label,eager in [('eager','1'),('lazy','0')]:
    os.environ['CROSS_EAGER_LEVEL_PARAMS']=eager
    cache=make_cache();params=cache._level(0)
    try:
        encoded=pickle.dumps(params)
        restored=pickle.loads(encoded)
        assert_public_equal(restored,params)
        report['cases'][label]={'passed':True,'serialized_bytes':len(encoded),
                               'public_controls_exact':True}
    except Exception as error:
        report['cases'][label]={'passed':False,'error':f'{type(error).__name__}: {error}'}
after=hashes()
report.update(source_hashes_after=after,sources_unchanged=before==after,
              passed=before==after and all(row['passed'] for row in report['cases'].values()))
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
raise SystemExit(0 if report['passed'] else 1)
