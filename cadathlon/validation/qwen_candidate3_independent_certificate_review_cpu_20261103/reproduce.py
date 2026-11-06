"""Recompute the exact review using only this archive's public artifacts."""
from pathlib import Path
import argparse,importlib.util,json,shutil,tarfile,tempfile
ROOT=Path(__file__).resolve().parent

def run(output):
    output=Path(output).absolute();output.mkdir(parents=True,exist_ok=False)
    with tempfile.TemporaryDirectory(prefix='qwen-public-certificate-review-') as temporary:
        lab=Path(temporary)
        with tarfile.open(ROOT/'public-selection/sources.tar.gz') as archive:
            members=archive.getmembers()
            if len(members)!=12 or len({m.name for m in members})!=12:raise ValueError('exact public source inventory required')
            for member in members:
                name=Path(member.name)
                if name.is_absolute() or '..' in name.parts or not member.isfile() or not 0<=member.size<=8<<20:raise ValueError('bounded ordinary source required')
                target=lab/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(archive.extractfile(member).read())
        shutil.copytree(ROOT/'public-selection',lab/'source-bound-v1')
        spec=importlib.util.spec_from_file_location('independent_public_certificate_review',ROOT/'review.py')
        review=importlib.util.module_from_spec(spec);spec.loader.exec_module(review);review.LAB=lab;review.OUT=output;review.main()
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.output)
