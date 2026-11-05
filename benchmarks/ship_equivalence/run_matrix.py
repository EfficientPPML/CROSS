"""Run a four-way, serial comparison on one otherwise idle TPU chip."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from compare import compare

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--cross', type=Path, required=True, help='CROSS checkout root')
parser.add_argument('--jaxite-patched', type=Path, required=True)
parser.add_argument('--jaxite-optimized', type=Path, required=True)
parser.add_argument('--manifest', type=Path, required=True)
parser.add_argument('--output-dir', type=Path, required=True)
parser.add_argument('--chip', type=int, default=0)
parser.add_argument('--iterations', type=int, default=3)
args = parser.parse_args()
if args.iterations < 3:
    parser.error('At least three timing samples are required')
directory = args.output_dir.resolve()
directory.mkdir(parents=True, exist_ok=False)
script_dir = Path(__file__).resolve().parent
environment = dict(os.environ, JAX_PLATFORMS='tpu', TPU_VISIBLE_CHIPS=str(args.chip),
                   TPU_PROCESS_BOUNDS='1,1,1', TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1',
                   BOOTSTRAP_LOW_MEM='0', BOOTSTRAP_DC_SUBTRACT='0',
                   JAX_ENABLE_COMPILATION_CACHE='false')
environment.pop('JAX_COMPILATION_CACHE_DIR', None)
reports = []
for variant in ('cross-native', 'cross-ship', 'jaxite-patched', 'jaxite-optimized'):
    paths = [str(args.cross.resolve()/'jaxite_word'), str(script_dir)]
    if variant.startswith('jaxite'):
        paths.append(str(getattr(args, variant.replace('-', '_')).resolve()))
    environment['PYTHONPATH'] = os.pathsep.join(paths)
    output = directory/f'{variant}.json'
    command = [sys.executable, str(script_dir/'run_common.py'), '--manifest',
               str(args.manifest.resolve()), '--variant', variant, '--output', str(output),
               '--iterations', str(args.iterations), '--profile-dir', str(directory/f'{variant}-traces')]
    print(f'Running {variant}; log: {directory / (variant + ".log")}', flush=True)
    with (directory/f'{variant}.log').open('w') as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
    reports.append(json.loads(output.read_text()))
summary = compare(reports)
(directory/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
print(json.dumps(summary, indent=2))
