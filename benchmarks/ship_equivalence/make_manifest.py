"""Generate an explicit common candidate; correctness must be checked separately."""
import argparse
import json
from pathlib import Path

import gmpy2
from composite_prime_gen import composite_prime_gen

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--degree', type=int, required=True)
parser.add_argument('--budget', type=int, default=2)
parser.add_argument('--scale-bits', type=int, choices=range(27, 31), default=30)
parser.add_argument('--first-bits', type=int, choices=(30, 31), default=31)
parser.add_argument('--correction', type=int, default=1)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
n = args.degree
if n < 32 or n & (n - 1):
    parser.error('degree must be a power of two, at least 32')
if not 1 <= args.budget <= (n//2).bit_length()-1:
    parser.error('budget must be between 1 and log2(degree/2)')
q = composite_prime_gen(1, 21 + 2 * args.budget, args.first_bits, args.scale_bits, 2*n, 31)
p = []
candidate = ((1 << args.scale_bits) - 2) // (2*n) * (2*n) + 1
while len(p) < len(q) + 1:
    if candidate not in q and gmpy2.is_prime(candidate):
        p.append(candidate)
    candidate -= 2*n
manifest = dict(degree=n, q=q, p=p, scale=float(q[-1]), dnum=2,
                level_budget=[args.budget]*2, meta_precision=4,
                native_correction=args.correction, weight=min(31, n-1),
                theta=min(9,n//2), gamma=8., seed=42, magnitude=.125,
                error_gate=2**-10)
args.output.write_text(json.dumps(manifest,indent=2)+'\n')
